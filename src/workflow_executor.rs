//! Side-effect-free preparation for declarative workflow transitions.
//!
//! The graph module owns which transitions are legal. This module turns a
//! task's persisted evidence into the guard facts required by the graph, and
//! produces the durable records a caller must commit together. Keeping this
//! step independent from the TUI means web, MCP, and terminal callers cannot
//! disagree about what an admission means.

use anyhow::{bail, Result};
use std::path::Path;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::agent::AgentRegistry;
use crate::config::{MergedConfig, WorkflowPlugin};
use crate::db::{Database, Task, TaskStatus, WorkflowTaskState, WorkflowTransitionRecord};
use crate::git::GitOperations;
use crate::tmux::TmuxOperations;
use crate::tui::app::{
    agtx_task_env, archive_workflow_artifact, build_policy_agent_command,
    ensure_project_tmux_session, ensure_review_addresses_failed_validation, generate_task_slug,
    plan_revision, planning_artifact_path, resolve_prompt, spawn_send_to_agent,
    switch_agent_in_tmux, workflow_artifact_path, workflow_artifact_sha256, workflow_artifact_value,
};
use crate::workflow::{GuardContext, WorkflowDefinition, WorkflowProjectConfig};

/// The records to persist after a task worktree has been created from the
/// frozen admission commit.
#[derive(Debug, Clone)]
pub struct Admission {
    pub state: WorkflowTaskState,
    pub transition: WorkflowTransitionRecord,
}

/// The durable result of any already-admitted workflow transition.
#[derive(Debug, Clone)]
pub struct PreparedTransition {
    pub state: WorkflowTaskState,
    pub transition: WorkflowTransitionRecord,
    /// The agent bound to the role that owns the destination state, if any.
    pub destination_agent: Option<String>,
}

/// Validate and prepare a transition for a task with existing workflow state.
///
/// No agent name or phase name is hard-coded here: the destination state's
/// stable role is resolved through the project's bindings. Callers can use the
/// returned agent to launch or switch an interactive session after persistence.
pub fn prepare_transition(
    workflow: &WorkflowDefinition,
    project: &WorkflowProjectConfig,
    current: &WorkflowTaskState,
    action: &str,
    guards: GuardContext,
) -> Result<PreparedTransition> {
    if current.target_branch != project.target_branch {
        bail!(
            "task '{}' was admitted to '{}', not configured target '{}'",
            current.task_id,
            current.target_branch,
            project.target_branch
        );
    }
    let edge = workflow.validate_transition(&current.state, action, guards)?;
    let destination = workflow
        .state(&edge.to)
        .ok_or_else(|| anyhow::anyhow!("workflow transition '{}' has no destination state", edge.action))?;
    let destination_agent = match &destination.role {
        Some(role) => Some(
            project
                .role_bindings
                .get(role)
                .ok_or_else(|| anyhow::anyhow!("workflow role '{role}' has no agent binding"))?
                .clone(),
        ),
        None => None,
    };

    let mut state = current.clone();
    state.state = edge.to.clone();
    // A new state_attempt identifies this exact entry into the destination
    // state, distinct from any prior entry (e.g. a rework loop back into
    // `engineering_review`). `advance_workflow_state` persists this value
    // verbatim, so every committed transition -- including one that
    // re-enters an already-visited state -- observably increments it by 1.
    state.state_attempt = current.state_attempt + 1;
    state.updated_at = chrono::Utc::now();
    let mut transition = WorkflowTransitionRecord::new(
        &current.task_id,
        &edge.action,
        &edge.from,
        &edge.to,
    );
    transition.actor_role = destination.role.clone();
    transition.actor_agent = destination_agent.clone();

    Ok(PreparedTransition {
        state,
        transition,
        destination_agent,
    })
}

/// Validate and prepare the `admit` transition.
///
/// `base_sha` must already have been resolved with git from the configured
/// target branch. The caller creates the worktree from that SHA before it
/// commits these records; this avoids recording an admission for a worktree
/// that failed to materialize.
pub fn prepare_admission(
    workflow: &WorkflowDefinition,
    project: &WorkflowProjectConfig,
    task: &Task,
    dependencies_resolved: bool,
    base_sha: impl Into<String>,
) -> Result<Admission> {
    if task.worktree_path.is_some() || task.branch_name.is_some() {
        bail!("task '{}' already has a worktree or branch", task.id);
    }

    let base_sha = base_sha.into();
    if base_sha.trim().is_empty() {
        bail!("admission requires a resolved immutable base commit");
    }

    let transition = workflow.validate_transition(
        &workflow.initial_state,
        "admit",
        GuardContext {
            dependencies_resolved,
            ..GuardContext::default()
        },
    )?;

    let mut state = WorkflowTaskState::new(&task.id, &transition.to, &project.target_branch);
    state.base_sha = Some(base_sha);

    let mut record = WorkflowTransitionRecord::new(
        &task.id,
        &transition.action,
        &transition.from,
        &transition.to,
    );
    record.reason = Some("admission base commit frozen before worktree creation".to_string());

    Ok(Admission {
        state,
        transition: record,
    })
}

/// Runtime dependencies shared by the workflow launch functions below.
///
/// These are exactly the non-TUI collaborators the manual `Shift+<key>`
/// handlers in `tui::app` already hold on `AppState`: bundling them here lets
/// each extracted function below take one reference instead of a long
/// parameter list, while still accepting nothing but trait objects/config
/// values -- never `&mut App` or any other TUI type.
pub struct WorkflowRuntime<'a> {
    pub tmux_ops: &'a Arc<dyn TmuxOperations>,
    pub agent_registry: &'a Arc<dyn AgentRegistry>,
    pub git_ops: &'a Arc<dyn GitOperations>,
    pub tmux_project_name: &'a str,
    pub project_path: &'a Path,
    pub config: &'a MergedConfig,
    pub flags: &'a crate::FeatureFlags,
}

/// The outcome of an extracted workflow step.
///
/// Every DB write and agent launch a manual keybinding would have performed
/// has already happened by the time this is returned; the caller (a TUI
/// handler today, an automation driver later) only updates its own
/// bookkeeping from it. A hard failure (malformed evidence, an invalid
/// transition, a failed git/db operation, ...) is still surfaced as `Err`,
/// exactly as the inline handlers propagated it before this extraction.
#[derive(Debug, Clone)]
pub enum WorkflowStepOutcome {
    /// The transition was durably recorded (and the destination agent
    /// launched, where the destination state calls for one). `task` carries
    /// every field the caller must reflect in its own board state; the task
    /// row and the workflow-state/transition-history rows are already
    /// committed to `db`.
    Advanced { task: Task, message: String },
    /// A precondition was not met (missing/malformed artifact, task not in
    /// the expected state, missing worktree/session, ...). No durable write
    /// happened. `message` is the exact operator-facing text the manual
    /// handler surfaced as its warning toast.
    Blocked { message: String },
    /// The manual handler would have silently returned without comment (for
    /// example: no persisted workflow state exists yet for this task). No
    /// durable write happened, and there is nothing to show the operator.
    NoOp,
}

/// Extracted body of `App::admit_selected_task`.
///
/// Freezes the admission base commit, creates the task worktree, and
/// persists the admission evidence. Callers resolve `plugin`/`workflow`/
/// `project_workflow` and check the task's own status/worktree preconditions
/// first, exactly as the manual handler's own early guards did.
pub fn admit_task(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let dependencies_resolved = db.deps_satisfied(&task);
    let base_sha = match crate::git::resolve_commit(runtime.project_path, &project_workflow.target_branch) {
        Ok(sha) => sha,
        Err(error) => {
            return Ok(WorkflowStepOutcome::Blocked {
                message: format!("Cannot admit task: {error}"),
            });
        }
    };
    let admission = match prepare_admission(workflow, project_workflow, &task, dependencies_resolved, base_sha.clone()) {
        Ok(admission) => admission,
        Err(error) => {
            return Ok(WorkflowStepOutcome::Blocked {
                message: format!("Cannot admit task: {error}"),
            });
        }
    };

    let slug = generate_task_slug(&task.id, &task.title);
    let worktree_path = match runtime.git_ops.create_worktree(
        runtime.project_path,
        &slug,
        &base_sha,
        &runtime.config.worktree_dir,
        &runtime.config.branch_prefix,
    ) {
        Ok(path) => path,
        Err(error) => {
            return Ok(WorkflowStepOutcome::Blocked {
                message: format!("Cannot create admission worktree: {error}"),
            });
        }
    };

    let copy_files = match (&runtime.config.copy_files, plugin.copy_files.is_empty()) {
        (Some(project_files), false) if !project_files.trim().is_empty() => {
            Some(format!("{project_files},{}", plugin.copy_files.join(",")))
        }
        (Some(project_files), _) if !project_files.trim().is_empty() => Some(project_files.clone()),
        (_, false) => Some(plugin.copy_files.join(",")),
        _ => None,
    };
    let init_script = if runtime.flags.no_init_scripts {
        None
    } else {
        runtime.config.init_script.clone()
    };
    let _warnings = runtime.git_ops.initialize_worktree(
        runtime.project_path,
        Path::new(&worktree_path),
        copy_files,
        init_script,
        plugin.copy_dirs.clone(),
    );

    task.worktree_path = Some(worktree_path.clone());
    task.branch_name = Some(format!("{}/{}", runtime.config.branch_prefix, slug));
    task.base_branch = Some(project_workflow.target_branch.clone());
    task.updated_at = chrono::Utc::now();

    if let Err(error) = db.record_workflow_admission(&task, &admission.state, &admission.transition) {
        let _ = runtime.git_ops.remove_worktree(runtime.project_path, &worktree_path);
        if let Some(branch) = &task.branch_name {
            let _ = runtime.git_ops.delete_branch(runtime.project_path, branch);
        }
        return Ok(WorkflowStepOutcome::Blocked {
            message: format!("Admission was rolled back: {error}"),
        });
    }

    Ok(WorkflowStepOutcome::Advanced {
        message: format!(
            "Admitted at {} — ready for planning",
            &base_sha[..base_sha.len().min(12)]
        ),
        task,
    })
}

/// Standalone execution of the `admission_complete` transition
/// (`admission` -> `ready_for_planning`).
///
/// No artifact, no agent launch, no `TaskStatus` change — this is a pure
/// dependency-projection step, safe for automation to perform on its own.
/// The manual `Shift+S` keybinding never calls this directly: it fuses this
/// same transition together with `start_planning` inside
/// [`start_workflow_planning`], since a human pressing Shift+S wants the
/// planner launched immediately. Automation is different: it stops here,
/// leaving a task visibly parked in `ready_for_planning` (still `Backlog` on
/// the board) until a human deliberately commits it into `planning`.
pub fn complete_admission(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    task: Task,
    db: &mut Database,
) -> Result<WorkflowStepOutcome> {
    let Some(current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    if current.state != "admission" {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "admission_complete is only available in Admission".into(),
        });
    }
    let ready = prepare_transition(
        workflow,
        project_workflow,
        &current,
        "admission_complete",
        GuardContext {
            admission_recorded: true,
            ..GuardContext::default()
        },
    )?;
    db.advance_workflow_state(&ready.state, &ready.transition)?;
    Ok(WorkflowStepOutcome::Advanced {
        message: "Admission complete — ready for planning".to_string(),
        task,
    })
}

/// Extracted body of `App::start_selected_workflow_planning`.
///
/// Planning is deliberately restartable: a task can retain its durable
/// admission evidence while a terminal or agent process exits, in which case
/// this relaunches the planner rather than merely realigning state.
pub fn start_workflow_planning(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(worktree) = task.worktree_path.clone() else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Admit the task before starting planning".into(),
        });
    };
    let Some(current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };

    let restarting = current.state == "planning";
    let (planner, planning_state, planning_attempt, transitions) = if restarting {
        let Some(role) = workflow.state("planning").and_then(|state| state.role.as_ref()) else {
            return Ok(WorkflowStepOutcome::Blocked {
                message: "Planning state has no workflow role".into(),
            });
        };
        let Some(agent) = project_workflow.role_bindings.get(role).cloned() else {
            return Ok(WorkflowStepOutcome::Blocked {
                message: format!("Workflow role '{role}' has no agent binding"),
            });
        };
        (agent, current.state.clone(), current.state_attempt, None)
    } else {
        // `current.state` is normally `ready_for_planning` here: automation
        // (or a prior manual `Shift+S`) has already driven `admission ->
        // ready_for_planning` on its own via `complete_admission`. A task
        // can still be found sitting in `admission` itself (automation
        // disabled, or this call races a not-yet-run automation tick), so
        // both starting points are handled by chaining only the transitions
        // actually needed from wherever the task currently is — never
        // assuming `admission` unconditionally.
        let ready = if current.state == "admission" {
            match prepare_transition(
                workflow,
                project_workflow,
                &current,
                "admission_complete",
                GuardContext {
                    admission_recorded: true,
                    ..GuardContext::default()
                },
            ) {
                Ok(value) => Some(value),
                Err(error) => {
                    return Ok(WorkflowStepOutcome::Blocked {
                        message: format!("Cannot start planning: {error}"),
                    });
                }
            }
        } else {
            None
        };
        let before_planning = ready.as_ref().map(|ready| &ready.state).unwrap_or(&current);
        let planning = match prepare_transition(workflow, project_workflow, before_planning, "start_planning", GuardContext::default()) {
            Ok(value) => value,
            Err(error) => {
                return Ok(WorkflowStepOutcome::Blocked {
                    message: format!("Cannot start planning: {error}"),
                });
            }
        };
        let Some(agent) = planning.destination_agent.clone() else {
            return Ok(WorkflowStepOutcome::Blocked {
                message: "Planning state has no bound agent".into(),
            });
        };
        let mut transitions = Vec::with_capacity(2);
        if let Some(ready) = ready {
            transitions.push(ready);
        }
        transitions.push(planning.clone());
        (agent, planning.state.state.clone(), planning.state.state_attempt, Some(transitions))
    };

    let agent_ops = runtime.agent_registry.get(&planner);
    let prompt = format!(
        "{}\n\nCurrent workflow attempt: {n}. Your output artifact MUST contain the line: workflow_attempt: {n}",
        resolve_prompt(&Some(plugin.clone()), "planning", &task.content_text(), &task.id, task.cycle),
        n = planning_attempt,
    );
    let slug = generate_task_slug(&task.id, &task.title);
    let window_name = format!("task-{slug}");
    let target = format!("{}:{window_name}", runtime.tmux_project_name);
    ensure_project_tmux_session(runtime.tmux_project_name, runtime.project_path, runtime.tmux_ops.as_ref());
    let policy = project_workflow.policy_for_state(workflow, &planning_state)?;
    let command = build_policy_agent_command(agent_ops.as_ref(), &planner, &prompt, policy.as_ref(), Some(Path::new(&worktree)));

    if let Some(transitions) = &transitions {
        for prepared in transitions {
            db.advance_workflow_state(&prepared.state, &prepared.transition)?;
        }
    }

    if restarting && runtime.tmux_ops.window_exists(&target).unwrap_or(false) {
        switch_agent_in_tmux(runtime.tmux_ops.as_ref(), &target, &task.agent, &command);
    } else {
        runtime.tmux_ops.create_window(
            runtime.tmux_project_name,
            &window_name,
            &worktree,
            Some(command),
            true,
            &agtx_task_env(&task.id, &worktree),
        )?;
    }

    task.status = TaskStatus::Planning;
    task.agent = planner;
    task.session_name = Some(target);
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;

    let message = if restarting {
        "Planning session relaunched; save the required plan artifact before Shift+V"
    } else {
        "Planning started in admitted worktree"
    }
    .to_string();
    Ok(WorkflowStepOutcome::Advanced { task, message })
}

/// Extracted body of `App::submit_selected_workflow_plan`.
///
/// Hashes the saved planning artifact, records it durably, and hands the
/// exact revision to the role bound to `plan_reviewer`.
pub fn submit_workflow_plan(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(worktree) = task.worktree_path.clone() else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Planning requires an admitted worktree".into(),
        });
    };
    let Some(current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    if current.state != "planning" {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Submit plan is only available in Planning".into(),
        });
    }
    let path = planning_artifact_path(&worktree, plugin, &task.id)?;
    let contents = std::fs::read(&path).map_err(|_| anyhow::anyhow!("Missing planning artifact: {}", path.display()))?;
    let revision = plan_revision(&contents).ok_or_else(|| {
        anyhow::anyhow!("Planning artifact must contain 'plan_revision: <positive integer>'")
    })?;
    if revision <= current.plan_revision {
        anyhow::bail!(
            "Plan revision {revision} is not newer than recorded revision {}",
            current.plan_revision
        );
    }
    let mut evidenced = current;
    evidenced.plan_revision = revision;
    evidenced.plan_hash = Some(format!("{:x}", Sha256::digest(&contents)));
    let handoff = prepare_transition(workflow, project_workflow, &evidenced, "submit_plan", GuardContext::default())?;
    let reviewer = handoff
        .destination_agent
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Plan review state has no bound agent"))?;
    let prompt = format!(
        "You are the plan reviewer for task {}. Review only {} (revision {}, SHA-256 {}). Do not implement code. Check it against the task, identify concrete changes if needed, then leave your decision for the operator: approve with Shift+Y or request changes with Shift+N.\n\nCurrent workflow attempt: {n}. Your output artifact MUST contain the line: workflow_attempt: {n}",
        task.id,
        path.strip_prefix(&worktree).unwrap_or(&path).display(),
        revision,
        evidenced.plan_hash.as_deref().unwrap_or_default(),
        n = handoff.state.state_attempt,
    );
    let target = task
        .session_name
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Planning session is unavailable"))?;
    let previous_agent = task.agent.clone();
    db.advance_workflow_state(&handoff.state, &handoff.transition)?;
    let policy = project_workflow.policy_for_state(workflow, &handoff.state.state)?;
    if let Some(policy) = policy.as_ref() {
        let command = build_policy_agent_command(
            runtime.agent_registry.get(&reviewer).as_ref(),
            &reviewer,
            &prompt,
            Some(policy),
            Some(Path::new(&worktree)),
        );
        switch_agent_in_tmux(runtime.tmux_ops.as_ref(), &target, &previous_agent, &command);
    } else {
        spawn_send_to_agent(
            Arc::clone(runtime.tmux_ops),
            Arc::clone(runtime.agent_registry),
            task.id.clone(),
            runtime.config.agent_hooks,
            runtime.config.auto_trust,
            target,
            previous_agent,
            reviewer.clone(),
            true,
            None,
            None,
            prompt,
            None,
            task.content_text(),
            Vec::new(),
            task.worktree_path.clone(),
            runtime.project_path.to_path_buf(),
            Some(plugin.clone()),
        );
    }
    task.status = TaskStatus::Review;
    task.agent = reviewer;
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;
    Ok(WorkflowStepOutcome::Advanced {
        message: format!("Plan revision {revision} submitted for review"),
        task,
    })
}

/// Extracted body of `App::decide_selected_workflow_plan`.
///
/// Persists the operator's reviewer decision. Approval freezes the exact
/// recorded hash; request-changes returns ownership to the configured
/// planner.
pub fn decide_workflow_plan(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    approve: bool,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(mut current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    if current.state != "plan_review" {
        return Ok(WorkflowStepOutcome::NoOp);
    }
    let action = if approve { "approve_plan" } else { "plan_changes_requested" };
    if approve {
        current.approved_plan_revision = Some(current.plan_revision);
        current.approved_plan_hash = current.plan_hash.clone();
    }
    let decision = prepare_transition(
        workflow,
        project_workflow,
        &current,
        action,
        GuardContext {
            approved_plan: approve && current.plan_hash.is_some(),
            ..GuardContext::default()
        },
    )?;
    let previous_agent = task.agent.clone();
    task.status = TaskStatus::Planning;
    task.agent = decision.destination_agent.clone().unwrap_or(task.agent);
    task.updated_at = chrono::Utc::now();
    db.advance_workflow_state(&decision.state, &decision.transition)?;
    db.update_task(&task)?;
    if !approve {
        if let Some(target) = task.session_name.clone() {
            let prompt = format!(
                "Plan review requested changes for task {}. Revise .agtx/plans/{}.md, increment plan_revision above {}, and do not implement code. When complete, save the artifact for another Shift+V submission.\n\nCurrent workflow attempt: {n}. Your output artifact MUST contain the line: workflow_attempt: {n}",
                task.id, task.id, current.plan_revision, n = decision.state.state_attempt
            );
            spawn_send_to_agent(
                Arc::clone(runtime.tmux_ops),
                Arc::clone(runtime.agent_registry),
                task.id.clone(),
                runtime.config.agent_hooks,
                runtime.config.auto_trust,
                target,
                previous_agent,
                task.agent.clone(),
                true,
                None,
                None,
                prompt,
                None,
                task.content_text(),
                Vec::new(),
                task.worktree_path.clone(),
                runtime.project_path.to_path_buf(),
                Some(plugin.clone()),
            );
        }
    }
    let message = if approve {
        "Plan approved"
    } else {
        "Plan changes requested; returned to Planning"
    }
    .to_string();
    Ok(WorkflowStepOutcome::Advanced { task, message })
}

/// Extracted body of `App::start_selected_workflow_implementation`.
///
/// Starts the bound implementer only after an exact plan revision and hash
/// have been approved by the workflow evidence store. Implementation is
/// restartable from durable state the same way planning is: a lost tmux
/// window is recovered rather than treated as a fresh launch.
pub fn start_workflow_implementation(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(worktree) = task.worktree_path.clone() else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Implementation requires an admitted worktree".into(),
        });
    };
    let Some(current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    let implementation = prepare_transition(
        workflow,
        project_workflow,
        &current,
        "start_implementation",
        GuardContext {
            approved_plan: current.approved_plan_revision == Some(current.plan_revision)
                && current.approved_plan_hash == current.plan_hash
                && current.plan_hash.is_some(),
            ..GuardContext::default()
        },
    )?;
    let implementer = implementation
        .destination_agent
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Implementation state has no bound agent"))?;
    let approved_revision = current.approved_plan_revision.unwrap_or_default();
    let prompt = format!(
        "{}\n\nApproved plan revision: {}\nApproved plan SHA-256: {}\n\nCurrent workflow attempt: {n}. Your output artifact MUST contain the line: workflow_attempt: {n}",
        resolve_prompt(&Some(plugin.clone()), "running", &task.content_text(), &task.id, task.cycle),
        approved_revision,
        current.approved_plan_hash.as_deref().unwrap_or_default(),
        n = implementation.state.state_attempt,
    );
    let policy = project_workflow.policy_for_state(workflow, &implementation.state.state)?;
    let command = build_policy_agent_command(
        runtime.agent_registry.get(&implementer).as_ref(),
        &implementer,
        &prompt,
        policy.as_ref(),
        Some(Path::new(&worktree)),
    );
    let existing_target = task.session_name.clone();
    let session_available = existing_target
        .as_ref()
        .is_some_and(|target| runtime.tmux_ops.window_exists(target).unwrap_or(false));
    let slug = generate_task_slug(&task.id, &task.title);
    let window_name = format!("task-{slug}");
    let target = if session_available {
        existing_target.expect("checked above")
    } else {
        format!("{}:{window_name}", runtime.tmux_project_name)
    };
    if session_available {
        switch_agent_in_tmux(runtime.tmux_ops.as_ref(), &target, &task.agent, &command);
    } else {
        ensure_project_tmux_session(runtime.tmux_project_name, runtime.project_path, runtime.tmux_ops.as_ref());
        runtime.tmux_ops.create_window(
            runtime.tmux_project_name,
            &window_name,
            &worktree,
            Some(command),
            true,
            &agtx_task_env(&task.id, &worktree),
        )?;
    }
    db.advance_workflow_state(&implementation.state, &implementation.transition)?;
    task.status = TaskStatus::Running;
    task.agent = implementer;
    task.session_name = Some(target);
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;
    Ok(WorkflowStepOutcome::Advanced {
        message: "Implementation started from the approved plan".into(),
        task,
    })
}

/// Extracted body of `App::submit_selected_workflow_implementation`.
///
/// Validates the implementer's durable result before handing the worktree to
/// the configured engineering reviewer.
pub fn submit_workflow_implementation(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(worktree) = task.worktree_path.clone() else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    let Some(current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    let artifact = Path::new(&worktree).join(".agent-flow/implementation-result.yaml");
    if !artifact.is_file() {
        return Ok(WorkflowStepOutcome::Blocked {
            message: format!("Missing implementation evidence: {}", artifact.display()),
        });
    }
    let implemented = prepare_transition(
        workflow,
        project_workflow,
        &current,
        "implementation_complete",
        GuardContext {
            implementation_recorded: true,
            ..GuardContext::default()
        },
    )?;
    let review = prepare_transition(workflow, project_workflow, &implemented.state, "start_engineering_review", GuardContext::default())?;
    let reviewer = review
        .destination_agent
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Engineering review state has no bound agent"))?;
    let prompt = format!(
        "{}\n\nCurrent workflow attempt: {n}. Your output artifact MUST contain the line: workflow_attempt: {n}",
        resolve_prompt(&Some(plugin.clone()), "review", &task.content_text(), &task.id, task.cycle),
        n = review.state.state_attempt,
    );
    let target = task
        .session_name
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Task session is unavailable"))?;
    let policy = project_workflow.policy_for_state(workflow, &review.state.state)?;
    let command = build_policy_agent_command(
        runtime.agent_registry.get(&reviewer).as_ref(),
        &reviewer,
        &prompt,
        policy.as_ref(),
        Some(Path::new(&worktree)),
    );
    db.advance_workflow_state_chain(&[
        (&implemented.state, &implemented.transition),
        (&review.state, &review.transition),
    ])?;
    switch_agent_in_tmux(runtime.tmux_ops.as_ref(), &target, &task.agent, &command);
    task.status = TaskStatus::Review;
    task.agent = reviewer;
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;
    Ok(WorkflowStepOutcome::Advanced {
        message: "Implementation evidence accepted; engineering review started".into(),
        task,
    })
}

/// Extracted body of `App::submit_selected_engineering_review`.
///
/// Consumes the engineering reviewer's durable verdict and hands the task to
/// the role that owns the next declared workflow state.
pub fn submit_engineering_review(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(worktree) = task.worktree_path.clone() else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    let Some(current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    if current.state != "engineering_review" {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Submit engineering review is only available in Engineering review".into(),
        });
    }
    let artifact = workflow_artifact_path(&worktree, plugin.artifacts.review.as_deref(), &task.id, ".agent-flow/engineering-review.yaml");
    let verdict = workflow_artifact_value(&artifact, "verdict")?;
    ensure_review_addresses_failed_validation(&worktree, plugin, &task.id, &artifact, &verdict)?;
    let (action, phase, status) = match verdict.as_str() {
        "corrections_required" => ("engineering_corrections_required", "running", TaskStatus::Running),
        "plan_issue" => ("engineering_plan_issue", "planning", TaskStatus::Planning),
        "approved_for_validation" => ("start_final_validation", "final_validation", TaskStatus::Review),
        _ => bail!("{} has unsupported engineering-review verdict '{verdict}'", artifact.display()),
    };
    let transition = prepare_transition(workflow, project_workflow, &current, action, GuardContext::default())?;
    let next_agent = transition
        .destination_agent
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Workflow state has no bound agent"))?;
    let target = task
        .session_name
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Task session is unavailable"))?;
    let prompt = format!(
        "{}\n\nEngineering-review verdict: {verdict}. Evidence: {}. Follow the declared role policy; do not commit, push, create a PR, merge, or bypass controls.\n\nCurrent workflow attempt: {n}. Your output artifact MUST contain the line: workflow_attempt: {n}",
        resolve_prompt(&Some(plugin.clone()), phase, &task.content_text(), &task.id, task.cycle),
        artifact.strip_prefix(&worktree).unwrap_or(&artifact).display(),
        n = transition.state.state_attempt,
    );
    let policy = project_workflow.policy_for_state(workflow, &transition.state.state)?;
    let command = build_policy_agent_command(
        runtime.agent_registry.get(&next_agent).as_ref(),
        &next_agent,
        &prompt,
        policy.as_ref(),
        Some(Path::new(&worktree)),
    );
    db.advance_workflow_state(&transition.state, &transition.transition)?;
    switch_agent_in_tmux(runtime.tmux_ops.as_ref(), &target, &task.agent, &command);
    task.status = status;
    task.agent = next_agent;
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;
    Ok(WorkflowStepOutcome::Advanced {
        message: format!("Engineering review recorded: {verdict}"),
        task,
    })
}

/// Extracted body of `App::submit_selected_final_validation`.
///
/// Records the final-validation artifact. A pass hands control to the
/// reviewer-owned integration state; a failure returns to engineering
/// review, archiving the superseded review artifact so a stale approval
/// cannot be reused.
pub fn submit_final_validation(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(worktree) = task.worktree_path.clone() else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    let Some(mut current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    if current.state != "final_validation" {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Submit final validation is only available in Final validation".into(),
        });
    }
    let artifact = workflow_artifact_path(&worktree, plugin.artifacts.final_validation.as_deref(), &task.id, ".agent-flow/final-validation.yaml");
    let verdict = workflow_artifact_value(&artifact, "verdict")?;
    let (action, phase, passed) = match verdict.as_str() {
        "passed" => ("begin_feature_integration", "integration", true),
        "failed" => ("validation_failed", "review", false),
        _ => bail!("{} has unsupported final-validation verdict '{verdict}'", artifact.display()),
    };
    if passed {
        current.validation_passed_at = Some(chrono::Utc::now());
    }
    let transition = prepare_transition(
        workflow,
        project_workflow,
        &current,
        action,
        GuardContext {
            final_validation_passed: passed,
            clean_worktree: true,
            ..GuardContext::default()
        },
    )?;
    let next_agent = transition
        .destination_agent
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Workflow state has no bound agent"))?;
    let target = task
        .session_name
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Task session is unavailable"))?;
    let prompt = format!(
        "{}\n\nFinal-validation verdict: {verdict}. Evidence: {}.{} Follow the declared role policy; do not merge feature/poc into main.\n\nCurrent workflow attempt: {n}. Your output artifact MUST contain the line: workflow_attempt: {n}",
        resolve_prompt(&Some(plugin.clone()), phase, &task.content_text(), &task.id, task.cycle),
        artifact.strip_prefix(&worktree).unwrap_or(&artifact).display(),
        if passed {
            String::new()
        } else {
            format!(
                " A previous validation failure is an active gate: read this exact evidence and write a fresh engineering review. Your review must include validation_failure_sha256: {} and validation_failure_resolution: <what you verified or changed>. Choose the normal engineering-review verdict: corrections_required for unresolved source/test failures, plan_issue for a material plan defect, or approved_for_validation only when a repeat validation is justified.",
                workflow_artifact_sha256(&artifact)?,
            )
        },
        n = transition.state.state_attempt,
    );
    let policy = project_workflow.policy_for_state(workflow, &transition.state.state)?;
    let command = build_policy_agent_command(
        runtime.agent_registry.get(&next_agent).as_ref(),
        &next_agent,
        &prompt,
        policy.as_ref(),
        Some(Path::new(&worktree)),
    );
    if !passed {
        let review_artifact = workflow_artifact_path(&worktree, plugin.artifacts.review.as_deref(), &task.id, ".agent-flow/engineering-review.yaml");
        archive_workflow_artifact(&review_artifact, "superseded-after-validation-failure")?;
    }
    db.advance_workflow_state(&transition.state, &transition.transition)?;
    switch_agent_in_tmux(runtime.tmux_ops.as_ref(), &target, &task.agent, &command);
    task.status = TaskStatus::Review;
    task.agent = next_agent;
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;
    Ok(WorkflowStepOutcome::Advanced {
        message: format!("Final validation recorded: {verdict}"),
        task,
    })
}

/// Extracted body of `App::complete_selected_feature_integration`.
///
/// Executes the narrowly-scoped, reviewer-authorized task integration. The
/// configured target must already be checked out and may never be `main`.
pub fn complete_feature_integration(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(worktree) = task.worktree_path.clone() else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    let Some(branch) = task.branch_name.clone() else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    let Some(current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    if current.state != "integrate_to_feature" {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Complete integration is only available in Integrate to feature".into(),
        });
    }
    let artifact = workflow_artifact_path(&worktree, plugin.artifacts.integration.as_deref(), &task.id, ".agent-flow/integration-ready.yaml");
    if workflow_artifact_value(&artifact, "verdict")? != "ready_for_integration" {
        bail!("{} must declare verdict: ready_for_integration", artifact.display());
    }
    let policy = project_workflow
        .policy_for_state(workflow, &current.state)?
        .ok_or_else(|| anyhow::anyhow!("Integration state has no role policy"))?;
    let target_branch = policy
        .merge_target
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Integration state has no merge target"))?;
    if target_branch != current.target_branch || target_branch == "main" {
        bail!("workflow integration may only target the admitted non-main branch");
    }
    if !(policy.role_policy.final_task_commit && policy.role_policy.push_task_branch && policy.role_policy.merge_task_into_target) {
        bail!("Integration state policy does not authorize commit, push, and target merge");
    }
    if runtime.git_ops.has_changes(runtime.project_path) {
        bail!("configured target checkout has uncommitted changes; integration is refused");
    }
    if crate::git::current_branch(runtime.project_path)? != target_branch {
        bail!("configured target checkout is not on '{target_branch}'; integration is refused");
    }
    let (has_conflicts, files) = crate::git::check_merge_conflicts(runtime.project_path, &target_branch, &branch)?;
    if has_conflicts {
        bail!("task branch conflicts with '{target_branch}': {}", files.join(", "));
    }
    if runtime.git_ops.has_changes(Path::new(&worktree)) {
        runtime.git_ops.add_all(Path::new(&worktree))?;
        runtime.git_ops.commit(Path::new(&worktree), &format!("workflow: complete task {}", task.id))?;
    }
    runtime.git_ops.push(Path::new(&worktree), &branch, true)?;
    crate::git::merge_branch(runtime.project_path, &branch, &format!("workflow: integrate task {}", task.id))?;
    runtime.git_ops.push(runtime.project_path, &target_branch, false)?;
    let integration_sha = crate::git::resolve_commit(runtime.project_path, &target_branch)?;
    let completed = prepare_transition(
        workflow,
        project_workflow,
        &current,
        "complete_feature_integration",
        GuardContext {
            integrated_into_target: true,
            ..GuardContext::default()
        },
    )?;
    let mut completed_state = completed.state;
    completed_state.integration_sha = Some(integration_sha);
    db.advance_workflow_state(&completed_state, &completed.transition)?;
    task.status = TaskStatus::Done;
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;
    Ok(WorkflowStepOutcome::Advanced {
        message: format!("Task integrated into {target_branch}"),
        task,
    })
}


/// What automation should do next for a task, computed from durable
/// evidence alone: the workflow graph, the project's guard-relevant facts,
/// and whatever the destination agent has (or has not) written to disk.
///
/// This mirrors the operator's own read of the board -- "is there something
/// for me to look at, or is nothing new here yet" -- without ever performing
/// a write itself. A later automation driver decides what to *do* with a
/// `Advance`; this only decides what the evidence currently means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutomationDecision {
    /// Nothing actionable yet: no legal transition, an agent state with no
    /// artifact on disk yet, or an artifact left over from a previous entry
    /// into this same state (a stale `workflow_attempt`). Automation should
    /// simply check again later.
    Wait,
    /// The named action is ready to fire.
    Advance(String),
    /// An artifact exists, matches the task's current `state_attempt`, but
    /// cannot be trusted as evidence: bad YAML, a missing required field, or
    /// (for planning) a `plan_revision` that has not actually advanced.
    InvalidArtifact(String),
    /// Evidence is valid but this specific outcome is never auto-advanced.
    /// Currently only final validation's `failed` verdict: a human must look
    /// before any rework loop restarts, by fixed rule rather than project
    /// configuration.
    HumanGate(String),
}

/// Facts about `task`/`state` needed to evaluate the graph's declared
/// guards, computed the same way the extracted launch functions above
/// already compute them for their own action -- reused here rather than
/// re-derived, since `assess` has no specific action in mind and instead
/// asks the graph which actions are currently legal at all.
pub fn guard_context_for(db: &Database, task: &Task, state: &WorkflowTaskState) -> GuardContext {
    let implementation_recorded = task
        .worktree_path
        .as_deref()
        .map(|worktree| Path::new(worktree).join(".agent-flow/implementation-result.yaml").is_file())
        .unwrap_or(false);
    GuardContext {
        dependencies_resolved: db.deps_satisfied(task),
        // Reaching any durable workflow state at all implies admission was
        // already recorded; `admit_task` is the only path that creates one.
        admission_recorded: task.worktree_path.is_some(),
        // Same formula `start_workflow_implementation` uses to gate
        // `start_implementation`: the recorded approval must name the exact
        // plan revision and hash currently on file, not merely "some"
        // earlier approval.
        approved_plan: state.approved_plan_revision == Some(state.plan_revision)
            && state.approved_plan_hash == state.plan_hash
            && state.plan_hash.is_some(),
        implementation_recorded,
        final_validation_passed: state.validation_passed_at.is_some(),
        // Git-dependent guards that `assess` cannot verify without shelling
        // out are intentionally left at their default (false/unproven); no
        // state this function currently understands is gated on them.
        clean_worktree: false,
        integrated_into_target: false,
    }
}

/// Read `workflow_attempt` from a workflow evidence file the same way
/// `workflow_artifact_value` reads `verdict`/`plan_revision`. `None` covers
/// both "the field is absent" and "the value does not parse as an
/// integer" -- both are treated identically by `assess`: a fresh agent
/// session simply has not written a valid attempt marker yet.
fn artifact_workflow_attempt(path: &Path) -> Option<i64> {
    workflow_artifact_value(path, "workflow_attempt")
        .ok()?
        .parse::<i64>()
        .ok()
}

/// Shared freshness gate for every artifact-backed state: `None` means the
/// artifact is present and stamped with the task's current `state_attempt`,
/// so the caller should go on to interpret its contents. `Some(decision)` is
/// the answer `assess` should return immediately, without reading further.
fn artifact_freshness(artifact: &Path, state: &WorkflowTaskState) -> Option<AutomationDecision> {
    if !artifact.is_file() {
        return Some(AutomationDecision::Wait);
    }
    match artifact_workflow_attempt(artifact) {
        Some(attempt) if attempt == state.state_attempt => None,
        // Missing/unparseable attempt field, or an attempt number left over
        // from a previous entry into this state: treated exactly like a
        // missing file, never as an error.
        _ => Some(AutomationDecision::Wait),
    }
}

/// Determine what automation should do about `task`, currently sitting in
/// `state`, given the declarative graph, the project's role/guard
/// configuration, and the plugin's artifact locations.
///
/// Pure DB-row + filesystem + graph logic: no TUI, tmux, or agent process is
/// touched. `db` is read-only here (`deps_satisfied` needs to see other
/// tasks' status); nothing is written.
pub fn assess(
    workflow: &WorkflowDefinition,
    _project: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    task: &Task,
    state: &WorkflowTaskState,
    db: &Database,
) -> AutomationDecision {
    let guards = guard_context_for(db, task, state);
    let available = workflow.available_transitions(&state.state, guards);
    if available.is_empty() {
        return AutomationDecision::Wait;
    }

    let Some(workflow_state) = workflow.state(&state.state) else {
        return AutomationDecision::Wait;
    };

    if workflow_state.role.is_none() {
        // No agent owns this state, so there is no artifact to wait for.
        // Only advance on a transition that is actually guard-gated (real
        // computed evidence, e.g. dependencies_resolved); a transition with
        // no guards at all (e.g. `ready_for_planning` -> `start_planning`)
        // is exclusively human-initiated and carries no automation signal
        // to act on, even though the graph trivially permits it.
        return match available.iter().find(|transition| !transition.guards.is_empty()) {
            Some(transition) => AutomationDecision::Advance(transition.action.clone()),
            None => AutomationDecision::Wait,
        };
    }

    let Some(worktree) = task.worktree_path.as_deref() else {
        return AutomationDecision::Wait;
    };

    match state.state.as_str() {
        "planning" => assess_planning(worktree, plugin, task, state),
        "engineering_review" => assess_engineering_review(worktree, plugin, task, state),
        "final_validation" => assess_final_validation(worktree, plugin, task, state),
        "implementing" | "running" => assess_implementation(worktree, state),
        // A role state this function does not yet know an artifact mapping
        // for (e.g. `plan_review`, which is decided by an operator keybind
        // rather than an artifact). Nothing to read, so nothing to report.
        _ => AutomationDecision::Wait,
    }
}

/// Mirrors `submit_workflow_plan`'s own evidence check: a plan artifact is
/// only meaningful once its `plan_revision` has actually moved past the
/// last recorded one.
fn assess_planning(worktree: &str, plugin: &WorkflowPlugin, task: &Task, state: &WorkflowTaskState) -> AutomationDecision {
    let path = match planning_artifact_path(worktree, plugin, &task.id) {
        Ok(path) => path,
        Err(_) => return AutomationDecision::Wait,
    };
    if let Some(decision) = artifact_freshness(&path, state) {
        return decision;
    }
    let contents = match std::fs::read(&path) {
        Ok(contents) => contents,
        Err(_) => return AutomationDecision::Wait,
    };
    let Some(revision) = plan_revision(&contents) else {
        return AutomationDecision::InvalidArtifact(
            "Planning artifact must contain 'plan_revision: <positive integer>'".to_string(),
        );
    };
    if revision <= state.plan_revision {
        return AutomationDecision::InvalidArtifact(format!(
            "Plan revision {revision} is not newer than recorded revision {}",
            state.plan_revision
        ));
    }
    AutomationDecision::Advance("submit_plan".to_string())
}

/// Mirrors `submit_engineering_review`'s verdict-to-action match exactly.
fn assess_engineering_review(worktree: &str, plugin: &WorkflowPlugin, task: &Task, state: &WorkflowTaskState) -> AutomationDecision {
    let artifact = workflow_artifact_path(worktree, plugin.artifacts.review.as_deref(), &task.id, ".agent-flow/engineering-review.yaml");
    if let Some(decision) = artifact_freshness(&artifact, state) {
        return decision;
    }
    let verdict = match workflow_artifact_value(&artifact, "verdict") {
        Ok(verdict) => verdict,
        Err(error) => return AutomationDecision::InvalidArtifact(error.to_string()),
    };
    let action = match verdict.as_str() {
        "corrections_required" => "engineering_corrections_required",
        "plan_issue" => "engineering_plan_issue",
        "approved_for_validation" => "start_final_validation",
        other => {
            return AutomationDecision::InvalidArtifact(format!(
                "{} has unsupported engineering-review verdict '{other}'",
                artifact.display()
            ))
        }
    };
    AutomationDecision::Advance(action.to_string())
}

/// Mirrors `submit_final_validation`'s verdict-to-action match, with one
/// fixed override: a `failed` verdict is always a `HumanGate`, never an
/// automatic `Advance("validation_failed")`. This is a hard automation-safety
/// rule, not a project-configurable choice -- a failed validation always
/// needs a human look before any rework loop restarts.
fn assess_final_validation(worktree: &str, plugin: &WorkflowPlugin, task: &Task, state: &WorkflowTaskState) -> AutomationDecision {
    let artifact = workflow_artifact_path(worktree, plugin.artifacts.final_validation.as_deref(), &task.id, ".agent-flow/final-validation.yaml");
    if let Some(decision) = artifact_freshness(&artifact, state) {
        return decision;
    }
    let verdict = match workflow_artifact_value(&artifact, "verdict") {
        Ok(verdict) => verdict,
        Err(error) => return AutomationDecision::InvalidArtifact(error.to_string()),
    };
    match verdict.as_str() {
        "passed" => AutomationDecision::Advance("begin_feature_integration".to_string()),
        "failed" => AutomationDecision::HumanGate("final validation failed".to_string()),
        other => AutomationDecision::InvalidArtifact(format!(
            "{} has unsupported final-validation verdict '{other}'",
            artifact.display()
        )),
    }
}

/// Mirrors `submit_workflow_implementation`'s evidence check: the
/// implementer's result file has no verdict of its own, it is either
/// present (and fresh) or it is not.
fn assess_implementation(worktree: &str, state: &WorkflowTaskState) -> AutomationDecision {
    let artifact = Path::new(worktree).join(".agent-flow/implementation-result.yaml");
    match artifact_freshness(&artifact, state) {
        Some(decision) => decision,
        None => AutomationDecision::Advance("implementation_complete".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{WorkflowState, WorkflowTransition};

    fn workflow() -> WorkflowDefinition {
        WorkflowDefinition {
            initial_state: "backlog".into(),
            states: vec![
                WorkflowState { id: "backlog".into(), label: "Backlog".into(), role: None, terminal: false },
                WorkflowState { id: "admission".into(), label: "Admission".into(), role: None, terminal: false },
                WorkflowState { id: "done".into(), label: "Done".into(), role: None, terminal: true },
            ],
            transitions: vec![WorkflowTransition {
                action: "admit".into(),
                from: "backlog".into(),
                to: "admission".into(),
                guards: vec![crate::workflow::WorkflowGuard::DependenciesResolved],
            }],
        }
    }

    fn project() -> WorkflowProjectConfig {
        WorkflowProjectConfig {
            target_branch: "feature/poc".into(),
            role_bindings: Default::default(),
            ..Default::default()
        }
    }

    #[test]
    fn admission_freezes_the_commit_and_records_history() {
        let task = Task::new("Seed cameras", "claude", "heaves");
        let admission = prepare_admission(&workflow(), &project(), &task, true, "a1b2c3").unwrap();

        assert_eq!(admission.state.state, "admission");
        assert_eq!(admission.state.target_branch, "feature/poc");
        assert_eq!(admission.state.base_sha.as_deref(), Some("a1b2c3"));
        assert_eq!(admission.transition.action, "admit");
        assert_eq!(admission.transition.from_state, "backlog");
        assert_eq!(admission.transition.to_state, "admission");
    }

    #[test]
    fn admission_refuses_unresolved_dependencies_or_existing_worktree() {
        let mut task = Task::new("Seed cameras", "claude", "heaves");
        assert!(prepare_admission(&workflow(), &project(), &task, false, "a1b2c3").is_err());

        task.worktree_path = Some(".agtx/worktrees/example".into());
        assert!(prepare_admission(&workflow(), &project(), &task, true, "a1b2c3").is_err());
    }

    #[test]
    fn transition_uses_project_role_binding_not_a_hard_coded_agent() {
        let mut graph = workflow();
        graph.states[1].role = Some("planner".into());
        graph.transitions[0].action = "start_planning".into();
        let mut current = WorkflowTaskState::new("task", "backlog", "feature/poc");
        current.base_sha = Some("a1b2c3".into());
        let mut project = project();
        project.role_bindings.insert("planner".into(), "claude".into());

        let transition = prepare_transition(
            &graph,
            &project,
            &current,
            "start_planning",
            GuardContext { dependencies_resolved: true, ..GuardContext::default() },
        )
        .unwrap();

        assert_eq!(transition.state.state, "admission");
        assert_eq!(transition.destination_agent.as_deref(), Some("claude"));
        assert_eq!(transition.transition.actor_role.as_deref(), Some("planner"));
    }

    fn plugin_for_tests(workflow: WorkflowDefinition) -> crate::config::WorkflowPlugin {
        crate::config::WorkflowPlugin {
            name: "test-plugin".into(),
            description: None,
            init_script: None,
            state_machine: Some(workflow),
            supported_agents: Vec::new(),
            artifacts: Default::default(),
            commands: Default::default(),
            prompts: Default::default(),
            prompt_triggers: Default::default(),
            copy_dirs: Vec::new(),
            copy_files: Vec::new(),
            cyclic: false,
            clear_context_on_advance: false,
            copy_back: Default::default(),
            auto_dismiss: Vec::new(),
        }
    }

    /// A graph shaped like the real project's conventional states, used only
    /// by the `assess` tests below: `backlog` and `admission` have no role,
    /// the rest are agent-owned states whose transitions are only
    /// distinguished by the verdict `assess` reads from their artifact
    /// (mirroring `submit_engineering_review`/`submit_final_validation`,
    /// which is why none of these transitions declare a graph guard).
    fn assess_workflow() -> WorkflowDefinition {
        WorkflowDefinition {
            initial_state: "backlog".into(),
            states: vec![
                WorkflowState { id: "backlog".into(), label: "Backlog".into(), role: None, terminal: false },
                WorkflowState { id: "admission".into(), label: "Admission".into(), role: None, terminal: false },
                WorkflowState { id: "planning".into(), label: "Planning".into(), role: Some("planner".into()), terminal: false },
                WorkflowState { id: "running".into(), label: "Running".into(), role: Some("implementer".into()), terminal: false },
                WorkflowState { id: "engineering_review".into(), label: "Engineering review".into(), role: Some("reviewer".into()), terminal: false },
                WorkflowState { id: "final_validation".into(), label: "Final validation".into(), role: Some("validator".into()), terminal: false },
                WorkflowState { id: "done".into(), label: "Done".into(), role: None, terminal: true },
            ],
            transitions: vec![
                WorkflowTransition {
                    action: "admit".into(),
                    from: "backlog".into(),
                    to: "admission".into(),
                    guards: vec![crate::workflow::WorkflowGuard::DependenciesResolved],
                },
                WorkflowTransition {
                    action: "engineering_corrections_required".into(),
                    from: "engineering_review".into(),
                    to: "running".into(),
                    guards: vec![],
                },
                WorkflowTransition {
                    action: "engineering_plan_issue".into(),
                    from: "engineering_review".into(),
                    to: "planning".into(),
                    guards: vec![],
                },
                WorkflowTransition {
                    action: "start_final_validation".into(),
                    from: "engineering_review".into(),
                    to: "final_validation".into(),
                    guards: vec![],
                },
                WorkflowTransition {
                    action: "begin_feature_integration".into(),
                    from: "final_validation".into(),
                    to: "done".into(),
                    guards: vec![],
                },
                WorkflowTransition {
                    action: "validation_failed".into(),
                    from: "final_validation".into(),
                    to: "engineering_review".into(),
                    guards: vec![],
                },
            ],
        }
    }

    fn admitted_task(worktree: &std::path::Path) -> Task {
        let mut task = Task::new("Review thing", "claude", "proj");
        task.worktree_path = Some(worktree.to_string_lossy().to_string());
        task
    }

    #[test]
    fn assess_advances_a_backlog_task_once_dependencies_resolve() {
        let graph = assess_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let task = Task::new("Seed cameras", "claude", "heaves");
        let state = WorkflowTaskState::new(&task.id, "backlog", "feature/poc");
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(decision, AutomationDecision::Advance("admit".to_string()));
    }

    #[test]
    fn assess_waits_when_the_completion_artifact_is_missing() {
        let graph = assess_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        let task = admitted_task(worktree.path());
        let state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(decision, AutomationDecision::Wait);
    }

    #[test]
    fn assess_advances_on_a_fresh_valid_artifact() {
        let graph = assess_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/engineering-review.yaml"),
            "verdict: approved_for_validation\nworkflow_attempt: 3\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let mut state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        state.state_attempt = 3;
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(decision, AutomationDecision::Advance("start_final_validation".to_string()));
    }

    #[test]
    fn assess_waits_on_an_artifact_stamped_with_a_stale_attempt() {
        let graph = assess_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        // Left over from a prior entry into this state: valid content, but
        // stamped with an attempt number that is not the task's current one.
        std::fs::write(
            worktree.path().join(".agent-flow/engineering-review.yaml"),
            "verdict: approved_for_validation\nworkflow_attempt: 1\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let mut state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        state.state_attempt = 2;
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(decision, AutomationDecision::Wait);
    }

    #[test]
    fn assess_flags_a_fresh_but_malformed_artifact_as_invalid() {
        let graph = assess_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        // Attempt matches, but there is no `verdict:` line at all.
        std::fs::write(
            worktree.path().join(".agent-flow/engineering-review.yaml"),
            "workflow_attempt: 1\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert!(matches!(decision, AutomationDecision::InvalidArtifact(_)));
    }

    #[test]
    fn assess_gates_a_failed_final_validation_to_a_human_instead_of_advancing() {
        let graph = assess_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/final-validation.yaml"),
            "verdict: failed\nworkflow_attempt: 1\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let state = WorkflowTaskState::new(&task.id, "final_validation", "main");
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(decision, AutomationDecision::HumanGate("final validation failed".to_string()));
    }

    /// A task re-entering `engineering_review` after a rework loop (e.g. a
    /// prior `corrections_required` sent it back to `running`, and
    /// implementation was resubmitted) must get a fresh `state_attempt`. The
    /// old `engineering-review.yaml` from the first pass is still sitting on
    /// disk, unchanged -- `assess` must not mistake it for evidence about
    /// the new pass.
    #[test]
    fn assess_ignores_a_stale_artifact_left_over_from_a_prior_pass_through_the_same_state() {
        let graph = assess_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        let artifact = worktree.path().join(".agent-flow/engineering-review.yaml");
        std::fs::write(&artifact, "verdict: approved_for_validation\nworkflow_attempt: 1\n").unwrap();
        let task = admitted_task(worktree.path());
        let db = Database::open_in_memory_project().unwrap();

        // First pass: state_attempt 1 matches the artifact's workflow_attempt.
        let first_state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        assert_eq!(
            assess(&graph, &project(), &plugin, &task, &first_state, &db),
            AutomationDecision::Advance("start_final_validation".to_string())
        );

        // Rework cycles the task back through `running` and it re-enters
        // `engineering_review` a second time; the artifact on disk is the
        // untouched leftover from the first pass.
        let mut second_state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        second_state.state_attempt = 2;
        assert_eq!(
            assess(&graph, &project(), &plugin, &task, &second_state, &db),
            AutomationDecision::Wait
        );
    }
}

/// Mock-backed tests for the launch functions extracted from `tui::app`'s
/// manual Shift+<key> handlers. These exercise the exact same collaborators
/// (`MockTmuxOperations`, `MockAgentRegistry`/`MockAgentOperations`,
/// `MockGitOperations`, an in-memory `Database`) the TUI test suite already
/// uses for `build_policy_agent_command`, without needing an `App` or a
/// terminal at all.
#[cfg(test)]
#[cfg(feature = "test-mocks")]
mod launch_tests {
    use super::*;
    use crate::agent::{AgentOperations, AgentRegistry, MockAgentOperations, MockAgentRegistry};
    use crate::config::{GlobalConfig, MergedConfig, ProjectConfig, WorkflowPlugin};
    use crate::db::Database;
    use crate::git::MockGitOperations;
    use crate::tmux::MockTmuxOperations;
    use crate::workflow::{WorkflowRolePolicy, WorkflowState, WorkflowTransition};
    use std::sync::{Arc, Mutex};

    fn plugin(workflow: WorkflowDefinition) -> WorkflowPlugin {
        WorkflowPlugin {
            name: "test-plugin".into(),
            description: None,
            init_script: None,
            state_machine: Some(workflow),
            supported_agents: Vec::new(),
            artifacts: Default::default(),
            commands: Default::default(),
            prompts: Default::default(),
            prompt_triggers: Default::default(),
            copy_dirs: Vec::new(),
            copy_files: Vec::new(),
            cyclic: false,
            clear_context_on_advance: false,
            copy_back: Default::default(),
            auto_dismiss: Vec::new(),
        }
    }

    fn merged_config() -> MergedConfig {
        MergedConfig::merge(&GlobalConfig::default(), &ProjectConfig::default())
    }

    fn feature_flags() -> crate::FeatureFlags {
        crate::FeatureFlags::default()
    }

    /// Same escaping `build_policy_agent_command` applies to the prompt before
    /// wrapping it in single quotes -- duplicated here (rather than calling the
    /// function under test) so the parity assertion is against an
    /// independently-derived expected string, not the helper checking itself.
    fn quote_for_shell(prompt: &str) -> String {
        prompt.replace('\'', "'\"'\"'")
    }

    /// Claude policy flags, computed independently of `claude_policy_flags` /
    /// `build_policy_agent_command` for the same reason as `quote_for_shell`.
    fn claude_allowed_tools(policy: &WorkflowRolePolicy) -> String {
        let mut tools = vec!["Read".to_string(), "Glob".to_string(), "Grep".to_string()];
        for command in &policy.allowed_commands {
            tools.push(format!("Bash({command} *)"));
        }
        if !policy.write_paths.is_empty() {
            tools.push("Edit".to_string());
            tools.push("Write".to_string());
        }
        tools.join(",")
    }

    /// `start_workflow_implementation` parity: the exact `claude ...` command
    /// text it hands to `TmuxOperations::create_window` must match what
    /// `build_policy_agent_command`'s documented behaviour computes for the
    /// implementer's resolved policy -- derived here independently instead of
    /// calling that helper, so a bug in either side would show up as a mismatch.
    #[test]
    fn start_workflow_implementation_launch_command_matches_the_resolved_policy() {
        let graph = WorkflowDefinition {
            initial_state: "plan_review".into(),
            states: vec![
                WorkflowState { id: "plan_review".into(), label: "Plan review".into(), role: None, terminal: false },
                WorkflowState { id: "implementation".into(), label: "Implementation".into(), role: Some("implementer".into()), terminal: true },
            ],
            transitions: vec![WorkflowTransition {
                action: "start_implementation".into(),
                from: "plan_review".into(),
                to: "implementation".into(),
                guards: vec![crate::workflow::WorkflowGuard::ApprovedPlan],
            }],
        };
        graph.validate().unwrap();

        let mut project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project.role_bindings.insert("implementer".into(), "claude".into());
        project.role_policies.roles.insert(
            "implementer".into(),
            WorkflowRolePolicy {
                allowed_commands: vec!["pytest".into()],
                write_paths: vec!["src/**".into()],
                model: Some("sonnet".into()),
                effort: Some("medium".into()),
                ..Default::default()
            },
        );

        let mut plugin = plugin(graph.clone());
        plugin.prompts.running = Some("Implement: {task}".into());

        let mut task = crate::db::Task::new("Implement thing", "codex", "proj");
        task.description = Some("do the work".into());
        task.worktree_path = Some("C:/work/wt".into());

        let mut db = Database::open_in_memory_project().unwrap();
        db.create_task(&task).unwrap();
        let mut current = WorkflowTaskState::new(&task.id, "plan_review", "main");
        current.plan_revision = 2;
        current.plan_hash = Some("abc123".into());
        current.approved_plan_revision = Some(2);
        current.approved_plan_hash = Some("abc123".into());
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "plan_review");
        db.record_workflow_admission(&task, &current, &record).unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux.expect_has_session().returning(|_| true);
        let captured = Arc::new(Mutex::new(None));
        let captured_create = captured.clone();
        mock_tmux.expect_create_window().returning(move |_, _, _, command, _, _| {
            *captured_create.lock().unwrap() = command;
            Ok(())
        });

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(mock_registry);
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = WorkflowRuntime {
            tmux_ops: &tmux_ops,
            agent_registry: &agent_registry,
            git_ops: &git_ops,
            tmux_project_name: "proj",
            project_path: Path::new("C:/work/project"),
            config: &config,
            flags: &flags,
        };

        let outcome = start_workflow_implementation(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        let WorkflowStepOutcome::Advanced { task: advanced, message } = outcome else {
            panic!("expected Advanced, got a Blocked/NoOp outcome");
        };
        assert_eq!(message, "Implementation started from the approved plan");
        assert_eq!(advanced.status, TaskStatus::Running);
        assert_eq!(advanced.agent, "claude");

        let policy = WorkflowRolePolicy {
            allowed_commands: vec!["pytest".into()],
            write_paths: vec!["src/**".into()],
            model: Some("sonnet".into()),
            effort: Some("medium".into()),
            ..Default::default()
        };
        let prompt = "Implement: do the work\n\nApproved plan revision: 2\nApproved plan SHA-256: abc123\n\nCurrent workflow attempt: 2. Your output artifact MUST contain the line: workflow_attempt: 2".to_string();
        let expected = format!(
            "claude --model sonnet --effort medium --permission-mode dontAsk --allowed-tools '{}' -- '{}'",
            claude_allowed_tools(&policy),
            quote_for_shell(&prompt),
        );
        let actual = captured.lock().unwrap().clone().expect("create_window should receive a command");
        assert_eq!(actual, expected);
    }

    /// `submit_engineering_review` parity: an `approved_for_validation`
    /// verdict must launch the validator under its own resolved policy with
    /// the same `claude ...` command shape as the implementer test above,
    /// delivered through `switch_agent_in_tmux`'s multi-line-prompt path
    /// (`paste_text` + Enter, since the built prompt always spans lines).
    #[test]
    fn submit_engineering_review_launch_command_matches_the_resolved_policy() {
        let graph = WorkflowDefinition {
            initial_state: "engineering_review".into(),
            states: vec![
                WorkflowState { id: "engineering_review".into(), label: "Engineering review".into(), role: Some("reviewer".into()), terminal: false },
                WorkflowState { id: "final_validation".into(), label: "Final validation".into(), role: Some("validator".into()), terminal: true },
            ],
            transitions: vec![WorkflowTransition {
                action: "start_final_validation".into(),
                from: "engineering_review".into(),
                to: "final_validation".into(),
                guards: vec![],
            }],
        };
        graph.validate().unwrap();

        let mut project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project.role_bindings.insert("reviewer".into(), "claude".into());
        project.role_bindings.insert("validator".into(), "claude".into());
        project.role_policies.roles.insert(
            "validator".into(),
            WorkflowRolePolicy {
                allowed_commands: vec!["ruff check".into()],
                write_paths: vec![".agent-flow/final-validation.yaml".into()],
                model: Some("opus".into()),
                effort: None,
                ..Default::default()
            },
        );

        let mut plugin = plugin(graph.clone());
        plugin.prompts.final_validation = Some("Review: {task}".into());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/engineering-review.yaml"),
            "verdict: approved_for_validation\n",
        )
        .unwrap();

        let mut task = crate::db::Task::new("Review thing", "claude", "proj");
        task.description = Some("review the change".into());
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-review".into());

        let mut db = Database::open_in_memory_project().unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "engineering_review");
        db.record_workflow_admission(&task, &current, &record).unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux.expect_send_keys().withf(|_, cmd: &str| cmd == "/exit").returning(|_, _| Ok(()));
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        mock_tmux.expect_pane_current_command().returning(|_| Some("bash".to_string()));
        mock_tmux.expect_capture_pane().returning(|_| Ok(String::new()));
        let captured = Arc::new(Mutex::new(None));
        let captured_paste = captured.clone();
        mock_tmux.expect_paste_text().returning(move |_, text| {
            *captured_paste.lock().unwrap() = Some(text.to_string());
            Ok(())
        });

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(mock_registry);
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = WorkflowRuntime {
            tmux_ops: &tmux_ops,
            agent_registry: &agent_registry,
            git_ops: &git_ops,
            tmux_project_name: "proj",
            project_path: Path::new("C:/work/project"),
            config: &config,
            flags: &flags,
        };

        let outcome = submit_engineering_review(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        let WorkflowStepOutcome::Advanced { task: advanced, message } = outcome else {
            panic!("expected Advanced, got a Blocked/NoOp outcome");
        };
        assert_eq!(message, "Engineering review recorded: approved_for_validation");
        assert_eq!(advanced.agent, "claude");
        assert_eq!(advanced.status, TaskStatus::Review);

        let policy = WorkflowRolePolicy {
            allowed_commands: vec!["ruff check".into()],
            write_paths: vec![".agent-flow/final-validation.yaml".into()],
            model: Some("opus".into()),
            effort: None,
            ..Default::default()
        };
        let artifact_rel = ".agent-flow/engineering-review.yaml";
        let prompt = format!(
            "Review: review the change

Engineering-review verdict: approved_for_validation. Evidence: {artifact_rel}. Follow the declared role policy; do not commit, push, create a PR, merge, or bypass controls.

Current workflow attempt: 2. Your output artifact MUST contain the line: workflow_attempt: 2"
        );
        let inner = format!(
            "claude --model opus --permission-mode dontAsk --allowed-tools '{}' -- '{}'",
            claude_allowed_tools(&policy),
            quote_for_shell(&prompt),
        );
        // `switch_agent_in_tmux` wraps every relaunch in a `cd`/`env -u` prefix
        // that clears Claude Code's nesting-detection vars, then -- because the
        // engineering-review prompt always spans lines -- delivers it via
        // `paste_text` rather than `send_keys`.
        let expected = format!(
            "cd -- \"$AGTX_WORKTREE\" && env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT {inner}"
        );
        let sent = captured.lock().unwrap().clone().expect("switch_agent_in_tmux should paste the new command");
        assert_eq!(sent, expected);
    }

    /// Dedicated coverage for the `state_attempt` prompt injection itself
    /// (separate from the exact-match parity tests above, which happen to
    /// also cover it): the destination state's `state_attempt` -- the value
    /// the task will have *after* this transition commits -- must appear in
    /// the prompt handed to the launched agent, in the documented form.
    #[test]
    fn submit_engineering_review_prompt_declares_the_destination_workflow_attempt() {
        let graph = WorkflowDefinition {
            initial_state: "engineering_review".into(),
            states: vec![
                WorkflowState { id: "engineering_review".into(), label: "Engineering review".into(), role: Some("reviewer".into()), terminal: false },
                WorkflowState { id: "final_validation".into(), label: "Final validation".into(), role: Some("validator".into()), terminal: true },
            ],
            transitions: vec![WorkflowTransition {
                action: "start_final_validation".into(),
                from: "engineering_review".into(),
                to: "final_validation".into(),
                guards: vec![],
            }],
        };
        graph.validate().unwrap();

        let mut project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project.role_bindings.insert("reviewer".into(), "claude".into());
        project.role_bindings.insert("validator".into(), "claude".into());
        project.role_policies.roles.insert("validator".into(), WorkflowRolePolicy::default());

        let plugin = plugin(graph.clone());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/engineering-review.yaml"),
            "verdict: approved_for_validation\n",
        )
        .unwrap();

        let mut task = crate::db::Task::new("Review thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-review".into());

        let mut db = Database::open_in_memory_project().unwrap();
        db.create_task(&task).unwrap();
        // `record_workflow_admission` always seeds state_attempt at 1, so the
        // one transition this test fires (engineering_review ->
        // final_validation) lands the task at state_attempt 2 -- that is the
        // number the launched prompt must carry.
        let current = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        assert_eq!(current.state_attempt, 1);
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "engineering_review");
        db.record_workflow_admission(&task, &current, &record).unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux.expect_send_keys().withf(|_, cmd: &str| cmd == "/exit").returning(|_, _| Ok(()));
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        mock_tmux.expect_pane_current_command().returning(|_| Some("bash".to_string()));
        mock_tmux.expect_capture_pane().returning(|_| Ok(String::new()));
        let captured = Arc::new(Mutex::new(None));
        let captured_paste = captured.clone();
        mock_tmux.expect_paste_text().returning(move |_, text| {
            *captured_paste.lock().unwrap() = Some(text.to_string());
            Ok(())
        });

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(mock_registry);
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = WorkflowRuntime {
            tmux_ops: &tmux_ops,
            agent_registry: &agent_registry,
            git_ops: &git_ops,
            tmux_project_name: "proj",
            project_path: Path::new("C:/work/project"),
            config: &config,
            flags: &flags,
        };

        let outcome = submit_engineering_review(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));

        let sent = captured.lock().unwrap().clone().expect("switch_agent_in_tmux should paste the new command");
        assert!(
            sent.contains("Current workflow attempt: 2. Your output artifact MUST contain the line: workflow_attempt: 2"),
            "expected the destination state_attempt (2) in the launched prompt, got: {sent}"
        );
    }

    /// Opens an independent connection to the same on-disk database `path`
    /// points at and asserts the task's durable workflow state has already
    /// reached `expected_state`.
    ///
    /// Called from inside a tmux-mock closure that fires during a function's
    /// launch step: `db` itself is exclusively borrowed by the call under
    /// test for its whole duration, so this cannot read through that same
    /// handle. `Database::open_project_at_path` (gated behind `test-mocks`
    /// for exactly this "concurrency tests" purpose, see its doc comment)
    /// opens a second, independent connection to the same file instead. If
    /// persistence were ever moved back to *after* the launch call -- the bug
    /// this reorder fixed -- this assertion would see the pre-transition
    /// state and fail.
    fn assert_state_already_persisted(path: &Path, task_id: &str, expected_state: &str) {
        let check_db = Database::open_project_at_path(path).unwrap();
        let state = check_db
            .get_workflow_task_state(task_id)
            .unwrap()
            .expect("workflow state row must exist by the time the agent launch fires");
        assert_eq!(
            state.state, expected_state,
            "durable state must already reflect '{expected_state}' by the time the agent launch \
             fires -- persistence must happen before launch, not after"
        );
    }

    /// `start_workflow_planning` reorder-proof: `admission_complete` and
    /// `start_planning` are each persisted via `advance_workflow_state`
    /// *before* the planner is launched. Forces the `admission` ->
    /// `ready_for_planning` -> `planning` two-hop path (see the doc comment
    /// on `start_workflow_planning` for why this case stays two separate
    /// calls rather than one `advance_workflow_state_chain`) and checks, from
    /// inside the `create_window` mock, that both hops already landed.
    #[test]
    fn start_workflow_planning_persists_before_launching_the_planner() {
        let graph = WorkflowDefinition {
            initial_state: "admission".into(),
            states: vec![
                WorkflowState { id: "admission".into(), label: "Admission".into(), role: None, terminal: false },
                WorkflowState { id: "ready_for_planning".into(), label: "Ready for planning".into(), role: None, terminal: false },
                WorkflowState { id: "planning".into(), label: "Planning".into(), role: Some("planner".into()), terminal: true },
            ],
            transitions: vec![
                WorkflowTransition {
                    action: "admission_complete".into(),
                    from: "admission".into(),
                    to: "ready_for_planning".into(),
                    guards: vec![crate::workflow::WorkflowGuard::AdmissionRecorded],
                },
                WorkflowTransition {
                    action: "start_planning".into(),
                    from: "ready_for_planning".into(),
                    to: "planning".into(),
                    guards: vec![],
                },
            ],
        };
        graph.validate().unwrap();

        let mut project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project.role_bindings.insert("planner".into(), "claude".into());
        project.role_policies.roles.insert("planner".into(), WorkflowRolePolicy::default());

        let plugin = plugin(graph.clone());

        let worktree = tempfile::tempdir().unwrap();
        let mut task = crate::db::Task::new("Plan thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());

        let db_dir = tempfile::tempdir().unwrap();
        let db_path = db_dir.path().join("wf.db");
        let mut db = Database::open_project_at_path(&db_path).unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "admission", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "admission");
        db.record_workflow_admission(&task, &current, &record).unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux.expect_has_session().returning(|_| true);
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux.expect_create_window().returning(move |_, _, _, _, _, _| {
            assert_state_already_persisted(&db_path_for_check, &task_id_for_check, "planning");
            Ok(())
        });

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(mock_registry);
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = WorkflowRuntime {
            tmux_ops: &tmux_ops,
            agent_registry: &agent_registry,
            git_ops: &git_ops,
            tmux_project_name: "proj",
            project_path: Path::new("C:/work/project"),
            config: &config,
            flags: &flags,
        };

        let outcome = start_workflow_planning(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));

        // Both hops committed for real, not just observed mid-flight.
        assert_eq!(db.get_workflow_task_state(&task.id).unwrap().unwrap().state, "planning");
        assert_eq!(db.workflow_transition_history(&task.id).unwrap().len(), 3);
    }

    /// `submit_workflow_plan` reorder-proof: `submit_plan` is persisted
    /// before the plan reviewer is launched. Configures a role policy for
    /// `plan_reviewer` so the function takes its synchronous
    /// `switch_agent_in_tmux` path rather than the fire-and-forget
    /// `spawn_send_to_agent` fallback, then checks -- from inside the mock's
    /// exit-command `send_keys` call, the first thing `switch_agent_in_tmux`
    /// does -- that the new state already landed.
    #[test]
    fn submit_workflow_plan_persists_before_launching_the_reviewer() {
        let graph = WorkflowDefinition {
            initial_state: "planning".into(),
            states: vec![
                WorkflowState { id: "planning".into(), label: "Planning".into(), role: Some("planner".into()), terminal: false },
                WorkflowState { id: "plan_review".into(), label: "Plan review".into(), role: Some("plan_reviewer".into()), terminal: true },
            ],
            transitions: vec![WorkflowTransition {
                action: "submit_plan".into(),
                from: "planning".into(),
                to: "plan_review".into(),
                guards: vec![],
            }],
        };
        graph.validate().unwrap();

        let mut project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project.role_bindings.insert("planner".into(), "claude".into());
        project.role_bindings.insert("plan_reviewer".into(), "claude".into());
        project.role_policies.roles.insert("plan_reviewer".into(), WorkflowRolePolicy::default());

        let mut plugin = plugin(graph.clone());
        plugin.artifacts.planning = Some(".agent-flow/plan.yaml".into());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan.yaml"),
            "plan_revision: 1\n",
        )
        .unwrap();

        let mut task = crate::db::Task::new("Plan thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-plan".into());

        let db_dir = tempfile::tempdir().unwrap();
        let db_path = db_dir.path().join("wf.db");
        let mut db = Database::open_project_at_path(&db_path).unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "planning", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "planning");
        db.record_workflow_admission(&task, &current, &record).unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux.expect_send_keys().withf(|_, cmd: &str| cmd == "/exit").returning(move |_, _| {
            assert_state_already_persisted(&db_path_for_check, &task_id_for_check, "plan_review");
            Ok(())
        });
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        mock_tmux.expect_pane_current_command().returning(|_| Some("bash".to_string()));
        mock_tmux.expect_capture_pane().returning(|_| Ok(String::new()));
        mock_tmux.expect_paste_text().returning(|_, _| Ok(()));

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(mock_registry);
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = WorkflowRuntime {
            tmux_ops: &tmux_ops,
            agent_registry: &agent_registry,
            git_ops: &git_ops,
            tmux_project_name: "proj",
            project_path: Path::new("C:/work/project"),
            config: &config,
            flags: &flags,
        };

        let outcome = submit_workflow_plan(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));
        assert_eq!(db.get_workflow_task_state(&task.id).unwrap().unwrap().state, "plan_review");
    }

    /// `submit_workflow_implementation` reorder-proof: both chained
    /// transitions (`implementation_complete`, `start_engineering_review`)
    /// must already be committed -- via `advance_workflow_state_chain` in one
    /// transaction -- before the engineering reviewer is launched.
    #[test]
    fn submit_workflow_implementation_persists_the_chain_before_launching_the_reviewer() {
        let graph = WorkflowDefinition {
            initial_state: "running".into(),
            states: vec![
                WorkflowState { id: "running".into(), label: "Running".into(), role: Some("implementer".into()), terminal: false },
                WorkflowState { id: "implementing_complete".into(), label: "Implementation complete".into(), role: None, terminal: false },
                WorkflowState { id: "engineering_review".into(), label: "Engineering review".into(), role: Some("reviewer".into()), terminal: true },
            ],
            transitions: vec![
                WorkflowTransition {
                    action: "implementation_complete".into(),
                    from: "running".into(),
                    to: "implementing_complete".into(),
                    guards: vec![],
                },
                WorkflowTransition {
                    action: "start_engineering_review".into(),
                    from: "implementing_complete".into(),
                    to: "engineering_review".into(),
                    guards: vec![],
                },
            ],
        };
        graph.validate().unwrap();

        let mut project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project.role_bindings.insert("implementer".into(), "claude".into());
        project.role_bindings.insert("reviewer".into(), "claude".into());
        project.role_policies.roles.insert("reviewer".into(), WorkflowRolePolicy::default());

        let plugin = plugin(graph.clone());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/implementation-result.yaml"),
            "status: done\n",
        )
        .unwrap();

        let mut task = crate::db::Task::new("Implement thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-impl".into());

        let db_dir = tempfile::tempdir().unwrap();
        let db_path = db_dir.path().join("wf.db");
        let mut db = Database::open_project_at_path(&db_path).unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "running", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "running");
        db.record_workflow_admission(&task, &current, &record).unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux.expect_send_keys().withf(|_, cmd: &str| cmd == "/exit").returning(move |_, _| {
            assert_state_already_persisted(&db_path_for_check, &task_id_for_check, "engineering_review");
            Ok(())
        });
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        mock_tmux.expect_pane_current_command().returning(|_| Some("bash".to_string()));
        mock_tmux.expect_capture_pane().returning(|_| Ok(String::new()));
        mock_tmux.expect_paste_text().returning(|_, _| Ok(()));

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(mock_registry);
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = WorkflowRuntime {
            tmux_ops: &tmux_ops,
            agent_registry: &agent_registry,
            git_ops: &git_ops,
            tmux_project_name: "proj",
            project_path: Path::new("C:/work/project"),
            config: &config,
            flags: &flags,
        };

        let outcome = submit_workflow_implementation(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));

        // Both chained hops committed for real, in one transaction.
        assert_eq!(db.get_workflow_task_state(&task.id).unwrap().unwrap().state, "engineering_review");
        let history = db.workflow_transition_history(&task.id).unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[1].action, "implementation_complete");
        assert_eq!(history[2].action, "start_engineering_review");
    }

    /// `submit_engineering_review` reorder-proof: the resolved verdict
    /// transition is persisted before the next agent (the validator, for an
    /// `approved_for_validation` verdict) is launched.
    #[test]
    fn submit_engineering_review_persists_before_launching_the_next_agent() {
        let graph = WorkflowDefinition {
            initial_state: "engineering_review".into(),
            states: vec![
                WorkflowState { id: "engineering_review".into(), label: "Engineering review".into(), role: Some("reviewer".into()), terminal: false },
                WorkflowState { id: "final_validation".into(), label: "Final validation".into(), role: Some("validator".into()), terminal: true },
            ],
            transitions: vec![WorkflowTransition {
                action: "start_final_validation".into(),
                from: "engineering_review".into(),
                to: "final_validation".into(),
                guards: vec![],
            }],
        };
        graph.validate().unwrap();

        let mut project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project.role_bindings.insert("reviewer".into(), "claude".into());
        project.role_bindings.insert("validator".into(), "claude".into());
        project.role_policies.roles.insert("validator".into(), WorkflowRolePolicy::default());

        let plugin = plugin(graph.clone());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/engineering-review.yaml"),
            "verdict: approved_for_validation\n",
        )
        .unwrap();

        let mut task = crate::db::Task::new("Review thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-review".into());

        let db_dir = tempfile::tempdir().unwrap();
        let db_path = db_dir.path().join("wf.db");
        let mut db = Database::open_project_at_path(&db_path).unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "engineering_review");
        db.record_workflow_admission(&task, &current, &record).unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux.expect_send_keys().withf(|_, cmd: &str| cmd == "/exit").returning(move |_, _| {
            assert_state_already_persisted(&db_path_for_check, &task_id_for_check, "final_validation");
            Ok(())
        });
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        mock_tmux.expect_pane_current_command().returning(|_| Some("bash".to_string()));
        mock_tmux.expect_capture_pane().returning(|_| Ok(String::new()));
        mock_tmux.expect_paste_text().returning(|_, _| Ok(()));

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(mock_registry);
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = WorkflowRuntime {
            tmux_ops: &tmux_ops,
            agent_registry: &agent_registry,
            git_ops: &git_ops,
            tmux_project_name: "proj",
            project_path: Path::new("C:/work/project"),
            config: &config,
            flags: &flags,
        };

        let outcome = submit_engineering_review(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));
        assert_eq!(db.get_workflow_task_state(&task.id).unwrap().unwrap().state, "final_validation");
    }

    /// `submit_final_validation` reorder-proof: a `passed` verdict's
    /// `begin_feature_integration` transition is persisted before the next
    /// agent is launched. `archive_workflow_artifact` only runs on the
    /// `failed` path (see the function's own doc comment), so it is not
    /// exercised here; this test only covers the persist/launch ordering the
    /// plan changed.
    #[test]
    fn submit_final_validation_persists_before_launching_the_next_agent() {
        let graph = WorkflowDefinition {
            initial_state: "final_validation".into(),
            states: vec![
                WorkflowState { id: "final_validation".into(), label: "Final validation".into(), role: Some("validator".into()), terminal: false },
                WorkflowState { id: "integrate_to_feature".into(), label: "Integrate to feature".into(), role: Some("integrator".into()), terminal: true },
            ],
            transitions: vec![WorkflowTransition {
                action: "begin_feature_integration".into(),
                from: "final_validation".into(),
                to: "integrate_to_feature".into(),
                guards: vec![crate::workflow::WorkflowGuard::FinalValidationPassed],
            }],
        };
        graph.validate().unwrap();

        let mut project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project.role_bindings.insert("validator".into(), "claude".into());
        project.role_bindings.insert("integrator".into(), "claude".into());
        project.role_policies.roles.insert("integrator".into(), WorkflowRolePolicy::default());

        let plugin = plugin(graph.clone());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/final-validation.yaml"),
            "verdict: passed\n",
        )
        .unwrap();

        let mut task = crate::db::Task::new("Validate thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-validate".into());

        let db_dir = tempfile::tempdir().unwrap();
        let db_path = db_dir.path().join("wf.db");
        let mut db = Database::open_project_at_path(&db_path).unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "final_validation", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "final_validation");
        db.record_workflow_admission(&task, &current, &record).unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux.expect_send_keys().withf(|_, cmd: &str| cmd == "/exit").returning(move |_, _| {
            assert_state_already_persisted(&db_path_for_check, &task_id_for_check, "integrate_to_feature");
            Ok(())
        });
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        mock_tmux.expect_pane_current_command().returning(|_| Some("bash".to_string()));
        mock_tmux.expect_capture_pane().returning(|_| Ok(String::new()));
        mock_tmux.expect_paste_text().returning(|_, _| Ok(()));

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(mock_registry);
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = WorkflowRuntime {
            tmux_ops: &tmux_ops,
            agent_registry: &agent_registry,
            git_ops: &git_ops,
            tmux_project_name: "proj",
            project_path: Path::new("C:/work/project"),
            config: &config,
            flags: &flags,
        };

        let outcome = submit_final_validation(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));
        assert_eq!(db.get_workflow_task_state(&task.id).unwrap().unwrap().state, "integrate_to_feature");
    }
}
