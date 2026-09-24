//! Side-effect-free preparation for declarative workflow transitions.
//!
//! The graph module owns which transitions are legal. This module turns a
//! task's persisted evidence into the guard facts required by the graph, and
//! produces the durable records a caller must commit together. Keeping this
//! step independent from the TUI means web, MCP, and terminal callers cannot
//! disagree about what an admission means.

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::agent::AgentRegistry;
use crate::config::{MergedConfig, WorkflowPlugin};
use crate::db::{
    Database, ProviderSession, Task, TaskExecutionEvent, TaskStatus, TaskStepReport,
    WorkflowArtifact, WorkflowStepInput, WorkflowTaskState, WorkflowTransitionRecord,
};
use crate::git::{GitOperations, GitProviderOperations, PullRequestState};
use crate::tmux::TmuxOperations;
use crate::tui::app::{
    agtx_task_env, archive_workflow_artifact, build_policy_agent_command,
    ensure_project_tmux_session, ensure_review_addresses_failed_validation, generate_task_slug,
    planning_artifact_path, resolve_prompt, switch_agent_in_tmux, wait_for_agent_ready,
    workflow_artifact_path, workflow_artifact_sha256, workflow_artifact_value,
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

// The journal is intended for post-mortem analysis, not as an unbounded copy
// of a terminal transcript. Keep enough context to explain the handoff while
// preventing a noisy command or generated file from making the project DB
// impractical to retain.
const JOURNAL_TEXT_LIMIT: usize = 128 * 1024;
const JOURNAL_PANE_LINE_LIMIT: usize = 200;

fn bounded_journal_text(text: &str) -> String {
    let mut chars = text.chars();
    let mut bounded: String = chars.by_ref().take(JOURNAL_TEXT_LIMIT).collect();
    if chars.next().is_some() {
        bounded.push_str("\n\n[truncated by AGTX execution journal]");
    }
    bounded
}

/// Save exactly the prompt that was delivered to an agent for a particular
/// state entry. This is intentionally separate from transitions: a launch
/// may be retried without entering a different state, and both deliveries are
/// useful in a later investigation.
fn record_agent_prompt(
    db: &Database,
    task: &Task,
    state: &str,
    workflow_attempt: i64,
    agent: &str,
    prompt: &str,
    confirmed_session: Option<&str>,
) -> Result<()> {
    let mut report = TaskStepReport::new(&task.id, workflow_attempt, state);
    report.agent = Some(agent.to_string());
    report.prompt_sha256 = Some(format!("{:x}", Sha256::digest(prompt.as_bytes())));
    report.prompt_text = Some(bounded_journal_text(prompt));
    db.upsert_task_step_report(&report)?;
    // A confirmed session is the one that demonstrably holds this state's
    // prompt; recovery resumes exactly it. Without confirmation fall back to
    // the provider's newest session for the worktree, as before.
    match confirmed_session {
        Some(session) => record_provider_session(
            db,
            task,
            state,
            workflow_attempt,
            agent,
            session.to_string(),
        ),
        None => {
            if let Some(worktree) = task.worktree_path.as_deref() {
                record_provider_session_if_known(
                    db,
                    task,
                    state,
                    workflow_attempt,
                    agent,
                    worktree,
                );
            }
        }
    }

    let mut event = TaskExecutionEvent::new(&task.id, "agent_prompt_delivered");
    event.workflow_attempt = Some(workflow_attempt);
    event.state = Some(state.to_string());
    event.agent = Some(agent.to_string());
    event.outcome = Some("started".to_string());
    event.message = Some(match confirmed_session {
        Some(session) => format!("Agent prompt delivered and confirmed in session {session}"),
        None => "Agent prompt delivered to the task session".to_string(),
    });
    db.record_task_execution_event(&event)
}

/// Attach a provider-native session when the provider exposes one. Hook-based
/// agents (Codex/Claude) report their current id; Codex and OpenCode are also
/// discovered from their local session stores by the exact worktree they were
/// started in (`agent::native_session`). Rows are append-only so a fallback or
/// relaunch remains visible to usage analysis.
fn record_provider_session_if_known(
    db: &Database,
    task: &Task,
    state: &str,
    attempt: i64,
    agent: &str,
    worktree: &str,
) {
    let hook_id = crate::agent::hook_status::read_status(
        Path::new(worktree),
        &task.id,
        chrono::Utc::now().timestamp(),
    )
    .and_then(|status| status.session_id);
    let Some(provider_session_id) = hook_id.or_else(|| {
        crate::agent::native_session::session_id_for_worktree(agent, Path::new(worktree))
    }) else {
        return;
    };
    record_provider_session(db, task, state, attempt, agent, provider_session_id);
}

fn record_provider_session(
    db: &Database,
    task: &Task,
    state: &str,
    attempt: i64,
    agent: &str,
    provider_session_id: String,
) {
    let _ = db.record_provider_session(&ProviderSession {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: task.id.clone(),
        workflow_attempt: attempt,
        state: state.into(),
        workflow_session_id: format!("agtx:{}:{}:{}", task.id, attempt, state),
        provider: agent.into(),
        provider_session_id,
        agent: Some(agent.into()),
        started_at: chrono::Utc::now(),
        ended_at: None,
    });
}

/// Record the file bytes present when AGTX hands a state to an agent.  This is
/// the freshness boundary for automation: a later submission is accepted only
/// after the agent changed the artifact.  It replaces the fragile
/// agent-authored `workflow_attempt` marker.
fn record_artifact_baseline(
    db: &Database,
    task: &Task,
    state: &str,
    workflow_attempt: i64,
    artifact: &Path,
) -> Result<()> {
    let mut report = TaskStepReport::new(&task.id, workflow_attempt, state);
    report.artifact_path = Some(artifact.display().to_string());
    report.artifact_sha256 = std::fs::read(artifact)
        .ok()
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)));
    db.upsert_task_step_report(&report)
}

/// Preserve the durable artifact, the agent's optional explicit final report,
/// and a bounded tail of the pane before it is reused by the next role.
fn record_step_evidence(
    db: &Database,
    task: &Task,
    state: &WorkflowTaskState,
    agent: &str,
    artifact: &Path,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowArtifact> {
    let evidence = snapshot_step_evidence(task, state, artifact)?;
    persist_step_evidence(db, task, state, agent, artifact, evidence, runtime)
}

/// Read an artifact into an immutable candidate without writing workflow
/// state.  This is deliberately separate from persistence so a failed agent
/// hand-off cannot strand an otherwise retryable state with partial evidence.
fn snapshot_step_evidence(
    task: &Task,
    state: &WorkflowTaskState,
    artifact: &Path,
) -> Result<WorkflowArtifact> {
    let bytes = std::fs::read(artifact).map_err(|error| {
        anyhow::anyhow!(
            "Could not read workflow evidence '{}' for the execution journal: {error}",
            artifact.display()
        )
    })?;
    let artifact_hash = format!("{:x}", Sha256::digest(&bytes));
    let evidence = WorkflowArtifact {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: task.id.clone(),
        workflow_attempt: state.state_attempt,
        state: state.state.clone(),
        kind: "step_evidence".to_string(),
        source_path: artifact.display().to_string(),
        sha256: artifact_hash.clone(),
        content: bytes.clone(),
        created_at: chrono::Utc::now(),
    };
    Ok(evidence)
}

/// Persist a snapshot that was read before an external hand-off.  The caller
/// must only invoke this after the receiving agent has been verified running:
/// an unsuccessful tmux switch must leave no provisional evidence behind.
fn persist_step_evidence(
    db: &Database,
    task: &Task,
    state: &WorkflowTaskState,
    agent: &str,
    artifact: &Path,
    evidence: WorkflowArtifact,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowArtifact> {
    if let Some(worktree) = task.worktree_path.as_deref() {
        record_provider_session_if_known(
            db,
            task,
            &state.state,
            state.state_attempt,
            agent,
            worktree,
        );
    }
    let artifact_text = String::from_utf8_lossy(&evidence.content);
    let artifact_hash = evidence.sha256.clone();
    let evidence = db.store_workflow_artifact(&evidence)?;
    let mut report = TaskStepReport::new(&task.id, state.state_attempt, &state.state);
    report.agent = Some(agent.to_string());
    report.artifact_path = Some(artifact.display().to_string());
    report.artifact_sha256 = Some(artifact_hash);
    report.artifact_text = Some(bounded_journal_text(&artifact_text));
    report.final_report = workflow_artifact_value(artifact, "final_report")
        .ok()
        .map(|value| bounded_journal_text(&value));
    report.pane_tail = task
        .session_name
        .as_deref()
        .and_then(|target| runtime.tmux_ops.capture_pane(target).ok())
        .map(|pane| bounded_journal_text(&tail_lines(&pane, JOURNAL_PANE_LINE_LIMIT)))
        .filter(|pane| !pane.trim().is_empty());
    db.upsert_task_step_report(&report)?;

    let mut event = TaskExecutionEvent::new(&task.id, "step_evidence_recorded");
    event.workflow_attempt = Some(state.state_attempt);
    event.state = Some(state.state.clone());
    event.agent = Some(agent.to_string());
    event.outcome = Some("completed".to_string());
    event.message = Some(format!(
        "Captured durable evidence from {}",
        artifact.display()
    ));
    db.record_task_execution_event(&event)?;
    Ok(evidence)
}

/// How many automation ticks a hand-off waits for the source agent to finish
/// its turn before handing over anyway (~12s per tick, so about five minutes).
/// The exit confirmation in `switch_agent_in_tmux` still refuses to type into
/// an agent that does not leave, so giving up the wait cannot mis-deliver.
const MAX_IDLE_DEFERRALS: usize = 25;

/// One workflow hand-off: the pane `target` passes from whichever agent is
/// running in it to `destination_agent`, which must end up holding `prompt`.
struct Handoff<'a> {
    /// The workflow state and attempt the hand-off happens in, for the journal.
    journal_state: &'a str,
    journal_attempt: i64,
    /// The agent the task record says owns the pane.
    source_agent: &'a str,
    destination_agent: &'a str,
    target: &'a str,
    worktree: &'a str,
    command: &'a str,
    prompt: &'a str,
    /// The prompt is pasted after launch rather than carried in `command`.
    paste_prompt: bool,
}

fn record_handoff_event(
    db: &Database,
    task: &Task,
    handoff: &Handoff,
    event_type: &str,
    outcome: &str,
    message: String,
) {
    let mut event = TaskExecutionEvent::new(&task.id, event_type);
    event.workflow_attempt = Some(handoff.journal_attempt);
    event.state = Some(handoff.journal_state.to_string());
    event.agent = Some(handoff.destination_agent.to_string());
    event.outcome = Some(outcome.to_string());
    event.message = Some(message);
    event.metadata_json = Some(
        serde_json::json!({
            "source_agent": handoff.source_agent,
            "destination_agent": handoff.destination_agent,
            "tmux_target": handoff.target,
        })
        .to_string(),
    );
    let _ = db.record_task_execution_event(&event);
}

/// The agent spec name whose process the pane is running, if any.
fn agent_in_pane(runtime: &WorkflowRuntime, target: &str) -> Option<&'static str> {
    let command = runtime.tmux_ops.pane_current_command(target)?;
    let command = command.trim();
    crate::agent::AGENT_SPECS
        .iter()
        .find(|spec| spec.process_names.iter().any(|name| *name == command))
        .map(|spec| spec.name)
}

/// Hand a task pane to the next agent and deliver its prompt, verifying every
/// step the 2026-09-24 incidents showed can silently fail:
///
///  1. the agent in the pane has finished its turn (an artifact is written
///     mid-turn; hand-off keystrokes in a busy agent become chat input),
///  2. the previous process has exited and the *destination* process runs
///     (`switch_agent_in_tmux`, which reads real process identity),
///  3. a pasted prompt is sent only once the destination is ready, and
///  4. the destination's own session recorded the prompt.
///
/// Any failure is journalled and returned as a retryable error *before* the
/// caller persists evidence or advances the workflow state, so the next
/// automation tick repeats the hand-off instead of trusting a lost prompt.
/// Returns the destination's provider session id when delivery was confirmed.
fn hand_off_and_deliver(
    db: &Database,
    task: &Task,
    handoff: &Handoff,
    runtime: &WorkflowRuntime,
) -> Result<Option<String>> {
    // The pane may not run what the task record says -- e.g. a previous
    // attempt already launched the destination before failing to confirm
    // delivery. Exit whatever is really there, with its own exit command.
    let running = agent_in_pane(runtime, handoff.target);
    let current = running.unwrap_or(handoff.source_agent);

    if let Some(running) = running {
        let busy =
            runtime
                .session_probe
                .turn_activity(running, Path::new(handoff.worktree), &task.id)
                == Some(crate::agent::native_session::TurnActivity::Busy);
        if busy {
            let deferrals = db
                .task_execution_events(&task.id)
                .map(|events| {
                    events
                        .iter()
                        .filter(|event| {
                            event.event_type == "agent_handoff_deferred"
                                && event.state.as_deref() == Some(handoff.journal_state)
                                && event.workflow_attempt == Some(handoff.journal_attempt)
                        })
                        .count()
                })
                .unwrap_or(0);
            if deferrals < MAX_IDLE_DEFERRALS {
                let message = format!(
                    "'{running}' is still working in tmux pane '{}'; hand-off to '{}' deferred until its turn ends",
                    handoff.target, handoff.destination_agent
                );
                record_handoff_event(
                    db,
                    task,
                    handoff,
                    "agent_handoff_deferred",
                    "retryable",
                    message.clone(),
                );
                bail!(message);
            }
            record_handoff_event(
                db,
                task,
                handoff,
                "agent_handoff_idle_timeout",
                "proceeding",
                format!("'{running}' still reported a turn in progress after {deferrals} deferrals; handing off anyway"),
            );
        }
    }

    let since = std::time::SystemTime::now();
    if let Err(error) = switch_agent_in_tmux(
        runtime.tmux_ops.as_ref(),
        handoff.target,
        current,
        handoff.destination_agent,
        handoff.command,
    ) {
        record_handoff_event(
            db,
            task,
            handoff,
            "agent_handoff_failed",
            "retryable",
            error.to_string(),
        );
        return Err(error);
    }
    deliver_to_launched_agent(db, task, handoff, since, runtime)
}

/// Steps 3 and 4 of [`hand_off_and_deliver`], shared with fresh-window
/// launches: wait for readiness before a paste, then confirm the prompt
/// reached the destination agent's own session.
fn deliver_to_launched_agent(
    db: &Database,
    task: &Task,
    handoff: &Handoff,
    since: std::time::SystemTime,
    runtime: &WorkflowRuntime,
) -> Result<Option<String>> {
    use crate::agent::native_session::Delivery;

    if handoff.paste_prompt {
        let ready = wait_for_agent_ready(
            runtime.tmux_ops,
            handoff.target,
            Some(handoff.destination_agent),
            runtime.config.auto_trust,
        );
        if ready.is_none() {
            let message = format!(
                "'{}' never became ready in tmux pane '{}'; prompt not pasted",
                handoff.destination_agent, handoff.target
            );
            record_handoff_event(
                db,
                task,
                handoff,
                "agent_handoff_failed",
                "retryable",
                message.clone(),
            );
            bail!(message);
        }
        runtime
            .tmux_ops
            .paste_text(handoff.target, handoff.prompt)?;
        runtime.tmux_ops.send_key(handoff.target, "C-m")?;
    }

    let marker = crate::agent::native_session::prompt_marker(handoff.prompt);
    let worktree = Path::new(handoff.worktree);
    let timeout = runtime.session_probe.delivery_timeout();
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match runtime.session_probe.find_delivered_prompt(
            handoff.destination_agent,
            worktree,
            &marker,
            since,
        ) {
            Delivery::Confirmed(session) => return Ok(Some(session)),
            Delivery::Unverifiable => {
                record_handoff_event(
                    db,
                    task,
                    handoff,
                    "agent_prompt_unverified",
                    "unverified",
                    format!(
                        "'{}' keeps no readable session store here; prompt delivery could not be confirmed",
                        handoff.destination_agent
                    ),
                );
                return Ok(None);
            }
            Delivery::Missing if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(500).min(timeout));
            }
            Delivery::Missing => {
                let message = format!(
                    "prompt for '{}' did not appear in its session within {}s (tmux pane '{}' reports '{}')",
                    handoff.destination_agent,
                    timeout.as_secs(),
                    handoff.target,
                    runtime.tmux_ops.pane_current_command(handoff.target).unwrap_or_default()
                );
                record_handoff_event(
                    db,
                    task,
                    handoff,
                    "agent_prompt_unconfirmed",
                    "retryable",
                    message.clone(),
                );
                bail!(message);
            }
        }
    }
}

/// Restore the immutable inputs bound to the current workflow-state attempt.
///
/// This is deliberately strict: a current worktree file is left untouched when
/// it already matches the expected digest, but a conflicting file is a
/// recoverable error rather than something AGTX silently overwrites. Callers
/// can therefore replay an interrupted prompt without accidentally reviewing a
/// newer plan or an older verdict.
pub fn restore_workflow_step_inputs(
    db: &Database,
    task: &Task,
    state: &WorkflowTaskState,
) -> Result<usize> {
    let worktree = task
        .worktree_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Workflow recovery requires an admitted worktree"))?;
    let worktree = Path::new(worktree);
    let inputs = db.workflow_step_inputs(&task.id, state.state_attempt, &state.state)?;
    let mut restored = 0;

    for input in inputs {
        let artifact = db
            .workflow_artifact(&input.artifact_id)?
            .ok_or_else(|| anyhow::anyhow!("Missing immutable artifact {}", input.artifact_id))?;
        if artifact.task_id != task.id
            || artifact.workflow_attempt >= state.state_attempt
            || artifact.sha256 != input.expected_sha256
        {
            anyhow::bail!(
                "Workflow input '{}' does not match its persisted binding",
                input.name
            );
        }
        let actual = format!("{:x}", Sha256::digest(&artifact.content));
        if actual != input.expected_sha256 {
            anyhow::bail!(
                "Workflow input '{}' failed its content digest check",
                input.name
            );
        }
        let source = Path::new(&artifact.source_path);
        let relative = source.strip_prefix(worktree).map_err(|_| {
            anyhow::anyhow!(
                "Workflow input '{}' is outside its task worktree",
                input.name
            )
        })?;
        let destination = worktree.join(relative);
        if destination.is_file() {
            let existing = std::fs::read(&destination)?;
            let existing_hash = format!("{:x}", Sha256::digest(&existing));
            if existing_hash != input.expected_sha256 {
                anyhow::bail!(
                    "Workflow input '{}' conflicts with {}",
                    input.name,
                    destination.display()
                );
            }
            continue;
        }
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&destination, &artifact.content)?;
        restored += 1;
    }
    Ok(restored)
}
/// Replay an interrupted agent-owned step without creating a new workflow
/// attempt. The exact stored prompt and exact bound inputs are the contract;
/// a completed artifact is never replayed.
pub fn restart_workflow_step(
    task: &Task,
    db: &Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(state) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    let reports = db.task_step_reports(&task.id)?;
    let Some(report) = reports.into_iter().find(|report| {
        report.workflow_attempt == state.state_attempt && report.state == state.state
    }) else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "This workflow step has no persisted prompt to restart".to_string(),
        });
    };
    if report.artifact_sha256.is_some() {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "This workflow step already has durable output; submit or resolve it instead of restarting".to_string(),
        });
    }
    let Some(prompt) = report.prompt_text else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "This workflow step has no persisted prompt to restart".to_string(),
        });
    };
    let Some(target) = task.session_name.as_deref() else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "This workflow step has no task session to restart".to_string(),
        });
    };
    if !runtime.tmux_ops.window_exists(target)? {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "This workflow step's task session is unavailable; recover the session first"
                .to_string(),
        });
    }
    restore_workflow_step_inputs(db, task, &state)?;
    runtime.tmux_ops.paste_text(target, &prompt)?;
    runtime.tmux_ops.send_key(target, "C-m")?;

    let mut event = TaskExecutionEvent::new(&task.id, "agent_prompt_restarted");
    event.workflow_attempt = Some(state.state_attempt);
    event.state = Some(state.state);
    event.agent = task.agent.clone().into();
    event.outcome = Some("restarted".to_string());
    event.message = Some("Replayed persisted workflow prompt with verified inputs".to_string());
    db.record_task_execution_event(&event)?;
    Ok(WorkflowStepOutcome::Recovered {
        message: "Workflow step restarted from persisted inputs".to_string(),
    })
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
    let destination = workflow.state(&edge.to).ok_or_else(|| {
        anyhow::anyhow!(
            "workflow transition '{}' has no destination state",
            edge.action
        )
    })?;
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
    let mut transition =
        WorkflowTransitionRecord::new(&current.task_id, &edge.action, &edge.from, &edge.to);
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
    /// Reads the providers' own session stores to tell whether an agent is
    /// mid-turn and whether a prompt reached the intended agent.
    pub session_probe: &'a dyn crate::agent::native_session::SessionProbe,
    /// Opens the pull request for a task whose integration conflicts, so the
    /// conflict can be resolved outside the task's container. `None` skips
    /// that step; the branch is still pushed.
    pub git_provider_ops: Option<&'a Arc<dyn GitProviderOperations>>,
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
    /// Work was safely replayed without changing workflow state or attempt.
    Recovered { message: String },
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
    let base_sha =
        match crate::git::resolve_commit(runtime.project_path, &project_workflow.target_branch) {
            Ok(sha) => sha,
            Err(error) => {
                return Ok(WorkflowStepOutcome::Blocked {
                    message: format!("Cannot admit task: {error}"),
                });
            }
        };
    let admission = match prepare_admission(
        workflow,
        project_workflow,
        &task,
        dependencies_resolved,
        base_sha.clone(),
    ) {
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

    if let Err(error) = db.record_workflow_admission(&task, &admission.state, &admission.transition)
    {
        let _ = runtime
            .git_ops
            .remove_worktree(runtime.project_path, &worktree_path);
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

/// Revoke a pre-planning admission only when no task work could be lost.
///
/// Git resources are removed before the database allocation is cleared. If a
/// git operation fails, the durable admission is deliberately left intact so
/// the operator can retry or recover it rather than being told a missing
/// checkout is still usable.
pub fn revoke_workflow_admission(
    task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    if task.status != TaskStatus::Backlog {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Only Backlog tasks can revoke admission".into(),
        });
    }
    if task.session_name.is_some() {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Stop the active task session before revoking admission".into(),
        });
    }
    let Some(worktree) = task.worktree_path.as_deref() else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Task has no admission to revoke".into(),
        });
    };
    let Some(branch) = task.branch_name.as_deref() else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Admission has no task branch; refusing cleanup".into(),
        });
    };
    let Some(state) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Admission has no workflow state; refusing cleanup".into(),
        });
    };
    if !matches!(state.state.as_str(), "admission" | "ready_for_planning") {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Only an unstarted admission can be revoked".into(),
        });
    }
    let Some(base_sha) = state.base_sha.as_deref() else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Admission has no frozen base; refusing cleanup".into(),
        });
    };
    if runtime.git_ops.has_changes(Path::new(worktree)) {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Worktree has uncommitted changes; refusing to revoke admission".into(),
        });
    }
    let head = match crate::git::resolve_commit(runtime.project_path, branch) {
        Ok(head) => head,
        Err(error) => {
            return Ok(WorkflowStepOutcome::Blocked {
                message: format!("Cannot verify task branch before revocation: {error}"),
            })
        }
    };
    if head != base_sha {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Task branch contains commits beyond its admission base; refusing to revoke"
                .into(),
        });
    }
    if let Err(error) = runtime
        .git_ops
        .remove_worktree(runtime.project_path, worktree)
    {
        return Ok(WorkflowStepOutcome::Blocked {
            message: format!("Could not remove admitted worktree: {error}"),
        });
    }
    if let Err(error) = runtime.git_ops.delete_branch(runtime.project_path, branch) {
        return Ok(WorkflowStepOutcome::Blocked { message: format!("Worktree removed but branch cleanup failed; admission remains recorded for recovery: {error}") });
    }
    db.revoke_workflow_admission(&task, &state)?;
    Ok(WorkflowStepOutcome::Advanced {
        task,
        message: "Admission revoked; task is Ready without an allocated worktree".into(),
    })
}

/// Reset a declarative-workflow task from any state. Explicit UI confirmation
/// authorizes discarding its worktree, branch commits, and workflow evidence.
/// Durable state is not changed until external cleanup succeeds.
pub fn reset_workflow_to_backlog(
    task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(state) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Task has no declarative workflow attempt to reset".into(),
        });
    };
    // Keep the standard planning artifact locally for reuse. The backup is
    // made before any destructive operation and its failure aborts the reset.
    if let Some(worktree) = task.worktree_path.as_deref() {
        let plan = Path::new(worktree).join(".agtx").join("plan.md");
        if plan.is_file() {
            backup_plan(&plan, runtime.project_path, &task.title)?;
        }
    }
    if let Some(session) = task.session_name.as_deref() {
        if let Err(error) = runtime.tmux_ops.kill_window(session) {
            return Ok(WorkflowStepOutcome::Blocked {
                message: format!("Could not stop task session; reset was not started: {error}"),
            });
        }
    }
    if let Some(worktree) = task.worktree_path.as_deref() {
        if let Err(error) = runtime
            .git_ops
            .remove_worktree(runtime.project_path, worktree)
        {
            return Ok(WorkflowStepOutcome::Blocked {
                message: format!(
                    "Could not remove task worktree; reset remains recoverable: {error}"
                ),
            });
        }
    }
    if let Some(branch) = task.branch_name.as_deref() {
        if let Err(error) = runtime.git_ops.delete_branch(runtime.project_path, branch) {
            return Ok(WorkflowStepOutcome::Blocked {
                message: format!(
                    "Worktree removed but branch cleanup failed; reset remains recoverable: {error}"
                ),
            });
        }
    }
    db.reset_workflow_to_backlog(&task, &state)?;
    Ok(WorkflowStepOutcome::Advanced {
        task,
        message: "Task reset to Backlog; planning evidence was cleared".into(),
    })
}

fn backup_plan(source: &Path, project_path: &Path, title: &str) -> Result<PathBuf> {
    let safe: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let stem = safe.trim_matches('-');
    let dir = project_path.join(".plans-backup");
    std::fs::create_dir_all(&dir)?;
    let mut destination = dir.join(format!("{stem}-plan.md"));
    let mut suffix = 2;
    while destination.exists() {
        destination = dir.join(format!("{stem}-plan-{suffix}.md"));
        suffix += 1;
    }
    std::fs::copy(source, &destination)?;
    Ok(destination)
}
/// Extracted body of `App::start_selected_workflow_planning`.
///
/// Planning is deliberately restartable: a task can retain its durable
/// admission evidence while a terminal or agent process exits, in which case
/// this relaunches the planner rather than merely realigning state.
///
/// `require_plan_approval` answers the Shift+S "require my approval" popup;
/// it only takes effect on a fresh start (not a restart of an
/// already-`planning` task, which has no fresh transition to stamp it onto)
/// and is stored on the task's `WorkflowTaskState::human_gate_plan_approval`
/// so `assess`'s `apply_human_gates` can hold this one task's `approve_plan`
/// for a human even when the project-wide `[automation].human_gates` list
/// does not name it.
pub fn start_workflow_planning(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    mut task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
    require_plan_approval: bool,
) -> Result<WorkflowStepOutcome> {
    // In just-in-time mode a dependency-ready card deliberately has no
    // allocation yet. Admission and planning start share this one operator
    // action so the frozen base is as current as possible.
    if task.worktree_path.is_none() {
        match admit_task(
            workflow,
            project_workflow,
            plugin,
            task.clone(),
            db,
            runtime,
        )? {
            WorkflowStepOutcome::Advanced { task: admitted, .. } => task = admitted,
            outcome => return Ok(outcome),
        }
    }
    let Some(worktree) = task.worktree_path.clone() else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Could not create an admitted worktree for planning".into(),
        });
    };
    let Some(current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Admission did not persist workflow state".into(),
        });
    };

    let restarting = current.state == "planning";
    let (planner, planning_state, planning_attempt, transitions) = if restarting {
        let Some(role) = workflow
            .state("planning")
            .and_then(|state| state.role.as_ref())
        else {
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
        let mut planning = match prepare_transition(
            workflow,
            project_workflow,
            before_planning,
            "start_planning",
            GuardContext::default(),
        ) {
            Ok(value) => value,
            Err(error) => {
                return Ok(WorkflowStepOutcome::Blocked {
                    message: format!("Cannot start planning: {error}"),
                });
            }
        };
        planning.state.human_gate_plan_approval = require_plan_approval;
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
        (
            agent,
            planning.state.state.clone(),
            planning.state.state_attempt,
            Some(transitions),
        )
    };

    let agent_ops = runtime.agent_registry.get(&planner);
    // The plan's revision and workflow attempt are state-machine facts, not
    // values a replacement terminal agent should infer from a missing file or
    // a prior pane.  A planner can be relaunched after its original agent
    // exits before saving an artifact, so give every launch the exact next
    // revision as well as the current attempt.  The SHA-256 deliberately is
    // absent: it belongs to the bytes the executor reads after the plan is
    // saved, and cannot be predicted or copied into the plan artifact.
    let prompt = format!(
        "{}\n\nAGTX owns revision, workflow-attempt, and SHA-256 metadata in its database. Write only the plan content at the required path; do not add, infer, or copy orchestration metadata into the artifact.",
        resolve_prompt(&Some(plugin.clone()), "planning", &task.content_text(), &task.id, task.cycle),
    );
    let slug = generate_task_slug(&task.id, &task.title);
    let window_name = format!("task-{slug}");
    let target = format!("{}:{window_name}", runtime.tmux_project_name);
    ensure_project_tmux_session(
        runtime.tmux_project_name,
        runtime.project_path,
        runtime.tmux_ops.as_ref(),
    );
    let policy = project_workflow.policy_for_state(workflow, &planning_state)?;

    if let Some(transitions) = &transitions {
        for prepared in transitions {
            db.advance_workflow_state(&prepared.state, &prepared.transition)?;
        }
    }

    // Only pass a prompt through argv for agent forms that are verified to
    // submit it. OpenCode's `--prompt` merely opens its interactive composer,
    // so it must use the post-readiness paste path even on a planner relaunch.
    let can_embed =
        crate::agent::spec::can_launch_with_prompt(agent_ops.prompt_injection(), &prompt);
    let confirmed_session =
        if restarting && runtime.tmux_ops.window_exists(&target).unwrap_or(false) {
            let command = build_policy_agent_command(
                agent_ops.as_ref(),
                &planner,
                if can_embed { &prompt } else { "" },
                policy.as_ref(),
                Some(Path::new(&worktree)),
            );
            hand_off_and_deliver(
                db,
                &task,
                &Handoff {
                    journal_state: &planning_state,
                    journal_attempt: planning_attempt,
                    source_agent: &task.agent,
                    destination_agent: &planner,
                    target: &target,
                    worktree: &worktree,
                    command: &command,
                    prompt: &prompt,
                    paste_prompt: !can_embed,
                },
                runtime,
            )?
        } else {
            // A prompt embedded directly in the launch command becomes part of
            // one tmux client/server message; large enough (see
            // `MAX_LAUNCH_PROMPT_BYTES`'s doc comment for the measured ceiling)
            // and tmux itself rejects it with "command too long" before the
            // window is ever created. Past that ceiling, launch with no prompt
            // and deliver it afterward via paste_text (stdin, no such limit)
            // instead — same approach already used by the TUI's agent-switch
            // launch flow.
            let command = build_policy_agent_command(
                agent_ops.as_ref(),
                &planner,
                if can_embed { &prompt } else { "" },
                policy.as_ref(),
                Some(Path::new(&worktree)),
            );
            let since = std::time::SystemTime::now();
            runtime.tmux_ops.create_window(
                runtime.tmux_project_name,
                &window_name,
                &worktree,
                Some(command.clone()),
                true,
                &agtx_task_env(&task.id, &worktree),
            )?;
            deliver_to_launched_agent(
                db,
                &task,
                &Handoff {
                    journal_state: &planning_state,
                    journal_attempt: planning_attempt,
                    source_agent: &task.agent,
                    destination_agent: &planner,
                    target: &target,
                    worktree: &worktree,
                    command: &command,
                    prompt: &prompt,
                    paste_prompt: !can_embed,
                },
                since,
                runtime,
            )?
        };
    record_agent_prompt(
        db,
        &task,
        &planning_state,
        planning_attempt,
        &planner,
        &prompt,
        confirmed_session.as_deref(),
    )?;
    record_artifact_baseline(
        db,
        &task,
        &planning_state,
        planning_attempt,
        &planning_artifact_path(&worktree, plugin, &task.id)?,
    )?;

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
    let contents = std::fs::read(&path)
        .map_err(|_| anyhow::anyhow!("Missing planning artifact: {}", path.display()))?;
    // Revision is allocated by AGTX when this exact byte snapshot is accepted;
    // planners never author workflow metadata.
    let revision = current.plan_revision + 1;
    let mut evidenced = current.clone();
    evidenced.plan_revision = revision;
    evidenced.plan_hash = Some(format!("{:x}", Sha256::digest(&contents)));
    let handoff = prepare_transition(
        workflow,
        project_workflow,
        &evidenced,
        "submit_plan",
        GuardContext::default(),
    )?;
    let reviewer = handoff
        .destination_agent
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Plan review state has no bound agent"))?;
    let review_artifact_path = workflow_artifact_path(
        &worktree,
        plugin.artifacts.plan_review.as_deref(),
        &task.id,
        ".agent-flow/plan-review.yaml",
    );
    let prompt = format!(
        "You are the plan reviewer for task {}. Review only {} (AGTX revision {}, immutable artifact {}). Do not implement code. Review only the submitted revision and its stated acceptance criteria. Request changes only for a contradiction with the task, approved specification, repository rule, or API contract; a safety, data-integrity, tenancy, migration, or transaction-ownership defect; or an acceptance criterion that cannot be delivered or validated from the plan. Classify every finding as BLOCKING, REQUIRED-NONBLOCKING, or SUGGESTION. Every BLOCKING finding must cite the exact conflicting plan text and governing requirement. Do not request changes merely because a summary is less detailed than executable steps, an implied documentation update is not repeated elsewhere, an ordinary targeted test is not enumerated, or a resolved finding is phrased differently in non-normative text. On a revision, verify whether prior findings are resolved and do not reopen them or introduce adjacent scope unless the revised text creates a new material contradiction. Prefer one consolidated set of actionable findings. Use changes_requested only when at least one BLOCKING finding exists; otherwise use approved and record required-nonblocking items as implementation/checklist notes. Then write {} in this task worktree containing: verdict: approved or verdict: changes_requested (exactly one of these two strings), findings: a non-empty folded scalar with specific, concrete findings, and final_report: a concise reviewer handoff summary. AGTX owns all revision, attempt, and SHA-256 metadata; do not write any of them into the review artifact.",
        task.id,
        path.strip_prefix(&worktree).unwrap_or(&path).display(),
        revision,
        "stored internally",
        review_artifact_path
            .strip_prefix(&worktree)
            .unwrap_or(&review_artifact_path)
            .display(),
    );
    let target = task
        .session_name
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Planning session is unavailable"))?;
    let previous_agent = task.agent.clone();
    let policy = project_workflow.policy_for_state(workflow, &handoff.state.state)?;
    // This handoff reuses the planner's pane. Do not expose `plan_review` to
    // automation until the reviewer is actually running; otherwise a prior
    // asynchronous switch can still own the pane and leave this task at bash.
    let command = build_policy_agent_command(
        runtime.agent_registry.get(&reviewer).as_ref(),
        &reviewer,
        &prompt,
        policy.as_ref(),
        Some(Path::new(&worktree)),
    );
    // Snapshot before the switch so the reviewer receives one exact revision,
    // but do not persist it yet.  A failed tmux hand-off is recoverable and
    // must not leave immutable evidence in the still-current planning state.
    let plan_evidence = snapshot_step_evidence(&task, &current, &path)?;
    let confirmed_session = hand_off_and_deliver(
        db,
        &task,
        &Handoff {
            journal_state: &current.state,
            journal_attempt: current.state_attempt,
            source_agent: &previous_agent,
            destination_agent: &reviewer,
            target: &target,
            worktree: &worktree,
            command: &command,
            prompt: &prompt,
            paste_prompt: false,
        },
        runtime,
    )?;
    let plan_evidence = persist_step_evidence(
        db,
        &task,
        &current,
        &task.agent,
        &path,
        plan_evidence,
        runtime,
    )?;
    record_agent_prompt(
        db,
        &task,
        &handoff.state.state,
        handoff.state.state_attempt,
        &reviewer,
        &prompt,
        confirmed_session.as_deref(),
    )?;
    db.bind_workflow_step_input(&WorkflowStepInput {
        task_id: task.id.clone(),
        workflow_attempt: handoff.state.state_attempt,
        state: handoff.state.state.clone(),
        name: "plan".to_string(),
        artifact_id: plan_evidence.id.clone(),
        expected_sha256: plan_evidence.sha256.clone(),
        created_at: chrono::Utc::now(),
    })?;
    db.advance_workflow_state(&handoff.state, &handoff.transition)?;
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
    let action = if approve {
        "approve_plan"
    } else {
        "plan_changes_requested"
    };
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
    if !approve {
        let Some(worktree) = task.worktree_path.as_deref() else {
            return Ok(WorkflowStepOutcome::Blocked {
                message: "Plan revision requires an admitted worktree".into(),
            });
        };
        let review_path = workflow_artifact_path(
            worktree,
            plugin.artifacts.plan_review.as_deref(),
            &task.id,
            ".agent-flow/plan-review.yaml",
        );
        if !review_path.is_file() {
            return Ok(WorkflowStepOutcome::Blocked {
                message: "Plan changes require a persisted plan-review artifact; write the review before returning to Planning".into(),
            });
        }
        // Validate the mutable file *before* promoting its bytes into the
        // immutable evidence slot.  A malformed review must remain editable,
        // not strand the workflow behind an unreplaceable snapshot.
        let findings = match workflow_artifact_value(&review_path, "findings") {
            Ok(findings) => findings,
            Err(_) => {
                return Ok(WorkflowStepOutcome::Blocked {
                    message: "Plan changes require non-empty findings in plan-review.yaml".into(),
                });
            }
        };
        // Keep the review in memory until the planner process has actually
        // launched.  Persisting it first used to leave a bound-looking
        // `step_evidence` row behind whenever tmux failed, turning a retry
        // into an artifact conflict instead of a clean hand-off.
        let review_evidence = snapshot_step_evidence(&task, &current, &review_path)?;

        let Some(target) = task.session_name.clone() else {
            return Ok(WorkflowStepOutcome::Blocked {
                message: "Planning session is unavailable".into(),
            });
        };
        let findings: String = findings.chars().take(12 * 1024).collect();
        let prompt = format!(
            "Plan review requested changes for task {}. Revise .agtx/plans/{}.md and do not implement code. The exact review input is AGTX artifact {}; its recorded findings follow:\n---\n{}\n---\n\nAGTX owns revision, workflow-attempt, and SHA-256 metadata. Write only the revised plan content; do not add or infer orchestration fields. When complete, save the artifact for another Shift+V submission.",
            task.id, task.id, review_evidence.id,
            findings,
        );
        let policy = project_workflow.policy_for_state(workflow, &decision.state.state)?;
        let planner_ops = runtime.agent_registry.get(&task.agent);
        // OpenCode's `--prompt` is a composer prefill, not a submitted
        // interactive turn.  More importantly, reviewer findings are free
        // text and must never be interpolated into the shell command used to
        // switch away from Codex.  Use the same verified-launch gate as the
        // ordinary planning entry point: launch bare, wait for the new agent,
        // then paste and submit the revision request.
        let can_embed =
            crate::agent::spec::can_launch_with_prompt(planner_ops.prompt_injection(), &prompt);
        let command = build_policy_agent_command(
            planner_ops.as_ref(),
            &task.agent,
            if can_embed { &prompt } else { "" },
            policy.as_ref(),
            Some(Path::new(worktree)),
        );
        let confirmed_session = hand_off_and_deliver(
            db,
            &task,
            &Handoff {
                journal_state: &current.state,
                journal_attempt: current.state_attempt,
                source_agent: &previous_agent,
                destination_agent: &task.agent,
                target: &target,
                worktree,
                command: &command,
                prompt: &prompt,
                paste_prompt: !can_embed,
            },
            runtime,
        )?;
        let review_evidence = persist_step_evidence(
            db,
            &task,
            &current,
            &previous_agent,
            &review_path,
            review_evidence,
            runtime,
        )?;
        db.bind_workflow_step_input(&WorkflowStepInput {
            task_id: task.id.clone(),
            workflow_attempt: decision.state.state_attempt,
            state: decision.state.state.clone(),
            name: "plan_review".to_string(),
            artifact_id: review_evidence.id.clone(),
            expected_sha256: review_evidence.sha256.clone(),
            created_at: chrono::Utc::now(),
        })?;
        record_agent_prompt(
            db,
            &task,
            &decision.state.state,
            decision.state.state_attempt,
            &task.agent,
            &prompt,
            confirmed_session.as_deref(),
        )?;
    }
    db.advance_workflow_state(&decision.state, &decision.transition)?;
    db.update_task(&task)?;
    let message = if approve {
        "Plan approved"
    } else {
        "Plan changes requested; returned to Planning"
    }
    .to_string();
    Ok(WorkflowStepOutcome::Advanced { task, message })
}

/// Artifact-driven counterpart to `decide_workflow_plan`: reads the plan
/// reviewer's durable `.agent-flow/plan-review.yaml` verdict and dispatches
/// to the exact same approve/reject logic a human's Shift+Y/Shift+N would
/// invoke, rather than duplicating it. Automation acts only on durable
/// evidence written to disk, never by scraping the reviewer's tmux pane.
pub fn submit_plan_review(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> Result<WorkflowStepOutcome> {
    let Some(worktree) = task.worktree_path.clone() else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    let Some(current) = db.get_workflow_task_state(&task.id)? else {
        return Ok(WorkflowStepOutcome::NoOp);
    };
    if current.state != "plan_review" {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Submit plan review is only available in Plan review".into(),
        });
    }
    let artifact = workflow_artifact_path(
        &worktree,
        plugin.artifacts.plan_review.as_deref(),
        &task.id,
        ".agent-flow/plan-review.yaml",
    );
    let verdict = workflow_artifact_value(&artifact, "verdict")?;
    let approve = match verdict.as_str() {
        "approved" => true,
        "changes_requested" => false,
        _ => bail!(
            "{} has unsupported plan-review verdict '{verdict}'",
            artifact.display()
        ),
    };
    // A rejected review must carry usable scalar findings before its mutable
    // bytes are promoted to immutable evidence.  `decide_workflow_plan` has
    // the same guard for manual decisions; keep the automated entry point
    // equally atomic.
    if !approve && workflow_artifact_value(&artifact, "findings").is_err() {
        return Ok(WorkflowStepOutcome::Blocked {
            message: "Plan changes require non-empty findings in plan-review.yaml".into(),
        });
    }
    // Approval has no external process hand-off, so it can promote the review
    // immediately. A requested revision does switch agents; its exact snapshot
    // is deliberately held by `decide_workflow_plan` until that switch is
    // acknowledged, otherwise a failed launch strands provisional evidence.
    if approve {
        record_step_evidence(db, &task, &current, &task.agent, &artifact, runtime)?;
    }
    decide_workflow_plan(
        workflow,
        project_workflow,
        plugin,
        task,
        approve,
        db,
        runtime,
    )
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
    let prompt = format!(
        "{}\n\nAGTX has bound the approved plan internally. Do not write revision, workflow-attempt, or SHA-256 metadata into your result artifact.",
        resolve_prompt(&Some(plugin.clone()), "running", &task.content_text(), &task.id, task.cycle),
    );
    let policy = project_workflow.policy_for_state(workflow, &implementation.state.state)?;
    let agent_ops = runtime.agent_registry.get(&implementer);
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
    // Keep unverified prompt forms out of both a fresh launch and an existing
    // session switch. In particular, `opencode --prompt` leaves text unsent in
    // the composer, which makes a successful process launch look like a
    // completed workflow action.
    let can_embed =
        crate::agent::spec::can_launch_with_prompt(agent_ops.prompt_injection(), &prompt);
    let confirmed_session = if session_available {
        let command = build_policy_agent_command(
            agent_ops.as_ref(),
            &implementer,
            if can_embed { &prompt } else { "" },
            policy.as_ref(),
            Some(Path::new(&worktree)),
        );
        hand_off_and_deliver(
            db,
            &task,
            &Handoff {
                journal_state: &implementation.state.state,
                journal_attempt: implementation.state.state_attempt,
                source_agent: &task.agent,
                destination_agent: &implementer,
                target: &target,
                worktree: &worktree,
                command: &command,
                prompt: &prompt,
                paste_prompt: !can_embed,
            },
            runtime,
        )?
    } else {
        ensure_project_tmux_session(
            runtime.tmux_project_name,
            runtime.project_path,
            runtime.tmux_ops.as_ref(),
        );
        // See the matching comment in `start_workflow_planning`: a prompt
        // embedded directly in the launch command can overflow tmux's own
        // command-line re-exec ("command too long"). Defer oversized prompts
        // to a post-launch paste_text instead.
        let command = build_policy_agent_command(
            agent_ops.as_ref(),
            &implementer,
            if can_embed { &prompt } else { "" },
            policy.as_ref(),
            Some(Path::new(&worktree)),
        );
        let since = std::time::SystemTime::now();
        runtime.tmux_ops.create_window(
            runtime.tmux_project_name,
            &window_name,
            &worktree,
            Some(command.clone()),
            true,
            &agtx_task_env(&task.id, &worktree),
        )?;
        deliver_to_launched_agent(
            db,
            &task,
            &Handoff {
                journal_state: &implementation.state.state,
                journal_attempt: implementation.state.state_attempt,
                source_agent: &task.agent,
                destination_agent: &implementer,
                target: &target,
                worktree: &worktree,
                command: &command,
                prompt: &prompt,
                paste_prompt: !can_embed,
            },
            since,
            runtime,
        )?
    };
    record_agent_prompt(
        db,
        &task,
        &implementation.state.state,
        implementation.state.state_attempt,
        &implementer,
        &prompt,
        confirmed_session.as_deref(),
    )?;
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
    let artifact = workflow_artifact_path(
        &worktree,
        plugin.artifacts.running.as_deref(),
        &task.id,
        ".agent-flow/implementation-result.yaml",
    );
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
    let review = prepare_transition(
        workflow,
        project_workflow,
        &implemented.state,
        "start_engineering_review",
        GuardContext::default(),
    )?;
    let reviewer = review
        .destination_agent
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Engineering review state has no bound agent"))?;
    let prompt = format!(
        "{}\n\nAGTX owns workflow-attempt and SHA-256 metadata. Do not write orchestration metadata into your result artifact.",
        resolve_prompt(&Some(plugin.clone()), "review", &task.content_text(), &task.id, task.cycle),
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
    let evidence = snapshot_step_evidence(&task, &current, &artifact)?;
    // A failed hand-off must leave the current state fully retryable: retain the
    // evidence snapshot in memory, launch first, then persist it after launch.
    let confirmed_session = hand_off_and_deliver(
        db,
        &task,
        &Handoff {
            journal_state: &current.state,
            journal_attempt: current.state_attempt,
            source_agent: &task.agent,
            destination_agent: &reviewer,
            target: &target,
            worktree: &worktree,
            command: &command,
            prompt: &prompt,
            paste_prompt: false,
        },
        runtime,
    )?;
    persist_step_evidence(
        db,
        &task,
        &current,
        &task.agent,
        &artifact,
        evidence,
        runtime,
    )?;
    db.advance_workflow_state_chain(&[
        (&implemented.state, &implemented.transition),
        (&review.state, &review.transition),
    ])?;
    record_agent_prompt(
        db,
        &task,
        &review.state.state,
        review.state.state_attempt,
        &reviewer,
        &prompt,
        confirmed_session.as_deref(),
    )?;
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
    let artifact = workflow_artifact_path(
        &worktree,
        plugin.artifacts.review.as_deref(),
        &task.id,
        ".agent-flow/engineering-review.yaml",
    );
    let verdict = workflow_artifact_value(&artifact, "verdict")?;
    ensure_review_addresses_failed_validation(&worktree, plugin, &task.id, &artifact, &verdict)?;
    let (action, phase, status) = match verdict.as_str() {
        "corrections_required" => (
            "engineering_corrections_required",
            "running",
            TaskStatus::Running,
        ),
        "plan_issue" => ("engineering_plan_issue", "planning", TaskStatus::Planning),
        "approved_for_validation" => (
            "start_final_validation",
            "final_validation",
            TaskStatus::Review,
        ),
        _ => bail!(
            "{} has unsupported engineering-review verdict '{verdict}'",
            artifact.display()
        ),
    };
    let transition = prepare_transition(
        workflow,
        project_workflow,
        &current,
        action,
        GuardContext::default(),
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
        "{}\n\nEngineering-review verdict: {verdict}. Evidence: {}. Follow the declared role policy; do not commit, push, create a PR, merge, or bypass controls.\nAGTX owns workflow-attempt and SHA-256 metadata; do not write it into your artifact.",
        resolve_prompt(&Some(plugin.clone()), phase, &task.content_text(), &task.id, task.cycle),
        artifact.strip_prefix(&worktree).unwrap_or(&artifact).display(),
    );
    let policy = project_workflow.policy_for_state(workflow, &transition.state.state)?;
    let command = build_policy_agent_command(
        runtime.agent_registry.get(&next_agent).as_ref(),
        &next_agent,
        &prompt,
        policy.as_ref(),
        Some(Path::new(&worktree)),
    );
    let evidence = snapshot_step_evidence(&task, &current, &artifact)?;
    // Preserve no immutable evidence until the receiving role is confirmed.
    let confirmed_session = hand_off_and_deliver(
        db,
        &task,
        &Handoff {
            journal_state: &current.state,
            journal_attempt: current.state_attempt,
            source_agent: &task.agent,
            destination_agent: &next_agent,
            target: &target,
            worktree: &worktree,
            command: &command,
            prompt: &prompt,
            paste_prompt: false,
        },
        runtime,
    )?;
    persist_step_evidence(
        db,
        &task,
        &current,
        &task.agent,
        &artifact,
        evidence,
        runtime,
    )?;
    db.advance_workflow_state(&transition.state, &transition.transition)?;
    record_agent_prompt(
        db,
        &task,
        &transition.state.state,
        transition.state.state_attempt,
        &next_agent,
        &prompt,
        confirmed_session.as_deref(),
    )?;
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
    let artifact = workflow_artifact_path(
        &worktree,
        plugin.artifacts.final_validation.as_deref(),
        &task.id,
        ".agent-flow/final-validation.yaml",
    );
    let verdict = workflow_artifact_value(&artifact, "verdict")?;
    let (action, phase, passed) = match verdict.as_str() {
        "passed" => ("begin_feature_integration", "integration", true),
        "failed" => ("validation_failed", "review", false),
        _ => bail!(
            "{} has unsupported final-validation verdict '{verdict}'",
            artifact.display()
        ),
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
        "{}\n\nFinal-validation verdict: {verdict}. Evidence: {}.{} Follow the declared role policy; do not merge feature/poc into main. AGTX owns workflow-attempt and SHA-256 metadata; do not write it into your artifact.",
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
        let review_artifact = workflow_artifact_path(
            &worktree,
            plugin.artifacts.review.as_deref(),
            &task.id,
            ".agent-flow/engineering-review.yaml",
        );
        archive_workflow_artifact(&review_artifact, "superseded-after-validation-failure")?;
    }
    let evidence = snapshot_step_evidence(&task, &current, &artifact)?;
    // Preserve no immutable evidence until the receiving role is confirmed.
    let confirmed_session = hand_off_and_deliver(
        db,
        &task,
        &Handoff {
            journal_state: &current.state,
            journal_attempt: current.state_attempt,
            source_agent: &task.agent,
            destination_agent: &next_agent,
            target: &target,
            worktree: &worktree,
            command: &command,
            prompt: &prompt,
            paste_prompt: false,
        },
        runtime,
    )?;
    persist_step_evidence(
        db,
        &task,
        &current,
        &task.agent,
        &artifact,
        evidence,
        runtime,
    )?;
    db.advance_workflow_state(&transition.state, &transition.transition)?;
    record_agent_prompt(
        db,
        &task,
        &transition.state.state,
        transition.state.state_attempt,
        &next_agent,
        &prompt,
        confirmed_session.as_deref(),
    )?;
    task.status = TaskStatus::Review;
    task.agent = next_agent;
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;
    Ok(WorkflowStepOutcome::Advanced {
        message: format!("Final validation recorded: {verdict}"),
        task,
    })
}

/// How long the executor waits between retries of an integration it could not
/// finish ([`Task::has_unresolved_integration`]). Each retry fetches from the
/// remote, so it runs on this cadence rather than on every automation tick.
pub const INTEGRATION_RETRY_SECONDS: i64 = 60;

/// Upper bound for the reason stored in `escalation_note`: git and `gh`
/// errors can carry whole hook outputs.
const INTEGRATION_NOTE_LIMIT: usize = 600;

/// Where one integration attempt ended, before it is recorded on the task.
enum IntegrationAttempt {
    /// The task branch is in the target: merged by this attempt, or merged on
    /// the remote (its pull request) and fast-forwarded into the checkout.
    Integrated { target_branch: String },
    /// The pushed task branch conflicts with the target.
    Conflicts {
        target_branch: String,
        files: Vec<String>,
        /// Why no pull request could be opened, when one was due.
        pull_request_error: Option<String>,
    },
}

/// Extracted body of `App::complete_selected_feature_integration`.
///
/// Executes the narrowly-scoped, reviewer-authorized task integration into
/// the configured target, which may never be `main`. The task branch is
/// committed and pushed *before* the merge is attempted, so the work always
/// leaves the task's container, even when the merge cannot happen:
///
/// - a clean merge completes the task;
/// - a conflicting merge records [`INTEGRATION_CONFLICTS`] with the paths,
///   opens a pull request against the target (when the state policy allows
///   it) and escalates;
/// - any other failure records [`INTEGRATION_BLOCKED`] with the reason.
///
/// [`assess`] retries both unresolved states every
/// [`INTEGRATION_RETRY_SECONDS`]. A retry first adopts what happened outside
/// the container: a resolution pushed to the task branch is fast-forwarded
/// into the worktree, and a task branch already merged on the remote (its
/// pull request) completes the task.
///
/// [`INTEGRATION_CONFLICTS`]: crate::db::INTEGRATION_CONFLICTS
/// [`INTEGRATION_BLOCKED`]: crate::db::INTEGRATION_BLOCKED
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
    let artifact = workflow_artifact_path(
        &worktree,
        plugin.artifacts.integration.as_deref(),
        &task.id,
        ".agent-flow/integration-ready.yaml",
    );
    if workflow_artifact_value(&artifact, "verdict")? != "ready_for_integration" {
        bail!(
            "{} must declare verdict: ready_for_integration",
            artifact.display()
        );
    }
    // A retry re-reads the artifact the first attempt already journalled;
    // recording it again would duplicate that evidence once per retry.
    if !task.has_unresolved_integration() {
        record_step_evidence(db, &task, &current, &task.agent, &artifact, runtime)?;
    }

    let attempt = attempt_feature_integration(
        workflow,
        project_workflow,
        &mut task,
        &current,
        Path::new(&worktree),
        &branch,
        runtime,
    );
    let target_branch = match attempt {
        Ok(IntegrationAttempt::Integrated { target_branch }) => target_branch,
        Ok(IntegrationAttempt::Conflicts {
            target_branch,
            files,
            pull_request_error,
        }) => {
            let pull_request = match (&task.pr_url, pull_request_error) {
                (Some(url), _) => format!("Pull request: {url}."),
                (None, Some(error)) => format!(
                    "Branch '{branch}' is pushed; no pull request could be opened: {error}."
                ),
                (None, None) => format!("Branch '{branch}' is pushed."),
            };
            let note = format!(
                "Has merge conflicts with '{target_branch}': {}. {pull_request} Resolve them on \
                 the task branch and push, or merge the pull request; AGTX retries on its own.",
                files.join(", ")
            );
            return record_unresolved_integration(
                db,
                task,
                &current,
                crate::db::INTEGRATION_CONFLICTS,
                Some(files.join(",")),
                note,
            );
        }
        Err(error) => {
            let note = format!(
                "Integration blocked: {error:#}. AGTX retries on its own once this is resolved."
            );
            return record_unresolved_integration(
                db,
                task,
                &current,
                crate::db::INTEGRATION_BLOCKED,
                None,
                note,
            );
        }
    };

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
    if task.has_unresolved_integration() {
        // The note was the executor's own account of the block; it is settled.
        task.escalation_note = None;
        let mut event = TaskExecutionEvent::new(&task.id, "integration_resolved");
        event.workflow_attempt = Some(current.state_attempt);
        event.state = Some(current.state.clone());
        event.outcome = Some("integrated".to_string());
        event.message = Some(format!("Task integrated into {target_branch}"));
        db.record_task_execution_event(&event)?;
    }
    task.integration_status = None;
    task.integration_conflicts = None;
    task.status = TaskStatus::Done;
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;
    Ok(WorkflowStepOutcome::Advanced {
        message: format!("Task integrated into {target_branch}"),
        task,
    })
}

/// The git side of [`complete_feature_integration`]: everything that can
/// fail for reasons outside the workflow graph. `Err` means "blocked, retry
/// later"; the caller records it rather than propagating it.
fn attempt_feature_integration(
    workflow: &WorkflowDefinition,
    project_workflow: &WorkflowProjectConfig,
    task: &mut Task,
    current: &WorkflowTaskState,
    worktree: &Path,
    branch: &str,
    runtime: &WorkflowRuntime,
) -> Result<IntegrationAttempt> {
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
    if !(policy.role_policy.final_task_commit
        && policy.role_policy.push_task_branch
        && policy.role_policy.merge_task_into_target)
    {
        bail!("Integration state policy does not authorize commit, push, and target merge");
    }

    if task.has_unresolved_integration() {
        let remote_target = format!("origin/{target_branch}");
        let remote_branch = format!("origin/{branch}");
        let target_on_remote = crate::git::fetch_branch(runtime.project_path, &target_branch)?;
        let branch_on_remote = crate::git::fetch_branch(runtime.project_path, branch)?;
        let merged_on_remote = branch_on_remote
            && target_on_remote
            && (crate::git::is_ancestor(runtime.project_path, &remote_branch, &remote_target)
                || pull_request_merged(task, runtime));
        if merged_on_remote {
            ensure_target_checkout(runtime, &target_branch)?;
            crate::git::fast_forward(runtime.project_path, &remote_target)?;
            return Ok(IntegrationAttempt::Integrated { target_branch });
        }
        if branch_on_remote && !crate::git::is_ancestor(worktree, &remote_branch, "HEAD") {
            // A resolution was pushed to the task branch from outside the
            // container (the pull request's conflict editor, a local checkout).
            crate::opencode_profile::strip_opencode_permission_profile(worktree);
            if runtime.git_ops.has_changes(worktree) {
                bail!(
                    "'{remote_branch}' has new commits but the task worktree has uncommitted \
                     changes; commit or discard them so the worktree can take the pushed branch"
                );
            }
            crate::git::fast_forward(worktree, &remote_branch).map_err(|error| {
                anyhow::anyhow!(
                    "the task worktree and '{remote_branch}' have diverged; merge them on the \
                     task branch and push ({error})"
                )
            })?;
        }
    }

    // agtx's own OpenCode permission profile is launch-time runtime state, not
    // task work: never let it be committed and merged into the target branch.
    crate::opencode_profile::strip_opencode_permission_profile(worktree);
    if runtime.git_ops.has_changes(worktree) {
        runtime.git_ops.add_all(worktree)?;
        runtime
            .git_ops
            .commit(worktree, &format!("workflow: complete task {}", task.id))?;
    }
    runtime.git_ops.push(worktree, branch, true)?;

    // Refs only: the check needs neither checkout, so a target checkout with
    // uncommitted work still gets its conflicts reported.
    let (has_conflicts, files) =
        crate::git::check_merge_conflicts(runtime.project_path, &target_branch, branch)?;
    if has_conflicts {
        let pull_request_error = if policy.role_policy.create_or_update_task_pr {
            ensure_conflict_pull_request(task, &target_branch, branch, &files, worktree, runtime)
                .err()
                .map(|error| format!("{error:#}"))
        } else {
            None
        };
        return Ok(IntegrationAttempt::Conflicts {
            target_branch,
            files,
            pull_request_error,
        });
    }

    ensure_target_checkout(runtime, &target_branch)?;
    crate::git::merge_branch(
        runtime.project_path,
        branch,
        &format!("workflow: integrate task {}", task.id),
    )?;
    runtime
        .git_ops
        .push(runtime.project_path, &target_branch, false)?;
    Ok(IntegrationAttempt::Integrated { target_branch })
}

/// The configured target checkout must be clean and on the target branch
/// before the executor writes to it.
fn ensure_target_checkout(runtime: &WorkflowRuntime, target_branch: &str) -> Result<()> {
    if runtime.git_ops.has_changes(runtime.project_path) {
        bail!("configured target checkout has uncommitted changes; integration is refused");
    }
    if crate::git::current_branch(runtime.project_path)? != target_branch {
        bail!("configured target checkout is not on '{target_branch}'; integration is refused");
    }
    Ok(())
}

/// Whether the provider reports the task's pull request as merged. Covers a
/// squash or rebase merge, which leaves the task branch outside the target's
/// history.
fn pull_request_merged(task: &Task, runtime: &WorkflowRuntime) -> bool {
    match (task.pr_number, runtime.git_provider_ops) {
        (Some(number), Some(provider)) if number > 0 => matches!(
            provider.get_pr_state(runtime.project_path, number),
            Ok(PullRequestState::Merged)
        ),
        _ => false,
    }
}

/// Open the pull request a person resolves the conflict on, once per task:
/// later pushes to the branch update it on their own.
fn ensure_conflict_pull_request(
    task: &mut Task,
    target_branch: &str,
    branch: &str,
    files: &[String],
    worktree: &Path,
    runtime: &WorkflowRuntime,
) -> Result<()> {
    if task.pr_url.is_some() {
        return Ok(());
    }
    let Some(provider) = runtime.git_provider_ops else {
        return Ok(());
    };
    let file_list = files
        .iter()
        .map(|file| format!("- `{file}`"))
        .collect::<Vec<_>>()
        .join("\n");
    let body = format!(
        "AGTX workflow task `{}`.\n\nMerging `{branch}` into `{target_branch}` conflicts in:\n\n\
         {file_list}\n\nResolve the conflicts on this branch (here, or by merging \
         `{target_branch}` into it locally) and push. AGTX retries the integration on its own \
         and completes the task once the branch merges cleanly; merging this pull request \
         completes it as well.",
        task.id
    );
    let (number, url) = provider.create_pr(
        worktree,
        &task.title,
        &body,
        branch,
        Some(target_branch.to_string()),
    )?;
    task.pr_number = Some(number);
    task.pr_url = Some(url);
    Ok(())
}

/// Persist an integration that could not finish, and escalate it. The journal
/// gets one event per change of status or reason, not one per retry.
fn record_unresolved_integration(
    db: &Database,
    mut task: Task,
    current: &WorkflowTaskState,
    status: &str,
    conflicts: Option<String>,
    note: String,
) -> Result<WorkflowStepOutcome> {
    let note = bounded_note(&note);
    let changed = task.integration_status.as_deref() != Some(status)
        || task.escalation_note.as_deref() != Some(note.as_str());
    task.integration_status = Some(status.to_string());
    task.integration_conflicts = conflicts;
    task.escalation_note = Some(note.clone());
    task.updated_at = chrono::Utc::now();
    db.update_task(&task)?;
    if changed {
        let mut event = TaskExecutionEvent::new(&task.id, "integration_unresolved");
        event.workflow_attempt = Some(current.state_attempt);
        event.state = Some(current.state.clone());
        event.outcome = Some(status.to_string());
        event.message = Some(note.clone());
        db.record_task_execution_event(&event)?;
    }
    Ok(WorkflowStepOutcome::Blocked { message: note })
}

fn bounded_note(note: &str) -> String {
    let note = note.trim();
    if note.chars().count() <= INTEGRATION_NOTE_LIMIT {
        return note.to_string();
    }
    let mut bounded: String = note.chars().take(INTEGRATION_NOTE_LIMIT - 1).collect();
    bounded.push('…');
    bounded
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
    /// into this same state (unchanged since AGTX launched that state). Automation should
    /// simply check again later.
    Wait,
    /// The named action is ready to fire.
    Advance(String),
    /// An artifact exists but
    /// cannot be trusted as evidence: bad YAML, a missing required field, or
    /// has malformed content or a missing required field.
    InvalidArtifact(String),
    /// Evidence is valid but this specific outcome is never auto-advanced.
    /// Either a fixed rule (final validation's `failed` verdict: a human must
    /// look before any rework loop restarts) or a configured one (an action
    /// named in `[automation].human_gates`, or `approve_plan` for a task
    /// whose `human_gate_plan_approval` flag was set via the Shift+S
    /// "require my approval" popup) -- see `apply_human_gates`.
    HumanGate(String),
}

/// Facts about `task`/`state` needed to evaluate the graph's declared
/// guards, computed the same way the extracted launch functions above
/// already compute them for their own action -- reused here rather than
/// re-derived, since `assess` has no specific action in mind and instead
/// asks the graph which actions are currently legal at all.
pub fn guard_context_for(
    db: &Database,
    task: &Task,
    state: &WorkflowTaskState,
    plugin: &WorkflowPlugin,
) -> GuardContext {
    let implementation_recorded = task
        .worktree_path
        .as_deref()
        .map(|worktree| {
            workflow_artifact_path(
                worktree,
                plugin.artifacts.running.as_deref(),
                &task.id,
                ".agent-flow/implementation-result.yaml",
            )
            .is_file()
        })
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

/// The last `max_lines` lines of `text`, trimmed. Used to bound a captured
/// tmux pane to a reasonable prompt size while keeping whatever was printed
/// most recently -- typically where a reviewer's final verdict/reasoning
/// lands, even on a tall or scrolled-back pane.
fn tail_lines(text: &str, max_lines: usize) -> String {
    let trimmed = text.trim_end();
    let lines: Vec<&str> = trimmed.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

/// Shared freshness gate for every artifact-backed state.  AGTX records when
/// it launched the state; an unchanged pre-existing file is never accepted.
fn artifact_freshness(
    artifact: &Path,
    state: &WorkflowTaskState,
    task: &Task,
    db: &Database,
) -> Option<AutomationDecision> {
    if !artifact.is_file() {
        return Some(AutomationDecision::Wait);
    }
    let launched_at = db
        .task_step_reports(&task.id)
        .ok()?
        .into_iter()
        .find(|report| {
            report.workflow_attempt == state.state_attempt && report.state == state.state
        })?
        .updated_at;
    let modified_at = std::fs::metadata(artifact).ok()?.modified().ok()?;
    let modified_at: chrono::DateTime<chrono::Utc> = modified_at.into();
    (modified_at <= launched_at).then_some(AutomationDecision::Wait)
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
    project: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    task: &Task,
    state: &WorkflowTaskState,
    db: &Database,
) -> AutomationDecision {
    // `IntegratedIntoTarget` is a postcondition of `complete_feature_integration`,
    // not a precondition: the executor itself performs the merge before it records
    // that guard as satisfied. Assess the reviewer's fresh integration evidence
    // before asking the graph for currently available (pre-merge) transitions.
    if state.state == "integrate_to_feature" {
        let Some(worktree) = task.worktree_path.as_deref() else {
            return AutomationDecision::Wait;
        };
        return apply_human_gates(
            assess_feature_integration(worktree, plugin, task, state, db),
            project,
            plugin,
            task,
            state,
        );
    }

    let guards = guard_context_for(db, task, state, plugin);
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
        let decision = match available
            .iter()
            .find(|transition| !transition.guards.is_empty())
        {
            Some(transition) => AutomationDecision::Advance(transition.action.clone()),
            None => AutomationDecision::Wait,
        };
        return apply_human_gates(decision, project, plugin, task, state);
    }

    let Some(worktree) = task.worktree_path.as_deref() else {
        return AutomationDecision::Wait;
    };

    let decision = match state.state.as_str() {
        "planning" => assess_planning(worktree, plugin, task, state, db),
        "plan_review" => assess_plan_review(worktree, plugin, task, state, db),
        "engineering_review" => assess_engineering_review(worktree, plugin, task, state, db),
        "final_validation" => assess_final_validation(worktree, plugin, task, state, db),
        "implementing" | "running" => assess_implementation(worktree, plugin, task, state, db),
        // A role state this function does not yet know an artifact mapping
        // for. Nothing to read, so nothing to report.
        _ => AutomationDecision::Wait,
    };
    apply_human_gates(decision, project, plugin, task, state)
}

/// Downgrade an `Advance` into a `HumanGate` when the action is named in
/// either gate source: the project-wide `[automation].human_gates` list
/// (`WorkflowAutomationConfig::human_gates`), which applies to every task, or
/// this one task's own `WorkflowTaskState::human_gate_plan_approval` flag,
/// set per-task via the Shift+S "require my approval" popup. Anything other
/// than `Advance` passes through untouched -- a gate only ever holds back a
/// transition that was otherwise ready to fire, it never turns `Wait` or
/// `InvalidArtifact` into something else.
fn apply_human_gates(
    decision: AutomationDecision,
    project: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    task: &Task,
    state: &WorkflowTaskState,
) -> AutomationDecision {
    let AutomationDecision::Advance(action) = &decision else {
        return decision;
    };
    let gated = project
        .automation
        .human_gates
        .iter()
        .any(|gate| gate == action)
        || (action == "approve_plan" && state.human_gate_plan_approval);
    if !gated {
        return decision;
    }
    let reason = if action == "approve_plan" {
        task.worktree_path
            .as_deref()
            .and_then(|worktree| planning_artifact_path(worktree, plugin, &task.id).ok())
            .map(|path| {
                format!(
                    "Plan approved by the reviewer -- awaiting your sign-off: {}",
                    path.display()
                )
            })
            .unwrap_or_else(|| "Plan approved by the reviewer -- awaiting your sign-off".into())
    } else {
        format!("'{action}' is gated for human approval")
    };
    AutomationDecision::HumanGate(reason)
}

/// Mirrors `submit_workflow_plan`'s own evidence check. Revision allocation is
/// database-owned and happens only when this snapshot is accepted.
fn assess_planning(
    worktree: &str,
    plugin: &WorkflowPlugin,
    task: &Task,
    state: &WorkflowTaskState,
    db: &Database,
) -> AutomationDecision {
    let path = match planning_artifact_path(worktree, plugin, &task.id) {
        Ok(path) => path,
        Err(_) => return AutomationDecision::Wait,
    };
    if let Some(decision) = artifact_freshness(&path, state, task, db) {
        return decision;
    }
    AutomationDecision::Advance("submit_plan".to_string())
}

/// Mirrors `submit_plan_review`'s verdict-to-action match exactly. Both
/// `approved` and `changes_requested` are legitimate, always-auto-dispatchable
/// outcomes -- matching how `engineering_review`'s own rework loops
/// (`corrections_required`/`plan_issue`) already auto-dispatch without a
/// human gate.
fn assess_plan_review(
    worktree: &str,
    plugin: &WorkflowPlugin,
    task: &Task,
    state: &WorkflowTaskState,
    db: &Database,
) -> AutomationDecision {
    let artifact = workflow_artifact_path(
        worktree,
        plugin.artifacts.plan_review.as_deref(),
        &task.id,
        ".agent-flow/plan-review.yaml",
    );
    if let Some(decision) = artifact_freshness(&artifact, state, task, db) {
        return decision;
    }
    let verdict = match workflow_artifact_value(&artifact, "verdict") {
        Ok(verdict) => verdict,
        Err(error) => return AutomationDecision::InvalidArtifact(error.to_string()),
    };
    match verdict.as_str() {
        "approved" => AutomationDecision::Advance("approve_plan".to_string()),
        "changes_requested" => AutomationDecision::Advance("plan_changes_requested".to_string()),
        other => AutomationDecision::InvalidArtifact(format!(
            "{} has unsupported plan-review verdict '{other}'",
            artifact.display()
        )),
    }
}

/// Mirrors `submit_engineering_review`'s verdict-to-action match exactly.
fn assess_engineering_review(
    worktree: &str,
    plugin: &WorkflowPlugin,
    task: &Task,
    state: &WorkflowTaskState,
    db: &Database,
) -> AutomationDecision {
    let artifact = workflow_artifact_path(
        worktree,
        plugin.artifacts.review.as_deref(),
        &task.id,
        ".agent-flow/engineering-review.yaml",
    );
    if let Some(decision) = artifact_freshness(&artifact, state, task, db) {
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
fn assess_final_validation(
    worktree: &str,
    plugin: &WorkflowPlugin,
    task: &Task,
    state: &WorkflowTaskState,
    db: &Database,
) -> AutomationDecision {
    let artifact = workflow_artifact_path(
        worktree,
        plugin.artifacts.final_validation.as_deref(),
        &task.id,
        ".agent-flow/final-validation.yaml",
    );
    if let Some(decision) = artifact_freshness(&artifact, state, task, db) {
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

/// The integration reviewer has supplied the final required evidence. The
/// executor now owns the push, merge into the admitted non-main target, and
/// durable completion transition.
///
/// An integration the executor already started and could not finish is
/// retried every [`INTEGRATION_RETRY_SECONDS`] instead: its artifact was
/// consumed by the first attempt, so the freshness gate would otherwise keep
/// the task waiting forever.
fn assess_feature_integration(
    worktree: &str,
    plugin: &WorkflowPlugin,
    task: &Task,
    state: &WorkflowTaskState,
    db: &Database,
) -> AutomationDecision {
    let artifact = workflow_artifact_path(
        worktree,
        plugin.artifacts.integration.as_deref(),
        &task.id,
        ".agent-flow/integration-ready.yaml",
    );
    if task.has_unresolved_integration() {
        let since_last_attempt = chrono::Utc::now() - task.updated_at;
        if since_last_attempt < chrono::Duration::seconds(INTEGRATION_RETRY_SECONDS) {
            return AutomationDecision::Wait;
        }
    } else if let Some(decision) = artifact_freshness(&artifact, state, task, db) {
        return decision;
    }
    let verdict = match workflow_artifact_value(&artifact, "verdict") {
        Ok(verdict) => verdict,
        Err(error) => return AutomationDecision::InvalidArtifact(error.to_string()),
    };
    match verdict.as_str() {
        "ready_for_integration" => {
            AutomationDecision::Advance("complete_feature_integration".to_string())
        }
        other => AutomationDecision::InvalidArtifact(format!(
            "{} has unsupported integration verdict '{other}'",
            artifact.display()
        )),
    }
}

/// Mirrors `submit_workflow_implementation`'s evidence check: the
/// implementer's result file has no verdict of its own, it is either
/// present (and fresh) or it is not.
fn assess_implementation(
    worktree: &str,
    plugin: &WorkflowPlugin,
    task: &Task,
    state: &WorkflowTaskState,
    db: &Database,
) -> AutomationDecision {
    let artifact = workflow_artifact_path(
        worktree,
        plugin.artifacts.running.as_deref(),
        &task.id,
        ".agent-flow/implementation-result.yaml",
    );
    match artifact_freshness(&artifact, state, task, db) {
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
                WorkflowState {
                    id: "backlog".into(),
                    label: "Backlog".into(),
                    role: None,
                    terminal: false,
                },
                WorkflowState {
                    id: "admission".into(),
                    label: "Admission".into(),
                    role: None,
                    terminal: false,
                },
                WorkflowState {
                    id: "done".into(),
                    label: "Done".into(),
                    role: None,
                    terminal: true,
                },
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
    fn backup_plan_preserves_contents_and_uses_a_collision_suffix() {
        let source_dir = tempfile::tempdir().unwrap();
        let project_dir = tempfile::tempdir().unwrap();
        let source = source_dir.path().join("plan.md");
        std::fs::write(&source, "keep this exact metadata-free plan\n").unwrap();

        let first = backup_plan(&source, project_dir.path(), "Recover planning!").unwrap();
        let second = backup_plan(&source, project_dir.path(), "Recover planning!").unwrap();

        assert_eq!(
            std::fs::read(&first).unwrap(),
            std::fs::read(&source).unwrap()
        );
        assert_ne!(first, second);
        assert!(second
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with("-2.md"));
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
        project
            .role_bindings
            .insert("planner".into(), "claude".into());

        let transition = prepare_transition(
            &graph,
            &project,
            &current,
            "start_planning",
            GuardContext {
                dependencies_resolved: true,
                ..GuardContext::default()
            },
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
                WorkflowState {
                    id: "backlog".into(),
                    label: "Backlog".into(),
                    role: None,
                    terminal: false,
                },
                WorkflowState {
                    id: "admission".into(),
                    label: "Admission".into(),
                    role: None,
                    terminal: false,
                },
                WorkflowState {
                    id: "planning".into(),
                    label: "Planning".into(),
                    role: Some("planner".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "running".into(),
                    label: "Running".into(),
                    role: Some("implementer".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "engineering_review".into(),
                    label: "Engineering review".into(),
                    role: Some("reviewer".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "final_validation".into(),
                    label: "Final validation".into(),
                    role: Some("validator".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "integrate_to_feature".into(),
                    label: "Integrate to feature".into(),
                    role: Some("integrator".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "done".into(),
                    label: "Done".into(),
                    role: None,
                    terminal: true,
                },
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
                    to: "integrate_to_feature".into(),
                    guards: vec![],
                },
                WorkflowTransition {
                    action: "complete_feature_integration".into(),
                    from: "integrate_to_feature".into(),
                    to: "done".into(),
                    guards: vec![crate::workflow::WorkflowGuard::IntegratedIntoTarget],
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

    fn record_launch_for_test(db: &Database, task: &Task, state: &WorkflowTaskState) {
        let mut report = TaskStepReport::new(&task.id, state.state_attempt, &state.state);
        report.updated_at = chrono::Utc::now();
        db.upsert_task_step_report(&report).unwrap();
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
            "verdict: approved_for_validation\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let mut state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        state.state_attempt = 3;
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(
            decision,
            AutomationDecision::Advance("start_final_validation".to_string())
        );
    }

    #[test]
    fn assess_implementation_resolves_a_task_scoped_artifact_path() {
        // `running`'s artifact check used to hardcode a flat
        // `.agent-flow/implementation-result.yaml` path, ignoring
        // `plugin.artifacts.running` entirely -- unlike every sibling
        // `assess_*` function. This pins the fix: a `{task_id}`-templated
        // `plugin.artifacts.running` must actually be consulted.
        let mut graph = assess_workflow();
        // `assess_workflow()`'s fixture has no outgoing transition from
        // `running` at all; add one so `assess` reaches the
        // `available_transitions` check and falls through into the
        // `"running" => assess_implementation(...)` match arm this test
        // actually exercises, instead of short-circuiting to `Wait` earlier.
        graph.transitions.push(WorkflowTransition {
            action: "implementation_complete".into(),
            from: "running".into(),
            to: "engineering_review".into(),
            guards: vec![],
        });
        let mut plugin = plugin_for_tests(graph.clone());
        plugin.artifacts.running = Some(".agent-flow/{task_id}/implementation-result.yaml".into());
        let worktree = tempfile::tempdir().unwrap();
        let task = admitted_task(worktree.path());
        std::fs::create_dir_all(worktree.path().join(".agent-flow").join(&task.id)).unwrap();
        std::fs::write(
            worktree
                .path()
                .join(".agent-flow")
                .join(&task.id)
                .join("implementation-result.yaml"),
            "result: completed\n",
        )
        .unwrap();
        let mut state = WorkflowTaskState::new(&task.id, "running", "main");
        state.state_attempt = 1;
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(
            decision,
            AutomationDecision::Advance("implementation_complete".to_string())
        );

        // Sanity check the flat (pre-fix) path was deliberately left empty:
        // if the fix regressed to reading the old hardcoded path instead of
        // the task-scoped one, this artifact wouldn't exist there and the
        // decision above would be Wait, not Advance.
        assert!(!worktree
            .path()
            .join(".agent-flow/implementation-result.yaml")
            .exists());
    }

    #[test]
    fn guard_context_for_resolves_a_task_scoped_implementation_artifact() {
        let mut plugin = plugin_for_tests(assess_workflow());
        plugin.artifacts.running = Some(".agent-flow/{task_id}/implementation-result.yaml".into());
        let worktree = tempfile::tempdir().unwrap();
        let task = admitted_task(worktree.path());
        std::fs::create_dir_all(worktree.path().join(".agent-flow").join(&task.id)).unwrap();
        std::fs::write(
            worktree
                .path()
                .join(".agent-flow")
                .join(&task.id)
                .join("implementation-result.yaml"),
            "result: completed\n",
        )
        .unwrap();
        let state = WorkflowTaskState::new(&task.id, "running", "main");
        let db = Database::open_in_memory_project().unwrap();

        let guards = guard_context_for(&db, &task, &state, &plugin);
        assert!(guards.implementation_recorded);
    }

    #[test]
    fn assess_waits_on_an_unchanged_artifact_from_before_launch() {
        let graph = assess_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        // Left over from a prior entry into this state.
        std::fs::write(
            worktree.path().join(".agent-flow/engineering-review.yaml"),
            "verdict: approved_for_validation\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let mut state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        state.state_attempt = 2;
        let db = Database::open_in_memory_project().unwrap();
        record_launch_for_test(&db, &task, &state);

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
            "result: completed\n",
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
            "verdict: failed\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let state = WorkflowTaskState::new(&task.id, "final_validation", "main");
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(
            decision,
            AutomationDecision::HumanGate("final validation failed".to_string())
        );
    }

    #[test]
    fn assess_advances_a_fresh_ready_integration_artifact() {
        let graph = assess_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/integration-ready.yaml"),
            "verdict: ready_for_integration\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let mut state = WorkflowTaskState::new(&task.id, "integrate_to_feature", "feature/poc");
        state.state_attempt = 4;
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(
            decision,
            AutomationDecision::Advance("complete_feature_integration".to_string())
        );
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
        std::fs::write(&artifact, "verdict: approved_for_validation\n").unwrap();
        let task = admitted_task(worktree.path());
        let db = Database::open_in_memory_project().unwrap();

        // First pass: state_attempt 1 matches the artifact's workflow_attempt.
        let first_state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        // The first pass sees newly written output. The second launch records
        // the existing file as its baseline and must wait for a real rewrite.
        assert_eq!(
            assess(&graph, &project(), &plugin, &task, &first_state, &db),
            AutomationDecision::Advance("start_final_validation".to_string())
        );

        // Rework cycles the task back through `running` and it re-enters
        // `engineering_review` a second time; the artifact on disk is the
        // untouched leftover from the first pass.
        let mut second_state = WorkflowTaskState::new(&task.id, "engineering_review", "main");
        second_state.state_attempt = 2;
        record_launch_for_test(&db, &task, &second_state);
        assert_eq!(
            assess(&graph, &project(), &plugin, &task, &second_state, &db),
            AutomationDecision::Wait
        );
    }

    /// A minimal graph exercising only the `plan_review` state, mirroring
    /// `assess_workflow`'s own style: `assess_plan_review` distinguishes its
    /// two outcomes purely by the artifact's `verdict`, so neither transition
    /// needs a graph guard.
    fn assess_plan_review_workflow() -> WorkflowDefinition {
        WorkflowDefinition {
            initial_state: "planning".into(),
            states: vec![
                WorkflowState {
                    id: "planning".into(),
                    label: "Planning".into(),
                    role: Some("planner".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "plan_review".into(),
                    label: "Plan review".into(),
                    role: Some("plan_reviewer".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "plan_approved".into(),
                    label: "Plan approved".into(),
                    role: None,
                    terminal: true,
                },
            ],
            transitions: vec![
                WorkflowTransition {
                    action: "plan_changes_requested".into(),
                    from: "plan_review".into(),
                    to: "planning".into(),
                    guards: vec![],
                },
                WorkflowTransition {
                    action: "approve_plan".into(),
                    from: "plan_review".into(),
                    to: "plan_approved".into(),
                    guards: vec![],
                },
            ],
        }
    }

    #[test]
    fn assess_plan_review_advances_on_a_fresh_approved_verdict() {
        let graph = assess_plan_review_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: approved\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let state = WorkflowTaskState::new(&task.id, "plan_review", "main");
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(
            decision,
            AutomationDecision::Advance("approve_plan".to_string())
        );
    }

    /// Project-wide `[automation].human_gates = ["approve_plan"]` holds back
    /// every task's `approve_plan`, not just one that opted in via the
    /// Shift+S popup.
    #[test]
    fn assess_plan_review_gates_approve_plan_when_project_configured() {
        let graph = assess_plan_review_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: approved\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let state = WorkflowTaskState::new(&task.id, "plan_review", "main");
        let db = Database::open_in_memory_project().unwrap();
        let mut project = project();
        project.automation.human_gates = vec!["approve_plan".to_string()];

        let decision = assess(&graph, &project, &plugin, &task, &state, &db);
        let AutomationDecision::HumanGate(reason) = decision else {
            panic!("expected a HumanGate decision, got {decision:?}");
        };
        assert!(
            reason.contains("plan.md") || reason.contains("awaiting your sign-off"),
            "reason should reference the plan artifact or say what it's waiting on: {reason}"
        );
    }

    /// A task's own `human_gate_plan_approval` flag (set via the Shift+S
    /// "require my approval" popup) gates just that task's `approve_plan`,
    /// with no project-wide `[automation].human_gates` entry at all.
    #[test]
    fn assess_plan_review_gates_approve_plan_when_task_opted_in() {
        let graph = assess_plan_review_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: approved\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let mut state = WorkflowTaskState::new(&task.id, "plan_review", "main");
        state.human_gate_plan_approval = true;
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert!(
            matches!(decision, AutomationDecision::HumanGate(_)),
            "expected a HumanGate decision, got {decision:?}"
        );
    }

    #[test]
    fn assess_plan_review_advances_on_a_fresh_changes_requested_verdict() {
        let graph = assess_plan_review_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: changes_requested\nfindings: Missing tests for the new endpoint.\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let state = WorkflowTaskState::new(&task.id, "plan_review", "main");
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(
            decision,
            AutomationDecision::Advance("plan_changes_requested".to_string())
        );
    }

    #[test]
    fn assess_plan_review_waits_when_the_artifact_is_missing() {
        let graph = assess_plan_review_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        let task = admitted_task(worktree.path());
        let state = WorkflowTaskState::new(&task.id, "plan_review", "main");
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(decision, AutomationDecision::Wait);
    }

    #[test]
    fn assess_plan_review_waits_on_an_unchanged_artifact_from_before_launch() {
        let graph = assess_plan_review_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: approved\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let mut state = WorkflowTaskState::new(&task.id, "plan_review", "main");
        state.state_attempt = 2;
        let db = Database::open_in_memory_project().unwrap();
        record_launch_for_test(&db, &task, &state);

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert_eq!(decision, AutomationDecision::Wait);
    }

    #[test]
    fn assess_plan_review_flags_an_unsupported_verdict_as_invalid() {
        let graph = assess_plan_review_workflow();
        let plugin = plugin_for_tests(graph.clone());
        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: looks_fine_i_guess\n",
        )
        .unwrap();
        let task = admitted_task(worktree.path());
        let state = WorkflowTaskState::new(&task.id, "plan_review", "main");
        let db = Database::open_in_memory_project().unwrap();

        let decision = assess(&graph, &project(), &plugin, &task, &state, &db);
        assert!(matches!(decision, AutomationDecision::InvalidArtifact(_)));
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

    #[test]
    fn records_hook_reported_codex_or_claude_session_without_overwriting_prior_rows() {
        let worktree = tempfile::tempdir().unwrap();
        let task = crate::db::Task::new("Session mapping", "codex", "proj");
        let db = Database::open_in_memory_project().unwrap();
        db.create_task(&task).unwrap();
        let status = crate::agent::hook_status::AgentHookStatus {
            ts: chrono::Utc::now().timestamp(),
            state: crate::agent::hook_status::HookState::Waiting,
            session_id: Some("codex-native-session".into()),
            transcript_path: None,
            message: None,
            tool: None,
            agent: "codex".into(),
        };
        crate::agent::hook_status::write_status(worktree.path(), &task.id, &status).unwrap();

        record_provider_session_if_known(
            &db,
            &task,
            "planning",
            4,
            "codex",
            &worktree.path().to_string_lossy(),
        );
        let sessions = db.provider_sessions(&task.id).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].provider_session_id, "codex-native-session");
        assert_eq!(
            sessions[0].workflow_session_id,
            format!("agtx:{}:4:planning", task.id)
        );
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
                WorkflowState {
                    id: "plan_review".into(),
                    label: "Plan review".into(),
                    role: None,
                    terminal: false,
                },
                WorkflowState {
                    id: "implementation".into(),
                    label: "Implementation".into(),
                    role: Some("implementer".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("implementer".into(), "claude".into());
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
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux.expect_has_session().returning(|_| true);
        let captured = Arc::new(Mutex::new(None));
        let captured_create = captured.clone();
        mock_tmux
            .expect_create_window()
            .returning(move |_, _, _, command, _, _| {
                *captured_create.lock().unwrap() = command;
                Ok(())
            });

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| {
            let mut agent_ops = MockAgentOperations::new();
            agent_ops
                .expect_prompt_injection()
                .returning(|| crate::agent::PromptInjection::Argv);
            Arc::new(agent_ops) as Arc<dyn AgentOperations>
        });

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome = start_workflow_implementation(
            &graph,
            &project,
            &plugin,
            task.clone(),
            &mut db,
            &runtime,
        )
        .unwrap();
        let WorkflowStepOutcome::Advanced {
            task: advanced,
            message,
        } = outcome
        else {
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
        let prompt = "Implement: do the work\n\nAGTX has bound the approved plan internally. Do not write revision, workflow-attempt, or SHA-256 metadata into your result artifact.".to_string();
        let expected = format!(
            "claude --model sonnet --effort medium --permission-mode dontAsk --allowed-tools '{}' -- '{}'",
            claude_allowed_tools(&policy),
            quote_for_shell(&prompt),
        );
        let actual = captured
            .lock()
            .unwrap()
            .clone()
            .expect("create_window should receive a command");
        assert_eq!(actual, expected);
    }

    /// A task description large enough to push the built prompt past
    /// `MAX_LAUNCH_PROMPT_BYTES` must not be embedded in the `create_window`
    /// launch command -- past that (empirically measured) size, tmux itself
    /// rejects the command with "command too long" (the crash this test
    /// guards against). Instead the window must launch with an empty prompt
    /// and the real prompt must be delivered afterward via `paste_text` + a
    /// literal `Enter` keypress.
    #[test]
    fn start_workflow_implementation_defers_an_oversized_prompt_to_paste_text() {
        let graph = WorkflowDefinition {
            initial_state: "plan_review".into(),
            states: vec![
                WorkflowState {
                    id: "plan_review".into(),
                    label: "Plan review".into(),
                    role: None,
                    terminal: false,
                },
                WorkflowState {
                    id: "implementation".into(),
                    label: "Implementation".into(),
                    role: Some("implementer".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("implementer".into(), "claude".into());
        project
            .role_policies
            .roles
            .insert("implementer".into(), WorkflowRolePolicy::default());

        let mut plugin = plugin(graph.clone());
        plugin.prompts.running = Some("Implement: {task}".into());

        let mut task = crate::db::Task::new("Implement thing", "codex", "proj");
        // Comfortably past MAX_LAUNCH_PROMPT_BYTES (4 KiB).
        task.description = Some("x".repeat(2_000_000));
        task.worktree_path = Some("C:/work/wt".into());

        let mut db = Database::open_in_memory_project().unwrap();
        db.create_task(&task).unwrap();
        let mut current = WorkflowTaskState::new(&task.id, "plan_review", "main");
        current.plan_revision = 2;
        current.plan_hash = Some("abc123".into());
        current.approved_plan_revision = Some(2);
        current.approved_plan_hash = Some("abc123".into());
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "plan_review");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux.expect_has_session().returning(|_| true);
        let captured_command = Arc::new(Mutex::new(None));
        let captured_command_clone = captured_command.clone();
        mock_tmux
            .expect_create_window()
            .returning(move |_, _, _, command, _, _| {
                *captured_command_clone.lock().unwrap() = command;
                Ok(())
            });
        // Lets `wait_for_agent_ready`'s Step 1 break immediately instead of
        // polling for 30s: a non-shell `pane_current_command` reads as "the
        // agent process is already running".
        mock_tmux
            .expect_pane_current_command()
            .returning(|_| Some("claude".to_string()));
        // Constant content lets Step 2's stabilization check succeed on its
        // first few ticks instead of waiting out its own 30s ceiling.
        mock_tmux
            .expect_capture_pane()
            .returning(|_| Ok("ready".to_string()));
        let captured_paste = Arc::new(Mutex::new(None));
        let captured_paste_clone = captured_paste.clone();
        mock_tmux.expect_paste_text().returning(move |_, text| {
            *captured_paste_clone.lock().unwrap() = Some(text.to_string());
            Ok(())
        });
        let sent_enter = Arc::new(Mutex::new(false));
        let sent_enter_clone = sent_enter.clone();
        mock_tmux.expect_send_key().returning(move |_, key| {
            if key == "C-m" {
                *sent_enter_clone.lock().unwrap() = true;
            }
            Ok(())
        });

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| {
            let mut agent_ops = MockAgentOperations::new();
            agent_ops
                .expect_prompt_injection()
                .returning(|| crate::agent::PromptInjection::Argv);
            Arc::new(agent_ops) as Arc<dyn AgentOperations>
        });

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome = start_workflow_implementation(
            &graph,
            &project,
            &plugin,
            task.clone(),
            &mut db,
            &runtime,
        )
        .unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));

        let launch_command = captured_command
            .lock()
            .unwrap()
            .clone()
            .expect("create_window should receive a command");
        assert!(
            !launch_command.contains('x'),
            "oversized prompt must not be embedded in the launch command: {launch_command}"
        );

        let pasted = captured_paste
            .lock()
            .unwrap()
            .clone()
            .expect("paste_text should have delivered the deferred prompt");
        assert!(
            pasted.contains(&"x".repeat(2_000_000)),
            "paste_text should carry the full oversized prompt"
        );
        assert!(
            *sent_enter.lock().unwrap(),
            "the deferred prompt must be submitted with Enter"
        );
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
                WorkflowState {
                    id: "engineering_review".into(),
                    label: "Engineering review".into(),
                    role: Some("reviewer".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "final_validation".into(),
                    label: "Final validation".into(),
                    role: Some("validator".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("reviewer".into(), "claude".into());
        project
            .role_bindings
            .insert("validator".into(), "claude".into());
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
        let record =
            WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "engineering_review");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux
            .expect_send_keys()
            .withf(|_, cmd: &str| cmd == "/exit")
            .returning(|_, _| Ok(()));
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        // First two polls (pre-switch pane check, exit check): outgoing agent
        // already at a shell. Every poll after
        // that: the freshly launched agent, matching a real tmux pane once
        // `switch_agent_in_tmux` types the new command -- its final
        // launch-detection loop requires a *recognized* agent process name,
        // not merely any string.
        let pane_polls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        mock_tmux.expect_pane_current_command().returning(move |_| {
            if pane_polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                Some("bash".to_string())
            } else {
                Some("claude".to_string())
            }
        });
        mock_tmux
            .expect_capture_pane()
            .returning(|_| Ok(String::new()));
        let captured = Arc::new(Mutex::new(None));
        let captured_paste = captured.clone();
        mock_tmux.expect_paste_text().returning(move |_, text| {
            *captured_paste.lock().unwrap() = Some(text.to_string());
            Ok(())
        });

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry
            .expect_get()
            .returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome =
            submit_engineering_review(&graph, &project, &plugin, task.clone(), &mut db, &runtime)
                .unwrap();
        let WorkflowStepOutcome::Advanced {
            task: advanced,
            message,
        } = outcome
        else {
            panic!("expected Advanced, got a Blocked/NoOp outcome");
        };
        assert_eq!(
            message,
            "Engineering review recorded: approved_for_validation"
        );
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
AGTX owns workflow-attempt and SHA-256 metadata; do not write it into your artifact."
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
        let sent = captured
            .lock()
            .unwrap()
            .clone()
            .expect("switch_agent_in_tmux should paste the new command");
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
                WorkflowState {
                    id: "engineering_review".into(),
                    label: "Engineering review".into(),
                    role: Some("reviewer".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "final_validation".into(),
                    label: "Final validation".into(),
                    role: Some("validator".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("reviewer".into(), "claude".into());
        project
            .role_bindings
            .insert("validator".into(), "claude".into());
        project
            .role_policies
            .roles
            .insert("validator".into(), WorkflowRolePolicy::default());

        let mut plugin = plugin(graph.clone());
        plugin.artifacts.planning = Some(".agtx/plans/{task_id}.md".into());

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
        let record =
            WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "engineering_review");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux
            .expect_send_keys()
            .withf(|_, cmd: &str| cmd == "/exit")
            .returning(|_, _| Ok(()));
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        // First two polls (pre-switch pane check, exit check): outgoing agent
        // already at a shell. Every poll after
        // that: the freshly launched agent, matching a real tmux pane once
        // `switch_agent_in_tmux` types the new command -- its final
        // launch-detection loop requires a *recognized* agent process name,
        // not merely any string.
        let pane_polls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        mock_tmux.expect_pane_current_command().returning(move |_| {
            if pane_polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                Some("bash".to_string())
            } else {
                Some("claude".to_string())
            }
        });
        mock_tmux
            .expect_capture_pane()
            .returning(|_| Ok(String::new()));
        let captured = Arc::new(Mutex::new(None));
        let captured_paste = captured.clone();
        mock_tmux.expect_paste_text().returning(move |_, text| {
            *captured_paste.lock().unwrap() = Some(text.to_string());
            Ok(())
        });

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry
            .expect_get()
            .returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome =
            submit_engineering_review(&graph, &project, &plugin, task.clone(), &mut db, &runtime)
                .unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));

        let sent = captured
            .lock()
            .unwrap()
            .clone()
            .expect("switch_agent_in_tmux should paste the new command");
        assert!(
            sent.contains("AGTX owns workflow-attempt and SHA-256 metadata"),
            "expected metadata-free artifact instruction in the launched prompt, got: {sent}"
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
                WorkflowState {
                    id: "admission".into(),
                    label: "Admission".into(),
                    role: None,
                    terminal: false,
                },
                WorkflowState {
                    id: "ready_for_planning".into(),
                    label: "Ready for planning".into(),
                    role: None,
                    terminal: false,
                },
                WorkflowState {
                    id: "planning".into(),
                    label: "Planning".into(),
                    role: Some("planner".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("planner".into(), "claude".into());
        project
            .role_policies
            .roles
            .insert("planner".into(), WorkflowRolePolicy::default());

        let mut plugin = plugin(graph.clone());
        plugin.artifacts.planning = Some(".agtx/plans/{task_id}.md".into());

        let worktree = tempfile::tempdir().unwrap();
        let mut task = crate::db::Task::new("Plan thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());

        let db_dir = tempfile::tempdir().unwrap();
        let db_path = db_dir.path().join("wf.db");
        let mut db = Database::open_project_at_path(&db_path).unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "admission", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "admission");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux.expect_has_session().returning(|_| true);
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux
            .expect_create_window()
            .returning(move |_, _, _, command, _, _| {
                assert_state_already_persisted(&db_path_for_check, &task_id_for_check, "planning");
                let command = command.expect("planner launch must carry its prompt");
                assert!(
                    command.contains("AGTX owns revision, workflow-attempt, and SHA-256 metadata"),
                    "planner launch must prohibit agent-authored orchestration metadata, got: {command}"
                );
                assert!(
                    command.contains("Write only the plan content"),
                    "planner prompt must prohibit orchestration fields, got: {command}"
                );
                Ok(())
            });
        let captured_prompt = Arc::new(Mutex::new(None));
        let captured_prompt_clone = captured_prompt.clone();
        mock_tmux.expect_send_keys().returning(move |_, text| {
            *captured_prompt_clone.lock().unwrap() = Some(text.to_string());
            Ok(())
        });

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| {
            let mut agent_ops = MockAgentOperations::new();
            agent_ops
                .expect_prompt_injection()
                .returning(|| crate::agent::PromptInjection::Argv);
            Arc::new(agent_ops) as Arc<dyn AgentOperations>
        });

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome = start_workflow_planning(
            &graph,
            &project,
            &plugin,
            task.clone(),
            &mut db,
            &runtime,
            false,
        )
        .unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));

        // Both hops committed for real, not just observed mid-flight.
        assert_eq!(
            db.get_workflow_task_state(&task.id).unwrap().unwrap().state,
            "planning"
        );
        assert_eq!(db.workflow_transition_history(&task.id).unwrap().len(), 3);
    }

    /// A failed reviewer launch must leave no evidence or input binding behind.
    /// This keeps automation from observing `plan_review` while a competing
    /// handoff owns the shared tmux pane, and makes its next attempt clean.
    #[test]
    fn submit_workflow_plan_leaves_no_evidence_when_reviewer_launch_fails() {
        let graph = WorkflowDefinition {
            initial_state: "planning".into(),
            states: vec![
                WorkflowState {
                    id: "planning".into(),
                    label: "Planning".into(),
                    role: Some("planner".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "plan_review".into(),
                    label: "Plan review".into(),
                    role: Some("plan_reviewer".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("planner".into(), "claude".into());
        project
            .role_bindings
            .insert("plan_reviewer".into(), "claude".into());
        project
            .role_policies
            .roles
            .insert("plan_reviewer".into(), WorkflowRolePolicy::default());

        let mut plugin = plugin(graph.clone());
        plugin.artifacts.planning = Some(".agent-flow/plan.yaml".into());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan.yaml"),
            "# Metadata-free plan\n\nImplement the requested behaviour.\n",
        )
        .unwrap();

        let mut task = crate::db::Task::new("Plan thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-plan".into());

        let db_dir = tempfile::tempdir().unwrap();
        let db_path = db_dir.path().join("wf.db");
        let mut db = Database::open_project_at_path(&db_path).unwrap();
        db.create_task(&task).unwrap();
        let mut current = WorkflowTaskState::new(&task.id, "planning", "main");
        current.plan_revision = 7;
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "planning");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux
            .expect_send_keys()
            .withf(|_, cmd: &str| cmd == "/exit")
            .returning(move |_, _| {
                let db = Database::open_project_at_path(&db_path_for_check).unwrap();
                assert_eq!(
                    db.get_workflow_task_state(&task_id_for_check)
                        .unwrap()
                        .unwrap()
                        .state,
                    "planning"
                );
                assert!(
                    db.workflow_artifacts_for_task(&task_id_for_check)
                        .unwrap()
                        .is_empty(),
                    "a reviewer-launch attempt must not persist planning evidence before the replacement process is verified"
                );
                Ok(())
            });
        mock_tmux.expect_send_keys().returning(|_, _| Ok(()));
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        // The outgoing planner has exited, but the receiving reviewer never
        // becomes a process in the pane. This reproduces the observed tmux
        // hand-off timeout without relying on a real terminal.
        mock_tmux
            .expect_pane_current_command()
            .returning(|_| Some("bash".to_string()));
        mock_tmux
            .expect_capture_pane()
            .returning(|_| Ok(String::new()));
        mock_tmux.expect_paste_text().returning(|_, _| Ok(()));

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry
            .expect_get()
            .returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let error =
            submit_workflow_plan(&graph, &project, &plugin, task.clone(), &mut db, &runtime)
                .expect_err("a reviewer that never launches must fail the hand-off");
        assert!(error.to_string().contains("did not start a process"));
        assert_eq!(
            db.get_workflow_task_state(&task.id).unwrap().unwrap().state,
            "planning"
        );
        assert_eq!(
            db.get_workflow_task_state(&task.id)
                .unwrap()
                .unwrap()
                .plan_revision,
            7,
            "a failed hand-off must not allocate a plan revision"
        );
        assert!(db.workflow_artifacts_for_task(&task.id).unwrap().is_empty());
        assert!(db
            .workflow_step_inputs(&task.id, 2, "plan_review")
            .unwrap()
            .is_empty());
        assert!(db
            .task_execution_events(&task.id)
            .unwrap()
            .iter()
            .any(|event| {
                event.event_type == "agent_handoff_failed"
                    && event.outcome.as_deref() == Some("retryable")
            }));
    }

    /// `submit_workflow_plan`'s reviewer prompt used to hardcode a flat
    /// `.agent-flow/plan-review.yaml` write instruction regardless of
    /// `plugin.artifacts.plan_review`, unlike every other artifact-writing
    /// prompt in this plugin. Pins the fix: a `{task_id}`-templated
    /// `plugin.artifacts.plan_review` must be reflected in the prompt text
    /// actually delivered to the reviewer agent.
    #[test]
    fn submit_workflow_plan_prompt_references_the_task_scoped_review_path() {
        let graph = WorkflowDefinition {
            initial_state: "planning".into(),
            states: vec![
                WorkflowState {
                    id: "planning".into(),
                    label: "Planning".into(),
                    role: Some("planner".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "plan_review".into(),
                    label: "Plan review".into(),
                    role: Some("plan_reviewer".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("planner".into(), "claude".into());
        project
            .role_bindings
            .insert("plan_reviewer".into(), "claude".into());
        project
            .role_policies
            .roles
            .insert("plan_reviewer".into(), WorkflowRolePolicy::default());

        let mut plugin = plugin(graph.clone());
        plugin.artifacts.planning = Some(".agent-flow/plan.yaml".into());
        plugin.artifacts.plan_review = Some(".agent-flow/{task_id}/plan-review.yaml".into());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan.yaml"),
            "# Metadata-free plan\n\nImplement the requested behaviour.\n",
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
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux
            .expect_send_keys()
            .withf(|_, cmd: &str| cmd == "/exit")
            .returning(|_, _| Ok(()));
        let captured_prompt = Arc::new(Mutex::new(None));
        let captured_prompt_clone = captured_prompt.clone();
        mock_tmux.expect_send_keys().returning(move |_, text| {
            *captured_prompt_clone.lock().unwrap() = Some(text.to_string());
            Ok(())
        });
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        let command_checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let command_checks_for_mock = Arc::clone(&command_checks);
        mock_tmux.expect_pane_current_command().returning(move |_| {
            if command_checks_for_mock.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                Some("bash".to_string())
            } else {
                Some("claude".to_string())
            }
        });
        mock_tmux
            .expect_capture_pane()
            .returning(|_| Ok(String::new()));

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry
            .expect_get()
            .returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome =
            submit_workflow_plan(&graph, &project, &plugin, task.clone(), &mut db, &runtime)
                .unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));

        let prompt = captured_prompt
            .lock()
            .unwrap()
            .clone()
            .expect("paste_text should have delivered the reviewer prompt");
        let expected_path = format!(".agent-flow/{}/plan-review.yaml", task.id);
        assert!(
            prompt.contains(&expected_path),
            "prompt must tell the reviewer to write the task-scoped path {expected_path}, got: {prompt}"
        );
        assert!(
            !prompt.contains("write .agent-flow/plan-review.yaml"),
            "prompt must not still reference the old flat path"
        );
        assert!(
            prompt.contains("AGTX owns all revision, attempt, and SHA-256 metadata"),
            "reviewer prompt must prohibit agent-authored orchestration metadata, got: {prompt}"
        );
        assert!(
            prompt.contains(
                "Classify every finding as BLOCKING, REQUIRED-NONBLOCKING, or SUGGESTION"
            ),
            "reviewer prompt must require calibrated finding classifications, got: {prompt}"
        );
        assert!(
            prompt.contains("Use changes_requested only when at least one BLOCKING finding exists"),
            "reviewer prompt must reserve rejection for blocking findings, got: {prompt}"
        );
    }

    /// A failed reviewer launch must not advance the durable lane: both
    /// chained transitions (`implementation_complete`,
    /// `start_engineering_review`) are committed via
    /// `advance_workflow_state_chain` only after `switch_agent_in_tmux`
    /// confirms the engineering reviewer actually launched.
    #[test]
    fn submit_workflow_implementation_launches_the_reviewer_before_persisting_the_chain() {
        let graph = WorkflowDefinition {
            initial_state: "running".into(),
            states: vec![
                WorkflowState {
                    id: "running".into(),
                    label: "Running".into(),
                    role: Some("implementer".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "implementing_complete".into(),
                    label: "Implementation complete".into(),
                    role: None,
                    terminal: false,
                },
                WorkflowState {
                    id: "engineering_review".into(),
                    label: "Engineering review".into(),
                    role: Some("reviewer".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("implementer".into(), "claude".into());
        project
            .role_bindings
            .insert("reviewer".into(), "claude".into());
        project
            .role_policies
            .roles
            .insert("reviewer".into(), WorkflowRolePolicy::default());

        let plugin = plugin(graph.clone());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree
                .path()
                .join(".agent-flow/implementation-result.yaml"),
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
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux
            .expect_send_keys()
            .withf(|_, cmd: &str| cmd == "/exit")
            .returning(move |_, _| {
                let db = Database::open_project_at_path(&db_path_for_check).unwrap();
                assert_eq!(
                    db.get_workflow_task_state(&task_id_for_check)
                        .unwrap()
                        .unwrap()
                        .state,
                    "running",
                    "durable state must not advance to 'engineering_review' until \
                     switch_agent_in_tmux confirms the reviewer launched"
                );
                Ok(())
            });
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        let command_checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let command_checks_for_mock = Arc::clone(&command_checks);
        mock_tmux.expect_pane_current_command().returning(move |_| {
            if command_checks_for_mock.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                Some("bash".to_string())
            } else {
                Some("claude".to_string())
            }
        });
        mock_tmux
            .expect_capture_pane()
            .returning(|_| Ok(String::new()));
        mock_tmux.expect_paste_text().returning(|_, _| Ok(()));

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry
            .expect_get()
            .returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome = submit_workflow_implementation(
            &graph,
            &project,
            &plugin,
            task.clone(),
            &mut db,
            &runtime,
        )
        .unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));

        // Both chained hops committed for real, in one transaction.
        assert_eq!(
            db.get_workflow_task_state(&task.id).unwrap().unwrap().state,
            "engineering_review"
        );
        let history = db.workflow_transition_history(&task.id).unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[1].action, "implementation_complete");
        assert_eq!(history[2].action, "start_engineering_review");
    }

    /// A failed next-agent launch must not advance the durable lane: the
    /// resolved verdict transition (here, `approved_for_validation` to the
    /// validator) is committed only after `switch_agent_in_tmux` confirms
    /// the next agent actually launched.
    #[test]
    fn submit_engineering_review_launches_the_next_agent_before_persisting_the_verdict() {
        let graph = WorkflowDefinition {
            initial_state: "engineering_review".into(),
            states: vec![
                WorkflowState {
                    id: "engineering_review".into(),
                    label: "Engineering review".into(),
                    role: Some("reviewer".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "final_validation".into(),
                    label: "Final validation".into(),
                    role: Some("validator".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("reviewer".into(), "claude".into());
        project
            .role_bindings
            .insert("validator".into(), "claude".into());
        project
            .role_policies
            .roles
            .insert("validator".into(), WorkflowRolePolicy::default());

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
        let record =
            WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "engineering_review");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux
            .expect_send_keys()
            .withf(|_, cmd: &str| cmd == "/exit")
            .returning(move |_, _| {
                let db = Database::open_project_at_path(&db_path_for_check).unwrap();
                assert_eq!(
                    db.get_workflow_task_state(&task_id_for_check)
                        .unwrap()
                        .unwrap()
                        .state,
                    "engineering_review",
                    "durable state must not advance to 'final_validation' until \
                     switch_agent_in_tmux confirms the next agent launched"
                );
                Ok(())
            });
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        let command_checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let command_checks_for_mock = Arc::clone(&command_checks);
        mock_tmux.expect_pane_current_command().returning(move |_| {
            if command_checks_for_mock.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                Some("bash".to_string())
            } else {
                Some("claude".to_string())
            }
        });
        mock_tmux
            .expect_capture_pane()
            .returning(|_| Ok(String::new()));
        mock_tmux.expect_paste_text().returning(|_, _| Ok(()));

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry
            .expect_get()
            .returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome =
            submit_engineering_review(&graph, &project, &plugin, task.clone(), &mut db, &runtime)
                .unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));
        assert_eq!(
            db.get_workflow_task_state(&task.id).unwrap().unwrap().state,
            "final_validation"
        );
    }

    /// A failed next-agent launch must not advance the durable lane: a
    /// `passed` verdict's `begin_feature_integration` transition is
    /// committed only after `switch_agent_in_tmux` confirms the next agent
    /// actually launched. `archive_workflow_artifact` only runs on the
    /// `failed` path (see the function's own doc comment), so it is not
    /// exercised here; this test only covers the launch/persist ordering.
    #[test]
    fn submit_final_validation_launches_the_next_agent_before_persisting_the_verdict() {
        let graph = WorkflowDefinition {
            initial_state: "final_validation".into(),
            states: vec![
                WorkflowState {
                    id: "final_validation".into(),
                    label: "Final validation".into(),
                    role: Some("validator".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "integrate_to_feature".into(),
                    label: "Integrate to feature".into(),
                    role: Some("integrator".into()),
                    terminal: true,
                },
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
        project
            .role_bindings
            .insert("validator".into(), "claude".into());
        project
            .role_bindings
            .insert("integrator".into(), "claude".into());
        project
            .role_policies
            .roles
            .insert("integrator".into(), WorkflowRolePolicy::default());

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
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        let task_id_for_check = task.id.clone();
        let db_path_for_check = db_path.clone();
        mock_tmux
            .expect_send_keys()
            .withf(|_, cmd: &str| cmd == "/exit")
            .returning(move |_, _| {
                let db = Database::open_project_at_path(&db_path_for_check).unwrap();
                assert_eq!(
                    db.get_workflow_task_state(&task_id_for_check)
                        .unwrap()
                        .unwrap()
                        .state,
                    "final_validation",
                    "durable state must not advance to 'integrate_to_feature' until \
                     switch_agent_in_tmux confirms the next agent launched"
                );
                Ok(())
            });
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        let command_checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let command_checks_for_mock = Arc::clone(&command_checks);
        mock_tmux.expect_pane_current_command().returning(move |_| {
            if command_checks_for_mock.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                Some("bash".to_string())
            } else {
                Some("claude".to_string())
            }
        });
        mock_tmux
            .expect_capture_pane()
            .returning(|_| Ok(String::new()));
        mock_tmux.expect_paste_text().returning(|_, _| Ok(()));

        let mut mock_registry = MockAgentRegistry::new();
        mock_registry
            .expect_get()
            .returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome =
            submit_final_validation(&graph, &project, &plugin, task.clone(), &mut db, &runtime)
                .unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));
        assert_eq!(
            db.get_workflow_task_state(&task.id).unwrap().unwrap().state,
            "integrate_to_feature"
        );
    }

    /// `submit_plan_review` on an `approved` artifact must drive the exact
    /// same transition `decide_workflow_plan(..., true, ...)` would: it is a
    /// thin artifact-reading wrapper, not a second implementation. No tmux or
    /// agent launch happens on approval, so no mocks are needed for either
    /// side of the comparison.
    #[test]
    fn submit_plan_review_approved_matches_decide_workflow_plan_directly() {
        let graph = WorkflowDefinition {
            initial_state: "plan_review".into(),
            states: vec![
                WorkflowState {
                    id: "plan_review".into(),
                    label: "Plan review".into(),
                    role: Some("plan_reviewer".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "plan_approved".into(),
                    label: "Plan approved".into(),
                    role: None,
                    terminal: true,
                },
            ],
            transitions: vec![WorkflowTransition {
                action: "approve_plan".into(),
                from: "plan_review".into(),
                to: "plan_approved".into(),
                guards: vec![crate::workflow::WorkflowGuard::ApprovedPlan],
            }],
        };
        graph.validate().unwrap();

        let project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        let plugin = plugin(graph.clone());

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(MockTmuxOperations::new());
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(MockAgentRegistry::new());
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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        // Artifact-driven path.
        let worktree_a = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree_a.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree_a.path().join(".agent-flow/plan-review.yaml"),
            "verdict: approved\n",
        )
        .unwrap();
        let mut task_a = crate::db::Task::new("Review thing", "claude", "proj");
        task_a.worktree_path = Some(worktree_a.path().to_string_lossy().to_string());
        let mut db_a = Database::open_in_memory_project().unwrap();
        db_a.create_task(&task_a).unwrap();
        let mut state_a = WorkflowTaskState::new(&task_a.id, "plan_review", "main");
        state_a.plan_revision = 1;
        state_a.plan_hash = Some("deadbeef".into());
        let record_a = WorkflowTransitionRecord::new(&task_a.id, "seed", "backlog", "plan_review");
        db_a.record_workflow_admission(&task_a, &state_a, &record_a)
            .unwrap();

        let outcome_a = submit_plan_review(
            &graph,
            &project,
            &plugin,
            task_a.clone(),
            &mut db_a,
            &runtime,
        )
        .unwrap();
        assert!(matches!(outcome_a, WorkflowStepOutcome::Advanced { .. }));
        let final_state_a = db_a.get_workflow_task_state(&task_a.id).unwrap().unwrap();

        // Direct manual-path call with an equivalent starting state.
        let mut task_b = crate::db::Task::new("Review thing", "claude", "proj");
        task_b.worktree_path = task_a.worktree_path.clone();
        task_b.id = task_a.id.clone();
        let mut db_b = Database::open_in_memory_project().unwrap();
        db_b.create_task(&task_b).unwrap();
        db_b.record_workflow_admission(&task_b, &state_a, &record_a)
            .unwrap();

        let outcome_b = decide_workflow_plan(
            &graph,
            &project,
            &plugin,
            task_b.clone(),
            true,
            &mut db_b,
            &runtime,
        )
        .unwrap();
        assert!(matches!(outcome_b, WorkflowStepOutcome::Advanced { .. }));
        let final_state_b = db_b.get_workflow_task_state(&task_b.id).unwrap().unwrap();

        assert_eq!(final_state_a.state, "plan_approved");
        assert_eq!(final_state_a.state, final_state_b.state);
        assert_eq!(
            final_state_a.approved_plan_hash,
            final_state_b.approved_plan_hash
        );
        assert_eq!(
            final_state_a.approved_plan_revision,
            final_state_b.approved_plan_revision
        );
    }

    /// Graph/project/plugin shared by the `changes_requested` tests below:
    /// the destination (`planning`) role is bound to a mock agent so
    /// `decide_workflow_plan`'s reject branch has somewhere to send the
    /// revise prompt.
    fn changes_requested_fixtures() -> (WorkflowDefinition, WorkflowProjectConfig, WorkflowPlugin) {
        let graph = WorkflowDefinition {
            initial_state: "planning".into(),
            states: vec![
                WorkflowState {
                    id: "planning".into(),
                    label: "Planning".into(),
                    role: Some("planner".into()),
                    terminal: true,
                },
                WorkflowState {
                    id: "plan_review".into(),
                    label: "Plan review".into(),
                    role: Some("plan_reviewer".into()),
                    terminal: false,
                },
            ],
            transitions: vec![WorkflowTransition {
                action: "plan_changes_requested".into(),
                from: "plan_review".into(),
                to: "planning".into(),
                guards: vec![],
            }],
        };
        graph.validate().unwrap();

        let mut project = WorkflowProjectConfig {
            target_branch: "main".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project
            .role_bindings
            .insert("planner".into(), "claude".into());
        project
            .role_policies
            .roles
            .insert("planner".into(), WorkflowRolePolicy::default());

        let plugin = plugin(graph.clone());
        (graph, project, plugin)
    }

    /// A mock registry for the planner handoff. The acknowledged switch path
    /// launches the revise prompt directly, without a detached sender.
    fn mock_agent_registry_for_argv_launch() -> Arc<dyn AgentRegistry> {
        let mut mock_registry = MockAgentRegistry::new();
        mock_registry.expect_get().returning(|_| {
            let mut ops = MockAgentOperations::new();
            ops.expect_prompt_injection()
                .return_const(crate::agent::PromptInjection::Unknown);
            ops.expect_build_interactive_command()
                .returning(|prompt| format!("claude '{}'", prompt));
            Arc::new(ops) as Arc<dyn AgentOperations>
        });
        Arc::new(mock_registry)
    }

    /// Builds the tmux mock shared by the `changes_requested` tests. The
    /// first process probe sees a shell after the reviewer exits; the next
    /// confirms the planner process, which is the launch acknowledgement.
    fn mock_tmux_capturing_paste(
        pane_content: &'static str,
    ) -> (
        Arc<dyn TmuxOperations>,
        std::sync::mpsc::Receiver<()>,
        Arc<Mutex<Option<String>>>,
    ) {
        let (tx, rx) = std::sync::mpsc::channel();
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let captured_for_closure = captured.clone();
        let command_checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let command_checks_for_mock = Arc::clone(&command_checks);

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux.expect_window_exists().returning(|_| Ok(true));
        mock_tmux.expect_send_keys().returning(|_, _| Ok(()));
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        mock_tmux.expect_pane_current_command().returning(move |_| {
            if command_checks_for_mock.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                Some("bash".to_string())
            } else {
                Some("claude".to_string())
            }
        });
        mock_tmux
            .expect_capture_pane()
            .returning(move |_| Ok(pane_content.to_string()));
        mock_tmux.expect_paste_text().returning(move |_, text| {
            *captured_for_closure.lock().unwrap() = Some(text.to_string());
            let _ = tx.send(());
            Ok(())
        });

        (Arc::new(mock_tmux), rx, captured)
    }

    #[test]
    fn submit_plan_review_changes_requested_uses_artifact_findings_not_pane_capture() {
        let (graph, project, plugin) = changes_requested_fixtures();
        let agent_registry = mock_agent_registry_for_argv_launch();
        // If the artifact's findings were ignored, this is what a pane-scrape
        // would have produced instead -- distinct text so the assertions can
        // tell which source actually won.
        let (tmux_ops, rx, captured) =
            mock_tmux_capturing_paste("Plan reviewer pane: please also check the retry path.");
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: changes_requested\nfindings: Add input validation for the new endpoint before revising further.\n",
        )
        .unwrap();

        let mut task = crate::db::Task::new("Review thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-review".into());

        let mut db = Database::open_in_memory_project().unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "plan_review", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "plan_review");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome =
            submit_plan_review(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));

        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("revise prompt should be delivered to the planner");
        let sent = captured
            .lock()
            .unwrap()
            .clone()
            .expect("paste_text should have captured the revise command");
        assert!(
            sent.contains("Add input validation for the new endpoint before revising further."),
            "expected the artifact's findings verbatim in the revise prompt, got: {sent}"
        );
        assert!(
            !sent.contains("reviewer's pane, captured at the moment"),
            "findings were present on the artifact; the pane-capture fallback header must not appear, got: {sent}"
        );
        assert!(
            !sent.contains("please also check the retry path"),
            "the stubbed pane content must be ignored when the artifact has findings, got: {sent}"
        );
        let inputs = db.workflow_step_inputs(&task.id, 2, "planning").unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name, "plan_review");
        let review = db
            .workflow_artifact(&inputs[0].artifact_id)
            .unwrap()
            .unwrap();
        assert_eq!(review.workflow_attempt, 1);
        assert_eq!(review.state, "plan_review");
        assert_eq!(review.sha256, inputs[0].expected_sha256);
        assert!(
            sent.contains(&format!("artifact {}", review.id)),
            "planner prompt must identify the exact persisted review artifact, got: {sent}"
        );
        assert!(
            sent.contains("AGTX owns revision, workflow-attempt, and SHA-256 metadata"),
            "revision handoff must prohibit agent-authored orchestration metadata, got: {sent}"
        );
        assert!(
            sent.contains("Write only the revised plan content"),
            "revision handoff must prohibit orchestration fields, got: {sent}"
        );
    }

    /// A failed return to Planning must leave no review evidence or input
    /// binding behind. Otherwise the next automation tick can conflict with
    /// the first failed hand-off instead of retrying it safely.
    #[test]
    fn submit_plan_review_changes_requested_leaves_no_evidence_when_planner_launch_fails() {
        let (graph, project, plugin) = changes_requested_fixtures();
        let agent_registry = mock_agent_registry_for_argv_launch();
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: changes_requested\nfindings: Validate malformed input before persistence.\n",
        )
        .unwrap();
        let mut task = crate::db::Task::new("Review thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-review".into());

        let mut db = Database::open_in_memory_project().unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "plan_review", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "plan_review");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

        let mut mock_tmux = MockTmuxOperations::new();
        mock_tmux.expect_send_keys().returning(|_, _| Ok(()));
        mock_tmux.expect_send_key().returning(|_, _| Ok(()));
        // The reviewer exits, but the planner never appears in the pane.
        mock_tmux
            .expect_pane_current_command()
            .returning(|_| Some("bash".to_string()));
        mock_tmux
            .expect_capture_pane()
            .returning(|_| Ok(String::new()));
        mock_tmux.expect_paste_text().times(0);
        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let error = submit_plan_review(&graph, &project, &plugin, task.clone(), &mut db, &runtime)
            .expect_err("a planner that never launches must fail the hand-off");
        assert!(error.to_string().contains("did not start a process"));
        assert_eq!(
            db.get_workflow_task_state(&task.id).unwrap().unwrap().state,
            "plan_review"
        );
        assert!(db.workflow_artifacts_for_task(&task.id).unwrap().is_empty());
        assert!(db
            .workflow_step_inputs(&task.id, 2, "planning")
            .unwrap()
            .is_empty());
        assert!(db
            .task_execution_events(&task.id)
            .unwrap()
            .iter()
            .any(|event| {
                event.event_type == "agent_handoff_failed"
                    && event.outcome.as_deref() == Some("retryable")
            }));
    }

    #[test]
    fn submit_plan_review_changes_requested_blocks_when_findings_missing() {
        let (graph, project, plugin) = changes_requested_fixtures();
        let agent_registry = mock_agent_registry_for_argv_launch();
        let (tmux_ops, rx, captured) = mock_tmux_capturing_paste(
            "Plan reviewer pane: needs better error handling on the retry path.",
        );
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        // `verdict` present, no `findings:` line at all.
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: changes_requested\n",
        )
        .unwrap();

        let mut task = crate::db::Task::new("Review thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-review".into());

        let mut db = Database::open_in_memory_project().unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "plan_review", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "plan_review");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome =
            submit_plan_review(&graph, &project, &plugin, task.clone(), &mut db, &runtime).unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Blocked { .. }));
        assert!(
            db.task_step_reports(&task.id)
                .unwrap()
                .iter()
                .all(|report| report.artifact_sha256.is_none()),
            "invalid findings must not become immutable evidence"
        );
        assert!(
            db.workflow_step_inputs(&task.id, 2, "planning")
                .unwrap()
                .is_empty(),
            "a review with no findings must never be handed to a planner"
        );
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "a blocked review must not launch the planner"
        );
        assert!(captured.lock().unwrap().is_none());
    }

    /// A manual Shift+N decision without durable reviewer output is not a
    /// recoverable planner handoff. It must leave the state untouched rather
    /// than inventing feedback from terminal scrollback.
    #[test]
    fn decide_workflow_plan_blocks_without_a_persisted_review_artifact() {
        let (graph, project, plugin) = changes_requested_fixtures();
        let agent_registry = mock_agent_registry_for_argv_launch();
        let (tmux_ops, rx, captured) =
            mock_tmux_capturing_paste("Manual reviewer pane: tighten the retry logic.");
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());

        // No `.agent-flow/plan-review.yaml` written at all.
        let worktree = tempfile::tempdir().unwrap();

        let mut task = crate::db::Task::new("Review thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-review".into());

        let mut db = Database::open_in_memory_project().unwrap();
        db.create_task(&task).unwrap();
        let current = WorkflowTaskState::new(&task.id, "plan_review", "main");
        let record = WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "plan_review");
        db.record_workflow_admission(&task, &current, &record)
            .unwrap();

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
            session_probe: crate::agent::native_session::default_probe(),
            git_provider_ops: None,
        };

        let outcome = decide_workflow_plan(
            &graph,
            &project,
            &plugin,
            task.clone(),
            false,
            &mut db,
            &runtime,
        )
        .unwrap();
        assert!(matches!(outcome, WorkflowStepOutcome::Blocked { .. }));
        assert_eq!(
            db.get_workflow_task_state(&task.id).unwrap().unwrap().state,
            "plan_review"
        );
        assert!(rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err());
        assert!(captured.lock().unwrap().is_none());
    }
}

/// The verified hand-off, driven directly. Every test scripts the provider
/// session stores and uses a tmux mock with no expectation for keystrokes the
/// scenario must never send: a stray keystroke panics the test.
#[cfg(test)]
#[cfg(feature = "test-mocks")]
mod handoff_tests {
    use super::*;
    use crate::agent::native_session::{Delivery, SessionProbe, TurnActivity};
    use crate::agent::{AgentRegistry, MockAgentRegistry};
    use crate::config::{GlobalConfig, MergedConfig, ProjectConfig};
    use crate::git::MockGitOperations;
    use crate::tmux::MockTmuxOperations;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ScriptedProbe {
        activity: Option<TurnActivity>,
        delivery: Delivery,
    }

    impl SessionProbe for ScriptedProbe {
        fn turn_activity(&self, _: &str, _: &Path, _: &str) -> Option<TurnActivity> {
            self.activity
        }
        fn find_delivered_prompt(
            &self,
            _: &str,
            _: &Path,
            _: &str,
            _: std::time::SystemTime,
        ) -> Delivery {
            self.delivery.clone()
        }
        fn delivery_timeout(&self) -> std::time::Duration {
            std::time::Duration::ZERO
        }
    }

    /// Mirrors task 16581cba: the plan reviewer (Codex) hands the pane to
    /// the implementer (OpenCode) with the prompt carried in the launch.
    fn handoff(command: &str) -> Handoff<'_> {
        Handoff {
            journal_state: "plan_review",
            journal_attempt: 6,
            source_agent: "codex",
            destination_agent: "opencode",
            target: "proj:task-16581cba",
            worktree: "/agtx-work/worktrees/16581cba",
            command,
            prompt: "You are the implementer for task 16581cba. Implement only the approved plan.",
            paste_prompt: false,
        }
    }

    fn run(
        mock_tmux: MockTmuxOperations,
        probe: &ScriptedProbe,
        db: &Database,
        task: &Task,
        command: &str,
    ) -> Result<Option<String>> {
        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(mock_tmux);
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(MockAgentRegistry::new());
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = MergedConfig::merge(&GlobalConfig::default(), &ProjectConfig::default());
        let flags = crate::FeatureFlags::default();
        let runtime = WorkflowRuntime {
            tmux_ops: &tmux_ops,
            agent_registry: &agent_registry,
            git_ops: &git_ops,
            tmux_project_name: "proj",
            project_path: Path::new("/agtx-work/project"),
            config: &config,
            flags: &flags,
            session_probe: probe,
            git_provider_ops: None,
        };
        hand_off_and_deliver(db, task, &handoff(command), &runtime)
    }

    fn task_and_db() -> (Task, Database) {
        let db = Database::open_in_memory_project().unwrap();
        let task = Task::new("PHASE1 - Setup", "codex", "proj");
        db.create_task(&task).unwrap();
        (task, db)
    }

    fn events(db: &Database, task: &Task, event_type: &str) -> usize {
        db.task_execution_events(&task.id)
            .unwrap()
            .iter()
            .filter(|event| event.event_type == event_type)
            .count()
    }

    /// Pane polls answer `sequence[i]`, repeating the last entry.
    fn pane_sequence(mock: &mut MockTmuxOperations, sequence: &'static [&'static str]) {
        let polls = Arc::new(AtomicUsize::new(0));
        mock.expect_pane_current_command().returning(move |_| {
            let index = polls.fetch_add(1, Ordering::SeqCst).min(sequence.len() - 1);
            Some(sequence[index].to_string())
        });
    }

    /// RC1: the reviewer wrote its verdict but its turn is still running.
    /// Nothing may be typed into the pane; the hand-off waits for the turn.
    #[test]
    fn handoff_waits_while_the_source_agent_is_mid_turn() {
        let (task, db) = task_and_db();
        let mut mock = MockTmuxOperations::new();
        pane_sequence(&mut mock, &["codex"]);
        mock.expect_send_key().never();
        mock.expect_send_keys().never();
        mock.expect_paste_text().never();
        let probe = ScriptedProbe {
            activity: Some(TurnActivity::Busy),
            delivery: Delivery::Unverifiable,
        };

        let error = run(mock, &probe, &db, &task, "opencode").unwrap_err();

        assert!(error.to_string().contains("deferred"), "{error}");
        assert_eq!(events(&db, &task, "agent_handoff_deferred"), 1);
    }

    /// A turn that never reports its end cannot stall the task forever: after
    /// the deferral budget the hand-off proceeds (still exit-verified).
    #[test]
    fn handoff_proceeds_after_the_idle_wait_budget_is_spent() {
        let (task, db) = task_and_db();
        for _ in 0..MAX_IDLE_DEFERRALS {
            let mut event = TaskExecutionEvent::new(&task.id, "agent_handoff_deferred");
            event.state = Some("plan_review".into());
            event.workflow_attempt = Some(6);
            db.record_task_execution_event(&event).unwrap();
        }
        let mut mock = MockTmuxOperations::new();
        // Codex busy, then it exits to bash, then OpenCode runs.
        pane_sequence(&mut mock, &["codex", "bash", "opencode"]);
        mock.expect_send_key().returning(|_, _| Ok(()));
        mock.expect_send_keys().returning(|_, _| Ok(()));
        mock.expect_capture_pane().returning(|_| Ok(String::new()));
        let probe = ScriptedProbe {
            activity: Some(TurnActivity::Busy),
            delivery: Delivery::Confirmed("ses_new".into()),
        };

        let session = run(mock, &probe, &db, &task, "opencode").unwrap();

        assert_eq!(session.as_deref(), Some("ses_new"));
        assert_eq!(events(&db, &task, "agent_handoff_idle_timeout"), 1);
    }

    /// RC2/RC3, the 16581cba incident: Codex never leaves the pane. The
    /// launch command and prompt must never be typed into it, and Codex
    /// must not be accepted as the launched OpenCode.
    #[test]
    fn a_source_agent_that_never_exits_is_never_typed_into() {
        let (task, db) = task_and_db();
        let mut mock = MockTmuxOperations::new();
        pane_sequence(&mut mock, &["codex"]);
        mock.expect_capture_pane().returning(|_| Ok(String::new()));
        // Only the exit escalation (Ctrl+C, Ctrl+D) is allowed.
        mock.expect_send_key()
            .withf(|_, key: &str| key == "C-c" || key == "C-d")
            .returning(|_, _| Ok(()));
        mock.expect_send_keys().never();
        mock.expect_paste_text().never();
        let probe = ScriptedProbe {
            activity: Some(TurnActivity::Idle),
            delivery: Delivery::Confirmed("unused".into()),
        };

        let error = run(mock, &probe, &db, &task, "opencode").unwrap_err();

        assert!(error.to_string().contains("could not confirm"), "{error}");
        assert_eq!(events(&db, &task, "agent_handoff_failed"), 1);
    }

    /// RC3: the old agent exited but the pane then shows *another* agent
    /// than the destination; that is not a successful launch.
    #[test]
    fn only_the_destination_agent_counts_as_launched() {
        let (task, db) = task_and_db();
        let mut mock = MockTmuxOperations::new();
        pane_sequence(&mut mock, &["bash", "bash", "codex"]);
        mock.expect_capture_pane().returning(|_| Ok(String::new()));
        mock.expect_send_key().returning(|_, _| Ok(()));
        mock.expect_send_keys().returning(|_, _| Ok(()));
        let probe = ScriptedProbe {
            activity: None,
            delivery: Delivery::Confirmed("unused".into()),
        };

        let error = run(mock, &probe, &db, &task, "opencode").unwrap_err();

        assert!(
            error
                .to_string()
                .contains("did not start a process for 'opencode'"),
            "{error}"
        );
    }

    /// RC4: the destination runs, but its session never records the prompt.
    /// The hand-off fails (retryable) instead of reporting a delivery.
    #[test]
    fn an_unconfirmed_prompt_fails_the_handoff() {
        let (task, db) = task_and_db();
        let mut mock = MockTmuxOperations::new();
        pane_sequence(&mut mock, &["bash", "bash", "opencode"]);
        mock.expect_capture_pane().returning(|_| Ok(String::new()));
        mock.expect_send_key().returning(|_, _| Ok(()));
        mock.expect_send_keys().returning(|_, _| Ok(()));
        let probe = ScriptedProbe {
            activity: None,
            delivery: Delivery::Missing,
        };

        let error = run(mock, &probe, &db, &task, "opencode").unwrap_err();

        assert!(
            error.to_string().contains("did not appear in its session"),
            "{error}"
        );
        assert_eq!(events(&db, &task, "agent_prompt_unconfirmed"), 1);
    }

    #[test]
    fn a_confirmed_delivery_records_exactly_that_provider_session() {
        let (task, db) = task_and_db();
        let mut mock = MockTmuxOperations::new();
        pane_sequence(&mut mock, &["bash", "bash", "opencode"]);
        mock.expect_capture_pane().returning(|_| Ok(String::new()));
        mock.expect_send_key().returning(|_, _| Ok(()));
        mock.expect_send_keys().returning(|_, _| Ok(()));
        let probe = ScriptedProbe {
            activity: None,
            delivery: Delivery::Confirmed("ses_implementer".into()),
        };

        let session = run(mock, &probe, &db, &task, "opencode").unwrap();
        record_agent_prompt(
            &db,
            &task,
            "implementing",
            8,
            "opencode",
            "prompt",
            session.as_deref(),
        )
        .unwrap();

        let sessions = db.provider_sessions(&task.id).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].provider_session_id, "ses_implementer");
        assert_eq!(sessions[0].state, "implementing");
        assert_eq!(sessions[0].workflow_attempt, 8);
    }
}

/// `complete_feature_integration` against real repositories: a bare `origin`,
/// the configured target checkout, and a task worktree. The git side is not
/// mocked because the behaviour under test *is* what git does with the refs.
#[cfg(test)]
#[cfg(feature = "test-mocks")]
mod integration_tests {
    use super::*;
    use crate::agent::MockAgentRegistry;
    use crate::config::{GlobalConfig, MergedConfig, ProjectConfig};
    use crate::db::{INTEGRATION_BLOCKED, INTEGRATION_CONFLICTS};
    use crate::git::{MockGitProviderOperations, RealGitOps};
    use crate::tmux::MockTmuxOperations;
    use crate::workflow::{WorkflowGuard, WorkflowState, WorkflowTransition};

    const TARGET: &str = "feature/poc";
    const BRANCH: &str = "task/integrate";

    fn git(dir: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git must be on PATH for this test");
        assert!(
            output.status.success(),
            "git {args:?} failed in {}: {}",
            dir.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    /// Identity for commits, and byte-exact checkouts whatever the machine's
    /// global `core.autocrlf` says.
    fn identify(dir: &Path) {
        git(dir, &["config", "user.email", "test@example.com"]);
        git(dir, &["config", "user.name", "Test"]);
        git(dir, &["config", "core.autocrlf", "false"]);
    }

    /// `origin` (bare), the target checkout on `feature/poc`, and a task
    /// worktree on its own branch with the task's work left uncommitted, the
    /// way an agent leaves it for the executor.
    struct Repos {
        _root: tempfile::TempDir,
        origin: PathBuf,
        project: PathBuf,
        worktree: PathBuf,
    }

    impl Repos {
        fn new(target_edit: Option<&str>) -> Self {
            let root = tempfile::tempdir().unwrap();
            let origin = root.path().join("origin.git");
            let project = root.path().join("project");
            let worktree = root.path().join("worktree");
            std::fs::create_dir_all(&origin).unwrap();
            std::fs::create_dir_all(&project).unwrap();
            git(&origin, &["init", "-q", "--bare"]);
            git(&project, &["init", "-q"]);
            identify(&project);
            git(&project, &["checkout", "-q", "-b", TARGET]);
            std::fs::write(project.join("shared.txt"), "base\n").unwrap();
            git(&project, &["add", "."]);
            git(&project, &["commit", "-q", "-m", "seed"]);
            git(
                &project,
                &["remote", "add", "origin", &origin.to_string_lossy()],
            );
            git(&project, &["push", "-q", "-u", "origin", TARGET]);
            git(
                &project,
                &[
                    "worktree",
                    "add",
                    "-q",
                    "-b",
                    BRANCH,
                    &worktree.to_string_lossy(),
                ],
            );
            std::fs::write(worktree.join("shared.txt"), "task\n").unwrap();
            std::fs::create_dir_all(worktree.join(".agent-flow")).unwrap();
            std::fs::write(
                worktree.join(".agent-flow/integration-ready.yaml"),
                "verdict: ready_for_integration\nfindings: >-\n  ready\n",
            )
            .unwrap();
            if let Some(content) = target_edit {
                std::fs::write(project.join("shared.txt"), content).unwrap();
                git(&project, &["commit", "-q", "-am", "target moved"]);
                git(&project, &["push", "-q", "origin", TARGET]);
            }
            Self {
                _root: root,
                origin,
                project,
                worktree,
            }
        }

        /// A second clone standing in for a person outside the container: the
        /// pull request's conflict editor, or a local checkout.
        fn outside_clone(&self) -> PathBuf {
            let clone = self._root.path().join("outside");
            git(
                self._root.path(),
                &[
                    "clone",
                    "-q",
                    "-c",
                    "core.autocrlf=false",
                    &self.origin.to_string_lossy(),
                    &clone.to_string_lossy(),
                ],
            );
            identify(&clone);
            clone
        }

        fn remote_ref(&self, branch: &str) -> Option<String> {
            std::process::Command::new("git")
                .args(["rev-parse", "--verify", "--quiet", branch])
                .current_dir(&self.origin)
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        }
    }

    fn workflow() -> WorkflowDefinition {
        let graph = WorkflowDefinition {
            initial_state: "integrate_to_feature".into(),
            states: vec![
                WorkflowState {
                    id: "integrate_to_feature".into(),
                    label: "Integrate to feature".into(),
                    role: Some("engineering_reviewer".into()),
                    terminal: false,
                },
                WorkflowState {
                    id: "done".into(),
                    label: "Done".into(),
                    role: None,
                    terminal: true,
                },
            ],
            transitions: vec![WorkflowTransition {
                action: "complete_feature_integration".into(),
                from: "integrate_to_feature".into(),
                to: "done".into(),
                guards: vec![WorkflowGuard::IntegratedIntoTarget],
            }],
        };
        graph.validate().unwrap();
        graph
    }

    fn project_config(open_pull_requests: bool) -> WorkflowProjectConfig {
        let config: WorkflowProjectConfig = toml::from_str(&format!(
            r#"
target_branch = "{TARGET}"
[role_bindings]
engineering_reviewer = "codex"
[role_policies.engineering_reviewer]
states = ["integrate_to_feature"]
modify_source_and_tests = true
[state_policies.integrate_to_feature]
role = "engineering_reviewer"
final_task_commit = true
push_task_branch = true
create_or_update_task_pr = {open_pull_requests}
merge_task_into_target = true
merge_target = "{TARGET}"
"#
        ))
        .unwrap();
        config.validate().unwrap();
        config
    }

    fn plugin() -> WorkflowPlugin {
        WorkflowPlugin {
            name: "integration".into(),
            description: None,
            init_script: None,
            state_machine: Some(workflow()),
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

    struct Fixture {
        repos: Repos,
        db: Database,
        _db_dir: tempfile::TempDir,
        task_id: String,
        dependent_id: String,
    }

    impl Fixture {
        fn new(target_edit: Option<&str>) -> Self {
            let repos = Repos::new(target_edit);
            let db_dir = tempfile::tempdir().unwrap();
            let mut db = Database::open_project_at_path(&db_dir.path().join("wf.db")).unwrap();
            let mut task = Task::new("Integrate thing", "codex", "proj");
            task.status = TaskStatus::Review;
            task.worktree_path = Some(repos.worktree.to_string_lossy().to_string());
            task.branch_name = Some(BRANCH.to_string());
            db.create_task(&task).unwrap();
            let state = WorkflowTaskState::new(&task.id, "integrate_to_feature", TARGET);
            let record =
                WorkflowTransitionRecord::new(&task.id, "seed", "backlog", "integrate_to_feature");
            db.record_workflow_admission(&task, &state, &record)
                .unwrap();
            let mut dependent = Task::new("Builds on it", "codex", "proj");
            dependent.referenced_tasks = Some(task.id.clone());
            db.create_task(&dependent).unwrap();
            Self {
                repos,
                db,
                _db_dir: db_dir,
                task_id: task.id,
                dependent_id: dependent.id,
            }
        }

        fn task(&self) -> Task {
            self.db.get_task(&self.task_id).unwrap().unwrap()
        }

        fn dependent_ready(&self) -> bool {
            let dependent = self.db.get_task(&self.dependent_id).unwrap().unwrap();
            self.db.deps_satisfied(&dependent)
        }

        fn state(&self) -> String {
            self.db
                .get_workflow_task_state(&self.task_id)
                .unwrap()
                .unwrap()
                .state
        }

        /// Pretend the last attempt was long enough ago for a retry.
        fn age_last_attempt(&mut self) {
            let mut task = self.task();
            task.updated_at =
                chrono::Utc::now() - chrono::Duration::seconds(INTEGRATION_RETRY_SECONDS + 5);
            self.db.update_task(&task).unwrap();
        }

        fn run(
            &mut self,
            config: &WorkflowProjectConfig,
            provider: Option<&Arc<dyn GitProviderOperations>>,
        ) -> WorkflowStepOutcome {
            let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(MockTmuxOperations::new());
            let agent_registry: Arc<dyn AgentRegistry> = Arc::new(MockAgentRegistry::new());
            let git_ops: Arc<dyn GitOperations> = Arc::new(RealGitOps);
            let merged = MergedConfig::merge(&GlobalConfig::default(), &ProjectConfig::default());
            let flags = crate::FeatureFlags::default();
            let runtime = WorkflowRuntime {
                tmux_ops: &tmux_ops,
                agent_registry: &agent_registry,
                git_ops: &git_ops,
                tmux_project_name: "proj",
                project_path: &self.repos.project,
                config: &merged,
                flags: &flags,
                session_probe: crate::agent::native_session::default_probe(),
                git_provider_ops: provider,
            };
            let task = self.task();
            complete_feature_integration(
                &workflow(),
                config,
                &plugin(),
                task,
                &mut self.db,
                &runtime,
            )
            .unwrap()
        }

        fn assess(&self, config: &WorkflowProjectConfig) -> AutomationDecision {
            let task = self.task();
            let state = self
                .db
                .get_workflow_task_state(&self.task_id)
                .unwrap()
                .unwrap();
            assess(&workflow(), config, &plugin(), &task, &state, &self.db)
        }
    }

    fn provider_opening_pull_request() -> Arc<dyn GitProviderOperations> {
        let mut provider = MockGitProviderOperations::new();
        provider
            .expect_create_pr()
            .withf(|_, _, body, head, base| {
                head == BRANCH && base.as_deref() == Some(TARGET) && body.contains("shared.txt")
            })
            .times(1)
            .returning(|_, _, _, _, _| Ok((7, "https://example.test/pull/7".to_string())));
        provider
            .expect_get_pr_state()
            .returning(|_, _| Ok(PullRequestState::Open));
        Arc::new(provider)
    }

    #[test]
    fn a_clean_merge_completes_the_task() {
        let mut fixture = Fixture::new(None);
        let outcome = fixture.run(&project_config(true), None);

        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));
        let task = fixture.task();
        assert_eq!(task.status, TaskStatus::Done);
        assert_eq!(task.integration_status, None);
        assert_eq!(fixture.state(), "done");
        assert_eq!(
            std::fs::read_to_string(fixture.repos.project.join("shared.txt")).unwrap(),
            "task\n"
        );
        assert!(fixture.dependent_ready());
    }

    #[test]
    fn a_conflict_pushes_the_branch_opens_a_pull_request_and_blocks_dependents() {
        let mut fixture = Fixture::new(Some("target\n"));
        let provider = provider_opening_pull_request();
        let target_before = fixture.repos.remote_ref(TARGET);

        let outcome = fixture.run(&project_config(true), Some(&provider));

        let WorkflowStepOutcome::Blocked { message } = outcome else {
            panic!("a conflicting merge must not advance the task");
        };
        assert!(message.contains("Has merge conflicts"), "{message}");
        let task = fixture.task();
        assert_eq!(task.integration_status.as_deref(), Some(INTEGRATION_CONFLICTS));
        assert_eq!(task.integration_conflicts.as_deref(), Some("shared.txt"));
        assert_eq!(task.pr_url.as_deref(), Some("https://example.test/pull/7"));
        assert!(task
            .escalation_note
            .as_deref()
            .is_some_and(|note| note.contains("shared.txt") && note.contains("/pull/7")));
        assert_eq!(task.status, TaskStatus::Review);
        assert_eq!(fixture.state(), "integrate_to_feature");
        assert!(
            fixture.repos.remote_ref(BRANCH).is_some(),
            "the task branch must leave the container even though it cannot merge"
        );
        assert_eq!(fixture.repos.remote_ref(TARGET), target_before);
        assert!(
            !fixture.dependent_ready(),
            "a dependency in Review with unresolved conflicts must keep dependents blocked"
        );
        assert_eq!(
            fixture.assess(&project_config(true)),
            AutomationDecision::Wait,
            "the retry waits out INTEGRATION_RETRY_SECONDS"
        );
        fixture.age_last_attempt();
        assert_eq!(
            fixture.assess(&project_config(true)),
            AutomationDecision::Advance("complete_feature_integration".into())
        );
    }

    #[test]
    fn a_resolution_pushed_to_the_task_branch_completes_the_task() {
        let mut fixture = Fixture::new(Some("target\n"));
        let provider = provider_opening_pull_request();
        fixture.run(&project_config(true), Some(&provider));

        let outside = fixture.repos.outside_clone();
        git(&outside, &["checkout", "-q", BRANCH]);
        let merge = std::process::Command::new("git")
            .args(["merge", &format!("origin/{TARGET}")])
            .current_dir(&outside)
            .output()
            .unwrap();
        assert!(!merge.status.success(), "the fixture must really conflict");
        std::fs::write(outside.join("shared.txt"), "task and target\n").unwrap();
        git(&outside, &["commit", "-q", "-am", "resolve conflict"]);
        git(&outside, &["push", "-q", "origin", BRANCH]);

        fixture.age_last_attempt();
        let outcome = fixture.run(&project_config(true), Some(&provider));

        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));
        let task = fixture.task();
        assert_eq!(task.status, TaskStatus::Done);
        assert_eq!(task.integration_status, None);
        assert_eq!(task.integration_conflicts, None);
        assert_eq!(task.escalation_note, None);
        assert_eq!(fixture.state(), "done");
        assert_eq!(
            std::fs::read_to_string(fixture.repos.project.join("shared.txt")).unwrap(),
            "task and target\n"
        );
        assert!(fixture.dependent_ready());
        let events = fixture.db.task_execution_events(&fixture.task_id).unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event_type == "integration_unresolved")
                .count(),
            1
        );
        assert!(events
            .iter()
            .any(|event| event.event_type == "integration_resolved"));
    }

    #[test]
    fn a_task_branch_merged_on_the_remote_completes_the_task() {
        let mut fixture = Fixture::new(Some("target\n"));
        let provider = provider_opening_pull_request();
        fixture.run(&project_config(true), Some(&provider));

        // Merging the pull request: the task branch lands in the remote target.
        let outside = fixture.repos.outside_clone();
        git(&outside, &["checkout", "-q", TARGET]);
        let merge = std::process::Command::new("git")
            .args(["merge", &format!("origin/{BRANCH}")])
            .current_dir(&outside)
            .output()
            .unwrap();
        assert!(!merge.status.success(), "the fixture must really conflict");
        std::fs::write(outside.join("shared.txt"), "merged on the remote\n").unwrap();
        git(&outside, &["commit", "-q", "-am", "merge pull request"]);
        git(&outside, &["push", "-q", "origin", TARGET]);

        fixture.age_last_attempt();
        let outcome = fixture.run(&project_config(true), Some(&provider));

        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));
        assert_eq!(fixture.task().status, TaskStatus::Done);
        assert_eq!(fixture.state(), "done");
        assert_eq!(
            std::fs::read_to_string(fixture.repos.project.join("shared.txt")).unwrap(),
            "merged on the remote\n",
            "the target checkout is fast-forwarded to the remote merge"
        );
    }

    #[test]
    fn a_dirty_target_checkout_blocks_and_the_retry_completes_once_it_is_clean() {
        let mut fixture = Fixture::new(None);
        std::fs::write(fixture.repos.project.join("scratch.txt"), "wip\n").unwrap();

        let outcome = fixture.run(&project_config(true), None);

        assert!(matches!(outcome, WorkflowStepOutcome::Blocked { .. }));
        let task = fixture.task();
        assert_eq!(task.integration_status.as_deref(), Some(INTEGRATION_BLOCKED));
        assert!(task
            .escalation_note
            .as_deref()
            .is_some_and(|note| note.contains("uncommitted changes")));
        assert!(fixture.repos.remote_ref(BRANCH).is_some());
        assert!(!fixture.dependent_ready());

        std::fs::remove_file(fixture.repos.project.join("scratch.txt")).unwrap();
        fixture.age_last_attempt();
        let outcome = fixture.run(&project_config(true), None);

        assert!(matches!(outcome, WorkflowStepOutcome::Advanced { .. }));
        assert_eq!(fixture.task().status, TaskStatus::Done);
        assert_eq!(fixture.task().escalation_note, None);
        assert!(fixture.dependent_ready());
    }

    #[test]
    fn conflicts_without_pull_request_authority_still_push_and_escalate() {
        let mut fixture = Fixture::new(Some("target\n"));
        let mut provider = MockGitProviderOperations::new();
        provider.expect_create_pr().never();
        let provider: Arc<dyn GitProviderOperations> = Arc::new(provider);

        let outcome = fixture.run(&project_config(false), Some(&provider));

        assert!(matches!(outcome, WorkflowStepOutcome::Blocked { .. }));
        let task = fixture.task();
        assert_eq!(task.integration_status.as_deref(), Some(INTEGRATION_CONFLICTS));
        assert_eq!(task.pr_url, None);
        assert!(fixture.repos.remote_ref(BRANCH).is_some());
    }
}
