//! Automation driver for declarative workflow tasks.
//!
//! This is the periodic-tick counterpart to the manual `Shift+<key>`
//! handlers in `tui::app`: it looks at every non-terminal task the project's
//! database knows about and, for each one, does exactly what a human
//! operator watching the board would have done -- nothing more.
//!
//! Two lanes, matched to the two things a human operator actually decides:
//!
//! 1. **Dependency admission.** A `backlog` task (or one with no persisted
//!    workflow state yet) whose dependencies have resolved is admitted via
//!    the existing [`admit_task`] extracted function, exactly as `Shift+A`
//!    does. Admission only ever reaches the plugin's post-admission holding
//!    state (whatever the workflow calls it); starting planning from there
//!    is a deliberate, permanently human-only action (`Shift+S` /
//!    [`start_workflow_planning`]) that this module never calls.
//! 2. **Everything from `planning` onward.** [`assess`] reads the durable
//!    evidence for a task and reports what, if anything, is ready to fire.
//!    An [`AutomationDecision::Advance`] is dispatched to whichever of the
//!    extracted `workflow_executor` functions implements that action;
//!    `Wait`, `InvalidArtifact`, and `HumanGate` all mean "do nothing this
//!    tick" -- every decision is re-derived fresh on the next call, so none
//!    of these are ever cached, sticky, or specially retried.
//!
//! What this module deliberately does *not* do: recover a task whose tmux
//! window disappeared. That is a separate, already-existing tick step
//! (`recover_task_session` in `tui::app`) that the host runs alongside this
//! one; duplicating it here would just be two paths racing to fix the same
//! thing.
//!
//! No TUI, ratatui, or terminal type is reachable from this module or from
//! anything it calls -- `run_automation_tick` depends only on `Database` and
//! the same trait objects (`AgentRegistry`, `TmuxOperations`,
//! `GitOperations`) the extracted `workflow_executor` functions already
//! depend on via [`WorkflowRuntime`]. That keeps it unit-testable with the
//! same mock harness `workflow_executor`'s own tests use, and safe to call
//! from a background tick with no terminal attached.

use crate::config::WorkflowPlugin;
use crate::db::{Database, Task, TaskStatus};
use crate::workflow::{WorkflowDefinition, WorkflowProjectConfig};
use crate::workflow_executor::{
    admit_task, assess, complete_admission, complete_feature_integration,
    start_workflow_implementation, submit_engineering_review, submit_final_validation,
    submit_plan_review, submit_workflow_implementation, submit_workflow_plan, AutomationDecision,
    WorkflowRuntime, WorkflowStepOutcome,
};

/// What automation did (or considered, and declined to do) for one task
/// during a single [`run_automation_tick`] call.
#[derive(Debug, Clone)]
pub struct TaskAutomationResult {
    pub task_id: String,
    /// The dependency-admission lane synthesizes `Advance("admit")` here
    /// (rather than calling `assess`, which is reserved for `planning`
    /// onward) so a caller does not need two different shapes of result to
    /// tell what happened.
    pub decision: AutomationDecision,
    /// `None` when nothing was attempted (a `Wait`/`InvalidArtifact`/
    /// `HumanGate` decision, or a task sitting unresolved in the dependency
    /// lane). `Some` whenever an extracted `workflow_executor` function was
    /// actually invoked -- including when it returned `Blocked` or failed:
    /// a stale/duplicate `Advance` racing an already-moved-on task surfaces
    /// here as an ordinary `Blocked` outcome, never a panic or a propagated
    /// error.
    pub outcome: Option<WorkflowStepOutcome>,
}

/// Run one automation pass over every task the project database knows
/// about.
///
/// Pure with respect to which tasks get looked at: this reads
/// `db.get_all_tasks()` fresh every call and applies the two lanes described
/// in the module docs. Every durable write happens inside the extracted
/// `workflow_executor` functions this dispatches to; this function performs
/// none itself beyond what those calls do.
pub fn run_automation_tick(
    db: &mut Database,
    workflow: &WorkflowDefinition,
    project: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    runtime: &WorkflowRuntime,
) -> Vec<TaskAutomationResult> {
    let tasks = match db.get_all_tasks() {
        Ok(tasks) => tasks,
        Err(_) => return Vec::new(),
    };

    let mut results = Vec::with_capacity(tasks.len());
    for task in tasks {
        if task.status == TaskStatus::Done {
            continue;
        }

        let state = match db.get_workflow_task_state(&task.id) {
            Ok(state) => state,
            Err(_) => continue,
        };

        let at_initial_state = state
            .as_ref()
            .map(|state| state.state == workflow.initial_state)
            .unwrap_or(true);

        if at_initial_state {
            // Lane 1: dependency-driven admission only. No artifact is read,
            // no agent is launched, and `task.status` is never touched here
            // -- `admit_task` itself only records the worktree/branch and
            // the admission evidence, exactly as `Shift+A` does manually.
            if !db.deps_satisfied(&task) {
                results.push(TaskAutomationResult {
                    task_id: task.id.clone(),
                    decision: AutomationDecision::Wait,
                    outcome: None,
                });
                continue;
            }
            let task_id = task.id.clone();
            let outcome = match admit_task(workflow, project, plugin, task, db, runtime) {
                Ok(outcome) => outcome,
                Err(error) => WorkflowStepOutcome::Blocked {
                    message: error.to_string(),
                },
            };
            results.push(TaskAutomationResult {
                task_id,
                decision: AutomationDecision::Advance("admit".to_string()),
                outcome: Some(outcome),
            });
            continue;
        }

        // Reaching here means a workflow_task_states row exists and it is
        // not the initial state -- `at_initial_state`'s `unwrap_or(true)`
        // guarantees `state` is `Some` on every other path.
        let Some(state) = state else { continue };

        let is_terminal = workflow
            .state(&state.state)
            .map(|workflow_state| workflow_state.terminal)
            .unwrap_or(true);
        if is_terminal {
            continue;
        }

        // Lane 2: everything from `planning` onward is decided by `assess`,
        // which already encodes every automation-safety rule (artifact
        // freshness via `state_attempt`, the fixed `HumanGate` for a failed
        // final validation) that this driver must never second-guess or
        // bypass.
        let decision = assess(workflow, project, plugin, &task, &state, db);
        if let AutomationDecision::Advance(action) = &decision {
            let action = action.clone();
            let task_id = task.id.clone();
            let result = dispatch_advance(&action, workflow, project, plugin, task, db, runtime);
            let outcome = match result {
                Ok(outcome) => outcome,
                Err(error) => {
                    tracing::warn!(
                        task_id = %task_id,
                        action = %action,
                        error = %error,
                        "workflow automation advance failed"
                    );
                    WorkflowStepOutcome::Blocked {
                        message: error.to_string(),
                    }
                }
            };
            results.push(TaskAutomationResult {
                task_id,
                decision,
                outcome: Some(outcome),
            });
            continue;
        }
        results.push(TaskAutomationResult {
            task_id: task.id.clone(),
            decision,
            outcome: None,
        });
    }
    results
}

/// Map one `assess`-reported action to the extracted `workflow_executor`
/// function that implements it. These are exactly the action strings the
/// plugin's own transition table uses; three of `submit_engineering_review`'s
/// possible verdicts share one dispatch entry (and likewise for
/// `submit_final_validation`) because the destination function itself reads
/// the artifact and decides which of its own outgoing actions applies --
/// `assess` and the launch function must never disagree about that mapping,
/// so it is only ever written once, inside the launch function.
///
/// `start_workflow_planning` and its `start_planning` action are
/// deliberately absent: `assess` never emits `start_planning` for a
/// no-guard, no-role holding state (see `assess`'s own doc comment), so
/// there is nothing to route here, and this driver must never call it
/// itself regardless.
///
/// `admission_complete` (`admission` -> `ready_for_planning`) is dispatched
/// to the standalone, launch-free [`complete_admission`] rather than to
/// [`start_workflow_planning`] (which fuses the same transition with
/// launching the planner, for the manual `Shift+S` path) — this is what lets
/// automation carry a dependency-clean task all the way to
/// `ready_for_planning` without ever starting an agent on its own.
fn dispatch_advance(
    action: &str,
    workflow: &WorkflowDefinition,
    project: &WorkflowProjectConfig,
    plugin: &WorkflowPlugin,
    task: Task,
    db: &mut Database,
    runtime: &WorkflowRuntime,
) -> anyhow::Result<WorkflowStepOutcome> {
    match action {
        "admission_complete" => complete_admission(workflow, project, task, db),
        "submit_plan" => submit_workflow_plan(workflow, project, plugin, task, db, runtime),
        "approve_plan" | "plan_changes_requested" => {
            submit_plan_review(workflow, project, plugin, task, db, runtime)
        }
        "start_implementation" => {
            start_workflow_implementation(workflow, project, plugin, task, db, runtime)
        }
        "implementation_complete" => {
            submit_workflow_implementation(workflow, project, plugin, task, db, runtime)
        }
        "engineering_corrections_required" | "engineering_plan_issue" | "start_final_validation" => {
            submit_engineering_review(workflow, project, plugin, task, db, runtime)
        }
        "validation_failed" | "begin_feature_integration" => {
            submit_final_validation(workflow, project, plugin, task, db, runtime)
        }
        "complete_feature_integration" => {
            complete_feature_integration(workflow, project, plugin, task, db, runtime)
        }
        _ => Ok(WorkflowStepOutcome::NoOp),
    }
}

#[cfg(test)]
#[cfg(feature = "test-mocks")]
mod tests {
    use super::*;
    use crate::agent::{AgentOperations, AgentRegistry, MockAgentOperations, MockAgentRegistry};
    use crate::config::{GlobalConfig, MergedConfig, ProjectConfig};
    use crate::db::{TaskStatus, WorkflowTaskState};
    use crate::git::{GitOperations, MockGitOperations};
    use crate::tmux::{MockTmuxOperations, TmuxOperations};
    use crate::workflow::{WorkflowGuard, WorkflowRolePolicy, WorkflowState, WorkflowTransition};
    use std::path::Path;
    use std::sync::Arc;

    /// A graph shaped like the real project's states, with names distinct
    /// from any single legacy phase word so a bug that accidentally reused
    /// `TaskStatus`'s own labels as workflow-state ids would be visible
    /// immediately: `backlog` -> `admission` -> `ready_for_planning` ->
    /// `planning` -> `plan_review` -> `implementing` -> `engineering_review`
    /// -> `final_validation` -> `integrate_to_feature` -> `done`.
    fn full_workflow() -> WorkflowDefinition {
        WorkflowDefinition {
            initial_state: "backlog".into(),
            states: vec![
                WorkflowState { id: "backlog".into(), label: "Backlog".into(), role: None, terminal: false },
                WorkflowState { id: "admission".into(), label: "Admission".into(), role: None, terminal: false },
                WorkflowState { id: "ready_for_planning".into(), label: "Ready for planning".into(), role: None, terminal: false },
                WorkflowState { id: "planning".into(), label: "Planning".into(), role: Some("planner".into()), terminal: false },
                WorkflowState { id: "plan_review".into(), label: "Plan review".into(), role: Some("plan_reviewer".into()), terminal: false },
                WorkflowState { id: "implementing".into(), label: "Implementing".into(), role: Some("implementer".into()), terminal: false },
                WorkflowState { id: "engineering_review".into(), label: "Engineering review".into(), role: Some("reviewer".into()), terminal: false },
                WorkflowState { id: "final_validation".into(), label: "Final validation".into(), role: Some("validator".into()), terminal: false },
                WorkflowState { id: "integrate_to_feature".into(), label: "Integrate".into(), role: Some("reviewer".into()), terminal: false },
                WorkflowState { id: "done".into(), label: "Done".into(), role: None, terminal: true },
            ],
            transitions: vec![
                // `admit_task` lands a freshly-admitted task in `admission`
                // (real worktree/branch, no artifact, no agent). One more
                // guard-gated hop (`admission_complete`, mirroring the real
                // `admission_recorded` guard) carries it on to
                // `ready_for_planning`, automatically, since it too has no
                // artifact and no launch. `start_planning` past that point is
                // deliberately guardless: it is the human-only
                // `start_workflow_planning` handoff, carries no automation
                // signal, and this driver never fires it itself.
                WorkflowTransition { action: "admit".into(), from: "backlog".into(), to: "admission".into(), guards: vec![WorkflowGuard::DependenciesResolved] },
                WorkflowTransition { action: "admission_complete".into(), from: "admission".into(), to: "ready_for_planning".into(), guards: vec![WorkflowGuard::AdmissionRecorded] },
                WorkflowTransition { action: "start_planning".into(), from: "ready_for_planning".into(), to: "planning".into(), guards: vec![] },
                WorkflowTransition { action: "submit_plan".into(), from: "planning".into(), to: "plan_review".into(), guards: vec![] },
                WorkflowTransition { action: "approve_plan".into(), from: "plan_review".into(), to: "implementing".into(), guards: vec![WorkflowGuard::ApprovedPlan] },
                WorkflowTransition { action: "plan_changes_requested".into(), from: "plan_review".into(), to: "planning".into(), guards: vec![] },
                WorkflowTransition { action: "implementation_complete".into(), from: "implementing".into(), to: "engineering_review".into(), guards: vec![WorkflowGuard::ImplementationRecorded] },
                WorkflowTransition { action: "engineering_corrections_required".into(), from: "engineering_review".into(), to: "implementing".into(), guards: vec![] },
                WorkflowTransition { action: "engineering_plan_issue".into(), from: "engineering_review".into(), to: "planning".into(), guards: vec![] },
                WorkflowTransition { action: "start_final_validation".into(), from: "engineering_review".into(), to: "final_validation".into(), guards: vec![] },
                WorkflowTransition { action: "begin_feature_integration".into(), from: "final_validation".into(), to: "integrate_to_feature".into(), guards: vec![WorkflowGuard::FinalValidationPassed] },
                WorkflowTransition { action: "validation_failed".into(), from: "final_validation".into(), to: "engineering_review".into(), guards: vec![] },
                WorkflowTransition { action: "complete_feature_integration".into(), from: "integrate_to_feature".into(), to: "done".into(), guards: vec![WorkflowGuard::IntegratedIntoTarget] },
            ],
        }
    }

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

    fn project() -> WorkflowProjectConfig {
        let mut project = WorkflowProjectConfig {
            target_branch: "feature/poc".into(),
            role_bindings: Default::default(),
            ..Default::default()
        };
        project.role_bindings.insert("planner".into(), "claude".into());
        project.role_bindings.insert("implementer".into(), "claude".into());
        project.role_bindings.insert("reviewer".into(), "claude".into());
        project.role_bindings.insert("validator".into(), "claude".into());
        project.role_bindings.insert("plan_reviewer".into(), "claude".into());
        for role in ["planner", "implementer", "reviewer", "validator", "plan_reviewer"] {
            project
                .role_policies
                .roles
                .insert(role.into(), WorkflowRolePolicy::default());
        }
        project
    }

    fn merged_config() -> MergedConfig {
        MergedConfig::merge(&GlobalConfig::default(), &ProjectConfig::default())
    }

    fn feature_flags() -> crate::FeatureFlags {
        crate::FeatureFlags::default()
    }

    /// A harness that always reports tmux windows as present and answers
    /// every session operation successfully, so launch functions in the
    /// dispatch path never fail on tmux plumbing this test does not care
    /// about.
    fn permissive_tmux() -> MockTmuxOperations {
        let mut mock = MockTmuxOperations::new();
        mock.expect_has_session().returning(|_| true);
        mock.expect_window_exists().returning(|_| Ok(true));
        mock.expect_create_window().returning(|_, _, _, _, _, _| Ok(()));
        mock.expect_send_keys().returning(|_, _| Ok(()));
        mock.expect_send_key().returning(|_, _| Ok(()));
        // The first poll finds the outgoing agent already at a shell (exit
        // confirmed immediately); every poll after that reports the freshly
        // launched agent, matching what a real tmux pane shows once
        // `switch_agent_in_tmux` types the new command. Its final
        // launch-detection loop requires seeing a *recognized* agent process
        // (see `AGENT_COMMANDS` in `src/tui/app.rs`), not merely any string --
        // a mock that always reports "bash" makes that loop time out and
        // `switch_agent_in_tmux` return `Err`, which every caller now
        // propagates instead of silently discarding.
        let pane_polls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        mock.expect_pane_current_command().returning(move |_| {
            if pane_polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                Some("bash".to_string())
            } else {
                Some("claude".to_string())
            }
        });
        mock.expect_capture_pane().returning(|_| Ok(String::new()));
        mock.expect_paste_text().returning(|_, _| Ok(()));
        mock
    }

    fn permissive_registry() -> MockAgentRegistry {
        let mut mock = MockAgentRegistry::new();
        mock.expect_get()
            .returning(|_| Arc::new(MockAgentOperations::new()) as Arc<dyn AgentOperations>);
        mock
    }

    fn runtime_with<'a>(
        tmux_ops: &'a Arc<dyn TmuxOperations>,
        agent_registry: &'a Arc<dyn AgentRegistry>,
        git_ops: &'a Arc<dyn GitOperations>,
        project_path: &'a Path,
        config: &'a MergedConfig,
        flags: &'a crate::FeatureFlags,
    ) -> WorkflowRuntime<'a> {
        WorkflowRuntime {
            tmux_ops,
            agent_registry,
            git_ops,
            tmux_project_name: "proj",
            project_path,
            config,
            flags,
        }
    }

    fn backlog_task() -> Task {
        let mut task = Task::new("Seed cameras", "claude", "proj");
        task.description = Some("do the work".into());
        task
    }

    /// `admit_task` resolves its admission base commit with the real
    /// `crate::git::resolve_commit` (a free function, not part of the mocked
    /// `GitOperations` trait), so exercising it needs an actual git
    /// repository on disk with the configured target branch checked out.
    fn init_git_repo_on_branch(path: &Path, branch: &str) {
        let run = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(path)
                .status()
                .expect("git must be on PATH for this test");
            assert!(status.success(), "git {args:?} failed in {}", path.display());
        };
        run(&["init", "-q"]);
        run(&["config", "user.email", "test@example.com"]);
        run(&["config", "user.name", "Test"]);
        run(&["checkout", "-q", "-b", branch]);
        std::fs::write(path.join("README.md"), "seed").unwrap();
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "seed"]);
    }

    /// Lane 1: a `backlog` task with satisfied dependencies is admitted by
    /// one tick, with no agent launched and no `TaskStatus` change --
    /// exactly what `Shift+A` alone would have done.
    #[test]
    fn tick_admits_a_backlog_task_once_dependencies_resolve() {
        let graph = full_workflow();
        let plugin = plugin(graph.clone());
        let project = project();

        let repo = tempfile::tempdir().unwrap();
        init_git_repo_on_branch(repo.path(), "feature/poc");

        let mut db = Database::open_in_memory_project().unwrap();
        let task = backlog_task();
        db.create_task(&task).unwrap();

        let mut mock_git = MockGitOperations::new();
        mock_git.expect_create_worktree().returning(|_, _, _, _, _| Ok("C:/work/wt".to_string()));
        mock_git.expect_initialize_worktree().returning(|_, _, _, _, _| Vec::new());

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(permissive_tmux());
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(permissive_registry());
        let git_ops: Arc<dyn GitOperations> = Arc::new(mock_git);
        let config = merged_config();
        let flags = feature_flags();
        let runtime = runtime_with(&tmux_ops, &agent_registry, &git_ops, repo.path(), &config, &flags);

        let results = run_automation_tick(&mut db, &graph, &project, &plugin, &runtime);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].task_id, task.id);
        assert_eq!(results[0].decision, AutomationDecision::Advance("admit".to_string()));
        let Some(WorkflowStepOutcome::Advanced { task: advanced, .. }) = &results[0].outcome else {
            panic!("expected admission to advance, got {:?}", results[0].outcome);
        };
        // `admit_task` never touches `TaskStatus`; only the plugin workflow
        // state (recorded in `workflow_task_states`, checked below) moves --
        // and only as far as `admission`. Reaching `ready_for_planning` is a
        // second, separately-guarded hop (`admission_complete`), fired by a
        // later tick once `admission_recorded` is true, not by this same one.
        assert_eq!(advanced.status, TaskStatus::Backlog);
        let state = db.get_workflow_task_state(&task.id).unwrap().unwrap();
        assert_eq!(state.state, "admission");

        // One more tick carries it the rest of the way to `ready_for_planning`.
        let next = run_automation_tick(&mut db, &graph, &project, &plugin, &runtime);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].decision, AutomationDecision::Advance("admission_complete".to_string()));
        assert!(matches!(next[0].outcome, Some(WorkflowStepOutcome::Advanced { .. })));
        let state = db.get_workflow_task_state(&task.id).unwrap().unwrap();
        assert_eq!(state.state, "ready_for_planning");

        // The driver never calls `start_workflow_planning` on its own --
        // confirmed generally by `ready_lane_never_advances_past_itself`
        // below.
    }

    /// Ready-lane/plugin-state boundary: once a task reaches the workflow's
    /// no-role, no-guard holding state, automation leaves it there forever
    /// -- `start_planning` is exclusively human, and `TaskStatus` never
    /// moves off `Backlog` without that human call.
    #[test]
    fn ready_lane_never_advances_past_itself() {
        let graph = full_workflow();
        let plugin_config = plugin(graph.clone());
        let project = project();

        let mut db = Database::open_in_memory_project().unwrap();
        let mut task = Task::new("Seed cameras", "claude", "proj");
        task.worktree_path = Some("C:/work/wt".into());
        db.create_task(&task).unwrap();
        let mut state = crate::db::WorkflowTaskState::new(&task.id, "ready_for_planning", "feature/poc");
        state.base_sha = Some("a1b2c3".into());
        let record = crate::db::WorkflowTransitionRecord::new(&task.id, "admit", "backlog", "ready_for_planning");
        db.record_workflow_admission(&task, &state, &record).unwrap();
        let attempt_before = state.state_attempt;

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(permissive_tmux());
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(permissive_registry());
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let project_dir = tempfile::tempdir().unwrap();
        let runtime = runtime_with(&tmux_ops, &agent_registry, &git_ops, project_dir.path(), &config, &flags);

        // `ready_for_planning`'s only outgoing edge (`start_planning`) is
        // guardless -- it is the human-only handoff and carries no
        // automation signal, so `assess` reports `Wait` for 100 straight
        // ticks; neither `TaskStatus` nor `state_attempt` ever move without
        // the human call.
        for _ in 0..100 {
            let results = run_automation_tick(&mut db, &graph, &project, &plugin_config, &runtime);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].decision, AutomationDecision::Wait);
            assert!(results[0].outcome.is_none());
        }
        let task_after = db.get_task(&task.id).unwrap().unwrap();
        assert_eq!(task_after.status, TaskStatus::Backlog);
        let state_after = db.get_workflow_task_state(&task.id).unwrap().unwrap();
        assert_eq!(state_after.state, "ready_for_planning");
        assert_eq!(state_after.state_attempt, attempt_before);

        // Only the manual planning-start path moves `TaskStatus`.
        let outcome = crate::workflow_executor::start_workflow_planning(
            &graph, &project, &plugin_config, task_after, &mut db, &runtime,
        )
        .unwrap();
        let WorkflowStepOutcome::Advanced { task: started, .. } = outcome else {
            panic!("expected planning to start");
        };
        assert_eq!(started.status, TaskStatus::Planning);
    }

    /// Artifact missing/invalid, with recovery: a mid-pipeline task with no
    /// artifact waits every tick; once a malformed artifact appears it
    /// reports `InvalidArtifact` every tick with no transition and no
    /// panic; fixing the artifact makes the very next tick advance --
    /// nothing here is sticky.
    #[test]
    fn artifact_missing_then_invalid_then_recovers() {
        let graph = full_workflow();
        let plugin_config = plugin(graph.clone());
        let project = project();

        let worktree = tempfile::tempdir().unwrap();
        let mut db = Database::open_in_memory_project().unwrap();
        let mut task = Task::new("Review thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-review".into());
        db.create_task(&task).unwrap();
        let state = crate::db::WorkflowTaskState::new(&task.id, "engineering_review", "feature/poc");
        let record = crate::db::WorkflowTransitionRecord::new(&task.id, "seed", "implementing", "engineering_review");
        db.record_workflow_admission(&task, &state, &record).unwrap();

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(permissive_tmux());
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(permissive_registry());
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = runtime_with(&tmux_ops, &agent_registry, &git_ops, worktree.path(), &config, &flags);

        // No artifact yet: `Wait`, many times.
        for _ in 0..10 {
            let results = run_automation_tick(&mut db, &graph, &project, &plugin_config, &runtime);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].decision, AutomationDecision::Wait);
            assert!(results[0].outcome.is_none());
        }

        // A fresh but malformed artifact: `InvalidArtifact`, many times.
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        let artifact = worktree.path().join(".agent-flow/engineering-review.yaml");
        std::fs::write(&artifact, "workflow_attempt: 1\n").unwrap();
        for _ in 0..10 {
            let results = run_automation_tick(&mut db, &graph, &project, &plugin_config, &runtime);
            assert_eq!(results.len(), 1);
            assert!(
                matches!(results[0].decision, AutomationDecision::InvalidArtifact(_)),
                "expected InvalidArtifact, got {:?}",
                results[0].decision
            );
            assert!(results[0].outcome.is_none());
        }
        let unchanged = db.get_workflow_task_state(&task.id).unwrap().unwrap();
        assert_eq!(unchanged.state, "engineering_review");

        // Fix it: the very next tick advances, with no special "recovery"
        // logic involved -- `assess` simply re-reads the file fresh.
        std::fs::write(&artifact, "verdict: approved_for_validation\nworkflow_attempt: 1\n").unwrap();
        let results = run_automation_tick(&mut db, &graph, &project, &plugin_config, &runtime);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].decision,
            AutomationDecision::Advance("start_final_validation".to_string())
        );
        assert!(matches!(results[0].outcome, Some(WorkflowStepOutcome::Advanced { .. })));
    }

    /// Human-gate boundary: a `final_validation` task whose artifact says
    /// `verdict: failed` is never auto-advanced by any number of ticks. Only
    /// a human calling `submit_final_validation` directly may move it.
    #[test]
    fn final_validation_failed_is_never_auto_advanced() {
        let graph = full_workflow();
        let plugin_config = plugin(graph.clone());
        let project = project();

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/final-validation.yaml"),
            "verdict: failed\nworkflow_attempt: 1\n",
        )
        .unwrap();

        let mut db = Database::open_in_memory_project().unwrap();
        let mut task = Task::new("Validate thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        db.create_task(&task).unwrap();
        let state = crate::db::WorkflowTaskState::new(&task.id, "final_validation", "feature/poc");
        let record = crate::db::WorkflowTransitionRecord::new(&task.id, "seed", "engineering_review", "final_validation");
        db.record_workflow_admission(&task, &state, &record).unwrap();

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(permissive_tmux());
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(permissive_registry());
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = runtime_with(&tmux_ops, &agent_registry, &git_ops, worktree.path(), &config, &flags);

        for _ in 0..50 {
            let results = run_automation_tick(&mut db, &graph, &project, &plugin_config, &runtime);
            assert_eq!(results.len(), 1);
            assert_eq!(
                results[0].decision,
                AutomationDecision::HumanGate("final validation failed".to_string())
            );
            assert!(results[0].outcome.is_none());
        }
        let unchanged = db.get_workflow_task_state(&task.id).unwrap().unwrap();
        assert_eq!(unchanged.state, "final_validation");
    }

    /// Idempotency: once a tick fires a transition, the artifact that
    /// triggered it is now stale for the new `state_attempt` -- a second
    /// consecutive tick must not re-fire it (no duplicate history row, no
    /// duplicate agent relaunch).
    #[test]
    fn a_second_tick_does_not_refire_a_stale_artifact() {
        let graph = full_workflow();
        let plugin_config = plugin(graph.clone());
        let project = project();

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/engineering-review.yaml"),
            "verdict: approved_for_validation\nworkflow_attempt: 1\n",
        )
        .unwrap();

        let mut db = Database::open_in_memory_project().unwrap();
        let mut task = Task::new("Review thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-review".into());
        db.create_task(&task).unwrap();
        let state = crate::db::WorkflowTaskState::new(&task.id, "engineering_review", "feature/poc");
        let record = crate::db::WorkflowTransitionRecord::new(&task.id, "seed", "implementing", "engineering_review");
        db.record_workflow_admission(&task, &state, &record).unwrap();

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(permissive_tmux());
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(permissive_registry());
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = runtime_with(&tmux_ops, &agent_registry, &git_ops, worktree.path(), &config, &flags);

        let first = run_automation_tick(&mut db, &graph, &project, &plugin_config, &runtime);
        assert_eq!(first.len(), 1);
        assert!(matches!(first[0].outcome, Some(WorkflowStepOutcome::Advanced { .. })));
        let history_after_first = db.get_workflow_task_state(&task.id).unwrap().unwrap().state_attempt;

        // The artifact on disk is untouched -- still stamped `workflow_attempt: 1`,
        // which is now stale for the state the task just advanced into.
        let second = run_automation_tick(&mut db, &graph, &project, &plugin_config, &runtime);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].decision, AutomationDecision::Wait);
        assert!(second[0].outcome.is_none());
        let history_after_second = db.get_workflow_task_state(&task.id).unwrap().unwrap().state_attempt;
        assert_eq!(history_after_first, history_after_second);
    }

    /// Structural invariant: nothing in this module ever names the merge
    /// destination the human-only integration path is forbidden from
    /// targeting, nor sets the policy flag that would allow it. Mirrors the
    /// style of `workflow.rs`'s own hard-forbidden-flag enforcement, applied
    /// here as a source-text check over this file rather than a runtime
    /// assertion, since this module never constructs a `WorkflowStatePolicy`
    /// at all -- integration policy is entirely `complete_feature_integration`'s
    /// concern, already enforced there. The forbidden substrings are
    /// assembled at runtime rather than written literally, so this very
    /// assertion cannot trip on its own description of what it checks.
    #[test]
    fn never_names_the_forbidden_merge_destination() {
        let source = std::fs::read_to_string(file!()).unwrap();
        let trunk_branch: String = ["m", "a", "i", "n"].concat();
        let forbidden_flag: String = ["merge_", "feature_to_", &trunk_branch].concat();
        let forbidden_field: String = ["merge_", "target"].concat();
        let forbidden_branch: String = ["\"", &trunk_branch, "\""].concat();
        assert!(!source.contains(&forbidden_flag));
        assert!(!source.contains(&forbidden_field));
        assert!(!source.contains(&forbidden_branch));
    }

    /// An agent registry whose `PromptInjection::Argv` lets `decide_workflow_plan`'s
    /// reject branch (always `spawn_send_to_agent`, regardless of policy)
    /// deliver its revise prompt via the synchronous launch-argv path rather
    /// than the mid-session lane, avoiding readiness-wait machinery in a tick
    /// test.
    fn registry_with_argv_launch() -> MockAgentRegistry {
        let mut mock = MockAgentRegistry::new();
        mock.expect_get().returning(|_| {
            let mut ops = MockAgentOperations::new();
            ops.expect_prompt_injection().returning(|| crate::agent::PromptInjection::Argv);
            ops.expect_build_interactive_command().returning(|prompt| format!("claude '{}'", prompt));
            Arc::new(ops) as Arc<dyn AgentOperations>
        });
        mock
    }

    /// End-to-end: a `plan_review` task with a fresh `verdict: approved`
    /// artifact is advanced all the way to `implementing` (this graph's real
    /// `approve_plan` destination) by a single automation tick -- no
    /// Shift+Y keypress at all, matching `submit_plan_review`'s wiring into
    /// `dispatch_advance`.
    #[test]
    fn tick_auto_approves_plan_review_and_advances_past_it() {
        let graph = full_workflow();
        let plugin_config = plugin(graph.clone());
        let project = project();

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: approved\nworkflow_attempt: 1\n",
        )
        .unwrap();

        let mut db = Database::open_in_memory_project().unwrap();
        let mut task = Task::new("Plan thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-plan".into());
        db.create_task(&task).unwrap();
        let mut state = WorkflowTaskState::new(&task.id, "plan_review", "feature/poc");
        state.plan_revision = 1;
        state.plan_hash = Some("deadbeef".into());
        let record = crate::db::WorkflowTransitionRecord::new(&task.id, "seed", "planning", "plan_review");
        db.record_workflow_admission(&task, &state, &record).unwrap();

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(permissive_tmux());
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(permissive_registry());
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = runtime_with(&tmux_ops, &agent_registry, &git_ops, worktree.path(), &config, &flags);

        let results = run_automation_tick(&mut db, &graph, &project, &plugin_config, &runtime);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].decision, AutomationDecision::Advance("approve_plan".to_string()));
        assert!(matches!(results[0].outcome, Some(WorkflowStepOutcome::Advanced { .. })));

        let state_after = db.get_workflow_task_state(&task.id).unwrap().unwrap();
        assert_eq!(state_after.state, "implementing");
        assert_eq!(state_after.approved_plan_revision, Some(1));
        assert_eq!(state_after.approved_plan_hash.as_deref(), Some("deadbeef"));
    }

    /// End-to-end rework cycle: a `plan_review` task with a fresh
    /// `verdict: changes_requested` artifact is driven straight back to
    /// `planning` by a single automation tick, with the same `state_attempt`
    /// bump and agent relaunch the other rework-loop states already get
    /// from `assess`/`dispatch_advance` -- no Shift+N keypress at all.
    #[test]
    fn tick_auto_requests_plan_changes_and_returns_to_planning() {
        let graph = full_workflow();
        let plugin_config = plugin(graph.clone());
        let project = project();

        let worktree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worktree.path().join(".agent-flow")).unwrap();
        std::fs::write(
            worktree.path().join(".agent-flow/plan-review.yaml"),
            "verdict: changes_requested\nfindings: Add a rollback step for the migration.\nworkflow_attempt: 1\n",
        )
        .unwrap();

        let mut db = Database::open_in_memory_project().unwrap();
        let mut task = Task::new("Plan thing", "claude", "proj");
        task.worktree_path = Some(worktree.path().to_string_lossy().to_string());
        task.session_name = Some("proj:task-plan".into());
        db.create_task(&task).unwrap();
        let mut state = WorkflowTaskState::new(&task.id, "plan_review", "feature/poc");
        state.plan_revision = 1;
        let record = crate::db::WorkflowTransitionRecord::new(&task.id, "seed", "planning", "plan_review");
        db.record_workflow_admission(&task, &state, &record).unwrap();
        let attempt_before = db.get_workflow_task_state(&task.id).unwrap().unwrap().state_attempt;

        let tmux_ops: Arc<dyn TmuxOperations> = Arc::new(permissive_tmux());
        let agent_registry: Arc<dyn AgentRegistry> = Arc::new(registry_with_argv_launch());
        let git_ops: Arc<dyn GitOperations> = Arc::new(MockGitOperations::new());
        let config = merged_config();
        let flags = feature_flags();
        let runtime = runtime_with(&tmux_ops, &agent_registry, &git_ops, worktree.path(), &config, &flags);

        let results = run_automation_tick(&mut db, &graph, &project, &plugin_config, &runtime);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].decision, AutomationDecision::Advance("plan_changes_requested".to_string()));
        assert!(matches!(results[0].outcome, Some(WorkflowStepOutcome::Advanced { .. })));

        let state_after = db.get_workflow_task_state(&task.id).unwrap().unwrap();
        assert_eq!(state_after.state, "planning");
        assert!(
            state_after.state_attempt > attempt_before,
            "re-entering planning must bump state_attempt (was {attempt_before}, now {})",
            state_after.state_attempt
        );
        let task_after = db.get_task(&task.id).unwrap().unwrap();
        assert_eq!(task_after.agent, "claude", "the bound planner agent must be the one relaunched");
    }
}
