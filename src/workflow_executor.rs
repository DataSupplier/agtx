//! Side-effect-free preparation for declarative workflow transitions.
//!
//! The graph module owns which transitions are legal. This module turns a
//! task's persisted evidence into the guard facts required by the graph, and
//! produces the durable records a caller must commit together. Keeping this
//! step independent from the TUI means web, MCP, and terminal callers cannot
//! disagree about what an admission means.

use anyhow::{bail, Result};

use crate::db::{Task, WorkflowTaskState, WorkflowTransitionRecord};
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
}
