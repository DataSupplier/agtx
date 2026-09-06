//! Declarative workflow state graphs.
//!
//! The original agtx board has a fixed five-column lifecycle.  Projects that
//! need stronger delivery controls can declare a graph of named states and
//! transitions instead.  This module deliberately contains no TUI, database,
//! git, or agent code: every surface asks the same graph what a task may do.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// A workflow supplied by a plugin.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowDefinition {
    /// Stable identifier of the first state for a newly-created task.
    pub initial_state: String,
    /// States are a list rather than a map so the declared order can drive a
    /// board without a second ordering mechanism.
    pub states: Vec<WorkflowState>,
    #[serde(default)]
    pub transitions: Vec<WorkflowTransition>,
}

/// One durable state in a workflow.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowState {
    pub id: String,
    pub label: String,
    /// Named workflow role that owns this state.  Agent products are bound by
    /// the project, never by a plugin.
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub terminal: bool,
}

/// A named, guarded edge between workflow states.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowTransition {
    pub action: String,
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub guards: Vec<WorkflowGuard>,
}

/// Evidence that must exist before a transition is permitted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowGuard {
    DependenciesResolved,
    AdmissionRecorded,
    ApprovedPlan,
    ImplementationRecorded,
    FinalValidationPassed,
    CleanWorktree,
    IntegratedIntoTarget,
}

/// Facts collected by the transition executor before it changes durable state.
/// Git-dependent checks are supplied by the executor rather than reimplemented
/// in this pure graph module.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GuardContext {
    pub dependencies_resolved: bool,
    pub admission_recorded: bool,
    pub approved_plan: bool,
    pub implementation_recorded: bool,
    pub final_validation_passed: bool,
    pub clean_worktree: bool,
    pub integrated_into_target: bool,
}

impl WorkflowDefinition {
    /// Validate the graph before a project can use it.
    pub fn validate(&self) -> Result<()> {
        if self.states.is_empty() {
            bail!("workflow must declare at least one state");
        }

        let mut ids = BTreeSet::new();
        for state in &self.states {
            validate_identifier("state", &state.id)?;
            if state.label.trim().is_empty() {
                bail!("workflow state '{}' needs a label", state.id);
            }
            if !ids.insert(&state.id) {
                bail!("workflow declares state '{}' more than once", state.id);
            }
            if let Some(role) = &state.role {
                validate_identifier("role", role)?;
            }
        }

        if !ids.contains(&self.initial_state) {
            bail!(
                "workflow initial_state '{}' is not a declared state",
                self.initial_state
            );
        }

        let mut actions = BTreeSet::new();
        for transition in &self.transitions {
            validate_identifier("transition action", &transition.action)?;
            if !actions.insert(&transition.action) {
                bail!(
                    "workflow declares transition action '{}' more than once",
                    transition.action
                );
            }
            if !ids.contains(&transition.from) {
                bail!(
                    "transition '{}' has unknown source state '{}'",
                    transition.action,
                    transition.from
                );
            }
            if !ids.contains(&transition.to) {
                bail!(
                    "transition '{}' has unknown target state '{}'",
                    transition.action,
                    transition.to
                );
            }
        }

        if !self.states.iter().any(|state| state.terminal) {
            bail!("workflow must declare at least one terminal state");
        }
        Ok(())
    }

    pub fn state(&self, id: &str) -> Option<&WorkflowState> {
        self.states.iter().find(|state| state.id == id)
    }

    pub fn transition(&self, action: &str) -> Option<&WorkflowTransition> {
        self.transitions
            .iter()
            .find(|transition| transition.action == action)
    }

    pub fn allowed_transitions(&self, state: &str) -> Vec<&WorkflowTransition> {
        self.transitions
            .iter()
            .filter(|transition| transition.from == state)
            .collect()
    }

    /// Resolve and verify one edge from the task's current state.
    pub fn validate_transition(
        &self,
        current_state: &str,
        action: &str,
        context: GuardContext,
    ) -> Result<&WorkflowTransition> {
        let transition = self
            .transition(action)
            .ok_or_else(|| anyhow::anyhow!("unknown workflow action '{action}'"))?;
        if transition.from != current_state {
            bail!(
                "workflow action '{action}' is only valid from '{}', not '{current_state}'",
                transition.from
            );
        }
        for guard in &transition.guards {
            if !guard_is_satisfied(guard, context) {
                bail!(
                    "workflow action '{action}' is blocked: guard '{}' is not satisfied",
                    guard_name(guard)
                );
            }
        }
        Ok(transition)
    }

    /// Resolve a state owner from project-local role bindings.
    pub fn agent_for_state<'a>(
        &'a self,
        state: &str,
        role_bindings: &'a BTreeMap<String, String>,
    ) -> Result<Option<&'a str>> {
        let state = self
            .state(state)
            .ok_or_else(|| anyhow::anyhow!("unknown workflow state '{state}'"))?;
        let Some(role) = &state.role else {
            return Ok(None);
        };
        role_bindings
            .get(role)
            .map(|agent| Some(agent.as_str()))
            .ok_or_else(|| anyhow::anyhow!("workflow role '{role}' has no agent binding"))
    }
}

fn guard_is_satisfied(guard: &WorkflowGuard, context: GuardContext) -> bool {
    match guard {
        WorkflowGuard::DependenciesResolved => context.dependencies_resolved,
        WorkflowGuard::AdmissionRecorded => context.admission_recorded,
        WorkflowGuard::ApprovedPlan => context.approved_plan,
        WorkflowGuard::ImplementationRecorded => context.implementation_recorded,
        WorkflowGuard::FinalValidationPassed => context.final_validation_passed,
        WorkflowGuard::CleanWorktree => context.clean_worktree,
        WorkflowGuard::IntegratedIntoTarget => context.integrated_into_target,
    }
}

fn guard_name(guard: &WorkflowGuard) -> &'static str {
    match guard {
        WorkflowGuard::DependenciesResolved => "dependencies_resolved",
        WorkflowGuard::AdmissionRecorded => "admission_recorded",
        WorkflowGuard::ApprovedPlan => "approved_plan",
        WorkflowGuard::ImplementationRecorded => "implementation_recorded",
        WorkflowGuard::FinalValidationPassed => "final_validation_passed",
        WorkflowGuard::CleanWorktree => "clean_worktree",
        WorkflowGuard::IntegratedIntoTarget => "integrated_into_target",
    }
}

fn validate_identifier(kind: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || !value
            .chars()
            .all(|character| character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_')
    {
        bail!("{kind} '{value}' must use lowercase letters, digits, or underscores");
    }
    Ok(())
}

/// Bind workflow roles to installed agent names at the project boundary.
pub type RoleBindings = BTreeMap<String, String>;

/// Project-owned bindings for a declared workflow.
///
/// This is deliberately separate from `config.toml`: existing agtx versions
/// may still edit that file, and a workflow definition must not turn a normal
/// configuration save into an accidental role reassignment.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowProjectConfig {
    /// The branch into which the task reviewer integrates approved work.
    pub target_branch: String,
    #[serde(default)]
    pub role_bindings: RoleBindings,
}

impl WorkflowProjectConfig {
    pub const FILE_NAME: &'static str = "workflow.toml";

    pub fn load(project_path: &Path) -> Result<Option<Self>> {
        let path = project_path.join(".agtx").join(Self::FILE_NAME);
        if !path.exists() {
            return Ok(None);
        }
        let config: Self = toml::from_str(&std::fs::read_to_string(&path)?)
            .map_err(|error| anyhow::anyhow!("failed to parse {}: {error}", path.display()))?;
        config.validate()?;
        Ok(Some(config))
    }

    pub fn validate(&self) -> Result<()> {
        if self.target_branch.trim().is_empty() {
            bail!("workflow target_branch is required");
        }
        for (role, agent) in &self.role_bindings {
            validate_identifier("role", role)?;
            if agent.trim().is_empty() {
                bail!("workflow role '{role}' needs an agent binding");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    id: "planning".into(),
                    label: "Planning".into(),
                    role: Some("planner".into()),
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
                action: "start_planning".into(),
                from: "backlog".into(),
                to: "planning".into(),
                guards: vec![WorkflowGuard::DependenciesResolved],
            }],
        }
    }

    #[test]
    fn validates_a_role_based_graph() {
        workflow().validate().unwrap();
    }

    #[test]
    fn rejects_unknown_transition_state() {
        let mut definition = workflow();
        definition.transitions[0].to = "missing".into();
        assert!(definition.validate().unwrap_err().to_string().contains("unknown target"));
    }

    #[test]
    fn resolves_agents_by_role_not_phase_name() {
        let mut bindings = RoleBindings::new();
        bindings.insert("planner".into(), "claude".into());
        assert_eq!(
            workflow().agent_for_state("planning", &bindings).unwrap(),
            Some("claude")
        );
    }

    #[test]
    fn parses_project_owned_role_bindings() {
        let config: WorkflowProjectConfig = toml::from_str(
            "target_branch = \"feature/poc\"\n[role_bindings]\nplanner = \"claude\"\n",
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.role_bindings["planner"], "claude");
    }

    #[test]
    fn refuses_a_guarded_transition_without_evidence() {
        let error = workflow()
            .validate_transition("backlog", "start_planning", GuardContext::default())
            .unwrap_err();
        assert!(error.to_string().contains("dependencies_resolved"));
    }

    #[test]
    fn permits_a_guarded_transition_with_evidence() {
        let definition = workflow();
        let transition = definition
            .validate_transition(
                "backlog",
                "start_planning",
                GuardContext {
                    dependencies_resolved: true,
                    ..GuardContext::default()
                },
            )
            .unwrap();
        assert_eq!(transition.to, "planning");
    }
}
