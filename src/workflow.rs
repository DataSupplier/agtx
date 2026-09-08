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

/// Project-owned capabilities for one stable workflow role.  The state graph
/// determines when the role is active; the launcher maps these capabilities to
/// agent-specific sandbox and approval settings.
///
/// This struct, as declared in `.agtx/workflow.toml`, is the **only** source of
/// permission for an unattended agent. It is deliberately the whole story: no
/// ambient settings file contributes, and there is no bypass or one-off
/// elevation path. A denied command or path stays denied, with no prompt and no
/// fallback; widening a role means a reviewed edit to `allowed_commands` or
/// `write_paths` here, then rerunning the state.
///
/// The property being protected is that the permissions an agent actually ran
/// under can be reconstructed from one reviewed file in version control. Before
/// adding a second source, see the invariant on `build_policy_agent_command`
/// and `AGENT_CONFIG_SKIP_FILES`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowRolePolicy {
    #[serde(default)]
    pub states: Vec<String>,
    #[serde(default)]
    pub modify_plan_artifacts: bool,
    #[serde(default)]
    pub modify_review_artifacts: bool,
    #[serde(default)]
    pub modify_source_and_tests: bool,
    #[serde(default)]
    pub final_task_commit: bool,
    #[serde(default)]
    pub push_task_branch: bool,
    #[serde(default)]
    pub create_or_update_task_pr: bool,
    #[serde(default)]
    pub merge_task_into_target: bool,
    /// Executable command prefixes this role may invoke in its task worktree.
    /// Matching is token-boundary aware: `git status` permits `git status --short`,
    /// but never `git statusx` or a shell compound command.
    #[serde(default)]
    pub allowed_commands: Vec<String>,
    /// Worktree-relative glob paths this role may modify. An empty list means
    /// the role receives no declared filesystem write scope.
    #[serde(default)]
    pub write_paths: Vec<String>,
    /// Optional model override for this stable workflow role. This remains
    /// project-owned so a workflow can be reproduced without hard-coding an
    /// agent provider into the state graph.
    #[serde(default)]
    pub model: Option<String>,
    /// Optional provider-native reasoning/effort level for this role.
    #[serde(default)]
    pub effort: Option<String>,
}

/// Cross-role capabilities that default to the project policy rather than an
/// individual workflow state.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowPolicyDefaults {
    #[serde(default)]
    pub read_repository: bool,
    #[serde(default)]
    pub read_only_git: bool,
    #[serde(default)]
    pub run_approved_checks: bool,
    #[serde(default)]
    pub modify_outside_task_worktree: bool,
    #[serde(default)]
    pub merge_feature_to_main: bool,
    #[serde(default)]
    pub request_bypass: bool,
    #[serde(default)]
    pub approve_bypass: bool,
    #[serde(default)]
    pub network: bool,
}

/// TOML container for `[role_policies.defaults]` and one table per role.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowRolePolicies {
    #[serde(default)]
    pub defaults: WorkflowPolicyDefaults,
    #[serde(flatten)]
    pub roles: BTreeMap<String, WorkflowRolePolicy>,
}

/// A narrowly scoped elevation for a delivery state.  It preserves stable role
/// bindings while preventing engineering review from implicitly gaining merge
/// authority in every state it owns.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowStatePolicy {
    pub role: String,
    #[serde(default)]
    pub modify_source_and_tests: bool,
    #[serde(default)]
    pub final_task_commit: bool,
    #[serde(default)]
    pub push_task_branch: bool,
    #[serde(default)]
    pub create_or_update_task_pr: bool,
    #[serde(default)]
    pub merge_task_into_target: bool,
    pub merge_target: Option<String>,
    #[serde(default)]
    pub merge_feature_to_main: bool,
    /// Explicit network elevation for this one state. `None` preserves the
    /// project default, which is normally false.
    #[serde(default)]
    pub network: Option<bool>,
}

/// The fully resolved policy for one destination state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedWorkflowPolicy {
    pub role: String,
    pub defaults: WorkflowPolicyDefaults,
    pub role_policy: WorkflowRolePolicy,
    pub merge_target: Option<String>,
    pub network: bool,
}

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
    #[serde(default)]
    pub role_policies: WorkflowRolePolicies,
    #[serde(default)]
    pub state_policies: BTreeMap<String, WorkflowStatePolicy>,
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
        if self.role_policies.defaults.modify_outside_task_worktree {
            bail!("workflow policy may not allow modify_outside_task_worktree");
        }
        if self.role_policies.defaults.merge_feature_to_main {
            bail!("workflow policy may not allow merge_feature_to_main");
        }
        if self.role_policies.defaults.approve_bypass {
            bail!("workflow policy may not allow approve_bypass");
        }
        for (role, policy) in &self.role_policies.roles {
            validate_identifier("role", role)?;
            for state in &policy.states {
                validate_identifier("workflow state", state)?;
            }
            for command in &policy.allowed_commands {
                validate_command_prefix(command)?;
            }
            for path in &policy.write_paths {
                validate_worktree_glob(path)?;
            }
            if let Some(model) = &policy.model {
                validate_agent_option("model", model)?;
            }
            if let Some(effort) = &policy.effort {
                validate_agent_option("effort", effort)?;
            }
        }
        for (state, policy) in &self.state_policies {
            validate_identifier("workflow state", state)?;
            validate_identifier("role", &policy.role)?;
            if policy.merge_feature_to_main {
                bail!("workflow state policy '{state}' may not allow merge_feature_to_main");
            }
        }
        Ok(())
    }

    /// Resolve policy for an owned workflow state.  State policy may only
    /// elevate the role that owns that exact state; it can never authorize the
    /// human-owned feature-to-main merge.
    pub fn policy_for_state(
        &self,
        workflow: &WorkflowDefinition,
        state_id: &str,
    ) -> Result<Option<ResolvedWorkflowPolicy>> {
        let state = workflow
            .state(state_id)
            .ok_or_else(|| anyhow::anyhow!("unknown workflow state '{state_id}'"))?;
        let Some(role) = state.role.as_ref() else { return Ok(None); };
        let mut resolved = ResolvedWorkflowPolicy {
            role: role.clone(),
            defaults: self.role_policies.defaults.clone(),
            // Never default this. `WorkflowRolePolicy::default()` has empty
            // `allowed_commands` and empty `write_paths`, which resolves to a
            // policy granting nothing -- and because its `states` is empty too,
            // the state guard below is skipped rather than tripped. The agent
            // then launches under `--permission-mode dontAsk` with an allowlist
            // of just `Read,Glob,Grep` and silently fails partway through the
            // role, unable to run its checks or write its own workflow
            // artifact. A missing role entry is a workflow misconfiguration;
            // surface it here instead of degrading into an agent that looks
            // launched but cannot do its job.
            role_policy: self
                .role_policies
                .roles
                .get(role)
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "workflow role '{role}' (state '{state_id}') has no \
                         [role_policies.{role}] entry; refusing to launch an \
                         agent with an empty permission set"
                    )
                })?,
            merge_target: None,
            network: self.role_policies.defaults.network,
        };
        if !resolved.role_policy.states.is_empty()
            && !resolved.role_policy.states.iter().any(|state| state == state_id)
        {
            bail!("role policy '{role}' does not permit workflow state '{state_id}'");
        }
        if let Some(state_policy) = self.state_policies.get(state_id) {
            if state_policy.role != *role {
                bail!("workflow state policy '{state_id}' belongs to '{}', not '{role}'", state_policy.role);
            }
            resolved.role_policy.modify_source_and_tests |= state_policy.modify_source_and_tests;
            resolved.role_policy.final_task_commit |= state_policy.final_task_commit;
            resolved.role_policy.push_task_branch |= state_policy.push_task_branch;
            resolved.role_policy.create_or_update_task_pr |= state_policy.create_or_update_task_pr;
            resolved.role_policy.merge_task_into_target |= state_policy.merge_task_into_target;
            resolved.merge_target = state_policy.merge_target.clone();
            if let Some(network) = state_policy.network {
                resolved.network = network;
            }
        }
        Ok(Some(resolved))
    }
}

fn validate_agent_option(kind: &str, value: &str) -> anyhow::Result<()> {
    if value.is_empty()
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.'))
    {
        bail!("workflow {kind} must contain only ASCII letters, digits, '-', '_' or '.'");
    }
    Ok(())
}

impl WorkflowRolePolicy {
    /// Whether an already tokenized command line begins with a configured
    /// allowlisted command. Shell operators are rejected by validation, so a
    /// prefix cannot be extended into a second command.
    pub fn permits_command(&self, command: &str) -> bool {
        self.allowed_commands.iter().any(|allowed| {
            command == allowed
                || command
                    .strip_prefix(allowed)
                    .is_some_and(|suffix| suffix.chars().next().is_some_and(char::is_whitespace))
        })
    }
}

fn validate_command_prefix(command: &str) -> Result<()> {
    let trimmed = command.trim();
    if trimmed.is_empty() || trimmed != command {
        bail!("allowed command must be non-empty and trimmed");
    }
    if command.chars().any(|character| matches!(character, '\n' | '\r' | '|' | ';' | '&' | '>' | '<' | '`' | '$')) {
        bail!("allowed command '{command}' may not contain shell operators");
    }
    if command.split_whitespace().next().is_none() {
        bail!("allowed command '{command}' has no executable");
    }
    Ok(())
}

fn validate_worktree_glob(glob: &str) -> Result<()> {
    if glob.is_empty() || Path::new(glob).is_absolute() {
        bail!("write path '{glob}' must be a non-empty worktree-relative glob");
    }
    if Path::new(glob).components().any(|component| matches!(component, std::path::Component::ParentDir | std::path::Component::RootDir | std::path::Component::Prefix(_))) {
        bail!("write path '{glob}' may not escape the task worktree");
    }
    Ok(())
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
    fn resolves_role_and_state_scoped_policy_without_delivery_leaking_to_review() {
        let graph = WorkflowDefinition {
            initial_state: "planning".into(),
            states: vec![
                WorkflowState { id: "planning".into(), label: "Planning".into(), role: Some("planner".into()), terminal: false },
                WorkflowState { id: "integrate_to_feature".into(), label: "Integrate".into(), role: Some("engineering_reviewer".into()), terminal: false },
                WorkflowState { id: "done".into(), label: "Done".into(), role: None, terminal: true },
            ],
            transitions: vec![],
        };
        let config: WorkflowProjectConfig = toml::from_str(
            r#"
target_branch = "feature/poc"
[role_bindings]
planner = "claude"
engineering_reviewer = "codex"
[role_policies.defaults]
read_repository = true
[role_policies.planner]
states = ["planning"]
modify_plan_artifacts = true
[role_policies.engineering_reviewer]
states = ["integrate_to_feature"]
modify_source_and_tests = true
[state_policies.integrate_to_feature]
role = "engineering_reviewer"
final_task_commit = true
push_task_branch = true
merge_task_into_target = true
merge_target = "feature/poc"
"#,
        ).unwrap();
        config.validate().unwrap();
        let planning = config.policy_for_state(&graph, "planning").unwrap().unwrap();
        assert!(planning.role_policy.modify_plan_artifacts);
        assert!(!planning.role_policy.final_task_commit);
        let delivery = config.policy_for_state(&graph, "integrate_to_feature").unwrap().unwrap();
        assert!(delivery.role_policy.final_task_commit);
        assert_eq!(delivery.merge_target.as_deref(), Some("feature/poc"));
    }

    #[test]
    fn validates_command_and_worktree_path_policies() {
        let config: WorkflowProjectConfig = toml::from_str(
            r#"
target_branch = "feature/poc"
[role_policies.planner]
model = "sonnet"
effort = "medium"
allowed_commands = ["git status", "rg", "cat"]
write_paths = [".agtx/plans/**"]
"#,
        )
        .unwrap();
        config.validate().unwrap();
        let planner = &config.role_policies.roles["planner"];
        assert_eq!(planner.model.as_deref(), Some("sonnet"));
        assert_eq!(planner.effort.as_deref(), Some("medium"));
        assert!(planner.permits_command("git status --short"));
        assert!(planner.permits_command("rg workflow ."));
        assert!(!planner.permits_command("git statusx"));
        assert!(!planner.permits_command("git status; git push"));
    }

    #[test]
    fn rejects_shell_compounds_and_worktree_escapes_in_policies() {
        assert!(validate_command_prefix("git status; git push").is_err());
        assert!(validate_worktree_glob("../.git/config").is_err());
        assert!(validate_worktree_glob("/etc/passwd").is_err());
        assert!(validate_agent_option("model", "sonnet; rm").is_err());
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
