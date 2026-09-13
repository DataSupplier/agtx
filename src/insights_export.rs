//! Export AGTX's durable SQLite workflow evidence as Heaves delivery-insight events.
//!
//! The exporter deliberately emits metadata and hashes only: prompt, report, and
//! artifact content stored in AGTX's local SQLite database never leave the machine.

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

use crate::db::Database;

const SOURCE: &str = "agtx";

/// Build newline-delimited events from the current project database.
///
/// Re-emitting is safe: every append-only event has a stable AGTX row id as its
/// `source_event_id`, while the session id is deterministically derived from a
/// task/workflow-attempt/state tuple.
pub fn export_project(project_path: &Path) -> Result<Vec<Value>> {
    let db = Database::open_project(project_path)?;
    let mut events = Vec::new();
    for task in db.get_all_tasks()? {
        let artifacts = db.workflow_artifacts_for_task(&task.id)?;
        let telemetry_by_session: HashMap<String, Value> = artifacts
            .iter()
            .filter_map(|artifact| {
                telemetry_from_artifact(&artifact.content).map(|telemetry| {
                    (
                        session_id(
                            &artifact.task_id,
                            artifact.workflow_attempt,
                            &artifact.state,
                        ),
                        telemetry,
                    )
                })
            })
            .collect();
        for report in db.task_step_reports(&task.id)? {
            let session_id = session_id(&task.id, report.workflow_attempt, &report.state);
            let telemetry = telemetry_by_session.get(&session_id);
            let mut session = json!({
                "kind": "session",
                "session_id": session_id,
                "task_id": task.id,
                "workflow": "agtx",
                "phase": report.state,
                "provider": "agtx",
                "agent": report.agent.unwrap_or_else(|| task.agent.clone()),
                "started_at": report.created_at.to_rfc3339(),
                "source_session_reference": report.id,
            });
            if let Some(telemetry) = telemetry {
                for (session_field, telemetry_field) in [
                    ("effective_model", "effective_model"),
                    ("effective_reasoning_effort", "reasoning_effort"),
                    ("provider", "provider"),
                ] {
                    if !telemetry[telemetry_field].is_null() {
                        session[session_field] = telemetry[telemetry_field].clone();
                    }
                }
            }
            events.push(session);
            if let Some(usage) = telemetry.and_then(|value| value.get("usage")) {
                let telemetry = telemetry.expect("usage requires a telemetry envelope");
                events.push(json!({
                    "kind": "usage",
                    "session_id": session_id,
                    "source": SOURCE,
                    "source_event_id": format!("{}:{}", report.id, "usage"),
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "reasoning_tokens": usage["reasoning_tokens"],
                    "credits": usage["credits"],
                    "cost": usage["cost"],
                    "currency": usage["currency"],
                    "effective_model": telemetry["effective_model"],
                    "effective_reasoning_effort": telemetry["reasoning_effort"],
                    "recorded_at": report.updated_at.to_rfc3339(),
                }));
            }
            events.push(json!({
                "kind": "session_completed",
                "session_id": session_id,
                "ended_at": report.updated_at.to_rfc3339(),
                "outcome": report_outcome(&report.final_report),
            }));
        }
        for artifact in artifacts {
            events.push(json!({
                "kind": "artifact",
                "artifact_key": format!("agtx:{}", artifact.id),
                "task_id": artifact.task_id,
                "produced_by_session_id": session_id(&artifact.task_id, artifact.workflow_attempt, &artifact.state),
                "artifact_type": format!("{}:{}", artifact.state, artifact.kind),
                "path_or_uri": artifact.source_path,
                "content_hash": artifact.sha256,
                "produced_at": artifact.created_at.to_rfc3339(),
            }));
        }
        for event in db.task_execution_events(&task.id)? {
            if let Some(outcome) = event.outcome {
                events.push(json!({
                    "kind": "quality",
                    "task_id": event.task_id,
                    "session_id": event.workflow_attempt.zip(event.state.as_deref()).map(|(attempt, state)| session_id(&task.id, attempt, state)),
                    "source": SOURCE,
                    "source_event_id": event.id,
                    "signal_type": event.event_type,
                    "outcome": outcome,
                    "occurred_at": event.created_at.to_rfc3339(),
                    "detail": event.message,
                }));
            }
        }
    }
    Ok(events)
}

/// Parse the one-line hand-over envelope without parsing arbitrary artifact prose.
/// The artifact itself remains immutable SQLite evidence; malformed telemetry is ignored
/// so it can never block workflow recovery or export of its other evidence.
fn telemetry_from_artifact(content: &[u8]) -> Option<Value> {
    let text = std::str::from_utf8(content).ok()?;
    let line = text
        .lines()
        .find_map(|line| line.trim().strip_prefix("agtx_telemetry:"))?;
    let value: Value = serde_json::from_str(line.trim()).ok()?;
    value.is_object().then_some(value)
}

pub fn session_id(task_id: &str, workflow_attempt: i64, state: &str) -> String {
    format!("agtx:{task_id}:{workflow_attempt}:{state}")
}

fn report_outcome(final_report: &Option<String>) -> &'static str {
    match final_report.as_deref() {
        Some(report) if report.contains("verdict: failed") => "failed",
        Some(_) => "completed",
        None => "in_progress",
    }
}

#[cfg(test)]
mod tests {
    use super::{report_outcome, session_id};

    #[test]
    fn session_identity_is_stable_per_task_attempt_and_state() {
        assert_eq!(
            session_id("task-1", 2, "planning"),
            "agtx:task-1:2:planning"
        );
        assert_ne!(
            session_id("task-1", 2, "planning"),
            session_id("task-1", 3, "planning")
        );
    }

    #[test]
    fn report_outcome_does_not_treat_missing_evidence_as_completion() {
        assert_eq!(report_outcome(&None), "in_progress");
        assert_eq!(report_outcome(&Some("verdict: failed".into())), "failed");
        assert_eq!(
            report_outcome(&Some("verdict: approved".into())),
            "completed"
        );
    }

    #[test]
    fn parses_only_the_explicit_handover_telemetry_envelope() {
        let telemetry = super::telemetry_from_artifact(
            b"verdict: approved\nagtx_telemetry: {\"effective_model\":\"gpt-5.6\",\"usage\":{\"input_tokens\":12}}\n",
        )
        .unwrap();
        assert_eq!(telemetry["effective_model"], "gpt-5.6");
        assert_eq!(telemetry["usage"]["input_tokens"], 12);
    }
}
