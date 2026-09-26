//! Read-only, cursor-paged export of whole rows for downstream mirrors.
//!
//! Consumers (the heaves AGTX cache) keep a full copy of these tables, so every
//! column is returned as stored — including columns added by later migrations —
//! and nothing is renamed or interpreted. Paging is keyset-based on a per-table
//! change cursor plus `rowid`, so a consumer can resume after the last row it
//! stored and pick up rows that changed since.

use anyhow::{bail, Result};
use rusqlite::types::ValueRef;
use serde::Serialize;
use serde_json::{Map, Value};

use super::Database;

/// Tables that may be exported, the database they live in, and the expression
/// that moves forward whenever a row is written. `provider_sessions` has no
/// `updated_at`; a session is rewritten when it ends, so its end time counts.
const EXPORTABLE: &[(&str, ExportDb, &str)] = &[
    ("projects", ExportDb::Global, "last_opened"),
    ("tasks", ExportDb::Project, "updated_at"),
    ("workflow_task_states", ExportDb::Project, "updated_at"),
    (
        "workflow_transition_history",
        ExportDb::Project,
        "created_at",
    ),
    ("task_execution_events", ExportDb::Project, "created_at"),
    ("task_step_reports", ExportDb::Project, "updated_at"),
    (
        "provider_sessions",
        ExportDb::Project,
        "COALESCE(ended_at, started_at)",
    ),
    ("workflow_artifacts", ExportDb::Project, "created_at"),
    ("workflow_step_inputs", ExportDb::Project, "created_at"),
];

pub const MAX_EXPORT_LIMIT: u32 = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportDb {
    Global,
    Project,
}

/// Which database an exportable table lives in, or `None` for any other table.
pub fn export_db_for(table: &str) -> Option<ExportDb> {
    EXPORTABLE
        .iter()
        .find(|(name, _, _)| *name == table)
        .map(|(_, db, _)| *db)
}

/// Resume point: the cursor value and `rowid` of the last row returned.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ExportCursor {
    pub since: String,
    pub after_rowid: i64,
}

#[derive(Debug, Serialize)]
pub struct ExportPage {
    pub table: String,
    pub columns: Vec<String>,
    /// Each row maps every column to its stored value, plus `_rowid` and
    /// `_cursor`. BLOBs become `{"encoding": "utf8"|"hex", "data": ...}`.
    pub rows: Vec<Map<String, Value>>,
    /// Present when more rows may follow; pass it back to continue.
    pub next: Option<ExportCursor>,
}

fn value_of(value: ValueRef<'_>) -> Value {
    match value {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => Value::from(i),
        ValueRef::Real(f) => Value::from(f),
        ValueRef::Text(t) => Value::from(String::from_utf8_lossy(t).into_owned()),
        ValueRef::Blob(b) => {
            let mut blob = Map::new();
            match std::str::from_utf8(b) {
                Ok(text) => {
                    blob.insert("encoding".into(), Value::from("utf8"));
                    blob.insert("data".into(), Value::from(text));
                }
                Err(_) => {
                    let hex: String = b.iter().map(|byte| format!("{:02x}", byte)).collect();
                    blob.insert("encoding".into(), Value::from("hex"));
                    blob.insert("data".into(), Value::from(hex));
                }
            }
            Value::Object(blob)
        }
    }
}

impl Database {
    /// Export one page of `table`, ordered by its change cursor then `rowid`,
    /// strictly after `(since, after_rowid)` when given.
    pub fn export_records(
        &self,
        table: &str,
        since: Option<&str>,
        after_rowid: Option<i64>,
        limit: u32,
    ) -> Result<ExportPage> {
        let Some((name, _, cursor)) = EXPORTABLE.iter().find(|(name, _, _)| *name == table) else {
            bail!("table '{}' is not exportable", table);
        };
        let limit = limit.clamp(1, MAX_EXPORT_LIMIT);
        // `name` and `cursor` come from the fixed allowlist above, never from the caller.
        let base = format!("SELECT rowid AS _rowid, {cursor} AS _cursor, * FROM {name}");
        let (sql, since_value) = match since {
            Some(since) => (
                format!(
                    "{base} WHERE ({cursor} > ?1 OR ({cursor} = ?1 AND rowid > ?2)) \
                     ORDER BY {cursor}, rowid LIMIT ?3"
                ),
                Some(since.to_string()),
            ),
            None => (format!("{base} ORDER BY {cursor}, rowid LIMIT ?3"), None),
        };
        let mut stmt = self.conn.prepare(&sql)?;
        let columns: Vec<String> = stmt.column_names().iter().map(|c| c.to_string()).collect();
        let mut query = stmt.query(rusqlite::params![
            since_value.as_deref().unwrap_or(""),
            after_rowid.unwrap_or(0),
            limit
        ])?;
        let mut rows = Vec::new();
        while let Some(row) = query.next()? {
            let mut record = Map::new();
            for (index, column) in columns.iter().enumerate() {
                record.insert(column.clone(), value_of(row.get_ref(index)?));
            }
            rows.push(record);
        }
        let next = if rows.len() as u32 == limit {
            rows.last().map(|last| ExportCursor {
                since: last
                    .get("_cursor")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                after_rowid: last.get("_rowid").and_then(Value::as_i64).unwrap_or(0),
            })
        } else {
            None
        };
        Ok(ExportPage {
            table: name.to_string(),
            columns: columns
                .into_iter()
                .filter(|c| c != "_rowid" && c != "_cursor")
                .collect(),
            rows,
            next,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Database {
        Database::open_in_memory_project().expect("in-memory project db")
    }

    #[test]
    fn rejects_tables_outside_the_allowlist() {
        let err = db()
            .export_records("sqlite_master", None, None, 10)
            .unwrap_err();
        assert!(err.to_string().contains("not exportable"));
    }

    #[test]
    fn pages_every_column_in_cursor_order_and_resumes_after_the_last_row() {
        let db = db();
        for (id, at) in [
            ("e1", "2026-01-01T00:00:01Z"),
            ("e2", "2026-01-01T00:00:02Z"),
            ("e3", "2026-01-01T00:00:03Z"),
        ] {
            db.conn
                .execute(
                    "INSERT INTO task_execution_events (id, task_id, event_type, message, created_at) \
                     VALUES (?1, 't1', 'phase', 'm', ?2)",
                    rusqlite::params![id, at],
                )
                .unwrap();
        }
        let first = db
            .export_records("task_execution_events", None, None, 2)
            .unwrap();
        assert_eq!(first.rows.len(), 2);
        assert!(first.columns.contains(&"metadata_json".to_string()));
        assert_eq!(first.rows[0]["id"], "e1");
        let next = first.next.expect("more rows");
        let second = db
            .export_records(
                "task_execution_events",
                Some(&next.since),
                Some(next.after_rowid),
                2,
            )
            .unwrap();
        assert_eq!(second.rows.len(), 1);
        assert_eq!(second.rows[0]["id"], "e3");
        assert!(second.next.is_none());
    }

    #[test]
    fn blobs_are_returned_as_utf8_or_hex() {
        let db = db();
        db.conn
            .execute(
                "INSERT INTO workflow_artifacts (id, task_id, workflow_attempt, state, kind, source_path, sha256, content, created_at) \
                 VALUES ('a1', 't1', 1, 'planning', 'plan', 'plan.md', 'x', ?1, '2026-01-01T00:00:00Z'), \
                        ('a2', 't1', 1, 'planning', 'bin', 'b.bin', 'y', ?2, '2026-01-01T00:00:01Z')",
                rusqlite::params![b"# Plan".to_vec(), vec![0xffu8, 0x00]],
            )
            .unwrap();
        let page = db
            .export_records("workflow_artifacts", None, None, 10)
            .unwrap();
        assert_eq!(page.rows[0]["content"]["encoding"], "utf8");
        assert_eq!(page.rows[0]["content"]["data"], "# Plan");
        assert_eq!(page.rows[1]["content"]["encoding"], "hex");
        assert_eq!(page.rows[1]["content"]["data"], "ff00");
    }
}
