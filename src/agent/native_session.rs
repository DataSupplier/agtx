//! Discovery of an agent's provider-native session for one task worktree.
//!
//! Resuming "the most recent session" is not task-scoped for every agent:
//! Codex's `resume --last` picks the newest session across every directory,
//! and every AGTX worktree of a repository shares one OpenCode project, so
//! `opencode --continue` is repository-wide as well. After a container or
//! tmux restart that recency rule handed one task's pane another task's
//! conversation. These lookups instead select a session strictly by the
//! directory it was started in, so a recovered pane can resume by explicit id
//! or, when nothing matches, start fresh -- never guess.
//!
//! All lookups are read-only and best-effort: a missing or unreadable store is
//! normal (a just-launched agent has not written one yet) and yields `None`.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// The session this worktree's own agent started most recently, for agents
/// whose resume must be scoped by id. `None` for other agents and when no
/// session was started in `worktree`.
pub fn session_id_for_worktree(agent: &str, worktree: &Path) -> Option<String> {
    match agent {
        "codex" => codex_session_id_for_worktree(&codex_home()?, worktree),
        "opencode" => opencode_session_id_for_worktree(&opencode_data_home(), worktree),
        _ => None,
    }
}

/// `$CODEX_HOME`, else `~/.codex` -- the same resolution the Codex CLI uses.
fn codex_home() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("CODEX_HOME").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(home));
    }
    directories::BaseDirs::new().map(|dirs| dirs.home_dir().join(".codex"))
}

/// AGTX launches OpenCode with `XDG_DATA_HOME=/tmp/agtx-opencode` when the
/// environment does not provide one.
fn opencode_data_home() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/agtx-opencode"))
}

/// Newest interactive Codex session whose `session_meta.cwd` is `worktree`.
///
/// Ownership is the directory the session was *started* in, not the latest
/// `turn_context` cwd: a session that was (wrongly) continued from another
/// worktree still belongs to the task that created it.
pub fn codex_session_id_for_worktree(codex_home: &Path, worktree: &Path) -> Option<String> {
    let mut rollouts = Vec::new();
    collect_rollouts(&codex_home.join("sessions"), &mut rollouts);
    rollouts
        .into_iter()
        .filter_map(|path| {
            let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok()?;
            let (id, cwd) = read_codex_session_meta(&path)?;
            (Path::new(&cwd) == worktree).then_some((modified, id))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, id)| id)
}

fn collect_rollouts(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rollouts(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            out.push(path);
        }
    }
}

/// `(id, cwd)` from a rollout's first line, skipping non-interactive
/// (`codex exec`) sessions, which the interactive TUI does not resume.
fn read_codex_session_meta(path: &Path) -> Option<(String, String)> {
    let file = std::fs::File::open(path).ok()?;
    let mut line = String::new();
    BufReader::new(file).read_line(&mut line).ok()?;
    let value: serde_json::Value = serde_json::from_str(&line).ok()?;
    if value.get("type")?.as_str()? != "session_meta" {
        return None;
    }
    let payload = value.get("payload")?;
    if payload.get("source").and_then(|s| s.as_str()) == Some("exec") {
        return None;
    }
    let id = payload.get("id")?.as_str()?.to_string();
    let cwd = payload.get("cwd")?.as_str()?.to_string();
    Some((id, cwd))
}

/// Newest top-level OpenCode session whose directory is `worktree`, without
/// ever mutating the provider's local database. Child (subagent) sessions are
/// excluded: resuming one would drop the task's own conversation.
pub fn opencode_session_id_for_worktree(data_home: &Path, worktree: &Path) -> Option<String> {
    let database = data_home.join("opencode").join("opencode.db");
    let conn =
        rusqlite::Connection::open_with_flags(database, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .ok()?;
    conn.query_row(
        "SELECT id FROM session_v2 WHERE directory = ?1 AND parent_id IS NULL \
         ORDER BY time_updated DESC LIMIT 1",
        [worktree.to_string_lossy()],
        |row| row.get(0),
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_rollout(dir: &Path, name: &str, id: &str, cwd: &str, source: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let meta = serde_json::json!({
            "timestamp": "2026-09-24T05:22:32.000Z",
            "type": "session_meta",
            "payload": { "id": id, "cwd": cwd, "source": source, "originator": "codex-tui" }
        });
        std::fs::write(
            dir.join(name),
            format!("{meta}\n{{\"type\":\"turn_context\",\"payload\":{{\"cwd\":\"{cwd}\"}}}}\n"),
        )
        .unwrap();
    }

    /// Regression for the 2026-09-24 incident: task 12ced161's recovered
    /// pane ran `codex resume --last` and received task 44489f0d's session,
    /// because that session was newer. The lookup must pick the worktree's
    /// own session even when another worktree's session is more recent.
    #[test]
    fn codex_lookup_ignores_a_newer_session_from_another_worktree() {
        let home = tempfile::tempdir().unwrap();
        let day = home.path().join("sessions/2026/09/24");
        write_rollout(&day, "rollout-a.jsonl", "own-session", "/wt/12ced161", "cli");
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_rollout(&day, "rollout-b.jsonl", "other-task", "/wt/44489f0d", "cli");

        assert_eq!(
            codex_session_id_for_worktree(home.path(), Path::new("/wt/12ced161")),
            Some("own-session".to_string())
        );
        assert_eq!(
            codex_session_id_for_worktree(home.path(), Path::new("/wt/44489f0d")),
            Some("other-task".to_string())
        );
    }

    #[test]
    fn codex_lookup_picks_newest_own_session_across_days() {
        let home = tempfile::tempdir().unwrap();
        write_rollout(&home.path().join("sessions/2026/09/23"), "r1.jsonl", "old", "/wt/a", "cli");
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_rollout(&home.path().join("sessions/2026/09/24"), "r2.jsonl", "new", "/wt/a", "cli");

        assert_eq!(
            codex_session_id_for_worktree(home.path(), Path::new("/wt/a")),
            Some("new".to_string())
        );
    }

    #[test]
    fn codex_lookup_skips_exec_sessions_and_unknown_worktrees() {
        let home = tempfile::tempdir().unwrap();
        let day = home.path().join("sessions/2026/09/24");
        write_rollout(&day, "r1.jsonl", "headless", "/wt/a", "exec");
        std::fs::write(day.join("garbage.jsonl"), "not json\n").unwrap();

        assert_eq!(codex_session_id_for_worktree(home.path(), Path::new("/wt/a")), None);
        assert_eq!(codex_session_id_for_worktree(home.path(), Path::new("/wt/b")), None);
        assert_eq!(
            codex_session_id_for_worktree(&home.path().join("missing"), Path::new("/wt/a")),
            None
        );
    }

    #[test]
    fn opencode_lookup_is_worktree_scoped_and_skips_child_sessions() {
        let data_home = tempfile::tempdir().unwrap();
        let store = data_home.path().join("opencode");
        std::fs::create_dir_all(&store).unwrap();
        let conn = rusqlite::Connection::open(store.join("opencode.db")).unwrap();
        conn.execute(
            "CREATE TABLE session_v2 (id TEXT, parent_id TEXT, directory TEXT, time_updated INTEGER)",
            [],
        )
        .unwrap();
        for (id, parent, dir, updated) in [
            ("older", None, "/wt/task", 10_i64),
            ("newest", None, "/wt/task", 20),
            ("subagent", Some("newest"), "/wt/task", 30),
            ("other-task", None, "/wt/other", 40),
        ] {
            conn.execute(
                "INSERT INTO session_v2 (id, parent_id, directory, time_updated) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![id, parent, dir, updated],
            )
            .unwrap();
        }

        assert_eq!(
            opencode_session_id_for_worktree(data_home.path(), Path::new("/wt/task")),
            Some("newest".to_string())
        );
        assert_eq!(
            opencode_session_id_for_worktree(data_home.path(), Path::new("/wt/none")),
            None
        );
    }
}
