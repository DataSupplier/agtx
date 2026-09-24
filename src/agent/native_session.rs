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
use std::time::SystemTime;

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
    codex_rollouts_for_worktree(codex_home, worktree)
        .into_iter()
        .max_by_key(|rollout| rollout.modified)
        .map(|rollout| rollout.id)
}

struct CodexRollout {
    id: String,
    path: PathBuf,
    modified: SystemTime,
}

fn codex_rollouts_for_worktree(codex_home: &Path, worktree: &Path) -> Vec<CodexRollout> {
    let mut rollouts = Vec::new();
    collect_rollouts(&codex_home.join("sessions"), &mut rollouts);
    rollouts
        .into_iter()
        .filter_map(|path| {
            let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok()?;
            let (id, cwd) = read_codex_session_meta(&path)?;
            (Path::new(&cwd) == worktree).then_some(CodexRollout { id, path, modified })
        })
        .collect()
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

// ── Turn activity ─────────────────────────────────────────────────────────

/// Whether the agent working in a worktree is mid-turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnActivity {
    /// A turn is in progress: the agent may still write files or run tools.
    Busy,
    /// The last turn has ended; the agent is waiting for input.
    Idle,
}

/// The turn state of `agent`'s newest session in `worktree`, read from the
/// provider's own session store (Claude: AGTX's hook status file). `None`
/// when the provider records nothing AGTX can read.
///
/// A workflow artifact is often written *before* the agent's turn ends -- a
/// reviewer writes its verdict, then keeps running checks and a summary for
/// up to a minute. Handing the pane over at the artifact write lands the
/// hand-off keystrokes in a busy agent, where Ctrl+C only interrupts the turn
/// and the next agent's launch command becomes a chat message.
pub fn turn_activity(agent: &str, worktree: &Path, task_id: &str) -> Option<TurnActivity> {
    match agent {
        "codex" => codex_turn_activity(&codex_home()?, worktree),
        "opencode" => opencode_turn_activity(&opencode_data_home(), worktree),
        _ => {
            let status =
                super::hook_status::read_status(worktree, task_id, chrono::Utc::now().timestamp())?;
            Some(match status.state {
                super::hook_status::HookState::Working => TurnActivity::Busy,
                _ => TurnActivity::Idle,
            })
        }
    }
}

/// Codex journals `task_started` at the beginning of every turn and
/// `task_complete` / `turn_aborted` at its end.
pub fn codex_turn_activity(codex_home: &Path, worktree: &Path) -> Option<TurnActivity> {
    let rollout = codex_rollouts_for_worktree(codex_home, worktree)
        .into_iter()
        .max_by_key(|rollout| rollout.modified)?;
    let file = std::fs::File::open(&rollout.path).ok()?;
    let mut activity = TurnActivity::Idle;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        // Cheap pre-filter: only event lines can change the turn state.
        if !line.contains("\"event_msg\"") {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        match value.pointer("/payload/type").and_then(|t| t.as_str()) {
            Some("task_started") => activity = TurnActivity::Busy,
            Some("task_complete" | "turn_aborted") => activity = TurnActivity::Idle,
            _ => {}
        }
    }
    Some(activity)
}

/// OpenCode appends an `idle` message to a session each time a turn ends.
pub fn opencode_turn_activity(data_home: &Path, worktree: &Path) -> Option<TurnActivity> {
    let conn = open_opencode(data_home)?;
    let session: String = conn
        .query_row(
            "SELECT id FROM session_v2 WHERE directory = ?1 AND parent_id IS NULL \
             ORDER BY time_updated DESC LIMIT 1",
            [worktree.to_string_lossy()],
            |row| row.get(0),
        )
        .ok()?;
    let last: Option<String> = conn
        .query_row(
            "SELECT type FROM session_message WHERE session_id = ?1 ORDER BY seq DESC LIMIT 1",
            [&session],
            |row| row.get(0),
        )
        .ok();
    Some(match last.as_deref() {
        None | Some("idle") => TurnActivity::Idle,
        Some(_) => TurnActivity::Busy,
    })
}

fn open_opencode(data_home: &Path) -> Option<rusqlite::Connection> {
    let database = data_home.join("opencode").join("opencode.db");
    rusqlite::Connection::open_with_flags(database, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).ok()
}

// ── Probe used by the workflow executor ───────────────────────────────────

/// What the workflow executor asks the providers' session stores during a
/// hand-off. A trait so executor tests can script the answers.
pub trait SessionProbe: Send + Sync {
    fn turn_activity(&self, agent: &str, worktree: &Path, task_id: &str) -> Option<TurnActivity>;
    fn find_delivered_prompt(
        &self,
        agent: &str,
        worktree: &Path,
        marker: &str,
        since: SystemTime,
    ) -> Delivery;
    /// How long a delivered prompt may take to appear in the session store.
    fn delivery_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(45)
    }
}

/// Reads the real provider stores.
pub struct ProviderSessionProbe;

impl SessionProbe for ProviderSessionProbe {
    fn turn_activity(&self, agent: &str, worktree: &Path, task_id: &str) -> Option<TurnActivity> {
        turn_activity(agent, worktree, task_id)
    }
    fn find_delivered_prompt(
        &self,
        agent: &str,
        worktree: &Path,
        marker: &str,
        since: SystemTime,
    ) -> Delivery {
        find_delivered_prompt(agent, worktree, marker, since)
    }
}

/// Reports every agent as idle-unknown and every delivery as unverifiable,
/// i.e. the pre-verification behaviour. Used where no provider store exists.
pub struct UnverifiedSessionProbe;

impl SessionProbe for UnverifiedSessionProbe {
    fn turn_activity(&self, _: &str, _: &Path, _: &str) -> Option<TurnActivity> {
        None
    }
    fn find_delivered_prompt(&self, _: &str, _: &Path, _: &str, _: SystemTime) -> Delivery {
        Delivery::Unverifiable
    }
}

pub static PROVIDER_PROBE: ProviderSessionProbe = ProviderSessionProbe;
pub static UNVERIFIED_PROBE: UnverifiedSessionProbe = UnverifiedSessionProbe;

/// The probe production code hands the executor. Unit tests drive the TUI's
/// workflow paths against mocked tmux; they must not read the developer's own
/// `~/.codex` or OpenCode database, so they get the unverified probe and test
/// the probe logic directly instead.
pub fn default_probe() -> &'static dyn SessionProbe {
    if cfg!(test) {
        &UNVERIFIED_PROBE
    } else {
        &PROVIDER_PROBE
    }
}

// ── Prompt delivery ───────────────────────────────────────────────────────

/// Whether a prompt reached the intended agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delivery {
    /// The agent's own session recorded the prompt; carries that session id.
    Confirmed(String),
    /// The session store is readable but holds no such prompt.
    Missing,
    /// This agent keeps no session store AGTX can read here.
    Unverifiable,
}

/// The line AGTX looks for in the destination's session: the prompt's first
/// non-empty line, whitespace-normalised and bounded so it survives the
/// provider's own storage (JSON escaping, wrapping of long pastes).
pub fn prompt_marker(prompt: &str) -> String {
    let line = prompt
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    let normalised = line.split_whitespace().collect::<Vec<_>>().join(" ");
    normalised.chars().take(80).collect()
}

/// Look for a user message containing `marker` in a session of `agent` that
/// was started in `worktree` and written at or after `since`.
pub fn find_delivered_prompt(
    agent: &str,
    worktree: &Path,
    marker: &str,
    since: SystemTime,
) -> Delivery {
    if marker.is_empty() {
        return Delivery::Unverifiable;
    }
    match agent {
        "codex" => match codex_home() {
            Some(home) => codex_find_prompt(&home, worktree, marker, since),
            None => Delivery::Unverifiable,
        },
        "opencode" => opencode_find_prompt(&opencode_data_home(), worktree, marker, since),
        "claude" => match directories::BaseDirs::new() {
            Some(dirs) => {
                claude_find_prompt(&dirs.home_dir().join(".claude"), worktree, marker, since)
            }
            None => Delivery::Unverifiable,
        },
        _ => Delivery::Unverifiable,
    }
}

pub fn codex_find_prompt(
    codex_home: &Path,
    worktree: &Path,
    marker: &str,
    since: SystemTime,
) -> Delivery {
    if !codex_home.join("sessions").is_dir() {
        return Delivery::Unverifiable;
    }
    // Newest first: after a re-delivery, recovery must resume the latest
    // session that holds the prompt.
    let mut rollouts = codex_rollouts_for_worktree(codex_home, worktree);
    rollouts.sort_by(|a, b| b.modified.cmp(&a.modified));
    for rollout in rollouts {
        if rollout.modified < since {
            continue;
        }
        let Ok(file) = std::fs::File::open(&rollout.path) else {
            continue;
        };
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            let is_user_input = match value.get("type").and_then(|t| t.as_str()) {
                Some("event_msg") => {
                    value.pointer("/payload/type").and_then(|t| t.as_str()) == Some("user_message")
                }
                Some("response_item") => {
                    value.pointer("/payload/role").and_then(|r| r.as_str()) == Some("user")
                }
                _ => false,
            };
            if is_user_input && json_contains(&value["payload"], marker) {
                return Delivery::Confirmed(rollout.id);
            }
        }
    }
    Delivery::Missing
}

pub fn opencode_find_prompt(
    data_home: &Path,
    worktree: &Path,
    marker: &str,
    since: SystemTime,
) -> Delivery {
    let Some(conn) = open_opencode(data_home) else {
        return Delivery::Unverifiable;
    };
    let since_ms = since
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let Ok(mut statement) = conn.prepare(
        "SELECT m.session_id, m.data FROM session_message m \
         JOIN session_v2 s ON s.id = m.session_id \
         WHERE s.directory = ?1 AND s.parent_id IS NULL AND m.type = 'user' \
           AND m.time_created >= ?2 \
         ORDER BY m.time_created DESC",
    ) else {
        return Delivery::Unverifiable;
    };
    let Ok(rows) = statement.query_map(
        rusqlite::params![worktree.to_string_lossy(), since_ms],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
    ) else {
        return Delivery::Unverifiable;
    };
    for (session, data) in rows.flatten() {
        let found = serde_json::from_str::<serde_json::Value>(&data)
            .map(|value| json_contains(&value, marker))
            .unwrap_or_else(|_| normalise(&data).contains(marker));
        if found {
            return Delivery::Confirmed(session);
        }
    }
    Delivery::Missing
}

/// Claude Code keeps one transcript per session under
/// `~/.claude/projects/<cwd with separators replaced by '-'>/<id>.jsonl`.
pub fn claude_find_prompt(
    claude_home: &Path,
    worktree: &Path,
    marker: &str,
    since: SystemTime,
) -> Delivery {
    let path = worktree.to_string_lossy();
    let candidates = [
        path.replace(['/', '.'], "-"),
        path.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect::<String>(),
    ];
    let Some(dir) = candidates
        .iter()
        .map(|slug| claude_home.join("projects").join(slug))
        .find(|dir| dir.is_dir())
    else {
        return Delivery::Unverifiable;
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Delivery::Unverifiable;
    };
    let mut transcripts: Vec<(SystemTime, PathBuf)> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .filter_map(|path| {
            Some((
                std::fs::metadata(&path).and_then(|m| m.modified()).ok()?,
                path,
            ))
        })
        .filter(|(modified, _)| *modified >= since)
        .collect();
    transcripts.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, transcript) in transcripts {
        let Ok(file) = std::fs::File::open(&transcript) else {
            continue;
        };
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            if value.get("type").and_then(|t| t.as_str()) == Some("user")
                && json_contains(&value["message"], marker)
            {
                let id = transcript
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_default();
                return Delivery::Confirmed(id);
            }
        }
    }
    Delivery::Missing
}

fn normalise(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn json_contains(value: &serde_json::Value, marker: &str) -> bool {
    match value {
        serde_json::Value::String(text) => normalise(text).contains(marker),
        serde_json::Value::Array(items) => items.iter().any(|item| json_contains(item, marker)),
        serde_json::Value::Object(map) => map.values().any(|item| json_contains(item, marker)),
        _ => false,
    }
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
        write_rollout(
            &day,
            "rollout-a.jsonl",
            "own-session",
            "/wt/12ced161",
            "cli",
        );
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
        write_rollout(
            &home.path().join("sessions/2026/09/23"),
            "r1.jsonl",
            "old",
            "/wt/a",
            "cli",
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_rollout(
            &home.path().join("sessions/2026/09/24"),
            "r2.jsonl",
            "new",
            "/wt/a",
            "cli",
        );

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

        assert_eq!(
            codex_session_id_for_worktree(home.path(), Path::new("/wt/a")),
            None
        );
        assert_eq!(
            codex_session_id_for_worktree(home.path(), Path::new("/wt/b")),
            None
        );
        assert_eq!(
            codex_session_id_for_worktree(&home.path().join("missing"), Path::new("/wt/a")),
            None
        );
    }

    fn append(path: &Path, value: serde_json::Value) {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().append(true).open(path).unwrap();
        writeln!(file, "{value}").unwrap();
    }

    fn event(kind: &str) -> serde_json::Value {
        serde_json::json!({"timestamp": "2026-09-24T12:23:58Z", "type": "event_msg", "payload": {"type": kind}})
    }

    /// Codex's own journal for task 16581cba: the reviewer wrote its verdict
    /// at 12:27:15, but `task_complete` only followed at 12:28:14.
    #[test]
    fn codex_turn_activity_follows_task_started_and_task_complete() {
        let home = tempfile::tempdir().unwrap();
        let day = home.path().join("sessions/2026/09/24");
        write_rollout(&day, "r.jsonl", "reviewer", "/wt/a", "cli");
        let rollout = day.join("r.jsonl");
        append(&rollout, event("task_started"));
        assert_eq!(
            codex_turn_activity(home.path(), Path::new("/wt/a")),
            Some(TurnActivity::Busy)
        );
        append(&rollout, event("task_complete"));
        assert_eq!(
            codex_turn_activity(home.path(), Path::new("/wt/a")),
            Some(TurnActivity::Idle)
        );
        append(&rollout, event("task_started"));
        append(&rollout, event("turn_aborted"));
        assert_eq!(
            codex_turn_activity(home.path(), Path::new("/wt/a")),
            Some(TurnActivity::Idle)
        );
        assert_eq!(
            codex_turn_activity(home.path(), Path::new("/wt/other")),
            None
        );
    }

    #[test]
    fn prompt_marker_is_the_normalised_first_line() {
        assert_eq!(
            prompt_marker("\n  You are the   implementer for task 16581cba.\nImplement only…"),
            "You are the implementer for task 16581cba."
        );
        assert_eq!(prompt_marker(&"x".repeat(200)).len(), 80);
    }

    #[test]
    fn codex_finds_a_prompt_carried_in_the_launch_command() {
        let home = tempfile::tempdir().unwrap();
        let day = home.path().join("sessions/2026/09/24");
        write_rollout(&day, "r.jsonl", "01a0d35f", "/wt/a", "cli");
        let rollout = day.join("r.jsonl");
        let since = SystemTime::now() - std::time::Duration::from_secs(5);
        let marker = prompt_marker("You are the plan reviewer for task 16581cba. Review only …");
        assert_eq!(
            codex_find_prompt(home.path(), Path::new("/wt/a"), &marker, since),
            Delivery::Missing
        );

        append(
            &rollout,
            serde_json::json!({"type": "response_item", "payload": {"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "You are the plan reviewer for task 16581cba. Review only …"}]}}),
        );
        assert_eq!(
            codex_find_prompt(home.path(), Path::new("/wt/a"), &marker, since),
            Delivery::Confirmed("01a0d35f".into())
        );
        // Another worktree's session never confirms this task's prompt.
        assert_eq!(
            codex_find_prompt(home.path(), Path::new("/wt/b"), &marker, since),
            Delivery::Missing
        );
        assert_eq!(
            codex_find_prompt(
                &home.path().join("none"),
                Path::new("/wt/a"),
                &marker,
                since
            ),
            Delivery::Unverifiable
        );
    }

    fn opencode_store(dir: &Path) -> rusqlite::Connection {
        let store = dir.join("opencode");
        std::fs::create_dir_all(&store).unwrap();
        let conn = rusqlite::Connection::open(store.join("opencode.db")).unwrap();
        conn.execute_batch(
            "CREATE TABLE session_v2 (id TEXT, parent_id TEXT, directory TEXT, time_updated INTEGER);
             CREATE TABLE session_message (id TEXT, session_id TEXT, type TEXT, seq INTEGER, time_created INTEGER, data TEXT);",
        )
        .unwrap();
        conn
    }

    /// The 16581cba failure: the OpenCode planner session holds only the
    /// planner prompt; the implementer prompt went to Codex. Delivery to
    /// OpenCode must not be confirmed from that session.
    #[test]
    fn opencode_confirms_only_a_prompt_its_own_session_recorded() {
        let data_home = tempfile::tempdir().unwrap();
        let conn = opencode_store(data_home.path());
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        conn.execute(
            "INSERT INTO session_v2 VALUES ('ses_planner', NULL, '/wt/a', ?1)",
            [now_ms],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_message VALUES ('m1', 'ses_planner', 'user', 4, ?1, ?2)",
            rusqlite::params![
                now_ms,
                r#"{"text":"You are the planner for task 16581cba."}"#
            ],
        )
        .unwrap();
        let since = SystemTime::now() - std::time::Duration::from_secs(60);
        let implementer = prompt_marker("You are the implementer for task 16581cba.");

        assert_eq!(
            opencode_find_prompt(data_home.path(), Path::new("/wt/a"), &implementer, since),
            Delivery::Missing
        );
        conn.execute(
            "INSERT INTO session_v2 VALUES ('ses_impl', NULL, '/wt/a', ?1)",
            [now_ms],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_message VALUES ('m2', 'ses_impl', 'user', 1, ?1, ?2)",
            rusqlite::params![
                now_ms,
                r#"{"text":"You are the implementer for task 16581cba.\n\nImplement only"}"#
            ],
        )
        .unwrap();
        assert_eq!(
            opencode_find_prompt(data_home.path(), Path::new("/wt/a"), &implementer, since),
            Delivery::Confirmed("ses_impl".into())
        );
    }

    #[test]
    fn opencode_turn_activity_reads_the_idle_marker() {
        let data_home = tempfile::tempdir().unwrap();
        let conn = opencode_store(data_home.path());
        conn.execute("INSERT INTO session_v2 VALUES ('s', NULL, '/wt/a', 1)", [])
            .unwrap();
        conn.execute(
            "INSERT INTO session_message VALUES ('1', 's', 'user', 1, 1, '{}')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_message VALUES ('2', 's', 'assistant', 2, 2, '{}')",
            [],
        )
        .unwrap();
        assert_eq!(
            opencode_turn_activity(data_home.path(), Path::new("/wt/a")),
            Some(TurnActivity::Busy)
        );
        conn.execute(
            "INSERT INTO session_message VALUES ('3', 's', 'idle', 3, 3, '{}')",
            [],
        )
        .unwrap();
        assert_eq!(
            opencode_turn_activity(data_home.path(), Path::new("/wt/a")),
            Some(TurnActivity::Idle)
        );
    }

    #[test]
    fn claude_confirms_a_prompt_in_the_worktree_transcript() {
        let claude_home = tempfile::tempdir().unwrap();
        let project = claude_home.path().join("projects").join("-wt-a");
        std::fs::create_dir_all(&project).unwrap();
        let transcript = project.join("5f1c.jsonl");
        std::fs::write(&transcript, "").unwrap();
        let since = SystemTime::now() - std::time::Duration::from_secs(5);
        let marker = prompt_marker("You are the engineering reviewer for task 1.");
        append(
            &transcript,
            serde_json::json!({"type": "user", "message": {"role": "user", "content": "You are the engineering reviewer for task 1."}}),
        );
        assert_eq!(
            claude_find_prompt(claude_home.path(), Path::new("/wt/a"), &marker, since),
            Delivery::Confirmed("5f1c".into())
        );
        assert_eq!(
            claude_find_prompt(claude_home.path(), Path::new("/wt/zz"), &marker, since),
            Delivery::Unverifiable
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
