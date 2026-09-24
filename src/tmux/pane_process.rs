//! Which coding agent is running in a pane, read from its process tree.
//!
//! `#{pane_current_command}` is the name of the terminal's foreground
//! process-group leader, which is often not the agent:
//!
//!   * An agent launched by `create_window` runs under AGTX's
//!     `sh -c '… sh -c "<agent>"'` wrapper. A non-interactive `sh` does no job
//!     control, so the agent shares the wrapper's process group and tmux
//!     reports `sh` for the agent's whole lifetime.
//!   * An npm-installed agent is a Node launcher (`#!/usr/bin/env node`), so
//!     one typed into an interactive shell is reported as `node`.
//!
//! Neither name is an agent name, so the hand-off logic read a running Codex
//! as "back at the shell", typed the next agent's launch command into Codex's
//! composer, and then accepted Codex (`node`) as the "launched" successor. The
//! resolver below looks at the processes of the pane's foreground job instead
//! and reports the agent they belong to.

use std::path::Path;

/// One process, as far as agent identification needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcInfo {
    pub pid: u32,
    pub ppid: u32,
    pub pgrp: i64,
    /// The foreground process group of the process's controlling terminal.
    pub tpgid: i64,
    pub argv: Vec<String>,
}

/// The `display -p` format `pane_current_command` sends: pane pid, then
/// tmux's own command name.
///
/// Space-separated on purpose. tmux sanitises control characters in format
/// output -- tmux 3.5a prints a tab as `_` -- so a tab-separated `40\tnode`
/// arrived as the single word `40_node`, no agent was ever identified, and
/// every hand-off failed its launch check and was retried forever.
pub const PANE_IDENTITY_FORMAT: &str = "#{pane_pid} #{pane_current_command}";

/// Split what [`PANE_IDENTITY_FORMAT`] produces into the pane pid (if tmux
/// reported one) and tmux's command name. `None` for empty output.
pub fn parse_pane_identity(line: &str) -> Option<(Option<u32>, &str)> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let (pid, command) = match line.split_once(' ') {
        Some((pid, command)) => match pid.parse::<u32>() {
            Ok(pid) => (Some(pid), command.trim()),
            Err(_) => (None, line),
        },
        None => (None, line),
    };
    if command.is_empty() {
        return None;
    }
    Some((pid, command))
}

/// Launchers whose script argument, not their own name, identifies the agent.
const INTERPRETERS: &[&str] = &["node", "nodejs", "bun", "deno", "env"];

/// The agent process name (one of `agent_names`) running in the foreground
/// job of the pane rooted at `pane_pid`, or `None` when no agent is running
/// there -- e.g. the pane is back at its interactive shell.
pub fn resolve_agent(procs: &[ProcInfo], pane_pid: u32, agent_names: &[&str]) -> Option<String> {
    let root = procs.iter().find(|p| p.pid == pane_pid)?;
    let foreground = root.tpgid;
    if foreground <= 0 {
        return None;
    }
    // Descendants of the pane process (itself included), breadth first, so
    // the outermost agent process of the foreground job wins.
    let mut queue = vec![pane_pid];
    let mut index = 0;
    while index < queue.len() {
        let pid = queue[index];
        index += 1;
        if let Some(process) = procs.iter().find(|p| p.pid == pid) {
            if process.pgrp == foreground {
                if let Some(name) = agent_name_of(&process.argv, agent_names) {
                    return Some(name);
                }
            }
        }
        queue.extend(
            procs
                .iter()
                .filter(|p| p.ppid == pid && p.pid != pid)
                .map(|p| p.pid),
        );
    }
    None
}

fn agent_name_of(argv: &[String], agent_names: &[&str]) -> Option<String> {
    let first = argv.first()?;
    let program = base_name(first);
    if let Some(name) = agent_names.iter().find(|name| **name == program) {
        return Some((*name).to_string());
    }
    if INTERPRETERS.contains(&program.as_str()) {
        // `node /usr/bin/codex …` or `env node /usr/bin/codex …`: the first
        // non-flag argument that is not itself an interpreter is the script.
        let script = argv[1..]
            .iter()
            .filter(|arg| !arg.starts_with('-') && !arg.contains('='))
            .find(|arg| !INTERPRETERS.contains(&base_name(arg).as_str()))?;
        let script = base_name(script);
        let script = script
            .strip_suffix(".js")
            .or_else(|| script.strip_suffix(".mjs"))
            .unwrap_or(&script)
            .to_string();
        return agent_names
            .iter()
            .find(|name| **name == script)
            .map(|name| (*name).to_string());
    }
    None
}

fn base_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// Every process visible in `/proc`. Empty where `/proc` does not exist
/// (macOS, Windows), in which case callers keep tmux's own answer.
pub fn read_proc_table() -> Vec<ProcInfo> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .filter_map(read_proc)
        .collect()
}

fn read_proc(pid: u32) -> Option<ProcInfo> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // `comm` is parenthesised and may itself contain spaces or ')'.
    let fields: Vec<&str> = stat
        .get(stat.rfind(')')? + 1..)?
        .split_whitespace()
        .collect();
    // After `comm`: state, ppid, pgrp, session, tty_nr, tpgid, ...
    let ppid = fields.get(1)?.parse().ok()?;
    let pgrp = fields.get(2)?.parse().ok()?;
    let tpgid = fields.get(5)?.parse().ok()?;
    let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let argv = cmdline
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf8_lossy(part).into_owned())
        .collect();
    Some(ProcInfo {
        pid,
        ppid,
        pgrp,
        tpgid,
        argv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGENTS: &[&str] = &["claude", "codex", "opencode", "gemini", "agent"];

    fn proc(pid: u32, ppid: u32, pgrp: i64, tpgid: i64, argv: &[&str]) -> ProcInfo {
        ProcInfo {
            pid,
            ppid,
            pgrp,
            tpgid,
            argv: argv.iter().map(|arg| arg.to_string()).collect(),
        }
    }

    /// Recovered/fresh panes: the agent shares the wrapper `sh`'s process
    /// group, which tmux reports as `sh`. Observed live on 2026-09-24.
    #[test]
    fn resolves_an_agent_running_under_the_sh_c_wrapper() {
        let table = [
            proc(
                41,
                1,
                41,
                41,
                &[
                    "sh",
                    "-c",
                    "env -u CLAUDECODE sh -c 'opencode --session s'; exec $SHELL",
                ],
            ),
            proc(
                43,
                41,
                41,
                41,
                &[
                    "sh",
                    "-c",
                    "XDG_DATA_HOME=/tmp/agtx-opencode opencode --session s",
                ],
            ),
            proc(44, 43, 41, 41, &["opencode", "--session", "s"]),
        ];
        assert_eq!(resolve_agent(&table, 41, AGENTS), Some("opencode".into()));
    }

    /// Codex typed into an interactive shell: its own foreground job, led by
    /// the npm Node launcher, which tmux reports as `node`.
    #[test]
    fn resolves_codex_behind_its_node_launcher() {
        let table = [
            proc(30, 1, 30, 90, &["-bash"]),
            proc(
                90,
                30,
                90,
                90,
                &["node", "/usr/bin/codex", "--sandbox", "read-only"],
            ),
            proc(
                95,
                90,
                90,
                90,
                &[
                    "/usr/lib/node_modules/@openai/codex/vendor/x86_64/codex/codex",
                    "--sandbox",
                    "read-only",
                ],
            ),
        ];
        assert_eq!(resolve_agent(&table, 30, AGENTS), Some("codex".into()));
    }

    #[test]
    fn a_pane_back_at_its_shell_has_no_agent() {
        let table = [
            proc(30, 1, 30, 30, &["-bash"]),
            // A background helper the exited agent left behind is not the
            // foreground job and must not read as a running agent.
            proc(
                172,
                30,
                172,
                -1,
                &["/usr/lib/node_modules/@opencode/cli/bin/opencode", "serve"],
            ),
        ];
        assert_eq!(resolve_agent(&table, 30, AGENTS), None);
    }

    #[test]
    fn env_and_interpreter_flags_are_skipped() {
        let table = [
            proc(10, 1, 10, 10, &["bash"]),
            proc(
                11,
                10,
                11,
                10,
                &[
                    "/usr/bin/env",
                    "node",
                    "--no-warnings",
                    "/usr/local/bin/gemini",
                ],
            ),
        ];
        // pgrp 11 is not the foreground group (10): the shell owns the terminal.
        assert_eq!(resolve_agent(&table, 10, AGENTS), None);
        let table = [
            proc(10, 1, 10, 11, &["bash"]),
            proc(
                11,
                10,
                11,
                11,
                &[
                    "/usr/bin/env",
                    "node",
                    "--no-warnings",
                    "/usr/local/bin/gemini",
                ],
            ),
        ];
        assert_eq!(resolve_agent(&table, 10, AGENTS), Some("gemini".into()));
    }

    #[test]
    fn an_unrelated_node_program_is_not_an_agent() {
        let table = [
            proc(10, 1, 10, 11, &["bash"]),
            proc(
                11,
                10,
                11,
                11,
                &["node", "/app/node_modules/.bin/vitest", "run"],
            ),
        ];
        assert_eq!(resolve_agent(&table, 10, AGENTS), None);
    }

    /// Regression for 2026-09-24: tmux 3.5a printed a tab separator as `_`,
    /// so the pid and command never split and no hand-off was ever confirmed.
    #[test]
    fn pane_identity_uses_a_separator_tmux_prints_verbatim() {
        assert!(!PANE_IDENTITY_FORMAT.contains('\t'));
        assert_eq!(parse_pane_identity("40 node\n"), Some((Some(40), "node")));
        assert_eq!(parse_pane_identity("40 sh"), Some((Some(40), "sh")));
        assert_eq!(parse_pane_identity("bash"), Some((None, "bash")));
        assert_eq!(parse_pane_identity("  \n"), None);
    }

    #[test]
    fn missing_pane_process_resolves_to_none() {
        assert_eq!(resolve_agent(&[], 10, AGENTS), None);
    }
}
