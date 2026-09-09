use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default worktree directory relative to project root
pub const DEFAULT_WORKTREE_DIR: &str = ".agtx/worktrees";

/// Create a new git worktree for a task from the detected default branch.
pub fn create_worktree(project_path: &Path, task_slug: &str) -> Result<PathBuf> {
    let base_branch = detect_main_branch(project_path)?;
    create_worktree_from_base(project_path, task_slug, &base_branch, DEFAULT_WORKTREE_DIR)
}

/// Create a new git worktree for a task from the specified base branch.
pub fn create_worktree_from_base(
    project_path: &Path,
    task_slug: &str,
    base_branch: &str,
    worktree_dir: &str,
) -> Result<PathBuf> {
    create_worktree_with_prefix(project_path, task_slug, base_branch, worktree_dir, "task")
}

/// Create a new git worktree for a task with a configurable branch prefix.
pub fn create_worktree_with_prefix(
    project_path: &Path,
    task_slug: &str,
    base_branch: &str,
    worktree_dir: &str,
    branch_prefix: &str,
) -> Result<PathBuf> {
    let worktree_path = project_path.join(worktree_dir).join(task_slug);

    // If worktree already exists and is valid, return it
    if worktree_path.exists() && worktree_path.join(".git").exists() {
        return Ok(worktree_path);
    }

    // Clean up any partial worktree
    if worktree_path.exists() {
        let _ = std::fs::remove_dir_all(&worktree_path);
    }

    // Ensure parent directory exists
    if let Some(parent) = worktree_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let base_branch = resolve_base_branch(project_path, base_branch)?;

    // Create worktree with a new branch based on the requested base branch
    let branch_name = format!("{}/{}", branch_prefix, task_slug);

    // First, try to delete the branch if it exists (from a previous failed attempt)
    let _ = Command::new("git")
        .current_dir(project_path)
        .args(["branch", "-D", &branch_name])
        .output();

    let output = Command::new("git")
        .current_dir(project_path)
        .args(["worktree", "add"])
        .arg(&worktree_path)
        .args(["-b", &branch_name, &base_branch])
        .output()
        .context("Failed to create git worktree")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to create worktree: {}", stderr);
    }

    Ok(worktree_path)
}

fn resolve_base_branch(project_path: &Path, base_branch: &str) -> Result<String> {
    let base_branch = base_branch.trim();
    if base_branch.is_empty() {
        return detect_main_branch(project_path);
    }

    let output = Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--verify", base_branch])
        .output()
        .context("Failed to verify configured base branch")?;

    if output.status.success() {
        Ok(base_branch.to_string())
    } else {
        anyhow::bail!("Configured base branch '{}' was not found", base_branch);
    }
}

/// Agent config directories that are always copied from project root to worktrees.
/// These contain commands, skills, and configuration that agents need.
pub const AGENT_CONFIG_DIRS: &[&str] = &[
    ".claude",
    ".gemini",
    ".codex",
    ".github/agents",
    ".config/opencode",
];

/// File names deliberately **not** copied out of [`AGENT_CONFIG_DIRS`] into a
/// task worktree, matched by file name at any depth.
///
/// This is a permission-boundary decision, not an accidental omission. Please
/// do not "restore" these to the copy for convenience.
///
/// `settings.local.json` is Claude Code's machine-local settings file. It
/// accumulates a developer's interactive "always allow" approvals and is
/// normally untracked and gitignored: it is personal approval history for a
/// human working in the project root, with no review and no provenance.
///
/// A task worktree is a different setting entirely. Agents there run
/// unattended, and their authority is meant to come from exactly one reviewed,
/// version-controlled source -- the resolved role policy in
/// `.agtx/workflow.toml`, which agtx hands to the agent explicitly (for Claude,
/// as `--allowed-tools` under `--permission-mode dontAsk`). Claude merges a
/// present `settings.local.json` with those flags, so copying a personal
/// `permissions.allow` block into the worktree lets ambient, unreviewed state
/// take effect on equal footing with the policy, where it can:
///
///   * **expand** a role beyond `.agtx/workflow.toml` -- a stray
///     `Bash(git push *)` approved once at the project root silently grants an
///     autonomous agent an authority the workflow deliberately withholds; and
///   * **contradict** the declared policy -- the role's entry becomes only part
///     of what the agent may do, so what actually ran can no longer be
///     reconstructed from the file that is supposed to govern it.
///
/// Either way the effective permissions of an unattended agent stop being
/// reviewable and start depending on whichever prompts someone happened to
/// approve on that machine. Excluding the file keeps `.agtx/workflow.toml` the
/// single authority. agtx still writes the worktree's own
/// `settings.local.json` for the settings it genuinely owns (MCP pre-trust, the
/// bypass-dialog preflight, and its hooks); it just never inherits a personal
/// allowlist. See `write_skills_to_worktree`, which drops an inherited
/// `permissions` block if one reaches a worktree by some other route.
pub const AGENT_CONFIG_SKIP_FILES: &[&str] = &["settings.local.json"];

/// Output from a shell script run inside a worktree.
#[derive(Debug)]
pub(crate) struct ScriptOutput {
    pub status: std::process::ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

/// Run a shell script inside a worktree, capturing stdout/stderr.
pub(crate) fn run_worktree_script(
    script: &str,
    worktree_path: &Path,
    envs: &[(String, String)],
) -> Result<ScriptOutput> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(script)
        .current_dir(worktree_path)
        .envs(envs.iter().map(|(k, v)| (k, v)))
        .output()
        .with_context(|| format!("Failed to run script: {}", script))?;

    Ok(ScriptOutput {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

/// Initialize a worktree by copying agent config dirs, user-specified files, and running an init script.
///
/// Returns a Vec of warning messages for any issues encountered.
/// Does not fail fatally — errors are collected and returned for the caller to display.
pub fn initialize_worktree(
    project_path: &Path,
    worktree_path: &Path,
    copy_files: Option<&str>,
    init_script: Option<&str>,
    copy_dirs: &[String],
) -> Vec<String> {
    let mut warnings = Vec::new();

    // Always copy agent config directories, minus AGENT_CONFIG_SKIP_FILES. A
    // worktree is already a Git checkout: never overwrite one of its tracked
    // files with the project root's version. That would turn unrelated agent
    // configuration (and sometimes only its line endings) into task changes
    // that the integration step could accidentally stage and merge.
    let tracked_worktree_paths = tracked_worktree_paths(worktree_path);
    for dir_name in AGENT_CONFIG_DIRS {
        let src = project_path.join(dir_name);
        if src.is_dir() {
            let dst = worktree_path.join(dir_name);
            if let Err(e) =
                copy_agent_config_dir(&src, &dst, worktree_path, &tracked_worktree_paths)
            {
                warnings.push(format!("Failed to copy '{}' to worktree: {}", dir_name, e));
            }
        }
    }

    // Copy plugin-specific extra directories
    for dir_name in copy_dirs {
        let src = project_path.join(dir_name);
        if src.is_dir() {
            // Validate path stays within project root
            if let (Ok(canon_proj), Ok(canon_src)) =
                (project_path.canonicalize(), src.canonicalize())
            {
                if !canon_src.starts_with(&canon_proj) {
                    warnings.push(format!(
                        "copy_dirs: '{}' resolves outside project root, skipping (path traversal blocked)",
                        dir_name
                    ));
                    continue;
                }
            }
            let dst = worktree_path.join(dir_name);
            if let Err(e) = copy_dir_recursive(&src, &dst) {
                warnings.push(format!("Failed to copy '{}' to worktree: {}", dir_name, e));
            }
        }
    }

    // Copy user-specified files/directories
    if let Some(files_str) = copy_files {
        // Pre-compute canonical project path for traversal checks
        let canonical_project = project_path.canonicalize().ok();

        for entry in files_str.split(',') {
            let file_name = entry.trim();
            if file_name.is_empty() {
                continue;
            }

            // Reject obvious traversal patterns before touching the filesystem
            if file_name.contains("..") {
                warnings.push(format!(
                    "copy_files: '{}' contains '..', skipping (path traversal blocked)",
                    file_name
                ));
                continue;
            }

            let src = project_path.join(file_name);
            let dst = worktree_path.join(file_name);

            if !src.exists() {
                warnings.push(format!(
                    "copy_files: '{}' not found in project root, skipping",
                    file_name
                ));
                continue;
            }

            // Validate resolved path stays within project root
            if let Some(ref canon_proj) = canonical_project {
                if let Ok(canon_src) = src.canonicalize() {
                    if !canon_src.starts_with(canon_proj) {
                        warnings.push(format!(
                            "copy_files: '{}' resolves outside project root, skipping (path traversal blocked)",
                            file_name
                        ));
                        continue;
                    }
                }
            }

            if src.is_dir() {
                if let Err(e) = copy_dir_recursive(&src, &dst) {
                    warnings.push(format!(
                        "Failed to copy directory '{}' to worktree: {}",
                        file_name, e
                    ));
                }
            } else {
                if let Some(parent) = dst.parent() {
                    if !parent.exists() {
                        if let Err(e) = std::fs::create_dir_all(parent) {
                            warnings.push(format!(
                                "Failed to create directory for '{}': {}",
                                file_name, e
                            ));
                            continue;
                        }
                    }
                }
                if let Err(e) = std::fs::copy(&src, &dst) {
                    warnings.push(format!("Failed to copy '{}' to worktree: {}", file_name, e));
                }
            }
        }
    }

    if let Some(script) = init_script {
        let script = script.trim();
        if !script.is_empty() {
            tracing::info!(
                script = script,
                worktree = %worktree_path.display(),
                "Executing project init_script"
            );
            match run_worktree_script(script, worktree_path, &[]) {
                Ok(result) => {
                    if !result.status.success() {
                        warnings.push(format!(
                            "init_script exited with {}: {}",
                            result.status,
                            result.stderr.trim()
                        ));
                    }
                }
                Err(e) => warnings.push(format!("Failed to run init_script: {}", e)),
            }
        }
    }

    warnings
}

/// Copy an agent config directory into a worktree without overwriting tracked
/// checkout files, and skipping [`AGENT_CONFIG_SKIP_FILES`] at every depth.
///
/// Deliberately separate from [`copy_dir_recursive`], which stays a
/// general-purpose helper used for plugin and user-specified directories where
/// no permission boundary applies. The exclusion is applied here, at the copy
/// itself, rather than by deleting the file afterwards: a permission boundary
/// should never depend on a cleanup step that a later error path could skip.
fn copy_agent_config_dir(
    src: &Path,
    dst: &Path,
    worktree_path: &Path,
    tracked_worktree_paths: &HashSet<String>,
) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let name = entry.file_name();
        let dst_path = dst.join(&name);
        if src_path.is_dir() {
            copy_agent_config_dir(&src_path, &dst_path, worktree_path, tracked_worktree_paths)?;
        } else if !AGENT_CONFIG_SKIP_FILES
            .iter()
            .any(|skip| name.as_os_str() == *skip)
            && !is_tracked_worktree_path(worktree_path, &dst_path, tracked_worktree_paths)
        {
            std::fs::copy(&src_path, dst_path)?;
        }
    }
    Ok(())
}

fn tracked_worktree_paths(worktree_path: &Path) -> HashSet<String> {
    let Ok(output) = Command::new("git")
        .current_dir(worktree_path)
        .args(["ls-files", "-z"])
        .output()
    else {
        return HashSet::new();
    };
    if !output.status.success() {
        return HashSet::new();
    }

    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8_lossy(path).replace('\\', "/"))
        .collect()
}

fn is_tracked_worktree_path(
    worktree_path: &Path,
    destination: &Path,
    tracked_worktree_paths: &HashSet<String>,
) -> bool {
    destination
        .strip_prefix(worktree_path)
        .ok()
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .is_some_and(|path| tracked_worktree_paths.contains(&path))
}

/// Recursively copy a directory and its contents.
pub fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Detect the main branch name (main or master)
pub fn detect_main_branch(project_path: &Path) -> Result<String> {
    // Check if 'main' exists
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--verify", "main"])
        .output()
        .context("Failed to check for main branch")?;

    if output.status.success() {
        return Ok("main".to_string());
    }

    // Check if 'master' exists
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--verify", "master"])
        .output()
        .context("Failed to check for master branch")?;

    if output.status.success() {
        return Ok("master".to_string());
    }

    // Fallback: get the current branch
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .context("Failed to get current branch")?;

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// True when `path` is a repository's *main* working tree rather than a linked
/// worktree.
///
/// This is the shape `skip_worktree` produces: a task's `worktree_path` is the
/// user's own checkout, and every cleanup path then asks for that to be removed.
/// git already refuses ("is a main working tree"), so today the protection is
/// inherited rather than intended — and `fs::rename`, which a background trash
/// would use instead, has no such concept.
///
/// Two conditions, and both are needed:
///
/// - `--git-dir` == `--git-common-dir`. A linked worktree's git dir is
///   `{repo}/.git/worktrees/{name}` while its common dir is `{repo}/.git`.
/// - `--show-toplevel` is `path` itself.
///
/// The second is not redundant. The default `worktree_dir` is `.agtx/worktrees`,
/// *inside* the project, so a worktree there that has lost its `.git` link makes
/// git walk up to the main repository and report the first condition as true.
/// Without the toplevel check, exactly the half-deleted worktrees that most need
/// removing would be refused as if they were the user's checkout.
pub fn is_main_working_tree(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let rev_parse = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .current_dir(path)
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
    };

    let common = rev_parse(&["rev-parse", "--path-format=absolute", "--git-common-dir"]);
    let dir = rev_parse(&["rev-parse", "--path-format=absolute", "--git-dir"]);
    match (common, dir) {
        (Some(common), Some(dir)) if common == dir => {}
        // A linked worktree, not a repository at all, or a git too old for
        // --path-format. Not provably a main working tree, so this does not
        // block the removal; the caller's project-root comparison is the second
        // line of defence.
        _ => return false,
    }

    let Some(toplevel) = rev_parse(&["rev-parse", "--show-toplevel"]) else {
        return false;
    };
    match (Path::new(&toplevel).canonicalize(), path.canonicalize()) {
        (Ok(top), Ok(this)) => top == this,
        _ => false,
    }
}

/// Get the worktree path for a task
pub fn worktree_path(project_path: &Path, task_id: &str, worktree_dir: &str) -> PathBuf {
    project_path.join(worktree_dir).join(task_id)
}

/// Get the worktree path for a task using a custom worktree directory
pub fn worktree_path_with_dir(project_path: &Path, task_id: &str, worktree_dir: &str) -> PathBuf {
    worktree_path(project_path, task_id, worktree_dir)
}

/// Check if a worktree exists for a task
pub fn worktree_exists(project_path: &Path, task_id: &str) -> bool {
    worktree_path(project_path, task_id, DEFAULT_WORKTREE_DIR).exists()
}

/// Check if a worktree exists for a task using a custom worktree directory
pub fn worktree_exists_with_dir(project_path: &Path, task_id: &str, worktree_dir: &str) -> bool {
    worktree_path_with_dir(project_path, task_id, worktree_dir).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn run_git(path: &Path, args: &[&str]) {
        let output = Command::new("git")
            .current_dir(path)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn test_run_worktree_script_captures_output_and_env() {
        let temp_dir = TempDir::new().unwrap();
        let envs = vec![("AGTX_TASK_ID".to_string(), "task-123".to_string())];

        let output = run_worktree_script("echo $AGTX_TASK_ID", temp_dir.path(), &envs).unwrap();

        assert!(output.status.success());
        assert_eq!(output.stdout.trim(), "task-123");
    }

    #[test]
    fn test_run_worktree_script_nonzero_exit() {
        let temp_dir = TempDir::new().unwrap();

        let output = run_worktree_script("exit 42", temp_dir.path(), &[]).unwrap();

        assert!(!output.status.success());
    }

    #[test]
    fn initialize_worktree_does_not_overwrite_tracked_agent_config() {
        let temp_dir = TempDir::new().unwrap();
        let project = temp_dir.path().join("project");
        let worktree = temp_dir.path().join("task-worktree");
        std::fs::create_dir_all(project.join(".claude")).unwrap();
        std::fs::write(project.join(".claude/agent.md"), "task branch version\n").unwrap();

        std::fs::create_dir_all(&project).unwrap();
        run_git(&project, &["init"]);
        run_git(&project, &["config", "user.email", "test@example.invalid"]);
        run_git(&project, &["config", "user.name", "AGTX test"]);
        run_git(&project, &["add", "."]);
        run_git(&project, &["commit", "-m", "base"]);
        run_git(
            &project,
            &[
                "worktree",
                "add",
                "-b",
                "task/agent-config-copy",
                worktree.to_str().unwrap(),
            ],
        );
        let task_branch_bytes = std::fs::read(worktree.join(".claude/agent.md")).unwrap();

        // This models a newer or differently-normalized config in the project
        // root. It must not replace the task branch's tracked checkout file.
        std::fs::write(project.join(".claude/agent.md"), "project root version\r\n").unwrap();
        std::fs::create_dir_all(project.join(".claude/commands")).unwrap();
        std::fs::write(project.join(".claude/commands/local.md"), "local helper\n").unwrap();

        let warnings = initialize_worktree(&project, &worktree, None, None, &[]);

        assert!(warnings.is_empty(), "warnings: {warnings:?}");
        assert_eq!(
            std::fs::read(worktree.join(".claude/agent.md")).unwrap(),
            task_branch_bytes,
            "initialization must preserve the task checkout byte-for-byte"
        );
        assert_eq!(
            std::fs::read_to_string(worktree.join(".claude/commands/local.md")).unwrap(),
            "local helper\n"
        );
        let status = Command::new("git")
            .current_dir(&worktree)
            .args(["status", "--short", "--", ".claude/agent.md"])
            .output()
            .unwrap();
        assert!(status.stdout.is_empty(), "tracked config was modified");
    }
}
