//! Bookkeeping for the OpenCode permission profile agtx writes into a task
//! worktree's `opencode.json`, and its removal before a task commit.
//!
//! OpenCode has no CLI surface for per-invocation permissions, so before each
//! OpenCode role launch agtx merges that role's generated rules (and model)
//! into the worktree's `opencode.json` (`write_opencode_permission_profile` in
//! `tui/app.rs`). That file is usually project-tracked. Without the strip below,
//! the task's final `git add -A` committed the generated rules and merged them
//! into the target branch, where the next task's worktree saw them as
//! project-authored rules: they were never removed again, every task appended
//! another set (one real project grew to 242 entries, 38 unique), and one
//! role's grants (e.g. the implementer's `edit api/**`) silently applied to
//! every other OpenCode role (e.g. a read-only planner).
//!
//! The sidecar records exactly what agtx inserted, plus the file's original
//! text from before agtx first touched it, so [`strip_opencode_permission_profile`]
//! can restore the file byte-for-byte when nothing else in it changed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What agtx inserted into the worktree's `opencode.json`. Gitignored runtime
/// state under `.agtx/state/`, one file per worktree.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProfileSidecar {
    /// The generated rules agtx actually inserted last time (one entry per
    /// insertion; rules already present in the file are not inserted).
    pub rules: Vec<Value>,
    /// `opencode.json` exactly as it was before agtx first wrote a profile into
    /// this worktree; `None` when the file did not exist or predates this field.
    #[serde(default)]
    pub original_text: Option<String>,
}

pub fn sidecar_path(worktree: &Path) -> PathBuf {
    worktree
        .join(".agtx")
        .join("state")
        .join("opencode-permissions.json")
}

/// Read the sidecar, accepting the legacy format (a bare array of rules).
pub fn read_sidecar(worktree: &Path) -> Option<ProfileSidecar> {
    let text = std::fs::read_to_string(sidecar_path(worktree)).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    if let Some(rules) = value.as_array() {
        return Some(ProfileSidecar {
            rules: rules.clone(),
            original_text: None,
        });
    }
    serde_json::from_value(value).ok()
}

pub fn write_sidecar(worktree: &Path, sidecar: &ProfileSidecar) {
    let path = sidecar_path(worktree);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(
        &path,
        serde_json::to_string_pretty(sidecar).unwrap_or_default(),
    );
}

/// Remove one occurrence per previously generated rule. Treated as a multiset:
/// agtx owns precisely one insertion per recorded item, so a project-authored
/// rule that happens to equal a generated one survives.
pub fn remove_generated(permissions: &mut Vec<Value>, previous: &[Value]) {
    let mut remaining = previous.to_vec();
    permissions.retain(|entry| {
        if let Some(index) = remaining.iter().position(|prior| prior == entry) {
            remaining.remove(index);
            false
        } else {
            true
        }
    });
}

/// Undo agtx's permission profile in `worktree` before its changes are staged
/// for a commit: remove exactly the rules agtx inserted, restore the original
/// `model`, and delete the sidecar. When the result equals the original file,
/// the original bytes are written back so the commit carries no diff at all.
/// Changes the task itself made to `opencode.json` are preserved.
///
/// Best-effort like the writer: any read/parse failure leaves the file alone.
/// Returns true when `opencode.json` was rewritten.
pub fn strip_opencode_permission_profile(worktree: &Path) -> bool {
    let Some(sidecar) = read_sidecar(worktree) else {
        return false;
    };
    let cfg_path = worktree.join("opencode.json");
    let mut changed = false;
    if let Some(mut root) = std::fs::read_to_string(&cfg_path)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .filter(Value::is_object)
    {
        let original: Option<Value> = sidecar
            .original_text
            .as_deref()
            .and_then(|text| serde_json::from_str(text).ok());
        if let Some(permissions) = root.get_mut("permissions").and_then(Value::as_array_mut) {
            remove_generated(permissions, &sidecar.rules);
        }
        // `permissions` was introduced by agtx when the original had none.
        if let Some(original) = &original {
            if original.get("permissions").is_none()
                && root["permissions"].as_array().is_some_and(Vec::is_empty)
            {
                root.as_object_mut().unwrap().remove("permissions");
            }
            match original.get("model") {
                Some(model) => root["model"] = model.clone(),
                None => {
                    root.as_object_mut().unwrap().remove("model");
                }
            }
        }
        let text = match (&original, &sidecar.original_text) {
            (Some(original), Some(original_text)) if *original == root => original_text.clone(),
            _ => serde_json::to_string_pretty(&root).unwrap_or_default(),
        };
        if std::fs::write(&cfg_path, text).is_ok() {
            changed = true;
        }
    }
    let _ = std::fs::remove_file(sidecar_path(worktree));
    changed
}
