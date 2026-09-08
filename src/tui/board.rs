use crate::db::{DependencyState, Task, TaskStatus};
use std::collections::HashMap;

/// A column on the board. Six lanes over five statuses: Backlog splits by
/// dependency state so "ready to pick up" is visible without opening a card.
///
/// A lane is a projection, never a stored value — nothing writes a lane to the
/// database, and no transition targets one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayLane {
    Backlog,
    Ready,
    Planning,
    Running,
    Review,
    Done,
}

impl DisplayLane {
    /// The lanes in board order.
    pub fn lanes() -> &'static [DisplayLane] {
        &[
            DisplayLane::Backlog,
            DisplayLane::Ready,
            DisplayLane::Planning,
            DisplayLane::Running,
            DisplayLane::Review,
            DisplayLane::Done,
        ]
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            // Backlog keeps its research meaning and its name; the split only
            // lifts out the cards whose dependencies no longer stand in the way.
            DisplayLane::Backlog => "backlog/research",
            DisplayLane::Ready => "ready",
            DisplayLane::Planning => "planning",
            DisplayLane::Running => "running",
            DisplayLane::Review => "review",
            DisplayLane::Done => "done",
        }
    }

    /// The status a task in this lane holds. Both Backlog lanes map back to
    /// `TaskStatus::Backlog`.
    pub fn status(&self) -> TaskStatus {
        match self {
            DisplayLane::Backlog | DisplayLane::Ready => TaskStatus::Backlog,
            DisplayLane::Planning => TaskStatus::Planning,
            DisplayLane::Running => TaskStatus::Running,
            DisplayLane::Review => TaskStatus::Review,
            DisplayLane::Done => TaskStatus::Done,
        }
    }
}

/// Where a task belongs on the board, given its lifecycle status and whether
/// its dependencies let it be picked up.
pub fn display_lane(task: &Task, deps: &DependencyState) -> DisplayLane {
    match task.status {
        TaskStatus::Backlog if deps.is_ready() => DisplayLane::Ready,
        TaskStatus::Backlog => DisplayLane::Backlog,
        TaskStatus::Planning => DisplayLane::Planning,
        TaskStatus::Running => DisplayLane::Running,
        TaskStatus::Review => DisplayLane::Review,
        TaskStatus::Done => DisplayLane::Done,
    }
}

/// State for the kanban board view
#[derive(Debug)]
pub struct BoardState {
    pub tasks: Vec<Task>,
    /// Dependency state per task id, refreshed alongside `tasks`. Only tasks
    /// with references are stored; an absent entry means Ready, which is the
    /// right answer for a task that depends on nothing.
    pub dep_states: HashMap<String, DependencyState>,
    pub selected_column: usize,
    pub selected_row: usize,
}

impl BoardState {
    pub fn new() -> Self {
        Self {
            tasks: vec![],
            dep_states: HashMap::new(),
            selected_column: 0,
            selected_row: 0,
        }
    }

    /// The cached dependency state for a task, defaulting to Ready.
    pub fn dep_state(&self, task_id: &str) -> &DependencyState {
        static READY: DependencyState = DependencyState::Ready;
        self.dep_states.get(task_id).unwrap_or(&READY)
    }

    /// The lane a task currently renders in.
    pub fn lane_of(&self, task: &Task) -> DisplayLane {
        display_lane(task, self.dep_state(&task.id))
    }

    /// The board column index a task currently renders in.
    pub fn column_of(&self, task: &Task) -> usize {
        let lane = self.lane_of(task);
        DisplayLane::lanes()
            .iter()
            .position(|l| *l == lane)
            .unwrap_or(0)
    }

    /// Get tasks in a specific column
    pub fn tasks_in_column(&self, column: usize) -> Vec<&Task> {
        let lane = DisplayLane::lanes().get(column).copied();
        match lane {
            Some(l) => self.tasks.iter().filter(|t| self.lane_of(t) == l).collect(),
            None => vec![],
        }
    }

    /// Get the currently selected task (immutable)
    pub fn selected_task(&self) -> Option<&Task> {
        let column_tasks = self.tasks_in_column(self.selected_column);
        column_tasks.get(self.selected_row).copied()
    }

    /// Get the currently selected task (mutable)
    pub fn selected_task_mut(&mut self) -> Option<&mut Task> {
        let lane = DisplayLane::lanes().get(self.selected_column).copied()?;

        let matching_indices: Vec<usize> = self
            .tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| self.lane_of(t) == lane)
            .map(|(i, _)| i)
            .collect();

        matching_indices
            .get(self.selected_row)
            .and_then(|&idx| self.tasks.get_mut(idx))
    }

    /// Move selection left
    pub fn move_left(&mut self) {
        if self.selected_column > 0 {
            self.selected_column -= 1;
            self.clamp_row();
        }
    }

    /// Move selection right
    pub fn move_right(&mut self) {
        if self.selected_column < DisplayLane::lanes().len() - 1 {
            self.selected_column += 1;
            self.clamp_row();
        }
    }

    /// Move selection up
    pub fn move_up(&mut self) {
        if self.selected_row > 0 {
            self.selected_row -= 1;
        }
    }

    /// Move selection down
    pub fn move_down(&mut self) {
        let column_count = self.tasks_in_column(self.selected_column).len();
        if self.selected_row < column_count.saturating_sub(1) {
            self.selected_row += 1;
        }
    }

    /// Ensure selected_row is valid for current column
    fn clamp_row(&mut self) {
        let column_count = self.tasks_in_column(self.selected_column).len();
        if column_count == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= column_count {
            self.selected_row = column_count - 1;
        }
    }
}

impl Default for BoardState {
    fn default() -> Self {
        Self::new()
    }
}
