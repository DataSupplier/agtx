use agtx::db::{DependencyState, Task, TaskStatus};
use agtx::tui::board::{display_lane, BoardState, DisplayLane};

fn create_test_task(title: &str, status: TaskStatus) -> Task {
    let mut task = Task::new(title, "claude", "test-project");
    task.status = status;
    task
}

/// A Backlog task whose dependency has not reached Review/Done, registered in
/// the dependency cache on the board the way `refresh_tasks` registers it.
fn add_blocked_task(board: &mut BoardState, title: &str) -> String {
    let mut task = create_test_task(title, TaskStatus::Backlog);
    task.referenced_tasks = Some("dep-1".to_string());
    let id = task.id.clone();
    board.dep_states.insert(
        id.clone(),
        DependencyState::Blocked(vec!["dep-1".to_string()]),
    );
    board.tasks.push(task);
    id
}

// === BoardState Tests ===

#[test]
fn test_board_state_new() {
    let board = BoardState::new();

    assert!(board.tasks.is_empty());
    assert!(board.dep_states.is_empty());
    assert_eq!(board.selected_column, 0);
    assert_eq!(board.selected_row, 0);
}

#[test]
fn test_board_state_default() {
    let board = BoardState::default();

    assert!(board.tasks.is_empty());
    assert_eq!(board.selected_column, 0);
    assert_eq!(board.selected_row, 0);
}

#[test]
fn test_tasks_in_column_empty() {
    let board = BoardState::new();

    for i in 0..DisplayLane::lanes().len() {
        assert!(board.tasks_in_column(i).is_empty());
    }
}

#[test]
fn test_tasks_in_column_with_tasks() {
    let mut board = BoardState::new();
    board.tasks = vec![
        create_test_task("Task 1", TaskStatus::Backlog),
        create_test_task("Task 2", TaskStatus::Backlog),
        create_test_task("Task 3", TaskStatus::Running),
        create_test_task("Task 4", TaskStatus::Done),
    ];
    add_blocked_task(&mut board, "Task 5");

    assert_eq!(board.tasks_in_column(0).len(), 1); // Backlog, blocked
    assert_eq!(board.tasks_in_column(1).len(), 2); // Ready
    assert_eq!(board.tasks_in_column(2).len(), 0); // Planning
    assert_eq!(board.tasks_in_column(3).len(), 1); // Running
    assert_eq!(board.tasks_in_column(4).len(), 0); // Review
    assert_eq!(board.tasks_in_column(5).len(), 1); // Done
}

#[test]
fn test_tasks_in_column_invalid_column() {
    let board = BoardState::new();

    assert!(board.tasks_in_column(99).is_empty());
}

#[test]
fn test_selected_task_empty_board() {
    let board = BoardState::new();

    assert!(board.selected_task().is_none());
}

#[test]
fn test_selected_task_with_tasks() {
    let mut board = BoardState::new();
    board.tasks = vec![
        create_test_task("Task 1", TaskStatus::Backlog),
        create_test_task("Task 2", TaskStatus::Backlog),
    ];
    board.selected_column = 1; // Ready: neither task has dependencies
    board.selected_row = 1;

    let task = board.selected_task().unwrap();
    assert_eq!(task.title, "Task 2");
}

#[test]
fn test_move_left() {
    let mut board = BoardState::new();
    board.selected_column = 2;

    board.move_left();
    assert_eq!(board.selected_column, 1);

    board.move_left();
    assert_eq!(board.selected_column, 0);

    // Should not go below 0
    board.move_left();
    assert_eq!(board.selected_column, 0);
}

#[test]
fn test_move_right() {
    let mut board = BoardState::new();
    board.selected_column = 0;

    for expected in 1..DisplayLane::lanes().len() {
        board.move_right();
        assert_eq!(board.selected_column, expected);
    }

    // Should not go beyond last column
    board.move_right();
    assert_eq!(board.selected_column, DisplayLane::lanes().len() - 1);
}

#[test]
fn test_move_up() {
    let mut board = BoardState::new();
    board.tasks = vec![
        create_test_task("Task 1", TaskStatus::Backlog),
        create_test_task("Task 2", TaskStatus::Backlog),
        create_test_task("Task 3", TaskStatus::Backlog),
    ];
    board.selected_column = 1; // Ready
    board.selected_row = 2;

    board.move_up();
    assert_eq!(board.selected_row, 1);

    board.move_up();
    assert_eq!(board.selected_row, 0);

    // Should not go below 0
    board.move_up();
    assert_eq!(board.selected_row, 0);
}

#[test]
fn test_move_down() {
    let mut board = BoardState::new();
    board.tasks = vec![
        create_test_task("Task 1", TaskStatus::Backlog),
        create_test_task("Task 2", TaskStatus::Backlog),
        create_test_task("Task 3", TaskStatus::Backlog),
    ];
    board.selected_column = 1; // Ready
    board.selected_row = 0;

    board.move_down();
    assert_eq!(board.selected_row, 1);

    board.move_down();
    assert_eq!(board.selected_row, 2);

    // Should not go beyond last task
    board.move_down();
    assert_eq!(board.selected_row, 2);
}

#[test]
fn test_move_down_empty_column() {
    let mut board = BoardState::new();
    board.selected_row = 0;

    // Moving down in empty column should stay at 0
    board.move_down();
    assert_eq!(board.selected_row, 0);
}

#[test]
fn test_move_left_clamps_row() {
    let mut board = BoardState::new();
    board.tasks = vec![
        create_test_task("Task 1", TaskStatus::Backlog),
        create_test_task("Task 2", TaskStatus::Backlog),
        create_test_task("Task 3", TaskStatus::Backlog),
        create_test_task("Task 4", TaskStatus::Planning), // Only 1 task in Planning
    ];
    board.selected_column = 1; // Ready, with 3 tasks
    board.selected_row = 2; // Last task in Ready

    board.move_right(); // Move to Planning with 1 task

    // Row should be clamped to 0 (only task in Planning)
    assert_eq!(board.selected_column, 2);
    assert_eq!(board.selected_row, 0);
}

#[test]
fn test_move_to_empty_column_clamps_row() {
    let mut board = BoardState::new();
    board.tasks = vec![
        create_test_task("Task 1", TaskStatus::Backlog),
        create_test_task("Task 2", TaskStatus::Backlog),
        // Planning column is empty
    ];
    board.selected_column = 1; // Ready
    board.selected_row = 1;

    board.move_right(); // Move to empty Planning column

    assert_eq!(board.selected_column, 2);
    assert_eq!(board.selected_row, 0);
}

#[test]
fn test_selected_task_mut() {
    let mut board = BoardState::new();
    board.tasks = vec![
        create_test_task("Task 1", TaskStatus::Backlog),
        create_test_task("Task 2", TaskStatus::Backlog),
    ];
    board.selected_column = 1; // Ready
    board.selected_row = 0;

    if let Some(task) = board.selected_task_mut() {
        task.title = "Modified Task".to_string();
    }

    assert_eq!(board.tasks[0].title, "Modified Task");
}

#[test]
fn test_selected_task_mut_skips_other_lane() {
    let mut board = BoardState::new();
    add_blocked_task(&mut board, "Blocked");
    board
        .tasks
        .push(create_test_task("Ready", TaskStatus::Backlog));
    board.selected_column = 1; // Ready
    board.selected_row = 0;

    if let Some(task) = board.selected_task_mut() {
        task.title = "Modified Task".to_string();
    }

    // The blocked card sits earlier in `tasks` but in the other lane.
    assert_eq!(board.tasks[0].title, "Blocked");
    assert_eq!(board.tasks[1].title, "Modified Task");
}

// === DisplayLane Tests ===

#[test]
fn test_display_lane_backlog_splits_on_dependency_state() {
    let task = create_test_task("Task", TaskStatus::Backlog);

    assert_eq!(
        display_lane(&task, &DependencyState::Ready),
        DisplayLane::Ready
    );
    assert_eq!(
        display_lane(&task, &DependencyState::Missing(vec!["gone".to_string()])),
        DisplayLane::Ready
    );
    assert_eq!(
        display_lane(&task, &DependencyState::Blocked(vec!["a".to_string()])),
        DisplayLane::Backlog
    );
}

#[test]
fn test_display_lane_follows_status_past_backlog() {
    // Past Backlog the lane is the status: dependency state cannot move a task
    // that has already been picked up.
    let blocked = DependencyState::Blocked(vec!["a".to_string()]);
    for (status, lane) in [
        (TaskStatus::Planning, DisplayLane::Planning),
        (TaskStatus::Running, DisplayLane::Running),
        (TaskStatus::Review, DisplayLane::Review),
        (TaskStatus::Done, DisplayLane::Done),
    ] {
        let task = create_test_task("Task", status);
        assert_eq!(display_lane(&task, &DependencyState::Ready), lane);
        assert_eq!(display_lane(&task, &blocked), lane);
    }
}

#[test]
fn test_display_lane_status_maps_both_backlog_lanes() {
    assert_eq!(DisplayLane::Backlog.status(), TaskStatus::Backlog);
    assert_eq!(DisplayLane::Ready.status(), TaskStatus::Backlog);
    assert_eq!(DisplayLane::Planning.status(), TaskStatus::Planning);
    assert_eq!(DisplayLane::Running.status(), TaskStatus::Running);
    assert_eq!(DisplayLane::Review.status(), TaskStatus::Review);
    assert_eq!(DisplayLane::Done.status(), TaskStatus::Done);
}

#[test]
fn test_lanes_are_ordered_backlog_ready_then_workflow() {
    assert_eq!(
        DisplayLane::lanes(),
        &[
            DisplayLane::Backlog,
            DisplayLane::Ready,
            DisplayLane::Planning,
            DisplayLane::Running,
            DisplayLane::Review,
            DisplayLane::Done,
        ]
    );
}

#[test]
fn test_dep_state_defaults_to_ready() {
    let board = BoardState::new();

    // A task with no dependencies is never cached, and a task with no cache
    // entry must not read as blocked.
    assert_eq!(board.dep_state("unknown-id"), &DependencyState::Ready);
}

#[test]
fn test_column_of_moves_card_when_dependency_clears() {
    let mut board = BoardState::new();
    let id = add_blocked_task(&mut board, "B");

    assert_eq!(board.column_of(&board.tasks[0]), 0);

    // What a refresh does once the dependency reaches Review: recompute the
    // dependency state, and the card is in Ready without its status changing.
    board.dep_states.insert(id, DependencyState::Ready);

    assert_eq!(board.column_of(&board.tasks[0]), 1);
    assert_eq!(board.tasks[0].status, TaskStatus::Backlog);
}
