# Workspace Rules

## Critical Constraints

- **Do Not Modify `mjlab` Source Code**:
  - Never change, edit, patch, or modify anything in the `mjlab` library or source code (e.g., in `site-packages/mjlab/` or any external `mjlab` repository).
  - All project-specific tasks, reward functions, observations, curricula, commands, environments, and runner configurations must reside strictly within this repository (under `src/tasks/`, `src/assets/`, or `scripts/`).
  - Always use `mjlab`'s public APIs and extension interfaces (e.g., `register_mjlab_task`, manager configs, custom MDP functions) without altering `mjlab` internals.

## General Guidelines

- **Sim-to-Real Consistency**: Maintain exact parity between observation terms, sensor definitions, and action spaces in `src/tasks/` and their C++ counterparts in `deploy/` and `simulate/`.
- **Documentation & Code Integrity**: Preserve all existing comments, docstrings, and configurations unless explicit changes are requested.

## Token Conservation Mode

When operating in this workspace, strictly conserve tokens and resources:
- **Concise & Direct Responses**: Provide dense, actionable, high-signal responses. Eliminate conversational filler, redundant apologies, and generic boilerplate.
- **Minimal Necessary Executions**: Execute only the commands strictly required to inspect, test, or implement changes. Do not run exploratory or redundant commands, loop polls, or speculative steps.
- **Output Economy**: Avoid large stdout dumps. Truncate or filter command outputs with head/grep/tail when running commands.
- **Targeted Edits**: Make precise, scoped code modifications rather than whole-file rewrites.

