# Workspace Rules

## Critical Constraints

- **Do Not Modify `mjlab` Source Code**:
  - Never change, edit, patch, or modify anything in the `mjlab` library or source code (e.g., in `site-packages/mjlab/` or any external `mjlab` repository).
  - All project-specific tasks, reward functions, observations, curricula, commands, environments, and runner configurations must reside strictly within this repository (under `src/tasks/`, `src/assets/`, or `scripts/`).
  - Always use `mjlab`'s public APIs and extension interfaces (e.g., `register_mjlab_task`, manager configs, custom MDP functions) without altering `mjlab` internals.

## General Guidelines

- **Sim-to-Real Consistency**: Maintain exact parity between observation terms, sensor definitions, and action spaces in `src/tasks/` and their C++ counterparts in `deploy/` and `simulate/`.
- **Clean Code**: Write like a principal engineer. Comments are short and say only what the code cannot show (the why). No long explanations, no restating the code.
- **Artifacts**: Prefer visuals (architecture/flow diagrams, tables, charts) over prose. Use paragraphs only when a visual cannot carry the point.
- **Documentation & Code Integrity**: Preserve all existing comments, docstrings, and configurations unless explicit changes are requested.
- **Reward Kernel vs. Command Limits**: For any `exp(-error_sq/std**2)`-style tracking reward paired with a clipped/P-controlled command (e.g. `arm_max_vel`, `arm_kp` in `wbc_arm`), keep `std` sized to the command's actual achievable error range, including its saturated value. A `std` too tight relative to the command's clip limit makes the reward saturate to ~0 from the first step, killing the gradient for the entire training run with no error in logs or code to flag it — check this whenever adding or tuning a new P-controlled tracking term.

## Token Conservation Mode

When operating in this workspace, strictly conserve tokens and resources:
- **Concise & Direct Responses**: Provide dense, actionable, high-signal responses. Eliminate conversational filler, redundant apologies, and generic boilerplate.
- **Minimal Necessary Executions**: Execute only the commands strictly required to inspect, test, or implement changes. Do not run exploratory or redundant commands, loop polls, or speculative steps.
- **Output Economy**: Avoid large stdout dumps. Truncate or filter command outputs with head/grep/tail when running commands.
- **Targeted Edits**: Make precise, scoped code modifications rather than whole-file rewrites.
- **Context Reuse**: Use `CONTEXT.md` as the architecture reference. Do not re-explore directories it already describes; read only the files the change touches.

### Minimal Verification Ladder

Prefix commands with `conda run -n unitree_rl_mjlab`, since the default shell has no `mjlab`. Run the lowest level that covers the change, and stop there. Go up a level only if the change needs it or a lower level fails. Never start full training runs (`start_train.sh`, `num-envs=4096`) for verification.

1. **Syntax** (any `.py` edit): `python -m py_compile <changed_files>`
2. **Registration/import** (task config, MDP term, or `__init__.py` edit):
   `python -c "import mjlab.tasks, src.tasks; from mjlab.tasks.registry import load_env_cfg; load_env_cfg('<Task-Id>')" 2>&1 | tail -5`
3. **Smoke rollout** (reward/observation/command logic or tensor-shape changes only):
   `python scripts/train.py <Task-Id> --env.scene.num-envs=16 --agent.max-iterations=2 --agent.logger=tensorboard 2>&1 | tail -20`
   Afterwards, delete the smoke run directory it created under `logs/rsl_rl/`.
4. **C++** (`deploy/` or `simulate/` edits only): rebuild only the affected target with `make 2>&1 | grep -E "error|warning: unused" | head -20`.

Rules:
- Check only the task IDs the change affects (e.g. a `wbc_arm` change does not need `wbc` checks).
- Check docs, comments, or `.sh` edits by reading them; don't run anything.
- Always pipe output through `tail`/`grep`/`head`. Report pass/fail in one line, plus the error excerpt if it failed.

