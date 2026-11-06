"""Deterministic WBC-ARM checkpoint evaluation, optionally watching a run for new checkpoints.

Results are appended to <run_dir>/eval.csv; already-evaluated checkpoints are skipped.
"""

import csv
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK_BY_EXPERIMENT = {
  "g1_wbc": "Unitree-G1-Flat-WBC-ARM",
  "g1_wbc_split": "Unitree-G1-Flat-WBC-ARM-Split",
  "g1_wbc_supermimic": "Unitree-G1-Flat-WBC-ARM-SuperMimic",
}
FIELDS = (
  "iteration", "falls_per_env", "err_xy", "err_yaw", "err_vz", "err_height",
  "arm_err", "arm_err_frozen", "arm_closed_pct", "arm_pos_kernel", "arm_pos_kernel_frozen",
)


@dataclass
class Args:
  run_dir: Path | None = None
  """Run directory; defaults to the newest run across the known experiment folders."""
  task: str | None = None
  """Task id; inferred from the run's experiment folder when omitted."""
  watch: bool = True
  """Keep polling for new checkpoints; pass --no-watch for a single pass."""
  poll_s: float = 60.0
  num_envs: int = 64
  steps: int = 1000
  seed: int = 0
  device: str = "cuda:0"


def ckpt_iter(p: Path) -> int:
  return int(re.search(r"model_(\d+)\.pt", p.name).group(1))


def build(task: str, args: Args):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = args.num_envs
  # Play shrinks command ranges; evaluate on the training distribution.
  ranges = cfg.commands["twist"].ranges
  ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z = (-1.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)
  agent = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device), clip_actions=agent.clip_actions)
  runner = load_runner_cls(task)(env, asdict(agent), device=args.device)
  return env, runner


@torch.no_grad()
def evaluate(env, runner, ckpt: Path, args: Args) -> dict:
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  torch.manual_seed(args.seed)
  env.seed(args.seed)
  obs, _ = env.reset()

  u = env.unwrapped
  t = u.command_manager.get_term("twist")
  robot = u.scene["robot"]
  ids = t.arm_joint_ids
  default_arm = robot.data.default_joint_pos[:, ids]
  sums = dict.fromkeys(("xy", "yaw", "vz", "h", "arm", "frozen", "k", "kf"), 0.0)
  n_arm = n_cmd = 0
  falls = 0
  for _ in range(args.steps):
    obs, _, dones, _ = env.step(policy(obs))
    falls += int(dones.sum())
    cmd = t.command
    sums["xy"] += (cmd[:, :2] - t.lin_vel_heading()[:, :2]).norm(dim=-1).sum().item()
    sums["yaw"] += (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs().sum().item()
    sums["vz"] += (cmd[:, 3] - robot.data.root_link_lin_vel_w[:, 2]).abs().sum().item()
    sums["h"] += (t.height_target - t.base_height()).abs().sum().item()
    err = (t.arm_q_target - robot.data.joint_pos[:, ids]).abs()
    frozen = (t.arm_q_target - default_arm).abs()
    moving = frozen > 0.05
    cmd_env = moving.any(-1)
    sums["arm"] += err[moving].sum().item()
    sums["frozen"] += frozen[moving].sum().item()
    sums["k"] += torch.exp(-err.square().mean(-1) / 0.25)[cmd_env].sum().item()
    sums["kf"] += torch.exp(-frozen.square().mean(-1) / 0.25)[cmd_env].sum().item()
    n_arm += int(moving.sum())
    n_cmd += int(cmd_env.sum())

  n = args.num_envs * args.steps
  arm, fr = sums["arm"] / max(n_arm, 1), sums["frozen"] / max(n_arm, 1)
  return {
    "iteration": ckpt_iter(ckpt),
    "falls_per_env": falls / args.num_envs,
    "err_xy": sums["xy"] / n,
    "err_yaw": sums["yaw"] / n,
    "err_vz": sums["vz"] / n,
    "err_height": sums["h"] / n,
    "arm_err": arm,
    "arm_err_frozen": fr,
    "arm_closed_pct": 100.0 * (1.0 - arm / fr) if fr > 0 else 0.0,
    "arm_pos_kernel": sums["k"] / max(n_cmd, 1),
    "arm_pos_kernel_frozen": sums["kf"] / max(n_cmd, 1),
  }


def main(args: Args) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  runs = [p for e in TASK_BY_EXPERIMENT if (Path("logs/rsl_rl") / e).is_dir() for p in (Path("logs/rsl_rl") / e).iterdir()]
  run_dir = args.run_dir or max(runs, key=lambda p: p.stat().st_mtime)
  experiment = run_dir.resolve().parent.name
  if args.task is None and experiment not in TASK_BY_EXPERIMENT:
    raise SystemExit(f"Unknown experiment '{experiment}'; pass --task (known: {list(TASK_BY_EXPERIMENT)}).")
  out = run_dir / "eval.csv"
  done: set[int] = set()
  if out.exists():
    with out.open() as f:
      done = {int(r["iteration"]) for r in csv.DictReader(f)}
  task = args.task or TASK_BY_EXPERIMENT[experiment]
  print(f"[eval] {task} | {run_dir} -> {out}", flush=True)

  env, runner = build(task, args)
  while True:
    pending = sorted((p for p in run_dir.glob("model_*.pt") if ckpt_iter(p) not in done), key=ckpt_iter)
    for ckpt in pending:
      # Skip files still being written by the trainer.
      if time.time() - ckpt.stat().st_mtime < 10:
        continue
      row = evaluate(env, runner, ckpt, args)
      new = not out.exists()
      with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
          w.writeheader()
        w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
      done.add(row["iteration"])
      print(
        f"[eval] it {row['iteration']:>6} | falls {row['falls_per_env']:.2f} | xy {row['err_xy']:.3f} | "
        f"yaw {row['err_yaw']:.3f} | vz {row['err_vz']:.3f} | h {row['err_height']:.3f} | "
        f"arm {row['arm_err']:.3f}/{row['arm_err_frozen']:.3f} ({row['arm_closed_pct']:.0f}% closed) | "
        f"kernel {row['arm_pos_kernel']:.3f}/{row['arm_pos_kernel_frozen']:.3f}",
        flush=True,
      )
    if not args.watch:
      break
    time.sleep(args.poll_s)
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
