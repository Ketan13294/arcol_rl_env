"""Deterministic WBC-ARM checkpoint evaluation, optionally watching a run for new checkpoints.

Results are appended to <run_dir>/eval.csv; already-evaluated checkpoints are skipped.
A csv with older columns is kept as eval_old_<time>.csv and its checkpoints are evaluated again.

Column groups:
  *_active: steps where that command is non-zero (|cmd| > 0.1).
  err_height_{low,mid,high}: height target < 0.60, 0.60-0.70, >= 0.70 m. Blank if never commanded.
  Gait columns: walking commands only, except stand_air_frac.
"""

import csv
import re
import time
from collections import defaultdict
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
  "err_vx", "err_vy", "err_vx_active", "err_vy_active", "err_yaw_active", "err_vz_active",
  "err_height_low", "err_height_mid", "err_height_high", "height_bias_low",
  "arm_err_shoulder", "arm_err_elbow", "arm_err_wrist",
  "step_freq_hz", "duty_factor", "double_support_frac", "flight_frac", "swing_time_s", "stance_time_s",
  "swing_asym", "swing_height_m", "foot_slip_mps", "landing_force_n", "stand_air_frac",
)
ACTIVE_CMD = 0.1
ARM_GROUPS = ("shoulder", "elbow", "wrist")


class Means:
  """Running means; a key with no samples reads as NaN (blank in the csv)."""

  def __init__(self):
    self.sum, self.n = defaultdict(float), defaultdict(int)

  def add(self, key: str, x: torch.Tensor) -> None:
    self.sum[key] += x.float().sum().item()
    self.n[key] += x.numel()

  def __getitem__(self, key: str) -> float:
    return self.sum[key] / self.n[key] if self.n[key] else float("nan")


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
  m = Means()
  dt = u.step_dt
  groups = {g: torch.tensor([g in n for n in t.arm_joint_names], device=args.device) for g in ARM_GROUPS}
  feet = u.scene["feet_ground_contact"]
  prev = feet.data.found > 0
  swing, stance = torch.zeros_like(prev, dtype=torch.long), torch.zeros_like(prev, dtype=torch.long)
  peak, lift_z = torch.zeros_like(prev, dtype=torch.float), torch.zeros_like(prev, dtype=torch.float)
  # Only time phases that lie entirely within a walking command.
  seen_lift, seen_land = torch.zeros_like(prev), torch.zeros_like(prev)
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

    vel_err = (cmd[:, :2] - t.lin_vel_heading()[:, :2]).abs()
    yaw_err = (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs()
    vz_err = (cmd[:, 3] - robot.data.root_link_lin_vel_w[:, 2]).abs()
    m.add("err_vx", vel_err[:, 0])
    m.add("err_vy", vel_err[:, 1])
    m.add("err_vx_active", vel_err[:, 0][cmd[:, 0].abs() > ACTIVE_CMD])
    m.add("err_vy_active", vel_err[:, 1][cmd[:, 1].abs() > ACTIVE_CMD])
    m.add("err_yaw_active", yaw_err[cmd[:, 2].abs() > ACTIVE_CMD])
    m.add("err_vz_active", vz_err[cmd[:, 3].abs() > ACTIVE_CMD])
    bias = t.base_height() - t.height_target
    low, high = t.height_target < 0.60, t.height_target >= 0.70
    m.add("err_height_low", bias[low].abs())
    m.add("err_height_mid", bias[~low & ~high].abs())
    m.add("err_height_high", bias[high].abs())
    m.add("height_bias_low", bias[low])
    for g, mask in groups.items():
      m.add(f"arm_err_{g}", err[moving & mask])

    found = feet.data.found > 0
    alive = ~dones.bool()
    walking = (cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs() >= ACTIVE_CMD) & alive
    walk_feet = walking[:, None]
    in_contact = found.sum(-1)
    site_ids = t.height_site_ids
    foot_z = robot.data.site_pos_w[:, site_ids, 2]
    land, lift = found & ~prev & alive[:, None], ~found & prev & alive[:, None]
    swing_done, stance_done = land & walk_feet & seen_lift, lift & walk_feet & seen_land
    m.add("step_freq_hz", (land & walk_feet)[walking] / dt)
    m.add("duty_factor", found[walking])
    m.add("double_support_frac", in_contact[walking] == 2)
    m.add("flight_frac", in_contact[walking] == 0)
    m.add("swing_time_s", swing[swing_done] * dt)
    m.add("swing_left", swing[:, 0][swing_done[:, 0]] * dt)
    m.add("swing_right", swing[:, 1][swing_done[:, 1]] * dt)
    m.add("stance_time_s", stance[stance_done] * dt)
    m.add("swing_height_m", peak[swing_done])
    m.add("foot_slip_mps", robot.data.site_lin_vel_w[:, site_ids, :2].norm(dim=-1)[found & walk_feet])
    m.add("landing_force_n", feet.data.force.norm(dim=-1)[land & walk_feet])
    m.add("stand_air_frac", in_contact[~walking & alive] < 2)

    swing[land], stance[lift], peak[land] = 0, 0, 0
    lift_z[lift] = foot_z[lift]
    seen_lift |= lift
    seen_land |= land
    swing += ~found
    stance += found
    peak = torch.maximum(peak, (foot_z - lift_z) * ~found)
    swing[~alive], stance[~alive], peak[~alive] = 0, 0, 0
    seen_lift[~walking], seen_land[~walking] = False, False
    prev = found

  n = args.num_envs * args.steps
  arm, fr = sums["arm"] / max(n_arm, 1), sums["frozen"] / max(n_arm, 1)
  left, right = m["swing_left"], m["swing_right"]
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
    **{k: m[k] for k in FIELDS[FIELDS.index("err_vx"):] if k != "swing_asym"},
    "swing_asym": abs(left - right) / (left + right),
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
      reader = csv.DictReader(f)
      rows = list(reader)
    if tuple(reader.fieldnames or ()) == FIELDS:
      done = {int(r["iteration"]) for r in rows}
    else:
      out.rename(out.with_name(f"eval_old_{int(time.time())}.csv"))
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
        w.writerow({k: ("" if v != v else round(v, 4)) if isinstance(v, float) else v for k, v in row.items()})
      done.add(row["iteration"])
      print(
        f"[eval] it {row['iteration']:>6} | falls {row['falls_per_env']:.2f} | xy {row['err_xy']:.3f} | "
        f"yaw {row['err_yaw']:.3f} | vz {row['err_vz']:.3f} | h {row['err_height']:.3f} | "
        f"arm {row['arm_err']:.3f}/{row['arm_err_frozen']:.3f} ({row['arm_closed_pct']:.0f}% closed) | "
        f"kernel {row['arm_pos_kernel']:.3f}/{row['arm_pos_kernel_frozen']:.3f}\n"
        f"       cmd | vx {row['err_vx_active']:.3f} | vy {row['err_vy_active']:.3f} | "
        f"yaw {row['err_yaw_active']:.3f} | vz {row['err_vz_active']:.3f} | h low/mid/high "
        f"{row['err_height_low']:.3f}/{row['err_height_mid']:.3f}/{row['err_height_high']:.3f}\n"
        f"       gait | {row['step_freq_hz']:.2f} Hz | duty {row['duty_factor']:.2f} | "
        f"swing {row['swing_time_s']:.2f} s, {row['swing_height_m']:.3f} m | slip {row['foot_slip_mps']:.3f} m/s",
        flush=True,
      )
    if not args.watch:
      break
    time.sleep(args.poll_s)
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
