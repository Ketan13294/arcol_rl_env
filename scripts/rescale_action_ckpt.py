"""Carry a WBC-ARM checkpoint across an action-scale change without changing its behavior.

For each rescaled joint, the policy output row, its action std and the normalizer stats of
its last-action observation (actor and critic) are multiplied by old/new scale, so the joint
targets are identical. Optimizer state is reset because parameter magnitudes change.
"""

import copy
import re
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path

import torch
import tyro
from tensordict import TensorDict

G1_ACTION_JOINTS = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
  "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
  "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
LAST_ACTION_OBS = 84  # Start of the last-action term in both actor and critic obs.


@dataclass
class Args:
  checkpoint: Path
  task: str = "Unitree-G1-Flat-WBC-ARM-Split"
  joints: str = r".*_wrist_(pitch|yaw)_joint"
  old_scale: float = 0.0
  """Scale the checkpoint was trained with; defaults to mjlab's G1_ACTION_SCALE for the first match."""
  new_scale: float = 0.0
  """Defaults to the task's current action scale for the first match."""
  run_name: str = "Rescaled"


def output_rows(sd: dict) -> list[tuple[str, int]]:
  """(last-layer key prefix, row) for each action index, across split or monolithic heads."""
  heads = {}
  for k in sd:
    if m := re.match(r"(mlp(?:\.\w+)?)\.(\d+)\.weight$", k):
      heads[m.group(1)] = max(heads.get(m.group(1), -1), int(m.group(2)))
  order = [h for h in ("mlp.lower", "mlp.upper") if h in heads] or ["mlp"]
  return [(f"{h}.{heads[h]}", r) for h in order for r in range(sd[f"{h}.{heads[h]}.weight"].shape[0])]


def scale_obs_norm(sd: dict, cols: list[int], f: float, eps: float = 1e-2) -> None:
  """rsl_rl normalizes by (std + eps), so std maps to f * (std + eps) - eps."""
  for c in cols:
    std = f * (sd["obs_normalizer._std"][..., c] + eps) - eps
    if (std <= 0).any():
      raise ValueError(f"Obs column {c}: std too small to rescale by {f:.3f}")
    sd["obs_normalizer._mean"][..., c] *= f
    sd["obs_normalizer._std"][..., c] = std
    sd["obs_normalizer._var"][..., c] = std**2


def resolve_scale(scales: dict, name: str) -> float:
  return next(v for k, v in scales.items() if re.fullmatch(k, name))


def main(args: Args) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from src.tasks.wbc_arm.rl.split_actor import G1_ACTION_DIM

  ids = [i for i, n in enumerate(G1_ACTION_JOINTS) if re.fullmatch(args.joints, n)]
  if not ids:
    raise ValueError(f"No action joint matches {args.joints!r}")
  name = G1_ACTION_JOINTS[ids[0]]
  old = args.old_scale or resolve_scale(G1_ACTION_SCALE, name)
  new = args.new_scale or resolve_scale(load_env_cfg(args.task).actions["joint_pos"].scale, name)
  f = old / new

  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  actor_sd = {k: v.clone() for k, v in ckpt["actor_state_dict"].items()}
  critic_sd = {k: v.clone() for k, v in ckpt["critic_state_dict"].items()}
  rows = output_rows(actor_sd)
  for i in ids:
    prefix, r = rows[i]
    actor_sd[f"{prefix}.weight"][r] *= f
    actor_sd[f"{prefix}.bias"][r] *= f
    actor_sd["distribution.std_param"][i] *= f
  cols = [LAST_ACTION_OBS + i for i in ids]
  scale_obs_norm(actor_sd, cols, f)
  scale_obs_norm(critic_sd, cols, f)

  agent = load_rl_cfg(args.task)
  old_actor = build_model(ckpt["actor_state_dict"], agent.actor, G1_ACTION_DIM)
  actor = build_model(actor_sd, agent.actor, G1_ACTION_DIM)
  critic = build_model(critic_sd, agent.critic, 1)
  x = torch.randn(4096, actor.obs_dim, dtype=torch.float64) * 3.0
  x_new = x.clone()
  x_new[:, cols] *= f
  with torch.no_grad():
    a_old = copy.deepcopy(old_actor).double().eval()(TensorDict({"actor": x}, batch_size=[4096]))
    a_new = copy.deepcopy(actor).double().eval()(TensorDict({"actor": x_new}, batch_size=[4096]))
  a_old[:, ids] *= f
  max_diff = ((a_new - a_old).abs() / (a_old.abs() + 1.0)).max().item()
  # Scaled weights are stored in float32, so expect ~1e-7 relative rounding.
  if max_diff > 1e-5:
    raise RuntimeError(f"Rescaled policy diverges: max relative |diff| = {max_diff:.2e}")

  optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=agent.algorithm.learning_rate)
  out = {
    **ckpt,
    "actor_state_dict": actor_sd,
    "critic_state_dict": critic_sd,
    "optimizer_state_dict": optimizer.state_dict(),
  }
  run_dir = args.checkpoint.parent.parent / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
  run_dir.mkdir(parents=True, exist_ok=True)
  out_path = run_dir / args.checkpoint.name
  torch.save(out, out_path)
  print(f"{len(ids)} joints, scale {old:.4f} -> {new:.4f}; max rel |diff| = {max_diff:.2e}\nwrote {out_path}")


def build_model(sd: dict, model_cfg, output_dim: int):
  from rsl_rl.models import MLPModel

  from src.tasks.wbc_arm.rl.split_actor import SplitMLPModel

  cls = SplitMLPModel if any(k.startswith("mlp.lower.") for k in sd) else MLPModel
  obs = TensorDict({"actor": torch.zeros(1, sd["obs_normalizer._mean"].shape[-1])}, batch_size=[1])
  kw = {
    "hidden_dims": model_cfg.hidden_dims,
    "activation": model_cfg.activation,
    "obs_normalization": model_cfg.obs_normalization,
  }
  if model_cfg.distribution_cfg is not None:
    kw["distribution_cfg"] = dict(model_cfg.distribution_cfg)
  model = cls(obs, {"actor": ["actor"]}, "actor", output_dim, **kw)
  model.load_state_dict(sd, strict=True)
  return model


if __name__ == "__main__":
  main(tyro.cli(Args))
