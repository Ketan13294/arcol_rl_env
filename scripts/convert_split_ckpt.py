"""Warm-start a WBC-ARM-Split checkpoint from a monolithic WBC-ARM checkpoint.

Both heads get a full copy of the monolithic MLP; the last layer is sliced so the split
policy reproduces the original actions exactly. Optimizer state is reset because the
parameter list changes.
"""

import copy
import re
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path

import torch
import tyro
from rsl_rl.models import MLPModel
from tensordict import TensorDict

SPLIT_TASK = "Unitree-G1-Flat-WBC-ARM-Split"


@dataclass
class Args:
  checkpoint: Path
  run_name: str = "WarmStart"
  out_root: Path = Path("logs/rsl_rl/g1_wbc_split")


def split_actor_state(sd: dict, lower_dim: int) -> dict:
  layers = sorted({int(m.group(1)) for k in sd if (m := re.match(r"mlp\.(\d+)\.weight", k))})
  last = layers[-1]
  out = {k: v for k, v in sd.items() if not k.startswith("mlp.")}
  for k, v in sd.items():
    m = re.match(r"mlp\.(\d+)\.(weight|bias)", k)
    if not m:
      continue
    idx, kind = m.groups()
    if int(idx) == last:
      out[f"mlp.lower.{idx}.{kind}"] = v[:lower_dim].clone()
      out[f"mlp.upper.{idx}.{kind}"] = v[lower_dim:].clone()
    else:
      out[f"mlp.lower.{idx}.{kind}"] = v.clone()
      out[f"mlp.upper.{idx}.{kind}"] = v.clone()
  return out


def build_models(actor_sd: dict, critic_sd: dict, agent) -> tuple[torch.nn.Module, MLPModel, MLPModel]:
  from src.tasks.wbc_arm.rl.split_actor import G1_ACTION_DIM, SplitMLPModel

  obs = TensorDict(
    {
      "actor": torch.zeros(1, actor_sd["mlp.0.weight"].shape[1]),
      "critic": torch.zeros(1, critic_sd["mlp.0.weight"].shape[1]),
    },
    batch_size=[1],
  )
  groups = {"actor": ["actor"], "critic": ["critic"]}

  def kwargs(model_cfg) -> dict:
    kw = {
      "hidden_dims": model_cfg.hidden_dims,
      "activation": model_cfg.activation,
      "obs_normalization": model_cfg.obs_normalization,
    }
    if model_cfg.distribution_cfg is not None:
      kw["distribution_cfg"] = dict(model_cfg.distribution_cfg)
    return kw

  mono = MLPModel(obs, groups, "actor", G1_ACTION_DIM, **kwargs(agent.actor))
  split = SplitMLPModel(obs, groups, "actor", G1_ACTION_DIM, **kwargs(agent.actor))
  critic = MLPModel(obs, groups, "critic", 1, **kwargs(agent.critic))
  return mono, split, critic


def main(args: Args) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_rl_cfg
  from src.tasks.wbc_arm.rl.split_actor import G1_LOWER_DIM

  agent = load_rl_cfg(SPLIT_TASK)
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  split_sd = split_actor_state(ckpt["actor_state_dict"], G1_LOWER_DIM)

  mono, split, critic = build_models(ckpt["actor_state_dict"], ckpt["critic_state_dict"], agent)
  mono.load_state_dict(ckpt["actor_state_dict"], strict=True)
  split.load_state_dict(split_sd, strict=True)
  critic.load_state_dict(ckpt["critic_state_dict"], strict=True)

  # float64 copies: in float32 the sliced last layer differs from the full one by rounding.
  mono64, split64 = copy.deepcopy(mono).double().eval(), copy.deepcopy(split).double().eval()
  x = TensorDict({"actor": torch.randn(4096, mono.obs_dim, dtype=torch.float64) * 3.0}, batch_size=[4096])
  with torch.no_grad():
    max_diff = (mono64(x) - split64(x)).abs().max().item()
  if max_diff > 1e-9:
    raise RuntimeError(f"Split policy diverges from the monolithic one: max |diff| = {max_diff:.2e}")

  optimizer = torch.optim.Adam(
    chain(split.parameters(), critic.parameters()), lr=agent.algorithm.learning_rate
  )
  out = {
    **ckpt,
    "actor_state_dict": split.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
  }
  run_dir = args.out_root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
  run_dir.mkdir(parents=True, exist_ok=True)
  out_path = run_dir / args.checkpoint.name
  torch.save(out, out_path)
  print(f"max |a_mono - a_split| = {max_diff:.2e}\nwrote {out_path}")


if __name__ == "__main__":
  main(tyro.cli(Args))
