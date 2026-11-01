import os
import re

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

from .split_actor import G1_LOWER_DIM


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class SplitVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """Checks the action joint order matches the SplitMLPModel lower/upper split."""

  def __init__(self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    names = env.unwrapped.action_manager.get_term("joint_pos").target_names
    lower, upper = names[:G1_LOWER_DIM], names[G1_LOWER_DIM:]
    if not all(re.search(r"hip|knee|ankle", n) for n in lower) or any(
      re.search(r"hip|knee|ankle", n) for n in upper
    ):
      raise ValueError(f"Action joint order does not match the split at {G1_LOWER_DIM}: {names}")
    super().__init__(env, train_cfg, log_dir, device)
