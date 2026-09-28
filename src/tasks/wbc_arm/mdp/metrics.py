from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv




def error_lin_vel(env: ManagerBasedRlEnv, command_name: str = "twist", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """Error in linear velocity.

  Returns:
    Per-environment scalar. Shape: ``(B,)``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_lin_vel_b[:, :2]
  return torch.mean(torch.norm(env.command_manager.get_command(command_name)[:,:2] - actual,dim=1), dim=0)




def error_ang_vel(env: ManagerBasedRlEnv, command_name: str = "twist", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """Error in angular velocity.

  Returns:
    Per-environment scalar. Shape: ``(B,)``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_ang_vel_b[:, 2]
  return torch.mean(torch.abs(env.command_manager.get_command(command_name)[:,2] - actual), dim=0)


def error_lin_vel_z(env: ManagerBasedRlEnv, command_name: str = "twist", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """Error in linear velocity along the z-axis.

  Returns:
    Per-environment scalar. Shape: ``(B,)``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_lin_vel_w[:, 2]
  return torch.mean(torch.abs(env.command_manager.get_command(command_name)[:,3] - actual), dim=0)