from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None
  lin_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
      if "lin_vel_z" in stage and stage["lin_vel_z"] is not None:
        cfg.ranges.lin_vel_z = stage["lin_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    # "lin_vel_z_min": torch.tensor(cfg.ranges.lin_vel_z[0]),
    # "lin_vel_z_max": torch.tensor(cfg.ranges.lin_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])



class tracking_gated_disturbance:
  """Group curriculum for delayed_body_impulse events.

  Score = population mean of tracking rewards / weight, smoothed by an EMA.
  Disturbances stay off until the score reaches `threshold`; each promotion raises
  every cap by `increment`, restarts the cooldown and clears the score, so the gate
  is re-earned at the new level.
  Caps are force/torque norms: components are sampled in ±cap/√3, so |F| <= cap.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    names = env.reward_manager.active_terms
    p = cfg.params
    self._idx = [names.index(n) for n in p["tracking_terms"]]
    self._weights = torch.tensor(
      [env.reward_manager.get_term_cfg(n).weight for n in p["tracking_terms"]], device=env.device
    )
    self._score = torch.zeros((), device=env.device)
    self._level = 0
    self._last_promotion = 0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids  # Group state; nothing per env.

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    tracking_terms: tuple[str, ...],
    events: dict[str, dict[str, tuple[float, float]]],
    increment: float = 5.0,
    threshold: float = 0.8,
    cooldown_steps: int = 2400,
    ema_rate: float = 0.01,
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, tracking_terms  # Score uses all envs; terms resolved in __init__.
    # shortcut: reads RewardManager._step_reward (no public batch accessor), switch if mjlab adds one.
    step = env.reward_manager._step_reward
    score = (step[:, self._idx] / self._weights).mean()
    self._score += ema_rate * (score - self._score)

    since = env.common_step_counter - self._last_promotion
    if since >= cooldown_steps and self._score.item() >= threshold:
      self._level += 1
      self._last_promotion = env.common_step_counter
      self._score.zero_()
      for name, caps in events.items():
        params = env.event_manager.get_term_cfg(name).params
        for key, (start, cap) in caps.items():
          c = min(start + (self._level - 1) * increment, cap) / 3**0.5
          params[f"{key}_range"] = (-c, c)

    out = {"level": torch.tensor(float(self._level)), "score": self._score.clone()}
    for name in events:
      params = env.event_manager.get_term_cfg(name).params
      out[f"{name}_force_cap"] = torch.tensor(params["force_range"][1] * 3**0.5)
    return out
