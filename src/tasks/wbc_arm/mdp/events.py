from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import apply_body_impulse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class delayed_body_impulse(apply_body_impulse):
  """mjlab apply_body_impulse whose first pulse in each episode waits start_delay_s."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._start_delay_s = cfg.params["start_delay_s"]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._interval_time_left[slice(None) if env_ids is None else env_ids] = self._start_delay_s

  def __call__(self, env: ManagerBasedRlEnv, env_ids, start_delay_s: float, **kwargs) -> None:
    del start_delay_s  # Read once in __init__.
    super().__call__(env, env_ids, **kwargs)
