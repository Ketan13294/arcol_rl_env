from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import apply_body_impulse
from mjlab.utils.lab_api.math import quat_apply

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


class _WrenchRecorder:
  """Stands in for an impulse term's asset: records its writes instead of applying them."""

  def __init__(self, asset, forces: torch.Tensor, torques: torch.Tensor):
    self._asset, self._forces, self._torques = asset, forces, torques

  def __getattr__(self, name):
    return getattr(self._asset, name)

  def write_external_wrench_to_sim(self, forces, torques, env_ids=None, body_ids=None) -> None:
    del body_ids  # Same bodies as the load, in the same order (checked in palm_load).
    ids = slice(None) if env_ids is None else env_ids
    self._forces[ids] = forces
    self._torques[ids] = torques


class palm_load:
  """Constant per-episode force at each palm, inside a cone about world -z.

  Each palm samples its own magnitude in magnitude_range and a direction uniform over the
  cone's solid angle. magnitude_range is read at every reset, so a curriculum can raise it.
  An impulse term on the same bodies (`impulse_term`) would overwrite xfrc_applied, so its
  writes are recorded and summed with the load here instead.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._params = p
    self._asset = env.scene[p["asset_cfg"].name]
    self._body_ids = p["asset_cfg"].body_ids
    n, b = env.num_envs, len(self._body_ids)
    self._load = torch.zeros(n, b, 3, device=env.device)
    self._imp_f = torch.zeros(n, b, 3, device=env.device)
    self._imp_t = torch.zeros(n, b, 3, device=env.device)
    self._offset = torch.tensor(p["body_point_offset"], device=env.device).expand(n, b, 3)
    self._env, self._hooked = env, False

  def _hook_impulse(self) -> None:
    # Deferred: env.event_manager doesn't exist while its terms are being built.
    self._hooked = True
    name, em = self._params.get("impulse_term"), self._env.event_manager
    if not name or name not in em.active_terms.get("step", []):
      return
    term_cfg = em.get_term_cfg(name)
    assert list(term_cfg.params["asset_cfg"].body_ids) == list(self._body_ids)
    imp = term_cfg.func
    imp._asset = _WrenchRecorder(imp._asset, self._imp_f, self._imp_t)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    lo, hi = self._params["magnitude_range"]
    shape = self._load[ids].shape[:-1]
    dev = self._load.device
    mag = lo + (hi - lo) * torch.rand(shape, device=dev)
    cos_max = math.cos(math.radians(self._params["cone_deg"]))
    cos_t = 1.0 - torch.rand(shape, device=dev) * (1.0 - cos_max)
    sin_t = (1.0 - cos_t.square()).clamp(min=0.0).sqrt()
    phi = 2.0 * math.pi * torch.rand(shape, device=dev)
    d = torch.stack([sin_t * phi.cos(), sin_t * phi.sin(), -cos_t], dim=-1)
    self._load[ids] = (mag * (hi > 0.0)).unsqueeze(-1) * d
    self._imp_f[ids] = 0.0
    self._imp_t[ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv, env_ids, **kwargs) -> None:
    del env, env_ids, kwargs  # Read in __init__ / reset.
    if not self._hooked:
      self._hook_impulse()
    quat = self._asset.data.body_link_quat_w[:, self._body_ids]
    r = quat_apply(quat, self._offset)
    torques = self._imp_t + torch.cross(r, self._load, dim=-1)
    self._asset.write_external_wrench_to_sim(self._load + self._imp_f, torques, body_ids=self._body_ids)
