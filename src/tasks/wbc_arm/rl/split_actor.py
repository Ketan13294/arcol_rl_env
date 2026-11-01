"""Actor with separate lower-body (legs) and upper-body (waist + arms) MLP heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import GaussianDistribution

# G1 29-DoF action order: legs 0-11, then waist + arms 12-28.
G1_LOWER_DIM = 12
G1_ACTION_DIM = 29


class SplitMLP(nn.Module):
  """Two MLPs on the same input; outputs concatenated as [lower, upper]."""

  def __init__(
    self,
    input_dim: int,
    lower_dim: int,
    upper_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
  ) -> None:
    super().__init__()
    self.lower = MLP(input_dim, lower_dim, hidden_dims, activation)
    self.upper = MLP(input_dim, upper_dim, hidden_dims, activation)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([self.lower(x), self.upper(x)], dim=-1)


class SplitMLPModel(MLPModel):
  """MLPModel whose `mlp` is a SplitMLP; export and PPO see a single 29-D policy."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
  ) -> None:
    if output_dim != G1_ACTION_DIM:
      raise ValueError(f"SplitMLPModel expects {G1_ACTION_DIM} actions, got {output_dim}.")
    super().__init__(
      obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
    )
    # Other distributions read their params from the MLP output, which the split heads don't produce.
    if self.distribution is not None and type(self.distribution) is not GaussianDistribution:
      raise ValueError("SplitMLPModel supports only GaussianDistribution.")
    self.mlp = SplitMLP(
      self._get_latent_dim(), G1_LOWER_DIM, output_dim - G1_LOWER_DIM, hidden_dims, activation
    )
