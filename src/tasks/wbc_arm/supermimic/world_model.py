"""Deterministic differentiable world model: dynamics, reward residual and termination heads."""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...], layer_norm: bool = True) -> nn.Sequential:
  layers: list[nn.Module] = []
  for h in hidden:
    layers += [nn.Linear(in_dim, h)]
    if layer_norm:
      layers += [nn.LayerNorm(h)]
    layers += [nn.SiLU()]
    in_dim = h
  layers += [nn.Linear(in_dim, out_dim)]
  return nn.Sequential(*layers)


class RunningNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-4):
    super().__init__()
    self.eps = eps
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros(()))

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.reshape(-1, x.shape[-1]).float()
    n = x.shape[0]
    mean, var = x.mean(0), x.var(0, unbiased=False)
    total = self.count + n
    delta = mean - self.mean
    self.mean += delta * n / total
    self.var = (self.var * self.count + var * n + delta**2 * self.count * n / total) / total
    self.count = total

  @property
  def std(self) -> torch.Tensor:
    return torch.sqrt(self.var + self.eps)

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / self.std


class WorldModel(nn.Module):
  def __init__(
    self,
    state_dim: int,
    action_dim: int,
    xi_dim: int,
    cmd_feat_dim: int,
    hidden: tuple[int, ...],
    head_hidden: tuple[int, ...],
  ):
    super().__init__()
    self.state_norm = RunningNorm(state_dim)
    self.delta_norm = RunningNorm(state_dim)
    self.xi_norm = RunningNorm(xi_dim)
    self.dynamics = mlp(state_dim + action_dim + xi_dim, state_dim, hidden)
    # No action inputs: the policy could exploit an unconstrained dr/da. Action rate is analytic.
    self.reward_head = mlp(2 * state_dim + cmd_feat_dim, 1, head_hidden)
    self.term_head = mlp(state_dim, 1, head_hidden)

  @torch.no_grad()
  def update_stats(self, s: torch.Tensor, s_next: torch.Tensor, xi: torch.Tensor) -> None:
    self.state_norm.update(s)
    self.delta_norm.update(s_next - s)
    self.xi_norm.update(xi)

  def step(self, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    x = torch.cat([self.state_norm.normalize(s), a, self.xi_norm.normalize(xi)], dim=-1)
    delta = self.dynamics(x).float()
    return s + delta * self.delta_norm.std + self.delta_norm.mean

  def clamp_state(self, s: torch.Tensor, k: float) -> torch.Tensor:
    """Keep imagined states within k stds of the replay data, so one bad step can't blow up a rollout."""
    m, d = self.state_norm.mean, k * self.state_norm.std
    return torch.clamp(s, m - d, m + d)

  def reward_residual(self, s: torch.Tensor, s_next: torch.Tensor, cmd_feat: torch.Tensor) -> torch.Tensor:
    n = self.state_norm.normalize
    x = torch.cat([n(s), n(s_next), cmd_feat], dim=-1)
    return self.reward_head(x).float().squeeze(-1)

  def term_logit(self, s: torch.Tensor) -> torch.Tensor:
    return self.term_head(self.state_norm.normalize(s)).float().squeeze(-1)

  def state_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared error in normalized state units."""
    return torch.mean(((s_pred - s_true) / self.state_norm.std) ** 2, dim=-1)
