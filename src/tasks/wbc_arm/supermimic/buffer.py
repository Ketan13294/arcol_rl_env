"""GPU ring buffer of simulator transitions, stored per env in time order."""

from __future__ import annotations

import torch

PUSHED = 1  # State target jumps after the step (velocity push).
FORCED = 2  # External force acted during the step (impulse).


class ReplayBuffer:
  """Slot t holds the pre-step state/goal/last action and the step's action and outcomes.
  The next state of slot t is the state in slot t+1 (post-reset for done transitions)."""

  def __init__(self, capacity: int, num_envs: int, state_dim: int, goal_dim: int, action_dim: int, device: str):
    self.capacity, self.num_envs, self.device = capacity, num_envs, device
    shape = (capacity, num_envs)
    half = torch.float16
    self.state = torch.zeros(*shape, state_dim, dtype=half, device=device)
    self.goal = torch.zeros(*shape, goal_dim, dtype=torch.float32, device=device)
    self.last_action = torch.zeros(*shape, action_dim, dtype=half, device=device)
    self.action = torch.zeros(*shape, action_dim, dtype=half, device=device)
    self.reward = torch.zeros(*shape, dtype=torch.float32, device=device)
    self.terminated = torch.zeros(*shape, dtype=torch.bool, device=device)
    self.done = torch.zeros(*shape, dtype=torch.bool, device=device)
    self.flags = torch.zeros(*shape, dtype=torch.uint8, device=device)
    self.ptr = 0
    self.size = 0  # Complete transitions; slot `ptr` holds only the current state.

  def add_state(self, state: torch.Tensor, goal: torch.Tensor, last_action: torch.Tensor) -> None:
    self.state[self.ptr] = state
    self.goal[self.ptr] = goal
    self.last_action[self.ptr] = last_action

  def add_outcome(
    self, action: torch.Tensor, reward: torch.Tensor, terminated: torch.Tensor, done: torch.Tensor, flags: torch.Tensor
  ) -> None:
    p = self.ptr
    self.action[p], self.reward[p] = action, reward
    self.terminated[p], self.done[p], self.flags[p] = terminated, done, flags
    self.ptr = (p + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity - 1)

  def _slots(self, logical: torch.Tensor) -> torch.Tensor:
    return (self.ptr - self.size + logical) % self.capacity

  def sample_windows(self, batch: int, k: int) -> dict[str, torch.Tensor]:
    """k-step windows: state has k+1 entries, everything else k."""
    t0 = torch.randint(0, self.size - k + 1, (batch,), device=self.device)
    env = torch.randint(0, self.num_envs, (batch,), device=self.device)
    steps = torch.arange(k + 1, device=self.device)
    slots = self._slots(t0[:, None] + steps)  # [B, k+1]
    e = env[:, None].expand(-1, k + 1)
    s, a = slots[:, :k], e[:, :k]
    return {
      "state": self.state[slots, e].float(),
      "goal": self.goal[s, a],
      "last_action": self.last_action[s, a].float(),
      "action": self.action[s, a].float(),
      "reward": self.reward[s, a],
      "terminated": self.terminated[s, a],
      "done": self.done[s, a],
      "flags": self.flags[s, a],
      "env": env,
    }

  def sample_starts(self, batch: int) -> dict[str, torch.Tensor]:
    t = self._slots(torch.randint(0, self.size + 1, (batch,), device=self.device))
    env = torch.randint(0, self.num_envs, (batch,), device=self.device)
    return {
      "state": self.state[t, env].float(),
      "goal": self.goal[t, env],
      "last_action": self.last_action[t, env].float(),
      "env": env,
    }
