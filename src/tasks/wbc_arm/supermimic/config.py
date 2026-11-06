"""SuperMimic hyperparameters (paper Table A.5 unless noted), set via `--agent.supermimic.*`."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class SuperMimicCfg:
  # World model.
  wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
  wm_head_hidden: tuple[int, ...] = (512, 512)
  wm_window: int = 8
  wm_batch: int = 2048
  wm_steps_per_update: int = 10
  wm_lr: float = 2e-4
  wm_stats_updates: int = 2000  # Input/delta normalization is frozen after this many updates (not gradient steps).
  wm_init: str = ""  # World model + critic to start from (a wm_gate.pt), instead of fitting from scratch.
  # Imagination and policy.
  horizon: int = 32
  rollouts: int = 4096
  policy_lr: float = 5e-5  # Paper: 2e-4, which collapsed the warm-started policy within 20 updates.
  policy_betas: tuple[float, float] = (0.9, 0.95)
  gamma: float = 0.99
  bc_coef: float = 1.0  # Pull toward the warm-start actions; imagined returns alone exploit model error.
  state_clip: float = 6.0  # Imagined states are held within this many replay stds of the replay mean.
  # Critic.
  critic_hidden: tuple[int, ...] = (256, 128)
  critic_lr: float = 2e-4
  lam: float = 0.95
  critic_epochs: int = 5
  critic_minibatches: int = 4
  # Data collection.
  action_noise_std: float = 0.1
  action_clip: float = 5.0  # Same bound in sim and imagination, so BPTT can't push setpoints off the data.
  buffer_steps: int = 1000
  steps_per_iteration: int = 24
  update_to_data: float = 0.4
  # Policy updates start once the warmup is over and the world model's k-step error first drops under the gate.
  warmup_iterations: int = 200
  wm_gate_err: float = 0.3
  # Cosine decay of all learning rates to lr_final_frac over this many world-model updates, then constant.
  lr_decay_updates: int = 250_000
  lr_final_frac: float = 0.03
  # Fit the actor's observation normalizer on collected data (training from scratch); frozen with the WM stats.
  update_actor_obs_norm: bool = False
  # Speed (RTX 4090).
  bf16: bool = True
  compile: bool = True


@dataclass
class SuperMimicRunnerCfg(RslRlOnPolicyRunnerCfg):
  supermimic: SuperMimicCfg = field(default_factory=SuperMimicCfg)
