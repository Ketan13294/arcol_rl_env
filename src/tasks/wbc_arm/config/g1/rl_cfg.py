"""RL configuration for Unitree G1 velocity task."""

from dataclasses import fields

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_wbc",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=50001,
  )


def unitree_g1_split_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same as the WBC-ARM runner, with separate leg and waist+arm actor heads."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor.class_name = "src.tasks.wbc_arm.rl.split_actor:SplitMLPModel"
  cfg.experiment_name = "g1_wbc_split"
  return cfg


def unitree_g1_supermimic_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Split actor trained by SuperMimic; `supermimic` holds the world-model hyperparameters."""
  from src.tasks.wbc_arm.supermimic.config import SuperMimicRunnerCfg

  split = unitree_g1_split_ppo_runner_cfg()
  cfg = SuperMimicRunnerCfg(**{f.name: getattr(split, f.name) for f in fields(split)})
  cfg.experiment_name = "g1_wbc_supermimic"
  return cfg
