from mjlab.tasks.registry import register_mjlab_task
from src.tasks.wbc_arm.rl import SplitVelocityOnPolicyRunner, VelocityOnPolicyRunner

from src.tasks.wbc_arm.supermimic import SuperMimicRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_supermimic_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import (
  unitree_g1_ppo_runner_cfg,
  unitree_g1_split_ppo_runner_cfg,
  unitree_g1_supermimic_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-G1-Rough-WBC-ARM",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-WBC-ARM",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Rough-WBC-ARM-Split",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_split_ppo_runner_cfg(),
  runner_cls=SplitVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-WBC-ARM-Split",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_split_ppo_runner_cfg(),
  runner_cls=SplitVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-WBC-ARM-SuperMimic",
  env_cfg=unitree_g1_flat_supermimic_env_cfg(),
  play_env_cfg=unitree_g1_flat_supermimic_env_cfg(play=True),
  rl_cfg=unitree_g1_supermimic_runner_cfg(),
  runner_cls=SuperMimicRunner,
)
