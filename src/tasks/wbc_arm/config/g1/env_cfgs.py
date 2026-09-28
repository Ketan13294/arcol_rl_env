"""Unitree G1 velocity environment configurations."""

import re

import mujoco

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from src.tasks.wbc_arm import mdp
from src.tasks.wbc_arm.mdp.velocity_command import UniformVelocityCommandCfg
from src.tasks.wbc_arm.velocity_env_cfg import make_wbc_env_cfg
from src.assets.robots.unitree_g1.g1_constants import get_spec as get_g1_spec


G1_ARM_JOINT_NAMES = (
  ".*_shoulder_pitch_joint",
  ".*_shoulder_roll_joint",
  ".*_shoulder_yaw_joint",
  ".*_elbow_joint",
  ".*_wrist_roll_joint",
  ".*_wrist_pitch_joint",
  ".*_wrist_yaw_joint",
)
G1_NON_ARM_JOINT_REGEX = r"^(?!.*(shoulder|elbow|wrist)).*$"


def get_g1_spec_without_arm_self_collision() -> mujoco.MjSpec:
  """G1 spec where arm links never collide with the rest of the robot (or each
  other). Arm targets are sampled independently per joint, so many would
  otherwise put the arm inside the torso; contact with the world is unchanged."""
  spec = get_g1_spec()
  bodies = [b.name for b in spec.bodies if b.name and b.name != "world"]
  arm = [n for n in bodies if re.search(r"shoulder|elbow|wrist", n)]
  for i, a in enumerate(arm):
    for b in bodies:
      if b != a and (b not in arm or arm.index(b) > i):
        spec.add_exclude(bodyname1=a, bodyname2=b)
  return spec


# Upper limit of commanded base height (above the lowest foot site). For
# reference, the knees-straight height from g1.xml kinematics is 0.792 m.
G1_MAX_BASE_HEIGHT = 0.78


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_wbc_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  robot_cfg = get_g1_robot_cfg()
  robot_cfg.spec_fn = get_g1_spec_without_arm_self_collision
  cfg.scene.entities = {"robot": robot_cfg}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15
  twist_cmd.height_site_names = site_names
  assert twist_cmd.ranges.lin_pos_z is not None
  twist_cmd.ranges.lin_pos_z = (twist_cmd.ranges.lin_pos_z[0], G1_MAX_BASE_HEIGHT)
  twist_cmd.arm_joint_names = G1_ARM_JOINT_NAMES

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Arm joints are driven by the arm command, so they are excluded from the
  # default-pose regularizers (pose, stand_still) below.
  cfg.rewards["arm_joint_vel"].params["asset_cfg"].joint_names = G1_ARM_JOINT_NAMES
  cfg.rewards["pose"].params["asset_cfg"].joint_names = G1_NON_ARM_JOINT_REGEX
  cfg.rewards["stand_still"].params["asset_cfg"].joint_names = (
    r"^(?!.*(shoulder|elbow|wrist|hip_pitch|knee|ankle_pitch)).*$"
  )
  # Always-on upper-body regularizer covers only the waist here.
  cfg.rewards["stand_still_upper"].params["asset_cfg"].joint_names = r"^waist_.*$"

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Arm joints (shoulders/elbows/wrists) are omitted: they track the arm command.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  # Standing keeps the height-regime tolerance on the sagittal leg joints so a
  # commanded height can be held once reached (the base height is set by them).
  cfg.rewards["pose"].params["std_standing"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.05,
    r".*hip_yaw.*": 0.05,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.05,
    # Waist.
    r".*waist_yaw.*": 0.05,
    r".*waist_roll.*": 0.05,
    r".*waist_pitch.*": 0.05,
  }
  cfg.rewards["pose"].params["std_height"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.05,
    r".*hip_yaw.*": 0.05,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.05,
    # Waist.
    r".*waist_yaw.*": 0.05,
    r".*waist_roll.*": 0.05,
    r".*waist_pitch.*": 0.05,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  # cfg.rewards["foot_swing_height"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 0
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 32

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
    twist_cmd.ranges.lin_vel_z = (-0.5, 0.5)
    twist_cmd.ranges.lin_pos_z = (0.5, G1_MAX_BASE_HEIGHT)

  return cfg
