"""World-model view of the WBC-ARM env: state/goal/sim-context layout, plus the parts of
commands, observations and rewards that are recomputed from predicted state."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .. import mdp

# State: every quantity the actor obs, analytic rewards and termination need.
S_ANG_VEL_IMU = slice(0, 3)
S_GRAV = slice(3, 6)
S_JPOS = slice(6, 35)
S_JVEL = slice(35, 64)
S_LIN_VEL_H = slice(64, 67)  # Heading-frame xy, world z.
S_ANG_VEL_B = slice(67, 70)
S_BASE_H = slice(70, 71)
S_FOOT_Z = slice(71, 73)
S_CONTACT = slice(73, 75)
S_FORCE = slice(75, 81)
STATE_DIM = 81

# Goal: held through imagination, except the height target and episode clock advance.
G_VEL = slice(0, 3)
G_H_TARGET = 3
G_VZ_RATE = 4
G_IS_HEIGHT = 5
G_ARM_TARGET = slice(6, 20)
G_EP_TIME = 20
GOAL_DIM = 21

XI_DIM = 34  # Foot friction, joint damping, torso CoM offset (3), encoder bias (29).
CMD_FEAT_DIM = 20  # Command obs (4) + arm command (14) + phase (2).

_PHASE_PERIOD = 0.6
_IDLE_THRESHOLD = 0.1
_CONTACT_N = 50.0


class TaskMirror:
  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.robot = env.scene["robot"]
    self.twist = env.command_manager.get_term("twist")
    self.contact = env.scene["feet_ground_contact"]
    self.imu = env.scene["robot/imu_ang_vel"]
    self.dt = env.step_dt
    self.device = env.device

    cfg = self.twist.cfg
    self.h_lo, self.h_hi = cfg.ranges.lin_pos_z
    self.vz_lo, self.vz_hi = cfg.ranges.lin_vel_z
    self.h_kp = cfg.height_control_stiffness
    self.arm_ids = torch.as_tensor(self.twist.arm_joint_ids, device=self.device)
    self.arm_kp = self.twist.arm_kp
    self.arm_max_vel = cfg.arm_max_vel
    self.default_q = self.robot.data.default_joint_pos[0].clone()
    self.foot_sites = self.twist.height_site_ids

    rm = env.reward_manager
    terms = dict(zip(rm._term_names, rm._term_cfgs))
    self.w = {n: terms[n].weight for n in (
      "track_linear_velocity", "track_linear_velocity_z", "track_angular_velocity",
      "track_arm_joint_vel", "track_arm_joint_pos", "action_rate_l2", "is_terminated",
    )}
    self.std = {n: terms[n].params["std"] for n in (
      "track_linear_velocity", "track_linear_velocity_z", "track_angular_velocity",
      "track_arm_joint_vel", "track_arm_joint_pos",
    )}
    self.term_penalty = -self.w["is_terminated"] * self.dt

    self.obs_noise = torch.zeros(113, device=self.device)
    self.obs_noise[0:3], self.obs_noise[3:6] = 0.2, 0.05
    self.obs_noise[26:55], self.obs_noise[55:84] = 0.01, 1.5

    self.xi = self._read_xi()
    self._impulses = [
      c.func for cfgs in env.event_manager._mode_term_cfgs.values() for c in cfgs
      if hasattr(c.func, "_active")
    ]
    env._sm_pushed = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

  def _read_xi(self) -> torch.Tensor:
    em = self.env.event_manager
    cfgs = {n: c for m in em._mode_term_names for n, c in zip(em._mode_term_names[m], em._mode_term_cfgs[m])}
    model, idx, n = self.env.sim.model, self.robot.indexing, self.env.num_envs
    geom = idx.geom_ids[cfgs["foot_friction"].params["asset_cfg"].geom_ids[0]]
    torso = idx.body_ids[cfgs["base_com"].params["asset_cfg"].body_ids[0]]
    dof = idx.joint_v_adr[0]
    nominal_com = torch.as_tensor(self.env.sim.mj_model.body_ipos[int(torso)], device=self.device, dtype=torch.float)
    friction = model.geom_friction[:, geom, 0].expand(n).reshape(n, 1)
    damping = model.dof_damping[:, dof].expand(n).reshape(n, 1)
    com = (model.body_ipos[:, torso] - nominal_com).expand(n, 3)
    return torch.cat([friction, damping, com, self.robot.data.encoder_bias], dim=-1).float()

  # Reading from the simulator.

  def read_state(self) -> torch.Tensor:
    d = self.robot.data
    force = self.contact.data.force
    return torch.cat([
      self.imu.data,
      d.projected_gravity_b,
      d.joint_pos - d.default_joint_pos,
      d.joint_vel - d.default_joint_vel,
      quat_apply_inverse(yaw_quat(d.root_link_quat_w), d.root_link_lin_vel_w),
      d.root_link_ang_vel_b,
      self.twist.base_height().unsqueeze(-1),
      d.site_pos_w[:, self.foot_sites, 2],
      (torch.norm(force, dim=-1) > _CONTACT_N).float(),
      torch.sign(force.flatten(1)) * torch.log1p(force.flatten(1).abs()),
    ], dim=-1)

  def read_goal(self) -> torch.Tensor:
    t = self.twist
    return torch.cat([
      t.vel_command_b[:, :3],
      t.height_target.unsqueeze(-1),
      t.vz_rate.unsqueeze(-1),
      t.is_height_env.float().unsqueeze(-1),
      t.arm_q_target,
      (self.env.episode_length_buf * self.dt).unsqueeze(-1),
    ], dim=-1)

  def read_last_action(self) -> torch.Tensor:
    return self.env.action_manager.action.clone()

  def impulse_active(self) -> torch.Tensor:
    active = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
    for term in self._impulses:
      active |= term._active
    return active

  def pop_pushed(self) -> torch.Tensor:
    pushed = self.env._sm_pushed.clone()
    self.env._sm_pushed.zero_()
    return pushed

  # Recomputed from (predicted) state.

  def commands(self, s: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Command obs [vx, vy, wz, vz] and arm velocity command, as the command term builds them."""
    h = s[:, S_BASE_H].squeeze(-1)
    vz = torch.clamp(self.h_kp * (g[:, G_H_TARGET] - h), self.vz_lo, self.vz_hi)
    at_limit = ((h >= self.h_hi) & (vz > 0)) | ((h <= self.h_lo) & (vz < 0))
    vz = torch.where(at_limit, torch.zeros_like(vz), vz)
    q_arm = s[:, S_JPOS][:, self.arm_ids] + self.default_q[self.arm_ids]
    arm = torch.clamp(self.arm_kp * (g[:, G_ARM_TARGET] - q_arm), -self.arm_max_vel, self.arm_max_vel)
    return torch.cat([g[:, G_VEL], vz.unsqueeze(-1)], dim=-1), arm

  def phase(self, g: torch.Tensor) -> torch.Tensor:
    angle = (g[:, G_EP_TIME] % _PHASE_PERIOD) / _PHASE_PERIOD * 2.0 * torch.pi
    ph = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
    idle = (torch.norm(g[:, 0:2], dim=-1) + g[:, 2].abs()) < _IDLE_THRESHOLD
    return torch.where(idle.unsqueeze(-1), torch.zeros_like(ph), ph)

  def cmd_features(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    cmd, arm = self.commands(s, g)
    return torch.cat([cmd, arm, self.phase(g)], dim=-1)

  def obs(self, s: torch.Tensor, g: torch.Tensor, last_action: torch.Tensor, noise: bool) -> torch.Tensor:
    """Actor observation in the env's term order (113-D)."""
    o = torch.cat([
      s[:, S_ANG_VEL_IMU], s[:, S_GRAV], self.cmd_features(s, g), s[:, S_JPOS], s[:, S_JVEL], last_action,
    ], dim=-1)
    if noise:
      o = o + (torch.rand_like(o) * 2.0 - 1.0) * self.obs_noise
    return o

  def advance_goal(self, g: torch.Tensor) -> torch.Tensor:
    g = g.clone()
    free = g[:, G_IS_HEIGHT] < 0.5
    moved = torch.clamp(g[:, G_H_TARGET] + g[:, G_VZ_RATE] * self.dt, self.h_lo, self.h_hi)
    g[:, G_H_TARGET] = torch.where(free, moved, g[:, G_H_TARGET])
    g[:, G_EP_TIME] = g[:, G_EP_TIME] + self.dt
    return g

  def analytic_reward(
    self, s: torch.Tensor, s_next: torch.Tensor, g: torch.Tensor, a: torch.Tensor, a_prev: torch.Tensor
  ) -> torch.Tensor:
    """dt-scaled tracking and action-rate terms. The env scores the post-step state against
    commands built from the pre-step state, so this does the same."""
    cmd, arm_cmd = self.commands(s, g)
    v, w = s_next[:, S_LIN_VEL_H], s_next[:, S_ANG_VEL_B]
    std, wt = self.std, self.w
    lin = torch.exp(-torch.sum((cmd[:, :2] - v[:, :2]) ** 2, -1) / std["track_linear_velocity"] ** 2)
    lin_z = torch.exp(-((cmd[:, 3] - v[:, 2]) ** 2) / std["track_linear_velocity_z"] ** 2)
    yaw_err = (cmd[:, 2] - w[:, 2]) ** 2 + 0.05 * torch.sum(w[:, :2] ** 2, -1)
    yaw = torch.exp(-yaw_err / std["track_angular_velocity"] ** 2)
    qd_arm = s_next[:, S_JVEL][:, self.arm_ids]
    arm_vel = torch.exp(-torch.mean((qd_arm - arm_cmd) ** 2, -1) / std["track_arm_joint_vel"] ** 2)
    q_arm = s_next[:, S_JPOS][:, self.arm_ids] + self.default_q[self.arm_ids]
    arm_pos = torch.exp(
      -torch.mean((g[:, G_ARM_TARGET] - q_arm) ** 2, -1) / std["track_arm_joint_pos"] ** 2
    )
    rate = torch.sum((a - a_prev) ** 2, -1)
    total = (
      wt["track_linear_velocity"] * lin + wt["track_linear_velocity_z"] * lin_z
      + wt["track_angular_velocity"] * yaw + wt["track_arm_joint_vel"] * arm_vel
      + wt["track_arm_joint_pos"] * arm_pos + wt["action_rate_l2"] * rate
    )
    return total * self.dt


def push_by_setting_velocity_flagged(env: ManagerBasedRlEnv, env_ids: torch.Tensor, **kwargs) -> None:
  """The task's push event, also flagging pushed envs so their transitions skip the world-model fit."""
  mdp.push_by_setting_velocity(env, env_ids, **kwargs)
  if hasattr(env, "_sm_pushed"):
    env._sm_pushed[env_ids] = True
