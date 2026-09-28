from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")
    
    if self.cfg.height_command and self.cfg.ranges.lin_pos_z is None:
      raise ValueError("height_command=True but ranges.lin_pos_z is set to None.")
    if self.cfg.ranges.lin_pos_z and not self.cfg.height_command:
      raise ValueError("ranges.lin_pos_z is set but height_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
    self.height_target = torch.zeros(self.num_envs, device=self.device)
    self.height_error = torch.zeros(self.num_envs, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_height_env = torch.zeros_like(self.is_heading_env)
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    # --- Arm Command Additions ---
    self.arm_joint_ids = cfg.arm_joint_ids
    self.num_arm_joints = len(self.arm_joint_ids) if self.arm_joint_ids else 0
    
    if self.num_arm_joints > 0:
      # Fetch arm joint ranges and apply 0.9 safety scaling
      jnt_range = env.sim.model.jnt_range[self.arm_joint_ids]
      q_min = torch.tensor(jnt_range[:, 0], device=self.device)
      q_max = torch.tensor(jnt_range[:, 1], device=self.device)
      
      q_center = (q_max + q_min) / 2.0
      q_half = (q_max - q_min) / 2.0
      
      self.safe_arm_q_min = q_center - 0.9 * q_half
      self.safe_arm_q_max = q_center + 0.9 * q_half

      # Fetch maximum velocity limits from physics model and apply 0.9 safety factor
      # Note: env.sim.model.dof_armature / dof_limit / actuator_velocity_limit depend on MuJoCo spec
      max_arm_vel = torch.tensor(env.sim.model.actuator_gear[self.arm_joint_ids, 0], device=self.device) # or joint_vel_limit
      self.safe_arm_vel_limit = 0.9 * max_arm_vel
      
      self.arm_q_target = torch.zeros((self.num_envs, self.num_arm_joints), device=self.device)
      self.arm_q_dot_cmd = torch.zeros((self.num_envs, self.num_arm_joints), device=self.device)
      self.is_arm_commanded = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
    # -----------------------------

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_z"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_height_slider: viser.GuiSliderHandle | None = None
    self._joystick_arm_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None
    self._enable_posZ_handle = True

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )
    self.metrics["error_vel_z"] += (
      torch.abs(self.vel_command_b[:, 3] - self.robot.data.root_link_lin_vel_w[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    self.vel_command_b[env_ids, 3] = r.uniform_(*self.cfg.ranges.lin_vel_z)

    if self.cfg.ranges.lin_vel_x[0] < -0.2:
      num_red_envs_down = max(1, int(0.1 * len(env_ids)))
      red_env_ids_down = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_red_envs_down]]
      self.vel_command_b[red_env_ids_down, 0] = torch.zeros(len(red_env_ids_down), device=self.device).uniform_(-0.2, 0)

    if self.cfg.ranges.lin_vel_x[1] > 0.2:
      num_red_envs_up = max(1, int(0.1 * len(env_ids)))
      red_env_ids_up = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_red_envs_up]]
      self.vel_command_b[red_env_ids_up, 0] = torch.zeros(len(red_env_ids_up), device=self.device).uniform_(0, 0.2)

    if self.cfg.ranges.lin_vel_y[0] < -0.2:
      num_red_envs_down = max(1, int(0.1 * len(env_ids)))
      red_env_ids_down = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_red_envs_down]]
      self.vel_command_b[red_env_ids_down, 1] = torch.zeros(len(red_env_ids_down), device=self.device).uniform_(-0.2, 0)

    if self.cfg.ranges.lin_vel_y[1] > 0.2:
      num_red_envs_up = max(1, int(0.1 * len(env_ids)))
      red_env_ids_up = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_red_envs_up]]
      self.vel_command_b[red_env_ids_up, 1] = torch.zeros(len(red_env_ids_up), device=self.device).uniform_(0, 0.2)

    if self.cfg.ranges.ang_vel_z[0] < -0.2:
      num_red_envs_down = max(1, int(0.1 * len(env_ids)))
      red_env_ids_down = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_red_envs_down]]
      self.vel_command_b[red_env_ids_down, 2] = torch.zeros(len(red_env_ids_down), device=self.device).uniform_(-0.2, 0)

    if self.cfg.ranges.ang_vel_z[1] > 0.2:
      num_red_envs_up = max(1, int(0.1 * len(env_ids)))
      red_env_ids_up = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_red_envs_up]]
      self.vel_command_b[red_env_ids_up, 2] = torch.zeros(len(red_env_ids_up), device=self.device).uniform_(0, 0.2)

    if self.cfg.ranges.lin_vel_z[0] < -0.1:
      num_red_envs_down = max(1, int(0.1 * len(env_ids)))
      red_env_ids_down = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_red_envs_down]]
      self.vel_command_b[red_env_ids_down, 3] = torch.zeros(len(red_env_ids_down), device=self.device).uniform_(-0.1, 0)

    if self.cfg.ranges.lin_vel_z[1] > 0.1:
      num_red_envs_up = max(1, int(0.1 * len(env_ids)))
      red_env_ids_up = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_red_envs_up]]
      self.vel_command_b[red_env_ids_up, 3] = torch.zeros(len(red_env_ids_up), device=self.device).uniform_(0, 0.1)

    if self.cfg.height_command:
      assert self.cfg.ranges.lin_pos_z is not None
      self.height_target[env_ids] = r.uniform_(*self.cfg.ranges.lin_pos_z)
      self.is_height_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_height_envs

    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    self.vel_command_b[env_ids, :3] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=-1)+torch.abs(self.vel_command_b[env_ids, 2]) > 0.1).unsqueeze(1)
    self.vel_command_b[env_ids, 3] *= (torch.abs(self.vel_command_b[env_ids, 3]) > 0.1).float()

    # --- Arm Command Resampling ---
    if self.num_arm_joints > 0:
      # 50% probability that active arm commands are assigned, 50% idle default pose
      arm_cmd_mask = (torch.rand(len(env_ids), device=self.device) < self.cfg.arm_command_prob)
      
      rand_arm = torch.rand((len(env_ids), self.num_arm_joints), device=self.device)
      sampled_arm_q = self.safe_arm_q_min + rand_arm * (self.safe_arm_q_max - self.safe_arm_q_min)
      default_arm_q = self.robot.data.default_joint_pos[env_ids][:, self.arm_joint_ids]
      
      self.arm_q_target[env_ids] = torch.where(arm_cmd_mask.unsqueeze(-1), sampled_arm_q, default_arm_q)
    # ------------------------------

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      
      body_z_world = quat_apply(
        root_quat,
        torch.cat((torch.zeros_like(lin_vel_b[:, :2]), torch.ones_like(lin_vel_b[:, 2:])), dim=-1),
      )
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_lin_vel_w += (
        (self.vel_command_b[init_vel_env_ids, 3] - root_lin_vel_w[:, 2])
        / body_z_world[:, 2]
      ).unsqueeze(-1) * body_z_world

      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    if self.cfg.height_command:
      self.height_error = self.height_target - self.robot.data.root_link_pos_w[:, 2]
      env_ids = self.is_height_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 3] = torch.clip(
        self.cfg.height_control_stiffness * self.height_error[env_ids],
        min=self.cfg.ranges.lin_vel_z[0],
        max=self.cfg.ranges.lin_vel_z[1],
      )
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    
    # --- Arm Command Updates ---
    if self.num_arm_joints > 0:
      curr_arm_q = self.robot.data.joint_pos[:, self.arm_joint_ids]
      raw_q_dot_cmd = self.cfg.arm_kd * (self.arm_q_target - curr_arm_q)
      
      # Clamp commanded velocity to 90% of maximum joint velocity
      self.arm_q_dot_cmd = torch.clamp(
          raw_q_dot_cmd, 
          min=-self.safe_arm_vel_limit, 
          max=self.safe_arm_vel_limit
      )
      
      self.is_arm_commanded = (torch.norm(self.arm_q_dot_cmd, dim=-1, keepdim=True) > self.cfg.arm_vel_deadband)
    # ---------------------------

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
      ("lin_vel_z", ranges.lin_vel_z[1]),
    ]
    sliders: list = []
    height_slider = None
    arm_sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in filter(None, axes):
        gui_max = max(0.1, abs(max_val))
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=gui_max,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-gui_max,
          max=gui_max,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      if self.cfg.height_command and ranges.lin_pos_z is not None:
        height_slider = server.gui.add_slider(
          "lin_pos_z",
          min=ranges.lin_pos_z[0],
          max=ranges.lin_pos_z[1],
          step=0.01,
          initial_value=0.782,
        )

      if self.num_arm_joints > 0:
        with server.gui.add_folder("Arm Joint Targets"):
          for idx in range(self.num_arm_joints):
            min_val = float(self.safe_arm_q_min[idx].item())
            max_val = float(self.safe_arm_q_max[idx].item())
            arm_slider = server.gui.add_slider(
              f"Arm Joint {idx}",
              min=min_val,
              max=max_val,
              step=0.02,
              initial_value=0.0,
            )
            arm_sliders.append(arm_slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0
        if height_slider is not None and ranges.lin_pos_z is not None:
          height_slider.value = 0.782
        for s in arm_sliders:
          s.value = 0.0

      enable_posZ_btn = server.gui.add_button("Enable Z position:", icon=Icon.SQUARE_X)

      @enable_posZ_btn.on_click
      def _(_) -> None:
        self._enable_posZ_handle = not self._enable_posZ_handle

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_height_slider = height_slider
    self._joystick_arm_sliders = arm_sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value
      if self._joystick_height_slider is not None and self._enable_posZ_handle:
        self.height_target[idx] = self._joystick_height_slider.value
        self.is_height_env[idx] = True
        height_error = self.height_target[idx] - self.robot.data.root_link_pos_w[idx, 2]
        self.vel_command_b[idx, 3] = torch.clamp(
          self.cfg.height_control_stiffness * height_error,
          min=self.cfg.ranges.lin_vel_z[0],
          max=self.cfg.ranges.lin_vel_z[1],
        )
      if len(self._joystick_arm_sliders) > 0:
        for j_idx, s in enumerate(self._joystick_arm_sliders):
          self.arm_q_target[idx, j_idx] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], cmd[3]])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      cmd_ang_from = local_to_world(np.array([0.15, 0, z_offset]) * scale)
      cmd_ang_to = local_to_world(
        (np.array([0.15, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      act_ang_from = local_to_world(np.array([0.15, 0, z_offset]) * scale)
      act_ang_to = local_to_world(
        (np.array([0.15, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  height_command: bool = True
  height_control_stiffness: float = 8.0
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_height_envs: float = 1.0
  init_velocity_prob: float = 0.0

  # Arm Joint Parameters
  arm_joint_ids: list[int] = field(default_factory=list)
  arm_kd: float = 8.0
  arm_command_prob: float = 0.5
  arm_vel_deadband: float = 0.05

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    lin_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None
    lin_pos_z: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
    if self.height_command and self.ranges.lin_pos_z is None:
      raise ValueError(
        "Height commands are enabled (height_command=True) but "
        "the `ranges.lin_pos_z` range is not specified."
      )