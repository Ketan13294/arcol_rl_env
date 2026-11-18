"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  arm_sine: bool = False
  """WBC-ARM: drive every arm joint along a phase-shifted sinusoid and resample a random base velocity."""
  arm_sine_period: float = 20.0
  """Sinusoid period (s). Shoulders span ~2 rad, so short periods outrun the 0.5 rad/s arm command."""
  arm_sine_amplitude: float = 0.8
  """Fraction of each joint's safe half-range."""
  plots: bool = True
  """Viser, WBC-ARM: live plots of arm joints and base velocity/height, commanded vs actual."""
  base_vel_interval: float = 4.0
  """Seconds between random base-velocity commands (drawn from the play command ranges)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _ensure_wbc_lin_vel_z_command(env_cfg: Any) -> None:
  """Ensure the flat WBC command exposes all four velocity commands.

  The flat policy expects ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z`` and
  ``lin_vel_z``.  Some play configurations only list the first three, which
  makes the command vector differ from the one used by the policy.
  """
  commands = getattr(env_cfg, "commands", None)
  if not isinstance(commands, dict) or "wbc" not in commands:
    return

  wbc_cmd = commands["wbc"]

  required_commands = ("lin_vel_x", "lin_vel_y", "ang_vel_z", "lin_pos_z")

  # Activate explicit flags when present.  Existing x/y/yaw settings are
  # preserved; this primarily enables the missing vertical command.
  for flag_name in (
    "lin_vel_x",
    "lin_vel_y",
    "ang_vel_z",
    "lin_pos_z",
    "enable_lin_vel_z",
    "command_lin_vel_z",
  ):
    if hasattr(wbc_cmd, flag_name):
      flag_value = getattr(wbc_cmd, flag_name)
      if isinstance(flag_value, bool):
        setattr(wbc_cmd, flag_name, True)

  # Ensure lin_vel_z appears in command name containers when present.
  for names_attr in ("command_names", "commands"):
    if hasattr(wbc_cmd, names_attr):
      names = getattr(wbc_cmd, names_attr)
      missing = [name for name in required_commands if name not in names]
      if isinstance(names, list):
        names.extend(missing)
      elif isinstance(names, tuple):
        setattr(wbc_cmd, names_attr, (*names, *missing))

  # Validate we are commanding lin_vel_z in WBC config.
  configured_names = set()
  for names_attr in ("command_names", "commands"):
    names = getattr(wbc_cmd, names_attr, ())
    if isinstance(names, (list, tuple)):
      configured_names.update(names)

  if not all(name in configured_names for name in required_commands):
    raise ValueError(
      "WBC command config must include all four commands: "
      f"{', '.join(required_commands)}."
    )


SPLIT_ACTOR = "src.tasks.wbc_arm.rl.split_actor:SplitMLPModel"


def _match_actor_to_checkpoint(agent_cfg: Any, ckpt: Path) -> None:
  """Split and monolithic WBC-ARM actors share obs/action layouts; pick the one the checkpoint holds."""
  keys = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"].keys()
  is_split = any(k.startswith("mlp.lower.") for k in keys)
  if is_split != (agent_cfg.actor.class_name == SPLIT_ACTOR):
    agent_cfg.actor.class_name = SPLIT_ACTOR if is_split else "MLPModel"
    print(f"[INFO]: Checkpoint holds a {'split' if is_split else 'monolithic'} actor; using {agent_cfg.actor.class_name}")


class ArmSineDemo:
  """Wraps a policy and rewrites the twist command before each step."""

  def __init__(self, env: RslRlVecEnvWrapper, policy, cfg: PlayConfig):
    u = env.unwrapped
    self.t = u.command_manager.get_term("twist")
    if getattr(self.t, "num_arm_joints", 0) == 0:
      raise ValueError("--arm-sine needs a WBC-ARM task (twist command with arm joints).")
    self.policy, self.cfg, self.dt = policy, cfg, u.step_dt
    self.center = (self.t.safe_arm_q_min + self.t.safe_arm_q_max) / 2.0
    self.half = cfg.arm_sine_amplitude * (self.t.safe_arm_q_max - self.t.safe_arm_q_min) / 2.0
    n = self.t.num_arm_joints
    self.phase = 2.0 * torch.pi * torch.arange(n, device=u.device) / n
    self.step = 0

  def __call__(self, obs) -> torch.Tensor:
    t, r = self.t, self.t.cfg.ranges
    time_s = self.step * self.dt
    # Re-applied every step: an env reset resamples these flags.
    t.time_left[:] = 1e9
    t.is_standing_env[:] = False
    t.is_heading_env[:] = False
    t.is_height_env[:] = False
    t.vz_rate[:] = 0.0
    angle = 2.0 * torch.pi * time_s / self.cfg.arm_sine_period + self.phase
    t.arm_q_target[:] = self.center + self.half * torch.sin(angle)
    if self.step % max(1, round(self.cfg.base_vel_interval / self.dt)) == 0:
      for i, rng in enumerate((r.lin_vel_x, r.lin_vel_y, r.ang_vel_z)):
        t.vel_command_b[:, i].uniform_(*rng)
    self.step += 1
    return self.policy(obs)


class TelemetryPlots:
  """Wraps a policy and plots arm joints (target vs actual) and base velocity in the viser GUI."""

  def __init__(self, env: RslRlVecEnvWrapper, policy, server, history_s: float = 10.0, every: int = 5):
    import numpy as np
    import viser

    self.np, self.policy, self.every = np, policy, every
    u = env.unwrapped
    self.t, self.robot, self.dt = u.command_manager.get_term("twist"), u.scene["robot"], u.step_dt
    self.n = int(history_s / self.dt)
    names = self.t.arm_joint_names
    self.sides = [[i for i, n in enumerate(names) if n.startswith(s)] for s in ("left", "right")]
    self.buf = {k: np.full((self.n, d), np.nan) for k, d in (("q", len(names)), ("q_t", len(names)), ("v", 4), ("v_c", 4), ("h", 2))}
    self.x = (np.arange(self.n) - self.n + 1) * self.dt
    self.step = 0
    colors = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6")
    scales = {"x": viser.uplot.Scale(time=False, auto=False, range=(float(self.x[0]), 0.0)), "y": viser.uplot.Scale(auto=True)}

    def pair_series(labels):
      out = [viser.uplot.Series(label="t (s)")]
      for c, lab in zip(colors, labels):
        out += [viser.uplot.Series(label=lab, stroke=c, width=2), viser.uplot.Series(label=f"{lab} cmd", stroke=c, width=1, dash=[6, 4])]
      return tuple(out)

    def plot(title, series):
      return server.gui.add_uplot(
        data=(self.x,) + tuple(np.full(self.n, np.nan) for _ in series[1:]),
        series=series, scales=scales, title=title, aspect=1.6,
      )

    with server.gui.add_folder("Plots"):
      self.env_idx = server.gui.add_number("Env", initial_value=0, min=0, max=u.num_envs - 1, step=1)
      self.arm_plots = [
        plot(f"{side} arm joints (rad)", pair_series([names[i].removeprefix(f"{side}_").removesuffix("_joint") for i in ids]))
        for side, ids in zip(("left", "right"), self.sides)
      ]
      self.vel_plot = plot("Base velocity (m/s, rad/s)", pair_series(("vx", "vy", "yaw rate", "vz")))
      self.h_plot = plot("Base height (m)", pair_series(("height",)))

  def __call__(self, obs) -> torch.Tensor:
    actions = self.policy(obs)
    i, t, d = int(self.env_idx.value), self.t, self.robot.data
    v = torch.cat([t.lin_vel_heading()[i, :2], d.root_link_ang_vel_b[i, 2:3], d.root_link_lin_vel_w[i, 2:3]])
    new = {
      "q": d.joint_pos[i, t.arm_joint_ids], "q_t": t.arm_q_target[i], "v": v, "v_c": t.command[i, :4],
      "h": torch.stack([t.base_height()[i], t.height_target[i]]),
    }
    for k, val in new.items():
      self.buf[k] = self.np.roll(self.buf[k], -1, axis=0)
      self.buf[k][-1] = val.detach().cpu().numpy()
    self.step += 1
    if self.step % self.every == 0:
      b = self.buf
      for p, ids in zip(self.arm_plots, self.sides):
        p.data = (self.x,) + tuple(col for j in ids for col in (b["q"][:, j], b["q_t"][:, j]))
      self.vel_plot.data = (self.x,) + tuple(col for j in range(4) for col in (b["v"][:, j], b["v_c"][:, j]))
      self.h_plot.data = (self.x, b["h"][:, 0], b["h"][:, 1])
    return actions


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  _ensure_wbc_lin_vel_z_command(env_cfg)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    _match_actor_to_checkpoint(agent_cfg, resume_path)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  if cfg.arm_sine:
    policy = ArmSineDemo(env, policy, cfg)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    server = None
    cm = env.unwrapped.command_manager
    if cfg.plots and "twist" in cm.active_terms and getattr(cm.get_term("twist"), "num_arm_joints", 0) > 0:
      import viser

      server = viser.ViserServer(label="mjlab")
      policy = TelemetryPlots(env, policy, server)
    ViserPlayViewer(env, policy, viser_server=server).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
