"""SuperMimic training (arXiv 2610.28480): fit a differentiable world model on simulator
rollouts, then improve the policy by backpropagating short imagined returns through it."""

from __future__ import annotations

import copy
import math
import os
import statistics
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger

from . import mirror as M
from .buffer import FORCED, PUSHED, ReplayBuffer
from .config import SuperMimicCfg
from .world_model import WorldModel, mlp


class SuperMimicRunner:
  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    self.env, self.cfg, self.log_dir, self.device = env, train_cfg, log_dir, device
    self.sm = SuperMimicCfg(**train_cfg.get("supermimic", {}))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    obs = env.get_observations()
    self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], ["actor", "critic"])
    actor_kwargs = {k: v for k, v in self.cfg["actor"].items() if k not in ("class_name", "cnn_cfg") and v is not None}
    if "distribution_cfg" in actor_kwargs:
      actor_kwargs["distribution_cfg"] = dict(actor_kwargs["distribution_cfg"])
    actor_cls = resolve_callable(self.cfg["actor"]["class_name"])
    self.actor = actor_cls(obs, self.cfg["obs_groups"], "actor", env.num_actions, **actor_kwargs).to(device)
    self.ref_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()

    self.mirror = M.TaskMirror(env.unwrapped)
    self.wm = WorldModel(
      M.STATE_DIM, env.num_actions, M.XI_DIM, M.CMD_FEAT_DIM, self.sm.wm_hidden, self.sm.wm_head_hidden
    ).to(device)
    self.critic = mlp(M.STATE_DIM + M.CMD_FEAT_DIM + M.G_EP_TIME + M.XI_DIM, 1, self.sm.critic_hidden).to(device)
    fused = device.startswith("cuda")
    self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=self.sm.wm_lr, fused=fused)
    self.policy_opt = torch.optim.Adam(
      self.actor.parameters(), lr=self.sm.policy_lr, betas=self.sm.policy_betas, fused=fused
    )
    self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.sm.critic_lr, fused=fused)
    self._fit_step = torch.compile(self._fit_step_impl) if self.sm.compile else self._fit_step_impl
    self._imagine_step = torch.compile(self._imagine_step_impl) if self.sm.compile else self._imagine_step_impl

    self.buffer = ReplayBuffer(
      self.sm.buffer_steps, env.num_envs, M.STATE_DIM, M.GOAL_DIM, env.num_actions, device
    )
    self.cfg.setdefault("algorithm", {}).setdefault("rnd_cfg", None)
    self.logger = Logger(
      log_dir=log_dir, cfg=self.cfg, env_cfg=env.cfg, num_envs=env.num_envs, is_distributed=False,
      gpu_world_size=1, gpu_global_rank=0, device=device,
    )
    self.logger.writer = None  # Set by init_logging_writer() in learn().
    self.current_learning_iteration = 0
    self.wm_updates = 0
    self.wm_err_ema = float("inf")
    self.gate_opened = False
    self.skipped = {"wm": 0, "policy": 0, "critic": 0}

  # Policy helpers.

  def _act(self, obs: torch.Tensor) -> torch.Tensor:
    return self.actor(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))

  def _act_ref(self, obs: torch.Tensor) -> torch.Tensor:
    return self.ref_actor(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))

  def _opt_step(self, opt, params, loss: torch.Tensor, name: str, clip_value: float | None = None) -> bool:
    """Steps only on a finite gradient, so one diverged batch can't poison the weights."""
    opt.zero_grad(set_to_none=True)
    loss.backward()
    params = list(params)
    if not torch.isfinite(nn.utils.clip_grad_norm_(params, 1.0)):
      opt.zero_grad(set_to_none=True)
      self.skipped[name] += 1
      return False
    if clip_value is not None:
      nn.utils.clip_grad_value_(params, clip_value)
    opt.step()
    return True

  def _autocast(self):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.sm.bf16 and self.device.startswith("cuda"))

  def _critic_in(self, s: torch.Tensor, g: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    return torch.cat([
      self.wm.state_norm.normalize(s), self.mirror.cmd_features(s, g), g[:, :M.G_EP_TIME], self.wm.xi_norm.normalize(xi)
    ], dim=-1)

  def _value(self, s, g, xi) -> torch.Tensor:
    return self.critic(self._critic_in(s, g, xi)).squeeze(-1)

  # Data collection.

  @torch.no_grad()
  def _collect(self, steps: int) -> None:
    mirror, buf = self.mirror, self.buffer
    for _ in range(steps):
      state, goal, last_action = mirror.read_state(), mirror.read_goal(), mirror.read_last_action()
      buf.add_state(state, goal, last_action)
      obs = self.env.get_observations()
      if self.sm.update_actor_obs_norm and self._wm_cycles() < self.sm.wm_stats_updates:
        self.actor.update_normalization(obs)
      action = self.actor(obs) + self.sm.action_noise_std * torch.randn(
        self.env.num_envs, self.env.num_actions, device=self.device
      )
      action = action.clamp(-self.sm.action_clip, self.sm.action_clip)
      forced = mirror.impulse_active()
      _, reward, dones, extras = self.env.step(action)
      time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
      done = dones.bool()
      flags = mirror.pop_pushed().to(torch.uint8) * PUSHED + forced.to(torch.uint8) * FORCED
      buf.add_outcome(action, reward, done & ~time_outs, done, flags)
      self.logger.process_env_step(reward, dones, extras)
    buf.add_state(mirror.read_state(), mirror.read_goal(), mirror.read_last_action())

  def _wm_cycles(self) -> float:
    """World-model updates in paper units (one update = wm_steps_per_update gradient steps)."""
    return self.wm_updates / self.sm.wm_steps_per_update

  def _set_lr(self) -> None:
    sm = self.sm
    frac = min(self._wm_cycles() / sm.lr_decay_updates, 1.0)
    mult = sm.lr_final_frac + (1.0 - sm.lr_final_frac) * 0.5 * (1.0 + math.cos(math.pi * frac))
    for opt, lr in ((self.wm_opt, sm.wm_lr), (self.policy_opt, sm.policy_lr), (self.critic_opt, sm.critic_lr)):
      for group in opt.param_groups:
        group["lr"] = lr * mult

  # World model.

  def _wm_update(self) -> dict[str, float]:
    sm, mirror = self.sm, self.mirror
    stats = {"wm_state": 0.0, "wm_reward": 0.0, "wm_term": 0.0, "wm_err_last": 0.0}
    for _ in range(sm.wm_steps_per_update):
      w = self.buffer.sample_windows(sm.wm_batch, sm.wm_window)
      xi = mirror.xi[w["env"]]
      if self._wm_cycles() < sm.wm_stats_updates:
        self.wm.update_stats(w["state"][:, :-1], w["state"][:, 1:], xi)
      cut = w["done"] | (w["flags"] > 0)
      alive = torch.cumprod(torch.cat([torch.ones_like(cut[:, :1]), ~cut[:, :-1]], 1).float(), 1)
      state_ok = alive * (~cut).float()
      outcome_ok = alive * ((w["flags"] & FORCED) == 0).float()

      s = w["state"][:, 0]
      l_state = l_rew = l_term = 0.0
      with self._autocast():
        for j in range(sm.wm_window):
          term = w["terminated"][:, j].float()
          s_next, err, rew_sq, bce = self._fit_step(
            s, w["action"][:, j], w["last_action"][:, j], w["goal"][:, j], xi,
            w["state"][:, j + 1], w["reward"][:, j], term,
          )
          l_state = l_state + (err * state_ok[:, j]).sum() / state_ok[:, j].sum().clamp(min=1)
          l_rew = l_rew + (rew_sq * outcome_ok[:, j]).sum() / outcome_ok[:, j].sum().clamp(min=1)
          l_term = l_term + (bce * outcome_ok[:, j]).sum() / outcome_ok[:, j].sum().clamp(min=1)
          s = s_next
      loss = (l_state + l_rew + l_term) / sm.wm_window
      if self._opt_step(self.wm_opt, self.wm.parameters(), loss, "wm"):
        self.wm_updates += 1
      k = sm.wm_window
      err_last = float((err * state_ok[:, -1]).sum() / state_ok[:, -1].sum().clamp(min=1))
      if math.isfinite(err_last):
        self.wm_err_ema = err_last if self.wm_err_ema == float("inf") else 0.98 * self.wm_err_ema + 0.02 * err_last
      stats["wm_state"] += float(l_state) / k
      stats["wm_reward"] += float(l_rew) / k
      stats["wm_term"] += float(l_term) / k
      stats["wm_err_last"] += err_last
    return {key: v / sm.wm_steps_per_update for key, v in stats.items()}

  # One model step, compiled as a unit so the many small ops fuse.

  def _fit_step_impl(self, s, a, a_prev, g, xi, s_true, r_true, term):
    mirror = self.mirror
    s_next = self.wm.step(s, a, xi)
    err = self.wm.state_error(s_next, s_true)
    target = r_true - mirror.analytic_reward(s, s_next, g, a, a_prev).detach() + mirror.term_penalty * term
    pred = self.wm.reward_residual(s, s_next, mirror.cmd_features(s, g))
    rew_sq = ((pred - target) / mirror.dt) ** 2
    bce = F.binary_cross_entropy_with_logits(self.wm.term_logit(s_next), term, reduction="none")
    return s_next, err, rew_sq, bce

  def _imagine_step_impl(self, s, g, a_prev, xi):
    mirror = self.mirror
    obs = mirror.obs(s, g, a_prev, noise=True)
    a = self._act(obs).float().clamp(-self.sm.action_clip, self.sm.action_clip)
    with torch.no_grad():
      a_ref = self._act_ref(obs).float()
    s_next = self.wm.clamp_state(self.wm.step(s, a, xi), self.sm.state_clip)
    d = torch.sigmoid(self.wm.term_logit(s_next))
    r = (
      mirror.analytic_reward(s, s_next, g, a, a_prev)
      + self.wm.reward_residual(s, s_next, mirror.cmd_features(s, g))
      - mirror.term_penalty * d
    )
    return a, s_next, d, r, torch.sum((a - a_ref) ** 2, -1)

  # Imagination.

  def _imagine(self, train_policy: bool) -> dict[str, float]:
    sm, mirror = self.sm, self.mirror
    start = self.buffer.sample_starts(sm.rollouts)
    s, g, a_prev = start["state"], start["goal"], start["last_action"]
    xi = mirror.xi[start["env"]]
    self.wm.requires_grad_(False)
    self.critic.requires_grad_(False)

    survival = torch.ones(sm.rollouts, device=self.device)
    ret = torch.zeros(sm.rollouts, device=self.device)
    bc = torch.zeros(sm.rollouts, device=self.device)
    states, goals, rewards, terms = [s.detach()], [g], [], []
    with torch.set_grad_enabled(train_policy), self._autocast():
      for t in range(sm.horizon):
        a, s_next, d, r, bc_t = self._imagine_step(s, g, a_prev, xi)
        ret = ret + (sm.gamma**t) * survival * r
        bc = bc + bc_t / sm.horizon
        survival = survival * (1.0 - d)
        g = mirror.advance_goal(g)
        s, a_prev = s_next, a
        states.append(s.detach())
        goals.append(g)
        rewards.append(r.detach())
        terms.append(d.detach())
      v_last = self._value(s, g, xi)
      objective = ret + (sm.gamma**sm.horizon) * survival * v_last

    stats = {
      "imagined_return": float(objective.mean()), "imagined_survival": float(survival.mean()), "bc": float(bc.mean()),
    }
    if train_policy:
      loss = -objective.mean() + sm.bc_coef * bc.mean()
      self._opt_step(self.policy_opt, self.actor.parameters(), loss, "policy", clip_value=0.1)
    self.wm.requires_grad_(True)
    self.critic.requires_grad_(True)
    stats["critic_loss"] = self._critic_update(states, goals, rewards, terms, xi)
    return stats

  def _critic_update(self, states, goals, rewards, terms, xi) -> float:
    sm = self.sm
    with torch.no_grad():
      s_all = torch.stack(states)  # [H+1, B, S]
      g_all = torch.stack(goals)
      h, b = len(rewards), s_all.shape[1]
      values = self._value(s_all.flatten(0, 1), g_all.flatten(0, 1), xi.repeat(h + 1, 1)).view(h + 1, b)
      targets = torch.zeros(h, b, device=self.device)
      nxt = values[h]
      for t in reversed(range(h)):
        nxt = rewards[t] + sm.gamma * (1.0 - terms[t]) * ((1.0 - sm.lam) * values[t + 1] + sm.lam * nxt)
        targets[t] = nxt
      x = self._critic_in(s_all[:h].flatten(0, 1), g_all[:h].flatten(0, 1), xi.repeat(h, 1))
      y = targets.flatten()
      ok = torch.isfinite(y) & torch.isfinite(x).all(-1)
      x, y = x[ok], y[ok]
    total, n = 0.0, x.shape[0]
    for _ in range(sm.critic_epochs):
      for idx in torch.randperm(n, device=self.device).chunk(sm.critic_minibatches):
        loss = F.mse_loss(self.critic(x[idx]).squeeze(-1), y[idx])
        self._opt_step(self.critic_opt, self.critic.parameters(), loss, "critic")
        total += float(loss)
    return total / (sm.critic_epochs * sm.critic_minibatches)

  # Training loop.

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    self.logger.init_logging_writer()
    sm = self.sm
    if sm.wm_init:
      self._load_wm(sm.wm_init)
    updates_per_iter = max(1, round(sm.update_to_data * sm.steps_per_iteration))
    start_it = self.current_learning_iteration
    warmup_end = start_it + sm.warmup_iterations
    self._collect(max(sm.wm_window + 1, sm.steps_per_iteration))

    for it in range(start_it, start_it + num_learning_iterations):
      t0 = time.time()
      self._collect(sm.steps_per_iteration)
      t_collect = time.time() - t0
      self.actor.train()
      agg: dict[str, list[float]] = {}
      if not self.gate_opened and it >= warmup_end and self.wm_err_ema < sm.wm_gate_err:
        self.gate_opened = True
        if self.log_dir:
          self._save_wm(os.path.join(self.log_dir, "wm_gate.pt"))
      phase = "warmup" if it < warmup_end else "policy" if self.gate_opened else "gated"
      self._set_lr()
      for _ in range(updates_per_iter):
        for key, v in {**self._wm_update(), **self._imagine(train_policy=phase == "policy")}.items():
          agg.setdefault(key, []).append(v)
      self.current_learning_iteration = it + 1
      self._log(it, {k: statistics.fmean(v) for k, v in agg.items()}, t_collect, time.time() - t0, phase)
      if self.log_dir and (it + 1) % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
    if self.log_dir:
      self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

  def _log(self, it: int, stats: dict[str, float], t_collect: float, t_total: float, phase: str) -> None:
    lg = self.logger
    steps = self.sm.steps_per_iteration * self.env.num_envs
    if lg.writer is not None:
      for k, v in stats.items():
        lg.writer.add_scalar(f"SuperMimic/{k}", v, it)
      lg.writer.add_scalar("SuperMimic/wm_err_ema", self.wm_err_ema, it)
      lg.writer.add_scalar("SuperMimic/policy_lr", self.policy_opt.param_groups[0]["lr"], it)
      lg.writer.add_scalar("SuperMimic/policy_updating", float(phase == "policy"), it)
      for k, v in self.skipped.items():
        lg.writer.add_scalar(f"SuperMimic/skipped_{k}", v, it)
      lg.writer.add_scalar("Perf/collection_time", t_collect, it)
      lg.writer.add_scalar("Perf/iteration_time", t_total, it)
      lg.writer.add_scalar("Perf/total_fps", steps / t_total, it)
      if lg.rewbuffer:
        lg.writer.add_scalar("Train/mean_reward", statistics.fmean(lg.rewbuffer), it)
        lg.writer.add_scalar("Train/mean_episode_length", statistics.fmean(lg.lenbuffer), it)
      for key in (lg.ep_extras[0] if lg.ep_extras else {}):
        vals = [torch.as_tensor(e[key], device=self.device).float().mean() for e in lg.ep_extras if key in e]
        if vals:
          lg.writer.add_scalar(key if "/" in key else f"Episode/{key}", float(torch.stack(vals).mean()), it)
    lg.ep_extras.clear()
    ep_len = statistics.fmean(lg.lenbuffer) if lg.lenbuffer else float("nan")
    ep_rew = statistics.fmean(lg.rewbuffer) if lg.rewbuffer else float("nan")
    print(
      f"[supermimic] it {it + 1} ({phase}) | {t_total:.2f}s (collect {t_collect:.2f}s) | "
      f"ep_len {ep_len:.0f} ep_rew {ep_rew:.2f} | wm_state {stats['wm_state']:.4f} err@{self.sm.wm_window} "
      f"{stats['wm_err_last']:.4f} (ema {self.wm_err_ema:.3f}) rew {stats['wm_reward']:.4f} "
      f"term {stats['wm_term']:.4f} | J {stats['imagined_return']:.3f} surv {stats['imagined_survival']:.3f} "
      f"bc {stats['bc']:.4f} critic {stats['critic_loss']:.4f} | skipped {sum(self.skipped.values())}",
      flush=True,
    )

  # Checkpoints, export, inference.

  def save(self, path: str, infos: dict | None = None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    torch.save({
      "actor_state_dict": self.actor.state_dict(),
      "ref_actor_state_dict": self.ref_actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "world_model_state_dict": self.wm.state_dict(),
      "optimizer_state_dict": {
        "policy": self.policy_opt.state_dict(), "critic": self.critic_opt.state_dict(), "wm": self.wm_opt.state_dict(),
      },
      "iter": self.current_learning_iteration,
      "wm_updates": self.wm_updates,
      "supermimic_cfg": asdict(self.sm),
      "infos": {**(infos or {}), "env_state": env_state},
    }, path)
    policy_dir = os.path.dirname(path)
    self.export_policy_to_onnx(policy_dir, "policy.onnx")
    run_name = "local"
    if self.logger.writer is not None and getattr(self.logger, "logger_type", "") == "wandb":
      import wandb

      run_name = wandb.run.name if wandb.run else "local"
    attach_metadata_to_onnx(os.path.join(policy_dir, "policy.onnx"), get_base_metadata(self.env.unwrapped, run_name))
    if self.logger.writer is not None and self.cfg.get("upload_model", True):
      self.logger.save_model(path, self.current_learning_iteration)

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
    """Loads a SuperMimic checkpoint, or warm-starts the actor from a split PPO checkpoint."""
    ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
    load_cfg = load_cfg or {"actor": True, "critic": True, "optimizer": True, "iteration": True}
    if load_cfg.get("actor"):
      self.actor.load_state_dict(ckpt["actor_state_dict"], strict=strict)
      self.ref_actor.load_state_dict(ckpt.get("ref_actor_state_dict", ckpt["actor_state_dict"]), strict=strict)
    if "world_model_state_dict" in ckpt:
      if load_cfg.get("critic"):
        self.critic.load_state_dict(ckpt["critic_state_dict"], strict=strict)
        self.wm.load_state_dict(ckpt["world_model_state_dict"], strict=strict)
        self.wm_updates = ckpt.get("wm_updates", 0)
        self.sm.warmup_iterations = 0
        self.gate_opened = True
      if load_cfg.get("optimizer"):
        opt = ckpt["optimizer_state_dict"]
        self.policy_opt.load_state_dict(opt["policy"])
        self.critic_opt.load_state_dict(opt["critic"])
        self.wm_opt.load_state_dict(opt["wm"])
    if load_cfg.get("iteration"):
      self.current_learning_iteration = ckpt["iter"]
    infos = ckpt.get("infos") or {}
    if "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  def _save_wm(self, path: str) -> None:
    torch.save({
      "world_model_state_dict": self.wm.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "optimizer_state_dict": {"critic": self.critic_opt.state_dict(), "wm": self.wm_opt.state_dict()},
      "wm_updates": self.wm_updates,
    }, path)

  def _load_wm(self, path: str) -> None:
    ckpt = torch.load(path, map_location=self.device, weights_only=False)
    self.wm.load_state_dict(ckpt["world_model_state_dict"])
    self.critic.load_state_dict(ckpt["critic_state_dict"])
    self.wm_opt.load_state_dict(ckpt["optimizer_state_dict"]["wm"])
    self.critic_opt.load_state_dict(ckpt["optimizer_state_dict"]["critic"])
    self.wm_updates = ckpt["wm_updates"]
    print(f"[supermimic] world model and critic from {path} ({self.wm_updates} updates)", flush=True)

  def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
    onnx_model = self.actor.as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model, onnx_model.get_dummy_inputs(), os.path.join(path, filename), export_params=True,
      opset_version=18, verbose=verbose, input_names=onnx_model.input_names,
      output_names=onnx_model.output_names, dynamic_axes={}, dynamo=False,
    )

  def get_inference_policy(self, device: str | None = None):
    self.actor.eval()
    return self.actor.to(device)

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    self.logger.git_status_repos.append(repo_file_path)
