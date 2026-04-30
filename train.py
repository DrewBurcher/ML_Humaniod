"""
Training script for PPO and SAC on the T1 bipedal walking environment.

Features:
    - PyBullet GUI stays open the whole time (you watch the robot learn live)
    - Separate matplotlib dashboard shows all stats, plots, run info
    - Auto-commits to git before each run, saves commit hash in run metadata
    - Saves all hyperparams, reward weights, env config as JSON
    - Supports pause (Ctrl+C) and resume (--resume runs/my_run)

Usage:
    python train.py --algo sac --timesteps 100000 --name my_run
    python train.py --resume runs/my_run --algo sac --timesteps 50000
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pybullet as pb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import env as _  # noqa – triggers gym.register
import gymnasium as gym

from config import PPO_CONFIG, SAC_CONFIG, ENV_CONFIG, REWARD_WEIGHTS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _git_info():
    """Capture current git state."""
    info = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["commit_short"] = info["commit"][:8]
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
        info["remote_url"] = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception as e:
        info["error"] = str(e)
    return info


def _auto_commit():
    """Auto-commit all tracked + new .py files before a training run."""
    try:
        subprocess.run(["git", "add", "*.py", "config.py", "requirements.txt",
                         ".gitignore"],
                       capture_output=True)
        result = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True)
        if result.stdout.strip():
            subprocess.run(
                ["git", "commit", "-m",
                 f"Auto-commit before training run ({datetime.now().strftime('%Y-%m-%d %H:%M')})"],
                capture_output=True, text=True
            )
            print("[Git] Auto-committed changes before training run.")
            subprocess.run(["git", "push"], capture_output=True, text=True)
        else:
            print("[Git] Working tree clean, no commit needed.")
    except Exception as e:
        print(f"[Git] Warning: auto-commit failed: {e}")


def _save_run_metadata(log_dir, algo_name, total_timesteps, reward_weights,
                       resumed_from=None, extra=None):
    """Save all run metadata as JSON for reproducibility."""
    meta_path = os.path.join(log_dir, "run_metadata.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta.setdefault("training_history", [])
    else:
        meta = {"training_history": []}

    git = _git_info()

    session = {
        "timestamp": datetime.now().isoformat(),
        "algo": algo_name,
        "timesteps_this_session": total_timesteps,
        "resumed_from": resumed_from,
        "git": git,
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
    }
    if extra:
        session.update(extra)

    meta["training_history"].append(session)

    algo_cfg = PPO_CONFIG.copy() if algo_name == "ppo" else SAC_CONFIG.copy()
    meta["algo_config"] = algo_cfg
    meta["algo_config"]["policy_kwargs"] = str(algo_cfg.get("policy_kwargs", {}))
    meta["env_config"] = ENV_CONFIG.copy()
    meta["reward_weights"] = reward_weights if reward_weights else REWARD_WEIGHTS.copy()
    meta["algo"] = algo_name
    meta["run_name"] = os.path.basename(log_dir)
    meta["total_timesteps_all_sessions"] = sum(
        s.get("timesteps_this_session", 0) for s in meta["training_history"]
    )

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return meta


# ── Environment factory ──────────────────────────────────────────────────────

def make_env(render_mode=None, reward_weights=None, monitor_dir=None, mask_arms=False):
    def _init():
        e = gym.make("T1Walking-v0", render_mode=render_mode,
                     reward_weights=reward_weights, mask_arms=mask_arms)
        e = Monitor(e, filename=os.path.join(monitor_dir, "monitor") if monitor_dir else None)
        return e
    return _init


# ── Video eval callback ──────────────────────────────────────────────────────

class VideoEvalCallback(EvalCallback):
    """EvalCallback that also records one rgb_array episode per eval checkpoint.

    Saves to <log_dir>/videos/eval_step_XXXXXXX.mp4 (or .gif if ffmpeg missing).
    Recording runs on a separate headless rgb_array env so the GUI isn't affected.
    """

    def __init__(self, *args, video_dir: str = "", fps: int = 30,
                 video_freq: int = 100_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_dir = video_dir
        self.fps = fps
        self.video_freq = video_freq
        self._last_video_step = 0   # tracks last recorded floor(num_timesteps/video_freq)
        os.makedirs(video_dir, exist_ok=True)
        self._video_env = None  # created lazily on first eval

    def _init_video_env(self):
        """Build a headless rgb_array env with the same VecNormalize stats."""
        import env as _env_reg  # noqa — ensures T1Walking-v0 is registered
        rec_env = DummyVecEnv([lambda: gym.make("T1Walking-v0", render_mode="rgb_array")])
        # Clone normalisation stats from the eval env (read-only, no reward norm)
        rec_env = VecNormalize(rec_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, training=False)
        if hasattr(self.eval_env, "obs_rms"):
            rec_env.obs_rms = self.eval_env.obs_rms
            rec_env.ret_rms = self.eval_env.ret_rms
        self._video_env = rec_env

    def _on_step(self) -> bool:
        result = super()._on_step()
        # EvalCallback sets self.last_mean_reward after each eval round.
        # Record whenever num_timesteps crosses a video_freq boundary.
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            current_bucket = self.num_timesteps // self.video_freq
            if current_bucket > self._last_video_step:
                self._last_video_step = current_bucket
                self._record_episode()
        return result

    def _record_episode(self):
        try:
            if self._video_env is None:
                self._init_video_env()

            # Sync normalisation stats in case they've updated since last eval
            if hasattr(self.eval_env, "obs_rms"):
                self._video_env.obs_rms = self.eval_env.obs_rms

            frames = []
            obs = self._video_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, _ = self._video_env.step(action)
                done = dones[0]
                frame = self._video_env.envs[0].render()
                if frame is not None:
                    frames.append(frame)

            if not frames:
                return

            step_tag = f"{self.num_timesteps:07d}"
            vid_path = os.path.join(self.video_dir, f"eval_step_{step_tag}.mp4")
            self._save_video(frames, vid_path)
        except Exception as e:
            print(f"[VideoEval] Warning: recording failed: {e}")

    def _save_video(self, frames, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        im = ax.imshow(frames[0])

        def update(i):
            im.set_array(frames[i])
            return [im]

        anim = FuncAnimation(fig, update, frames=len(frames),
                             interval=1000 // self.fps, blit=True)
        try:
            anim.save(path, writer=FFMpegWriter(fps=self.fps))
            print(f"[VideoEval] Saved: {os.path.basename(path)}")
        except Exception:
            gif = path.replace(".mp4", ".gif")
            anim.save(gif, writer="pillow", fps=self.fps)
            print(f"[VideoEval] Saved (GIF): {os.path.basename(gif)}")
        plt.close(fig)

    def __del__(self):
        if self._video_env is not None:
            try:
                self._video_env.close()
            except Exception:
                pass


# ── Curriculum callback ──────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Unlocks arm joints at a given timestep (phase 1 → phase 2 transition).

    During phase 1 the env masks arm actions, holding arms at their neutral
    position so the network learns to balance/walk with fewer DOF. At
    `switch_at_timestep` we call set_mask_arms(False) on every env so the
    full 21-DOF policy takes over for phase 2.
    """

    def __init__(self, switch_at_timestep: int, verbose=0):
        super().__init__(verbose)
        self.switch_at = switch_at_timestep
        self.switched = False

    def _on_step(self) -> bool:
        if not self.switched and self.num_timesteps >= self.switch_at:
            try:
                self.training_env.env_method("set_mask_arms", False)
                self.switched = True
                print(f"\n[Curriculum] Phase 2 — arms unlocked at "
                      f"timestep {self.num_timesteps:,}")
            except Exception as e:
                print(f"[Curriculum] Warning: could not unlock arms: {e}")
        return True


# ── Camera follow callback ───────────────────────────────────────────────────

class PauseCallback(BaseCallback):
    """Blocks the training loop while .paused flag file contains '1'."""

    def __init__(self, pause_flag_path="", verbose=0):
        super().__init__(verbose)
        self.pause_flag_path = pause_flag_path
        self._was_paused = False

    def _on_step(self) -> bool:
        if not self.pause_flag_path:
            return True
        while True:
            try:
                with open(self.pause_flag_path, "r") as f:
                    paused = f.read().strip() == "1"
            except Exception:
                paused = False
            if not paused:
                if self._was_paused:
                    print("[Pause] Resumed.")
                    self._was_paused = False
                return True
            if not self._was_paused:
                print(f"[Pause] Paused at timestep {self.num_timesteps:,}. "
                      f"Click Resume in dashboard to continue.")
                self._was_paused = True
            time.sleep(0.5)


class CameraFollowCallback(BaseCallback):
    """Follows the robot with the PyBullet camera and handles render toggle."""

    def __init__(self, render_flag_path="", verbose=0):
        super().__init__(verbose)
        self.render_flag_path = render_flag_path
        self._rendering = True
        self._check_interval = 50  # check flag file every N steps

    def _on_step(self) -> bool:
        # Periodically check if dashboard toggled rendering off/on
        if self.num_timesteps % self._check_interval == 0 and self.render_flag_path:
            try:
                with open(self.render_flag_path, "r") as f:
                    want_render = f.read().strip() == "1"
                if want_render != self._rendering:
                    self._rendering = want_render
                    raw_env = self.training_env.envs[0].unwrapped
                    if want_render:
                        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1,
                                                     physicsClientId=raw_env.client)
                    else:
                        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0,
                                                     physicsClientId=raw_env.client)
            except Exception:
                pass

        return True


# ── Metrics logging callback ─────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """Logs reward components and SAC/PPO losses to a JSON file for the dashboard."""

    def __init__(self, log_dir="", save_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.metrics_path = os.path.join(log_dir, "metrics.json")
        self._reward_components = {}  # accumulate per episode
        self._episode_count = 0
        self._data = {
            "reward_components": [],   # per-episode avg breakdown
            "losses": [],              # actor/critic loss snapshots
        }
        # Load existing if resuming
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r") as f:
                    self._data = json.load(f)
                # Restore episode count so numbers don't restart and overlap
                if self._data["reward_components"]:
                    self._episode_count = self._data["reward_components"][-1]["episode"] + 1
            except Exception:
                pass

    def _on_step(self) -> bool:
        # Accumulate reward components from info
        infos = self.locals.get("infos", [{}])
        for info in infos:
            ri = info.get("reward_info", {})
            for k, v in ri.items():
                if k == "forward_vel":
                    # Track forward velocity separately (not abs, and it's a mean not sum)
                    self._reward_components["_fvel_sum"] = (
                        self._reward_components.get("_fvel_sum", 0) + float(v))
                else:
                    self._reward_components[k] = self._reward_components.get(k, 0) + abs(float(v))
            self._reward_components["_steps"] = self._reward_components.get("_steps", 0) + 1

            # Episode ended
            if info.get("episode"):
                steps = max(self._reward_components.pop("_steps", 1), 1)
                fvel_sum = self._reward_components.pop("_fvel_sum", 0.0)
                entry = {
                    "episode": self._episode_count,
                    "timestep": int(self.num_timesteps),
                    "forward_vel_mean": round(fvel_sum / steps, 4),
                    "x_distance": round(float(info.get("x_distance", 0.0)), 4),
                }
                for k, v in self._reward_components.items():
                    entry[k] = round(v / steps, 4)  # per-step average
                # Energy per meter (avoid div/0)
                dist = max(abs(entry["x_distance"]), 0.01)
                entry["energy_per_meter"] = round(
                    entry.get("energy_penalty", 0.0) * steps / dist, 4)
                self._data["reward_components"].append(entry)
                self._reward_components = {}
                self._episode_count += 1

        # Log losses from the model's logger
        if self.num_timesteps % self.save_freq == 0:
            loss_entry = {"timestep": int(self.num_timesteps)}
            try:
                logger = self.model.logger.name_to_value
                for key in ["train/actor_loss", "train/critic_loss",
                             "train/ent_coef_loss", "train/ent_coef",
                             "train/policy_gradient_loss", "train/value_loss",
                             "train/entropy_loss"]:
                    if key in logger:
                        loss_entry[key.split("/")[1]] = round(float(logger[key]), 6)
            except Exception:
                pass
            if len(loss_entry) > 1:
                self._data["losses"].append(loss_entry)

            # Save to disk
            try:
                with open(self.metrics_path, "w") as f:
                    json.dump(self._data, f)
            except Exception:
                pass

        return True


# ── Periodic save callback (for resume support) ─────────────────────────────

class PeriodicSaveCallback(BaseCallback):
    """Saves model + normalizer periodically so training can be resumed."""

    def __init__(self, save_freq=10_000, log_dir="", algo_name="", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.algo_name = algo_name

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.model.save(os.path.join(self.log_dir, f"{self.algo_name}_latest"))
            self.training_env.save(os.path.join(self.log_dir, "vecnormalize.pkl"))
            # Save replay buffer for SAC/off-policy (enables seamless resume)
            if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                buf_path = os.path.join(self.log_dir, "replay_buffer")
                self.model.save_replay_buffer(buf_path)
        return True


# ── Main train function ─────────────────────────────────────────────────────

def train(algo_name, total_timesteps, run_name, reward_weights=None,
          live_plot=True, resume=False, headless=False,
          mask_arms=False, curriculum_step=None, record_evals=False):
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    rw = reward_weights if reward_weights else REWARD_WEIGHTS

    # ── Auto-commit to git ──
    _auto_commit()

    # ── Determine if resuming ──
    latest_model_path = os.path.join(log_dir, f"{algo_name}_latest.zip")
    final_model_path = os.path.join(log_dir, f"{algo_name}_final.zip")
    vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")

    can_resume = resume and (os.path.exists(latest_model_path) or
                              os.path.exists(final_model_path))

    if can_resume:
        model_to_load = (latest_model_path if os.path.exists(latest_model_path)
                         else final_model_path)
        print(f"\n[Resume] Loading model from: {model_to_load}")

        # Archive current monitor CSV so dashboard can show all sessions
        import glob
        monitor_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        monitor_files = [f for f in monitor_files if "session_" not in f]
        existing_archives = glob.glob(os.path.join(log_dir, "monitor_session_*.csv"))
        next_session = len(existing_archives) + 1
        for mf in monitor_files:
            archive_name = os.path.join(log_dir, f"monitor_session_{next_session:03d}.csv")
            try:
                import shutil
                shutil.copy2(mf, archive_name)
                os.remove(mf)
                print(f"[Resume] Archived monitor CSV → {archive_name}")
            except Exception:
                pass
            next_session += 1

    # Write a render flag file — dashboard can toggle this
    render_flag_path = os.path.join(log_dir, ".render_on")
    with open(render_flag_path, "w") as f:
        f.write("1")

    # Pause flag file — dashboard can toggle this
    pause_flag_path = os.path.join(log_dir, ".paused")
    with open(pause_flag_path, "w") as f:
        f.write("0")

    # ── Create environments ──
    render = None if headless else "human"
    train_env = DummyVecEnv([make_env(render_mode=render, reward_weights=rw,
                                      monitor_dir=log_dir, mask_arms=mask_arms)])

    if can_resume and os.path.exists(vecnorm_path):
        print(f"[Resume] Loading VecNormalize from: {vecnorm_path}")
        train_env = VecNormalize.load(vecnorm_path, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                                 clip_obs=10.0)

    # Eval env: headless
    # Eval env always runs with arms unmasked (measure full-policy performance)
    eval_env = DummyVecEnv([make_env(reward_weights=rw, monitor_dir=None, mask_arms=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    eval_freq = 5_000

    # ── Load or create model ──
    if can_resume:
        AlgoCls = PPO if algo_name == "ppo" else SAC
        model = AlgoCls.load(
            model_to_load, env=train_env,
            tensorboard_log=os.path.join(log_dir, "tb"),
        )
        print(f"[Resume] Model loaded. Adding {total_timesteps:,} more steps.")

        # Load replay buffer if available (critical for SAC resume quality)
        replay_buf_path = os.path.join(log_dir, "replay_buffer.pkl")
        if hasattr(model, "replay_buffer") and os.path.exists(replay_buf_path):
            model.load_replay_buffer(replay_buf_path)
            print(f"[Resume] Replay buffer loaded ({model.replay_buffer.size()} transitions)")
        else:
            print("[Resume] WARNING: No replay buffer found — critic may be unstable initially")
    else:
        if algo_name == "ppo":
            cfg = PPO_CONFIG.copy()
            cfg.pop("total_timesteps")
            model = PPO("MlpPolicy", train_env, verbose=1,
                         tensorboard_log=os.path.join(log_dir, "tb"), **cfg)
        elif algo_name == "sac":
            cfg = SAC_CONFIG.copy()
            cfg.pop("total_timesteps")
            model = SAC("MlpPolicy", train_env, verbose=1,
                         tensorboard_log=os.path.join(log_dir, "tb"), **cfg)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

    if total_timesteps is None:
        cfg_ref = PPO_CONFIG if algo_name == "ppo" else SAC_CONFIG
        total_timesteps = cfg_ref["total_timesteps"]

    # ── Callbacks ──
    video_dir = os.path.join(log_dir, "videos")
    eval_callback = VideoEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        video_dir=video_dir if record_evals else "",
    ) if record_evals else EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Preserve eval history across resumes — SB3 otherwise overwrites the npz.
    if can_resume:
        prior_eval_path = os.path.join(log_dir, "eval_logs", "evaluations.npz")
        if os.path.exists(prior_eval_path):
            try:
                prior = np.load(prior_eval_path)
                eval_callback.evaluations_timesteps = prior["timesteps"].tolist()
                eval_callback.evaluations_results = prior["results"].tolist()
                eval_callback.evaluations_length = prior["ep_lengths"].tolist()
                print(f"[Resume] Preserved {len(eval_callback.evaluations_timesteps)}"
                      f" prior eval points")
            except Exception as e:
                print(f"[Resume] Warning: could not preload eval history: {e}")
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix=algo_name,
    )
    save_callback = PeriodicSaveCallback(
        save_freq=10_000,
        log_dir=log_dir,
        algo_name=algo_name,
    )
    camera_callback = CameraFollowCallback(render_flag_path=render_flag_path)
    metrics_callback = MetricsCallback(log_dir=log_dir, save_freq=100)
    pause_callback = PauseCallback(pause_flag_path=pause_flag_path)

    cb_list = [eval_callback, checkpoint_callback, save_callback,
               camera_callback, metrics_callback, pause_callback]

    # Curriculum: auto-unlock arms mid-run when --curriculum N is set
    if curriculum_step is not None:
        cb_list.append(CurriculumCallback(switch_at_timestep=curriculum_step))
        phase_msg = f"Curriculum: arms locked until step {curriculum_step:,}"
    elif mask_arms:
        phase_msg = "Phase 1: arms masked (run again with --phase 2 to unlock)"
    else:
        phase_msg = "Arms fully actuated"

    callbacks = CallbackList(cb_list)

    # ── Save metadata ──
    meta = _save_run_metadata(
        log_dir, algo_name, total_timesteps, rw,
        resumed_from=model_to_load if can_resume else None,
        extra={"eval_freq": eval_freq, "checkpoint_freq": 50_000,
               "save_freq": 10_000},
    )

    # ── Random baseline eval (only for fresh runs) ──
    if not can_resume:
        print("[Baseline] Running 5 episodes with random actions...")
        baseline_env = DummyVecEnv([make_env(reward_weights=rw, monitor_dir=None)])
        baseline_env = VecNormalize(baseline_env, norm_obs=True, norm_reward=False,
                                     clip_obs=10.0, training=False)
        baseline_rewards = []
        baseline_lengths = []
        for ep in range(5):
            obs = baseline_env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            while not done:
                action_rand = np.array([baseline_env.action_space.sample()])
                obs, reward, dones, infos = baseline_env.step(action_rand)
                ep_reward += reward[0]
                ep_len += 1
                done = dones[0]
            baseline_rewards.append(ep_reward)
            baseline_lengths.append(ep_len)
        baseline_env.close()
        mean_r = np.mean(baseline_rewards)
        mean_l = np.mean(baseline_lengths)
        print(f"[Baseline] Random policy: avg reward={mean_r:.1f}, "
              f"avg length={mean_l:.1f}")

        # Pre-seed the eval log so the dashboard shows timestep=0 baseline
        eval_log_dir = os.path.join(log_dir, "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)
        np.savez(
            os.path.join(eval_log_dir, "evaluations.npz"),
            timesteps=np.array([0]),
            results=np.array([baseline_rewards]),
            ep_lengths=np.array([baseline_lengths]),
        )
        print(f"[Baseline] Saved to eval_logs as timestep=0 reference\n")

    # ── Launch dashboard in separate process ──
    plot_proc = None
    if live_plot:
        plot_proc = subprocess.Popen(
            [sys.executable, "live_plot.py", "--run", log_dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # ── Train ──
    action = "Resuming" if can_resume else "Training"
    git_short = meta['training_history'][-1].get('git', {}).get('commit_short', 'N/A')
    gui_status = "OFF (headless)" if headless else "OPEN"
    print(f"\n{'='*60}")
    print(f"  {action} {algo_name.upper()} for {total_timesteps:,} timesteps")
    print(f"  Run: {run_name}  |  Log dir: {log_dir}")
    print(f"  Git: {git_short}")
    print(f"  PyBullet GUI: {gui_status}  |  Dashboard: {'OPEN' if live_plot else 'OFF'}")
    print(f"  {phase_msg}")
    print(f"  Ctrl+C to pause and save for later resume")
    print(f"{'='*60}\n")

    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    progress_bar=True, reset_num_timesteps=not can_resume)
    except KeyboardInterrupt:
        print("\n[Paused] Ctrl+C detected. Saving model for later resume...")

    elapsed = time.time() - start_time

    # ── Save everything ──
    final_path = os.path.join(log_dir, f"{algo_name}_final")
    model.save(final_path)
    model.save(os.path.join(log_dir, f"{algo_name}_latest"))
    train_env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    # Save replay buffer for seamless resume
    if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
        model.save_replay_buffer(os.path.join(log_dir, "replay_buffer"))
        print(f"[Save] Replay buffer saved ({model.replay_buffer.size()} transitions)")

    # Update metadata with final stats
    meta_path = os.path.join(log_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["training_history"][-1]["elapsed_seconds"] = elapsed
        meta["training_history"][-1]["elapsed_human"] = (
            f"{elapsed/3600:.2f}h" if elapsed > 3600 else f"{elapsed/60:.1f}min"
        )
        meta["training_history"][-1]["final_model"] = final_path + ".zip"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    print(f"\nTraining complete in {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")
    print(f"Final model saved to: {final_path}")
    print(f"Resume later with:  python train.py --resume runs/{run_name} "
          f"--algo {algo_name} --timesteps <N>")

    train_env.close()
    eval_env.close()

    if plot_proc is not None:
        try:
            plot_proc.terminate()
        except Exception:
            pass

    return model, log_dir


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO or SAC on T1Walking")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"],
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (default from config)")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name for logging directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to run dir to resume (e.g. runs/sac_test)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable live matplotlib dashboard")
    parser.add_argument("--headless", action="store_true",
                        help="No PyBullet GUI (faster, use with dashboard)")
    parser.add_argument("--mask-arms", action="store_true",
                        help="Phase 1: mask arm actions (arms held at zero). "
                             "Resume without this flag for phase 2.")
    parser.add_argument("--curriculum", type=int, default=None, metavar="N",
                        help="Auto-unlock arms after N timesteps in a single run "
                             "(e.g. --curriculum 1000000 with --timesteps 2000000)")
    parser.add_argument("--record-evals", action="store_true",
                        help="Record one mp4 per eval checkpoint to <run>/videos/")
    args = parser.parse_args()

    if args.resume:
        run_name = os.path.basename(args.resume.rstrip("/\\"))
        train(args.algo, args.timesteps, run_name,
              live_plot=not args.no_plot, resume=True,
              headless=args.headless,
              mask_arms=args.mask_arms,
              curriculum_step=args.curriculum,
              record_evals=args.record_evals)
    else:
        if args.name is None:
            args.name = f"{args.algo}_{int(time.time())}"
        train(args.algo, args.timesteps, args.name,
              live_plot=not args.no_plot, headless=args.headless,
              mask_arms=args.mask_arms,
              curriculum_step=args.curriculum,
              record_evals=args.record_evals)
