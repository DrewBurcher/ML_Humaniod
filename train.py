"""
Training script for PPO and SAC on the T1 bipedal walking environment.

Features:
    - Auto-commits to git before each run, saves commit hash in run metadata
    - Saves all hyperparams, reward weights, env config, and run info as JSON
    - Supports pause/resume: pass --resume runs/my_run to continue training
    - Live matplotlib dashboard auto-launches
    - Periodic GUI visualization of current policy

Usage:
    python train.py --algo sac --timesteps 100000 --name my_run
    python train.py --resume runs/my_run --timesteps 50000   # add 50k more steps
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
        # Stage all Python files and config
        subprocess.run(["git", "add", "*.py", "config.py", "requirements.txt"],
                       capture_output=True)
        # Check if there's anything to commit
        result = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True)
        if result.stdout.strip():
            subprocess.run(
                ["git", "commit", "-m",
                 f"Auto-commit before training run ({datetime.now().strftime('%Y-%m-%d %H:%M')})"],
                capture_output=True, text=True
            )
            print("[Git] Auto-committed changes before training run.")
        else:
            print("[Git] Working tree clean, no commit needed.")
        # Push
        subprocess.run(["git", "push"], capture_output=True, text=True)
    except Exception as e:
        print(f"[Git] Warning: auto-commit failed: {e}")


def _save_run_metadata(log_dir, algo_name, total_timesteps, reward_weights,
                       resumed_from=None, extra=None):
    """Save all run metadata as JSON for reproducibility."""
    meta_path = os.path.join(log_dir, "run_metadata.json")

    # Load existing metadata if resuming
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        # Append to training history
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

    # Always save current config snapshot
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


def _load_run_metadata(log_dir):
    """Load run metadata from a previous run."""
    meta_path = os.path.join(log_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


# ── Environment factory ──────────────────────────────────────────────────────

def make_env(render_mode=None, reward_weights=None):
    def _init():
        e = gym.make("T1Walking-v0", render_mode=render_mode,
                      reward_weights=reward_weights)
        e = Monitor(e)
        return e
    return _init


# ── GUI Visualization Callback ───────────────────────────────────────────────

class VisualEvalCallback(BaseCallback):
    """Opens a GUI window periodically to show 1 episode with debug overlay."""

    def __init__(self, eval_freq=10_000, eval_interval=10,
                 reward_weights=None, algo_name="", run_name="",
                 total_target_steps=0, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_interval = eval_interval
        self.reward_weights = reward_weights
        self.algo_name = algo_name
        self.run_name = run_name
        self.total_target_steps = total_target_steps
        self._eval_count = 0
        self._visual_count = 0
        self._best_visual_reward = -float("inf")
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        self._eval_count += 1
        if self._eval_count % self.eval_interval != 0:
            return True

        self._visual_count += 1
        elapsed = time.time() - self._start_time
        steps_per_sec = self.num_timesteps / max(elapsed, 1)

        print(f"\n[Visual eval #{self._visual_count}] Opening GUI "
              f"(timestep {self.num_timesteps:,})...")

        vis_env = gym.make("T1Walking-v0", render_mode="human",
                           reward_weights=self.reward_weights)
        try:
            obs, _ = vis_env.reset()
            vec_norm = self.training_env
            raw_env = vis_env.unwrapped

            total_reward = 0.0
            steps = 0
            done = False
            max_x = 0.0
            debug_ids = []

            while not done:
                obs_norm = vec_norm.normalize_obs(obs)
                action, _ = self.model.predict(obs_norm, deterministic=True)
                obs, reward, terminated, truncated, info = vis_env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
                max_x = max(max_x, info.get("x_distance", 0))

                # Camera follow
                try:
                    pos = raw_env.robot.get_state()["base-position"]
                    pb.resetDebugVisualizerCamera(
                        cameraDistance=3.0, cameraYaw=40.0, cameraPitch=-20.0,
                        cameraTargetPosition=[pos[0], pos[1], 0.8],
                        physicsClientId=raw_env.client)
                except Exception:
                    pass

                # Debug overlay text (update every 10 steps)
                if steps % 10 == 1:
                    try:
                        for did in debug_ids:
                            pb.removeUserDebugItem(did, physicsClientId=raw_env.client)
                        debug_ids.clear()

                        remaining = self.total_target_steps - self.num_timesteps
                        eta_sec = remaining / max(steps_per_sec, 0.1)
                        eta_str = (f"{eta_sec/3600:.1f}h" if eta_sec > 3600
                                   else f"{eta_sec/60:.0f}m")

                        lines = [
                            f"Run: {self.run_name}  |  Algo: {self.algo_name.upper()}",
                            f"Timestep: {self.num_timesteps:,} / {self.total_target_steps:,}",
                            f"Speed: {steps_per_sec:.1f} steps/sec  |  ETA: {eta_str}",
                            f"Visual eval: #{self._visual_count}  (every {self.eval_interval * self.eval_freq:,} steps)",
                            f"Evals done: {self._eval_count}",
                            f"Elapsed: {elapsed/60:.1f} min",
                            f"",
                            f"--- This Episode ---",
                            f"Step: {steps}  |  Reward: {total_reward:.1f}",
                            f"Max X: {max_x:.2f} m",
                            f"Best visual reward: {self._best_visual_reward:.1f}",
                        ]

                        for i, line in enumerate(lines):
                            did = pb.addUserDebugText(
                                line,
                                textPosition=[0, 0, 2.5 - i * 0.12],
                                textColorRGB=[1, 1, 1],
                                textSize=1.2,
                                lifeTime=0,
                                physicsClientId=raw_env.client,
                            )
                            debug_ids.append(did)
                    except Exception:
                        pass

                time.sleep(1.0 / 60)

            self._best_visual_reward = max(self._best_visual_reward, total_reward)
            print(f"[Visual eval] {steps} steps, reward={total_reward:.2f}, "
                  f"max_x={max_x:.2f}m")
        finally:
            vis_env.close()
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
            # Save latest model (overwrite)
            self.model.save(os.path.join(self.log_dir, f"{self.algo_name}_latest"))
            # Save normalizer
            self.training_env.save(os.path.join(self.log_dir, "vecnormalize.pkl"))
        return True


# ── Main train function ─────────────────────────────────────────────────────

def train(algo_name, total_timesteps, run_name, reward_weights=None,
          visualize_every=10, live_plot=True, resume=False):
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    rw = reward_weights if reward_weights else REWARD_WEIGHTS

    # ── Auto-commit to git ──────────────────────────────────────────────────
    _auto_commit()

    # ── Determine if resuming ───────────────────────────────────────────────
    latest_model_path = os.path.join(log_dir, f"{algo_name}_latest.zip")
    final_model_path = os.path.join(log_dir, f"{algo_name}_final.zip")
    vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")

    can_resume = resume and (os.path.exists(latest_model_path) or
                              os.path.exists(final_model_path))

    if can_resume:
        model_to_load = (latest_model_path if os.path.exists(latest_model_path)
                         else final_model_path)
        print(f"\n[Resume] Loading model from: {model_to_load}")

    # ── Create environments ─────────────────────────────────────────────────
    train_env = DummyVecEnv([make_env(reward_weights=rw)])

    if can_resume and os.path.exists(vecnorm_path):
        print(f"[Resume] Loading VecNormalize from: {vecnorm_path}")
        train_env = VecNormalize.load(vecnorm_path, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                                 clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(reward_weights=rw)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    eval_freq = 5_000

    # ── Load or create model ────────────────────────────────────────────────
    if can_resume:
        AlgoCls = PPO if algo_name == "ppo" else SAC
        model = AlgoCls.load(
            model_to_load, env=train_env,
            tensorboard_log=os.path.join(log_dir, "tb"),
        )
        print(f"[Resume] Model loaded. Adding {total_timesteps:,} more steps.")
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

    # ── Callbacks ───────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )
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

    cb_list = [eval_callback, checkpoint_callback, save_callback]

    if visualize_every and visualize_every > 0:
        visual_callback = VisualEvalCallback(
            eval_freq=eval_freq,
            eval_interval=visualize_every,
            reward_weights=rw,
            algo_name=algo_name,
            run_name=run_name,
            total_target_steps=total_timesteps,
        )
        cb_list.append(visual_callback)

    callbacks = CallbackList(cb_list)

    # ── Save metadata ───────────────────────────────────────────────────────
    meta = _save_run_metadata(
        log_dir, algo_name, total_timesteps, rw,
        resumed_from=model_to_load if can_resume else None,
        extra={
            "eval_freq": eval_freq,
            "visualize_every": visualize_every,
            "checkpoint_freq": 50_000,
            "save_freq": 10_000,
        },
    )

    # ── Launch live plot ────────────────────────────────────────────────────
    plot_proc = None
    if live_plot:
        plot_proc = subprocess.Popen(
            [sys.executable, "live_plot.py", "--run", log_dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # ── Train ───────────────────────────────────────────────────────────────
    action = "Resuming" if can_resume else "Training"
    print(f"\n{'='*60}")
    print(f"  {action} {algo_name.upper()} for {total_timesteps:,} timesteps")
    print(f"  Run: {run_name}  |  Log dir: {log_dir}")
    print(f"  Git: {meta['training_history'][-1].get('git', {}).get('commit_short', 'N/A')}")
    print(f"  Visual eval every {(visualize_every or 0) * eval_freq:,} steps")
    print(f"  Checkpoints every 50,000 steps  |  Auto-save every 10,000 steps")
    print(f"{'='*60}\n")

    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    progress_bar=True, reset_num_timesteps=not can_resume)
    except KeyboardInterrupt:
        print("\n[Paused] Ctrl+C detected. Saving model for later resume...")

    elapsed = time.time() - start_time

    # ── Save everything ─────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, f"{algo_name}_final")
    model.save(final_path)
    model.save(os.path.join(log_dir, f"{algo_name}_latest"))
    train_env.save(os.path.join(log_dir, "vecnormalize.pkl"))

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
    parser.add_argument("--visualize-every", type=int, default=10,
                        help="Show GUI every N-th eval (0 to disable, default 10)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable live matplotlib plot")
    args = parser.parse_args()

    if args.resume:
        # Resume mode: infer run name from path
        run_name = os.path.basename(args.resume.rstrip("/\\"))
        train(args.algo, args.timesteps, run_name,
              visualize_every=args.visualize_every,
              live_plot=not args.no_plot,
              resume=True)
    else:
        if args.name is None:
            args.name = f"{args.algo}_{int(time.time())}"
        train(args.algo, args.timesteps, args.name,
              visualize_every=args.visualize_every,
              live_plot=not args.no_plot)
