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

def make_env(render_mode=None, reward_weights=None):
    def _init():
        e = gym.make("T1Walking-v0", render_mode=render_mode,
                      reward_weights=reward_weights)
        e = Monitor(e)
        return e
    return _init


# ── Camera follow callback ───────────────────────────────────────────────────

class CameraFollowCallback(BaseCallback):
    """Follows the robot with the PyBullet camera. No text, just camera."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            # Get the raw env from inside DummyVecEnv -> VecNormalize -> Monitor
            raw_env = self.training_env.envs[0].unwrapped
            if raw_env.robot is not None and raw_env.render_mode == "human":
                pos = raw_env.robot.get_state()["base-position"]
                pb.resetDebugVisualizerCamera(
                    cameraDistance=3.0, cameraYaw=40.0, cameraPitch=-20.0,
                    cameraTargetPosition=[pos[0], pos[1], 0.8],
                    physicsClientId=raw_env.client)
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
        return True


# ── Main train function ─────────────────────────────────────────────────────

def train(algo_name, total_timesteps, run_name, reward_weights=None,
          live_plot=True, resume=False):
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

    # ── Create environments ──
    # Training env: GUI mode so PyBullet window stays open the whole time
    train_env = DummyVecEnv([make_env(render_mode="human", reward_weights=rw)])

    if can_resume and os.path.exists(vecnorm_path):
        print(f"[Resume] Loading VecNormalize from: {vecnorm_path}")
        train_env = VecNormalize.load(vecnorm_path, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                                 clip_obs=10.0)

    # Eval env: headless
    eval_env = DummyVecEnv([make_env(reward_weights=rw)])
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
    camera_callback = CameraFollowCallback()

    callbacks = CallbackList([
        eval_callback, checkpoint_callback, save_callback, camera_callback
    ])

    # ── Save metadata ──
    meta = _save_run_metadata(
        log_dir, algo_name, total_timesteps, rw,
        resumed_from=model_to_load if can_resume else None,
        extra={"eval_freq": eval_freq, "checkpoint_freq": 50_000,
               "save_freq": 10_000},
    )

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
    print(f"\n{'='*60}")
    print(f"  {action} {algo_name.upper()} for {total_timesteps:,} timesteps")
    print(f"  Run: {run_name}  |  Log dir: {log_dir}")
    print(f"  Git: {git_short}")
    print(f"  PyBullet GUI: OPEN  |  Dashboard: {'OPEN' if live_plot else 'OFF'}")
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
    args = parser.parse_args()

    if args.resume:
        run_name = os.path.basename(args.resume.rstrip("/\\"))
        train(args.algo, args.timesteps, run_name,
              live_plot=not args.no_plot, resume=True)
    else:
        if args.name is None:
            args.name = f"{args.algo}_{int(time.time())}"
        train(args.algo, args.timesteps, args.name,
              live_plot=not args.no_plot)
