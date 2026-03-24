"""
Visualization tools: training curves, comparison plots, and video recording.

Usage:
    python visualize.py --run runs/ppo_123 --algo ppo         # Plot training curves
    python visualize.py --compare runs/ppo_123 runs/sac_456   # Compare PPO vs SAC
    python visualize.py --record runs/ppo_123 --algo ppo      # Record video
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy

import env as _  # noqa
import gymnasium as gym


def plot_training_curves(run_dir, algo_name, save=True):
    """Plot reward curve from Monitor logs."""
    try:
        timesteps, rewards = ts2xy(load_results(run_dir), "timesteps")
    except Exception:
        print(f"No monitor logs found in {run_dir}. Checking eval logs...")
        eval_path = os.path.join(run_dir, "eval_logs", "evaluations.npz")
        if os.path.exists(eval_path):
            data = np.load(eval_path)
            timesteps = data["timesteps"]
            rewards = data["results"].mean(axis=1)
        else:
            print("No training data found to plot.")
            return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward curve
    ax = axes[0]
    ax.plot(timesteps, rewards, alpha=0.3, color="steelblue", label="Raw")
    # Smoothed curve (rolling average)
    window = max(len(rewards) // 50, 1)
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(timesteps[:len(smoothed)], smoothed, color="darkblue",
                linewidth=2, label=f"Smoothed (w={window})")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"{algo_name.upper()} Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode length curve
    ax = axes[1]
    try:
        _, ep_lengths = ts2xy(load_results(run_dir), "timesteps")
        # Actually load episode lengths from monitor
        df = load_results(run_dir)
        ep_lengths = df["l"].values
        ep_ts = np.cumsum(ep_lengths)
        ax.plot(ep_ts, ep_lengths, alpha=0.3, color="coral", label="Raw")
        if len(ep_lengths) >= window:
            smoothed_l = np.convolve(ep_lengths, np.ones(window) / window, mode="valid")
            ax.plot(ep_ts[:len(smoothed_l)], smoothed_l, color="darkred",
                    linewidth=2, label=f"Smoothed (w={window})")
    except Exception:
        ax.text(0.5, 0.5, "No episode length data", transform=ax.transAxes,
                ha="center")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title(f"{algo_name.upper()} Episode Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(run_dir, f"{algo_name}_training_curves.png")
        plt.savefig(path, dpi=150)
        print(f"Saved training curves to {path}")
    plt.show()


def plot_comparison(run_dirs, algo_names, save=True):
    """Compare training curves of PPO and SAC."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {"ppo": "steelblue", "sac": "coral"}

    for run_dir, algo in zip(run_dirs, algo_names):
        eval_path = os.path.join(run_dir, "eval_logs", "evaluations.npz")
        if os.path.exists(eval_path):
            data = np.load(eval_path)
            timesteps = data["timesteps"]
            mean_rewards = data["results"].mean(axis=1)
            std_rewards = data["results"].std(axis=1)
            ax.plot(timesteps, mean_rewards, color=colors.get(algo, "gray"),
                    linewidth=2, label=f"{algo.upper()}")
            ax.fill_between(timesteps, mean_rewards - std_rewards,
                            mean_rewards + std_rewards,
                            color=colors.get(algo, "gray"), alpha=0.2)
        else:
            try:
                ts, rewards = ts2xy(load_results(run_dir), "timesteps")
                window = max(len(rewards) // 50, 1)
                smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ax.plot(ts[:len(smoothed)], smoothed, color=colors.get(algo, "gray"),
                        linewidth=2, label=f"{algo.upper()}")
            except Exception:
                print(f"No data found for {algo} in {run_dir}")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.set_title("PPO vs SAC: Training Performance")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(run_dirs[0], "..", "ppo_vs_sac_comparison.png")
        path = os.path.normpath(path)
        plt.savefig(path, dpi=150)
        print(f"Saved comparison plot to {path}")
    plt.show()


def plot_eval_comparison(run_dirs, algo_names, save=True):
    """Bar chart comparing final evaluation metrics."""
    summaries = []
    for run_dir in run_dirs:
        results_path = os.path.join(run_dir, "eval_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                summaries.append(json.load(f))
        else:
            print(f"No eval_results.json in {run_dir}. Run evaluate.py first.")
            return

    metrics = ["mean_reward", "mean_distance", "mean_steps", "energy_per_meter"]
    labels = ["Reward", "Distance (m)", "Episode Length", "Energy / m"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(algo_names))

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [s[metric] for s in summaries]
        axes[i].bar(x, vals, color=["steelblue", "coral"][:len(algo_names)],
                    width=0.5)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([a.upper() for a in algo_names])
        axes[i].set_ylabel(label)
        axes[i].set_title(label)
        axes[i].grid(True, alpha=0.3, axis="y")

    plt.suptitle("PPO vs SAC: Evaluation Comparison", fontsize=14)
    plt.tight_layout()
    if save:
        path = os.path.join(run_dirs[0], "..", "eval_comparison.png")
        path = os.path.normpath(path)
        plt.savefig(path, dpi=150)
        print(f"Saved eval comparison to {path}")
    plt.show()


def record_video(run_dir, algo_name, n_steps=500, output_path=None):
    """Record frames and save as a video using matplotlib animation."""
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    ALGO_MAP = {"ppo": PPO, "sac": SAC}
    algo_cls = ALGO_MAP[algo_name]

    model_path = os.path.join(run_dir, f"{algo_name}_final.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(run_dir, "best_model", "best_model.zip")
    model = algo_cls.load(model_path)

    eval_env = DummyVecEnv([lambda: gym.make("T1Walking-v0", render_mode="rgb_array")])
    vec_path = os.path.join(run_dir, "vecnormalize.pkl")
    if os.path.exists(vec_path):
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    # Collect frames
    frames = []
    obs = eval_env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)
        frame = eval_env.envs[0].render()
        if frame is not None:
            frames.append(frame)
        if dones[0]:
            obs = eval_env.reset()

    eval_env.close()

    if not frames:
        print("No frames captured.")
        return

    if output_path is None:
        output_path = os.path.join(run_dir, f"{algo_name}_gait.mp4")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_array(frames[i])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=33, blit=True)
    try:
        writer = FFMpegWriter(fps=30)
        anim.save(output_path, writer=writer)
        print(f"Video saved to {output_path}")
    except Exception:
        # Fallback: save as GIF
        gif_path = output_path.replace(".mp4", ".gif")
        anim.save(gif_path, writer="pillow", fps=15)
        print(f"Video saved as GIF to {gif_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization tools")
    parser.add_argument("--run", type=str, help="Run directory")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"])
    parser.add_argument("--compare", type=str, nargs="+",
                        help="Run dirs to compare")
    parser.add_argument("--record", type=str, help="Run dir for video recording")
    parser.add_argument("--eval-compare", type=str, nargs="+",
                        help="Run dirs for eval bar chart comparison")
    args = parser.parse_args()

    if args.compare:
        algos = ["ppo", "sac"][:len(args.compare)]
        plot_comparison(args.compare, algos)
    elif args.eval_compare:
        algos = ["ppo", "sac"][:len(args.eval_compare)]
        plot_eval_comparison(args.eval_compare, algos)
    elif args.record and args.algo:
        record_video(args.record, args.algo)
    elif args.run and args.algo:
        plot_training_curves(args.run, args.algo)
    else:
        parser.print_help()
