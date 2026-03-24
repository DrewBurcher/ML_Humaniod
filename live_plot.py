"""
Live-updating plot of training progress. Run alongside training:
    python live_plot.py --run runs/sac_quick_test

Reads Monitor CSV + eval logs + run_metadata.json, refreshes every 2 seconds.
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd


def find_monitor_csv(run_dir):
    """Find the monitor CSV written by SB3's Monitor wrapper."""
    for pat in [os.path.join(run_dir, "*.monitor.csv"),
                os.path.join(run_dir, "**", "*.monitor.csv")]:
        files = glob.glob(pat, recursive=True)
        if files:
            return files[0]
    return None


def read_monitor_csv(path):
    try:
        return pd.read_csv(path, skiprows=1)
    except Exception:
        return None


def find_eval_log(run_dir):
    path = os.path.join(run_dir, "eval_logs", "evaluations.npz")
    return path if os.path.exists(path) else None


def load_metadata(run_dir):
    path = os.path.join(run_dir, "run_metadata.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def live_plot(run_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Training Progress: {os.path.basename(run_dir)}", fontsize=14, fontweight="bold")

    def smooth(y, window=20):
        if len(y) < window:
            return y
        return pd.Series(y).rolling(window, min_periods=1).mean().values

    def update(frame):
        for ax in axes.flat:
            ax.clear()

        meta = load_metadata(run_dir)
        monitor_path = find_monitor_csv(run_dir)
        df = read_monitor_csv(monitor_path) if monitor_path else None
        has_data = df is not None and len(df) > 0

        # ── Top-left: Episode Reward ──
        ax = axes[0, 0]
        if has_data:
            rewards = df["r"].values
            episodes = np.arange(1, len(rewards) + 1)
            ax.plot(episodes, rewards, alpha=0.25, color="steelblue", linewidth=0.5)
            ax.plot(episodes, smooth(rewards), color="steelblue", linewidth=2, label="Smoothed (20)")
            ax.legend(loc="upper left", fontsize=7)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.grid(True, alpha=0.3)

        # ── Top-center: Episode Length ──
        ax = axes[0, 1]
        if has_data:
            lengths = df["l"].values
            episodes = np.arange(1, len(lengths) + 1)
            ax.plot(episodes, lengths, alpha=0.25, color="darkorange", linewidth=0.5)
            ax.plot(episodes, smooth(lengths), color="darkorange", linewidth=2, label="Smoothed (20)")
            ax.legend(loc="upper left", fontsize=7)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Episode Length (survival)")
        ax.grid(True, alpha=0.3)

        # ── Top-right: Reward vs Timesteps ──
        ax = axes[0, 2]
        if has_data:
            rewards = df["r"].values
            lengths = df["l"].values
            cum_steps = np.cumsum(lengths)
            ax.plot(cum_steps, rewards, alpha=0.25, color="green", linewidth=0.5)
            ax.plot(cum_steps, smooth(rewards), color="green", linewidth=2, label="Smoothed (20)")
            ax.legend(loc="upper left", fontsize=7)
        ax.set_xlabel("Total Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title("Reward vs Timesteps")
        ax.grid(True, alpha=0.3)

        # ── Bottom-left: Evaluation Performance ──
        ax = axes[1, 0]
        eval_path = find_eval_log(run_dir)
        if eval_path:
            try:
                data = np.load(eval_path)
                timesteps = data["timesteps"]
                results = data["results"]
                mean_r = np.mean(results, axis=1)
                std_r = np.std(results, axis=1)
                ax.plot(timesteps, mean_r, color="purple", linewidth=2,
                        marker="o", markersize=3)
                ax.fill_between(timesteps, mean_r - std_r, mean_r + std_r,
                                alpha=0.2, color="purple")
            except Exception:
                ax.text(0.5, 0.5, "Waiting for eval data...",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10)
        else:
            ax.text(0.5, 0.5, "Waiting for eval data...",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Eval Reward")
        ax.set_title("Evaluation Performance")
        ax.grid(True, alpha=0.3)

        # ── Bottom-center: Reward distribution (histogram) ──
        ax = axes[1, 1]
        if has_data:
            rewards = df["r"].values
            n = len(rewards)
            if n > 50:
                recent = rewards[-50:]
                early = rewards[:50]
                ax.hist(early, bins=20, alpha=0.5, color="salmon", label="First 50 eps")
                ax.hist(recent, bins=20, alpha=0.5, color="steelblue", label="Last 50 eps")
                ax.legend(fontsize=7)
            else:
                ax.hist(rewards, bins=20, alpha=0.7, color="steelblue")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.set_title("Reward Distribution")
        ax.grid(True, alpha=0.3)

        # ── Bottom-right: Run Info panel ──
        ax = axes[1, 2]
        ax.axis("off")
        ax.set_title("Run Info", fontsize=12, fontweight="bold")

        lines = []
        if meta:
            lines.append(f"Run: {meta.get('run_name', '?')}")
            lines.append(f"Algo: {meta.get('algo', '?').upper()}")
            lines.append(f"Total target steps: {meta.get('total_timesteps_all_sessions', '?'):,}")
            lines.append("")

            # Config
            env_cfg = meta.get("env_config", {})
            lines.append(f"Policy freq: {env_cfg.get('policy_freq', '?')} Hz")
            lines.append(f"Target speed: {env_cfg.get('target_speed', '?')} m/s")
            lines.append(f"Max episode: {env_cfg.get('max_episode_steps', '?')} steps")
            lines.append("")

            # Algo config
            algo_cfg = meta.get("algo_config", {})
            lines.append(f"Learning rate: {algo_cfg.get('learning_rate', '?')}")
            lines.append(f"Gamma: {algo_cfg.get('gamma', '?')}")
            lines.append(f"Batch size: {algo_cfg.get('batch_size', '?')}")
            lines.append(f"Network: {algo_cfg.get('policy_kwargs', '?')}")
            lines.append("")

            # Reward weights
            rw = meta.get("reward_weights", {})
            lines.append("Reward weights:")
            for k, v in rw.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

            # Git info
            hist = meta.get("training_history", [{}])
            latest = hist[-1] if hist else {}
            git = latest.get("git", {})
            lines.append(f"Git: {git.get('commit_short', '?')} ({git.get('branch', '?')})")
            lines.append(f"Sessions: {len(hist)}")

        if has_data:
            rewards = df["r"].values
            lengths = df["l"].values
            cum_steps = np.cumsum(lengths)
            recent = rewards[-50:] if len(rewards) >= 50 else rewards
            lines.append("")
            lines.append("--- Live Stats ---")
            lines.append(f"Episodes: {len(rewards)}")
            lines.append(f"Timesteps: {cum_steps[-1]:,}")
            lines.append(f"Best reward: {rewards.max():.1f}")
            lines.append(f"Recent avg (50): {np.mean(recent):.1f}")
            lines.append(f"Best length: {lengths.max()}")

        text = "\n".join(lines)
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=7.5, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))

        fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3, w_pad=3)

    ani = animation.FuncAnimation(fig, update, interval=2000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live training progress plot")
    parser.add_argument("--run", required=True,
                        help="Path to run directory (e.g. runs/sac_test)")
    args = parser.parse_args()

    if not os.path.isdir(args.run):
        print(f"Run directory not found: {args.run}")
        print("Start training first, then run this script.")
        exit(1)

    live_plot(args.run)
