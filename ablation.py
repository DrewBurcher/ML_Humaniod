"""
Reward ablation study: train with each reward component removed to measure
its contribution to final performance.

Usage:
    python ablation.py --algo ppo --timesteps 500000
"""

import argparse
import os
import time
import json
import copy
import numpy as np
import matplotlib.pyplot as plt

from config import REWARD_WEIGHTS
from train import train
from evaluate import evaluate


ABLATION_CONFIGS = {
    "full": {},  # No changes — baseline
    "no_velocity": {"forward_velocity": 0.0},
    "no_survival": {"survival": 0.0},
    "no_energy_penalty": {"energy_penalty": 0.0},
    "no_orientation_penalty": {"orientation_penalty": 0.0},
    "no_joint_limit_penalty": {"joint_limit_penalty": 0.0},
}


def run_ablation(algo_name, total_timesteps):
    """Run training for each ablation configuration."""
    results = {}
    timestamp = int(time.time())

    for config_name, overrides in ABLATION_CONFIGS.items():
        print(f"\n{'#'*60}")
        print(f"  Ablation: {config_name}")
        print(f"{'#'*60}")

        # Create modified reward weights
        reward_weights = copy.deepcopy(REWARD_WEIGHTS)
        reward_weights.update(overrides)

        run_name = f"ablation_{algo_name}_{config_name}_{timestamp}"

        # Train
        model, log_dir = train(algo_name, total_timesteps, run_name,
                               reward_weights=reward_weights)

        # Evaluate
        summary, _ = evaluate(log_dir, algo_name, n_episodes=10)
        results[config_name] = summary

    # Save ablation results
    ablation_dir = os.path.join("runs", f"ablation_{algo_name}_{timestamp}")
    os.makedirs(ablation_dir, exist_ok=True)
    with open(os.path.join(ablation_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_ablation(results, algo_name, ablation_dir)
    return results


def plot_ablation(results, algo_name, save_dir):
    """Create bar charts showing the effect of removing each reward component."""
    configs = list(results.keys())
    metrics = {
        "mean_reward": "Mean Reward",
        "mean_distance": "Mean Distance (m)",
        "mean_steps": "Mean Episode Length",
        "energy_per_meter": "Energy / Meter",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (metric_key, metric_label) in enumerate(metrics.items()):
        ax = axes[i]
        vals = [results[c].get(metric_key, 0) for c in configs]
        colors = ["forestgreen" if c == "full" else "indianred" for c in configs]
        bars = ax.bar(range(len(configs)), vals, color=colors)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle(f"Reward Ablation Study ({algo_name.upper()})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f"ablation_{algo_name}.png")
    plt.savefig(path, dpi=150)
    print(f"Ablation plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward ablation study")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Timesteps per ablation run (shorter than full training)")
    args = parser.parse_args()

    run_ablation(args.algo, args.timesteps)
