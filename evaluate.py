"""
Evaluation framework for trained T1 walking policies.

Usage:
    python evaluate.py --run runs/ppo_123456 --algo ppo [--episodes 10] [--render]
    python evaluate.py --compare runs/ppo_123456 runs/sac_789012 --algo ppo sac
"""

import argparse
import os
import json
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import env as _  # noqa
import gymnasium as gym


ALGO_MAP = {"ppo": PPO, "sac": SAC}


def load_model(run_dir, algo_name):
    """Load a trained model and its VecNormalize stats."""
    algo_cls = ALGO_MAP[algo_name]
    model_path = os.path.join(run_dir, f"{algo_name}_final.zip")
    if not os.path.exists(model_path):
        # Try best model
        model_path = os.path.join(run_dir, "best_model", "best_model.zip")
    model = algo_cls.load(model_path)

    vec_path = os.path.join(run_dir, "vecnormalize.pkl")
    return model, vec_path


def evaluate(run_dir, algo_name, n_episodes=10, render=False):
    """Run evaluation episodes and collect metrics."""
    model, vec_path = load_model(run_dir, algo_name)

    render_mode = "human" if render else None
    eval_env = DummyVecEnv([lambda: gym.make("T1Walking-v0", render_mode=render_mode)])
    if os.path.exists(vec_path):
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    results = {
        "cumulative_rewards": [],
        "forward_distances": [],
        "episode_lengths": [],
        "energy_costs": [],
        "reward_components": [],
    }

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_energy = 0.0
        ep_steps = 0
        ep_components = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            done = dones[0]

            ep_reward += reward[0]
            ep_steps += 1

            info = infos[0]
            if "reward_info" in info:
                ep_components.append(info["reward_info"])
                ep_energy += abs(info["reward_info"].get("energy_penalty", 0.0))

        # Get final x-distance from the last info
        x_dist = info.get("x_distance", 0.0)

        results["cumulative_rewards"].append(ep_reward)
        results["forward_distances"].append(x_dist)
        results["episode_lengths"].append(ep_steps)
        results["energy_costs"].append(ep_energy)

        # Average reward components over episode
        if ep_components:
            avg_components = {}
            for key in ep_components[0]:
                avg_components[key] = np.mean([c[key] for c in ep_components])
            results["reward_components"].append(avg_components)

        print(f"Episode {ep+1}/{n_episodes}: "
              f"reward={ep_reward:.2f}, dist={x_dist:.2f}m, "
              f"steps={ep_steps}, energy={ep_energy:.2f}")

    eval_env.close()

    # Summary stats
    summary = {
        "algo": algo_name,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(results["cumulative_rewards"])),
        "std_reward": float(np.std(results["cumulative_rewards"])),
        "mean_distance": float(np.mean(results["forward_distances"])),
        "std_distance": float(np.std(results["forward_distances"])),
        "mean_steps": float(np.mean(results["episode_lengths"])),
        "mean_energy": float(np.mean(results["energy_costs"])),
        "energy_per_meter": float(
            np.mean(results["energy_costs"]) /
            max(np.mean(results["forward_distances"]), 0.01)
        ),
    }

    print(f"\n{'='*50}")
    print(f"  {algo_name.upper()} Evaluation Summary ({n_episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Mean reward:        {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  Mean distance:      {summary['mean_distance']:.2f} ± {summary['std_distance']:.2f} m")
    print(f"  Mean episode length: {summary['mean_steps']:.0f} steps")
    print(f"  Energy cost / meter: {summary['energy_per_meter']:.2f}")
    print(f"{'='*50}\n")

    # Save results
    results_path = os.path.join(run_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")

    return summary, results


def compare(run_dirs, algo_names):
    """Compare multiple trained models side by side."""
    summaries = []
    for run_dir, algo in zip(run_dirs, algo_names):
        print(f"\nEvaluating {algo.upper()} from {run_dir}...")
        summary, _ = evaluate(run_dir, algo, n_episodes=10, render=False)
        summaries.append(summary)

    print(f"\n{'='*70}")
    print(f"  {'Metric':<25} | {'PPO':>15} | {'SAC':>15}")
    print(f"{'='*70}")
    for key in ["mean_reward", "mean_distance", "mean_steps", "energy_per_meter"]:
        vals = [s.get(key, 0) for s in summaries]
        labels = [s["algo"].upper() for s in summaries]
        row = f"  {key:<25}"
        for v in vals:
            row += f" | {v:>15.2f}"
        print(row)
    print(f"{'='*70}")

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate T1 walking policies")
    parser.add_argument("--run", type=str, help="Path to run directory")
    parser.add_argument("--algo", type=str, nargs="+", choices=["ppo", "sac"],
                        help="Algorithm(s)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--compare", type=str, nargs="+",
                        help="Paths to run directories for comparison")
    args = parser.parse_args()

    if args.compare and args.algo:
        compare(args.compare, args.algo)
    elif args.run and args.algo:
        evaluate(args.run, args.algo[0], n_episodes=args.episodes,
                 render=args.render)
    else:
        parser.print_help()
