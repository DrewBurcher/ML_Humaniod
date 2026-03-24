"""
Main entry point for the T1 Bipedal Locomotion RL project.

Usage:
    python main.py train --algo ppo [--timesteps 2000000] [--name my_run]
    python main.py train --algo sac [--timesteps 2000000] [--name my_run]
    python main.py eval  --run runs/ppo_123 --algo ppo [--episodes 10] [--render]
    python main.py compare --runs runs/ppo_123 runs/sac_456 --algos ppo sac
    python main.py plot  --run runs/ppo_123 --algo ppo
    python main.py record --run runs/ppo_123 --algo ppo
    python main.py ablation --algo ppo [--timesteps 500000]
    python main.py demo  [--algo ppo --run runs/ppo_123]
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        description="T1 Bipedal Locomotion with RL (PPO & SAC)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ────────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train a policy")
    p_train.add_argument("--algo", required=True, choices=["ppo", "sac"])
    p_train.add_argument("--timesteps", type=int, default=None)
    p_train.add_argument("--name", type=str, default=None)
    p_train.add_argument("--resume", type=str, default=None,
                          help="Run dir to resume from (e.g. runs/sac_test)")
    p_train.add_argument("--no-plot", action="store_true")

    # ── eval ─────────────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("eval", help="Evaluate a trained policy")
    p_eval.add_argument("--run", required=True)
    p_eval.add_argument("--algo", required=True, choices=["ppo", "sac"])
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--render", action="store_true")

    # ── compare ──────────────────────────────────────────────────────────────
    p_cmp = subparsers.add_parser("compare", help="Compare PPO vs SAC")
    p_cmp.add_argument("--runs", nargs=2, required=True)
    p_cmp.add_argument("--algos", nargs=2, required=True)

    # ── plot ─────────────────────────────────────────────────────────────────
    p_plot = subparsers.add_parser("plot", help="Plot training curves")
    p_plot.add_argument("--run", required=True)
    p_plot.add_argument("--algo", required=True, choices=["ppo", "sac"])

    # ── record ───────────────────────────────────────────────────────────────
    p_rec = subparsers.add_parser("record", help="Record video of gait")
    p_rec.add_argument("--run", required=True)
    p_rec.add_argument("--algo", required=True, choices=["ppo", "sac"])

    # ── ablation ─────────────────────────────────────────────────────────────
    p_abl = subparsers.add_parser("ablation", help="Run reward ablation study")
    p_abl.add_argument("--algo", required=True, choices=["ppo", "sac"])
    p_abl.add_argument("--timesteps", type=int, default=500_000)

    # ── demo ─────────────────────────────────────────────────────────────────
    p_demo = subparsers.add_parser("demo",
                                    help="Run environment with random actions (no training)")

    args = parser.parse_args()

    if args.command == "train":
        from train import train
        if args.resume:
            import os
            run_name = os.path.basename(args.resume.rstrip("/\\"))
            train(args.algo, args.timesteps, run_name,
                  live_plot=not args.no_plot, resume=True)
        else:
            name = args.name or f"{args.algo}_{int(time.time())}"
            train(args.algo, args.timesteps, name,
                  live_plot=not args.no_plot)

    elif args.command == "eval":
        from evaluate import evaluate
        evaluate(args.run, args.algo, n_episodes=args.episodes, render=args.render)

    elif args.command == "compare":
        from evaluate import compare as eval_compare
        from visualize import plot_comparison, plot_eval_comparison
        eval_compare(args.runs, args.algos)
        plot_comparison(args.runs, args.algos)
        plot_eval_comparison(args.runs, args.algos)

    elif args.command == "plot":
        from visualize import plot_training_curves
        plot_training_curves(args.run, args.algo)

    elif args.command == "record":
        from visualize import record_video
        record_video(args.run, args.algo)

    elif args.command == "ablation":
        from ablation import run_ablation
        run_ablation(args.algo, args.timesteps)

    elif args.command == "demo":
        _run_demo()

    else:
        parser.print_help()


def _run_demo():
    """Quick demo: load the environment and run random actions with GUI."""
    import env as _  # noqa
    import gymnasium as gym
    import numpy as np

    e = gym.make("T1Walking-v0", render_mode="human")
    obs, _ = e.reset()
    print(f"Observation space: {e.observation_space.shape}")
    print(f"Action space:      {e.action_space.shape}")
    print("Running random actions... (close window or Ctrl+C to stop)\n")

    total_reward = 0.0
    step = 0
    try:
        while True:
            action = e.action_space.sample()
            obs, reward, terminated, truncated, info = e.step(action)
            total_reward += reward
            step += 1
            if step % 100 == 0:
                print(f"Step {step}: reward={total_reward:.2f}, "
                      f"z={info['torso_z']:.3f}, x={info['x_distance']:.3f}")
            if terminated or truncated:
                print(f"Episode ended at step {step} with reward {total_reward:.2f}")
                obs, _ = e.reset()
                total_reward = 0.0
                step = 0
    except (KeyboardInterrupt, Exception) as ex:
        if isinstance(ex, KeyboardInterrupt):
            print("\nStopped by user.")
        else:
            print(f"\nGUI window closed. Exiting.")
    finally:
        try:
            e.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
