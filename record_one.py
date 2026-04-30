"""Quick one-episode recorder. Loads a run's latest model and writes one mp4/gif."""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import env as _  # noqa
import gymnasium as gym


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--algo", required=True, choices=["ppo", "sac"])
    p.add_argument("--out", default=None, help="output path (mp4 or gif)")
    p.add_argument("--model", default=None,
                   help="model zip path (default: <run>/<algo>_latest.zip)")
    args = p.parse_args()

    AlgoCls = PPO if args.algo == "ppo" else SAC
    model_path = args.model or os.path.join(args.run, f"{args.algo}_latest.zip")
    model = AlgoCls.load(model_path)

    eval_env = DummyVecEnv([lambda: gym.make("T1Walking-v0", render_mode="rgb_array")])
    vec_path = os.path.join(args.run, "vecnormalize.pkl")
    if os.path.exists(vec_path):
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    obs = eval_env.reset()
    done = False
    frames = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)
        done = dones[0]
        f = eval_env.envs[0].render()
        if f is not None:
            frames.append(np.asarray(f))

    out = args.out or os.path.join(args.run, "videos", "manual_record.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    import imageio.v2 as imageio
    if out.endswith(".gif"):
        imageio.mimsave(out, frames, fps=30, loop=0)
    else:
        imageio.mimsave(out, frames, fps=30, codec="libx264", quality=8)
    print(f"Saved {len(frames)} frames -> {out}")
    eval_env.close()


if __name__ == "__main__":
    main()
