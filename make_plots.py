"""Generate presentation-quality plots from a training run.

Usage:
    python make_plots.py --run runs/sac_legs_only_v5
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "lines.linewidth": 2,
})

PALETTE = {
    "primary":   "#1f77b4",
    "accent":    "#ff7f0e",
    "good":      "#2ca02c",
    "bad":       "#d62728",
    "purple":    "#9467bd",
    "neutral":   "#7f7f7f",
}

COMPONENT_COLORS = {
    "velocity_reward":     "#2ca02c",
    "survival_reward":     "#1f77b4",
    "energy_penalty":      "#d62728",
    "orientation_penalty": "#ff7f0e",
    "joint_limit_penalty": "#9467bd",
    "z_velocity_penalty":  "#8c564b",
}


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_monitor(run_dir):
    dfs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "monitor_session_*.csv"))):
        try:
            dfs.append(pd.read_csv(path, skiprows=1))
        except Exception:
            pass
    cur = os.path.join(run_dir, "monitor.monitor.csv")
    if os.path.exists(cur):
        try:
            dfs.append(pd.read_csv(cur, skiprows=1))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_eval(run_dir):
    path = os.path.join(run_dir, "eval_logs", "evaluations.npz")
    return np.load(path) if os.path.exists(path) else None


def load_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_metadata(run_dir):
    path = os.path.join(run_dir, "run_metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def smooth(y, w):
    if len(y) < 2 or w <= 1:
        return np.asarray(y, dtype=float)
    w = min(w, len(y))
    kernel = np.ones(w) / w
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="valid")


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_learning_curve(df, out, smooth_w=50):
    if df is None:
        return
    r = df["r"].values
    ts = df["l"].cumsum().values
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, r, alpha=0.15, color=PALETTE["primary"], linewidth=0.6,
            label="raw episode reward")
    sm = smooth(r, smooth_w)
    ax.plot(ts[smooth_w - 1:] if len(sm) < len(ts) else ts, sm,
            color=PALETTE["primary"], linewidth=2.5,
            label=f"{smooth_w}-episode moving average")
    ax.set_xlabel("Total environment timesteps")
    ax.set_ylabel("Episode return")
    ax.set_title("Learning Curve — Episode Return")
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def plot_episode_length(df, out, smooth_w=50):
    if df is None:
        return
    l = df["l"].values
    ts = np.cumsum(l)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, l, alpha=0.15, color=PALETTE["accent"], linewidth=0.6,
            label="raw episode length")
    sm = smooth(l, smooth_w)
    ax.plot(ts[smooth_w - 1:] if len(sm) < len(ts) else ts, sm,
            color=PALETTE["accent"], linewidth=2.5,
            label=f"{smooth_w}-episode moving average")
    ax.axhline(1000, ls="--", color=PALETTE["neutral"], alpha=0.5,
               label="max episode length (1000)")
    ax.set_xlabel("Total environment timesteps")
    ax.set_ylabel("Steps survived")
    ax.set_title("Survival — Episode Length Over Training")
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def plot_eval(eval_data, out):
    if eval_data is None:
        return
    ts = eval_data["timesteps"]
    res = eval_data["results"]
    mean_r = np.mean(res, axis=1)
    std_r = np.std(res, axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(ts, mean_r - std_r, mean_r + std_r,
                    alpha=0.25, color=PALETTE["purple"])
    ax.plot(ts, mean_r, color=PALETTE["purple"], linewidth=2.5,
            marker="o", markersize=4, label="mean (5 episodes)")
    ax.set_xlabel("Total environment timesteps")
    ax.set_ylabel("Eval reward")
    ax.set_title("Deterministic Evaluation Performance")
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def plot_forward_velocity(metrics, out, smooth_w=50):
    rc = metrics.get("reward_components", []) if metrics else []
    if not rc:
        return
    ts = np.array([e["timestep"] for e in rc])
    v = np.array([e["forward_vel_mean"] for e in rc])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, v, alpha=0.15, color=PALETTE["good"], linewidth=0.6,
            label="raw")
    sm = smooth(v, smooth_w)
    ax.plot(ts[smooth_w - 1:] if len(sm) < len(ts) else ts, sm,
            color=PALETTE["good"], linewidth=2.5,
            label=f"{smooth_w}-episode moving average")
    ax.axhline(0, ls="--", color=PALETTE["neutral"], alpha=0.5)
    ax.set_xlabel("Total environment timesteps")
    ax.set_ylabel("Forward velocity (m/s)")
    ax.set_title("Walking Speed Over Training")
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def plot_x_distance(metrics, out, smooth_w=50):
    rc = metrics.get("reward_components", []) if metrics else []
    if not rc:
        return
    ts = np.array([e["timestep"] for e in rc])
    x = np.array([e["x_distance"] for e in rc])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, x, alpha=0.15, color=PALETTE["primary"], linewidth=0.6,
            label="raw")
    sm = smooth(x, smooth_w)
    ax.plot(ts[smooth_w - 1:] if len(sm) < len(ts) else ts, sm,
            color=PALETTE["primary"], linewidth=2.5,
            label=f"{smooth_w}-episode moving average")
    ax.set_xlabel("Total environment timesteps")
    ax.set_ylabel("Distance walked (m)")
    ax.set_title("Forward Distance Per Episode")
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def plot_reward_components(metrics, out, smooth_w=50):
    rc = metrics.get("reward_components", []) if metrics else []
    if not rc:
        return
    ts = np.array([e["timestep"] for e in rc])
    keys = [k for k in COMPONENT_COLORS if k in rc[0]]
    fig, ax = plt.subplots(figsize=(11, 6))
    for k in keys:
        v = np.array([e.get(k, 0.0) for e in rc])
        sm = smooth(v, smooth_w)
        x = ts[smooth_w - 1:] if len(sm) < len(ts) else ts
        ax.plot(x, sm, color=COMPONENT_COLORS[k],
                label=k.replace("_", " "), linewidth=2)
    ax.axhline(0, ls="--", color=PALETTE["neutral"], alpha=0.5)
    ax.set_xlabel("Total environment timesteps")
    ax.set_ylabel("Per-step reward magnitude")
    ax.set_title("Reward Component Magnitudes (smoothed)")
    ax.legend(loc="best", ncol=2)
    fig.savefig(out)
    plt.close(fig)


def plot_component_pie(metrics, out, last_n=200):
    rc = metrics.get("reward_components", []) if metrics else []
    if not rc:
        return
    tail = rc[-last_n:]
    keys = [k for k in COMPONENT_COLORS if k in tail[0]]
    sizes = [abs(np.mean([e.get(k, 0.0) for e in tail])) for k in keys]
    total = sum(sizes)
    if total == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=[k.replace("_", " ") for k in keys],
           colors=[COMPONENT_COLORS[k] for k in keys],
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title(f"Reward Composition — Last {len(tail)} Episodes")
    fig.savefig(out)
    plt.close(fig)


def plot_losses(metrics, out, smooth_w=200):
    losses = metrics.get("losses", []) if metrics else []
    if not losses:
        return
    has_actor = any("actor_loss" in e for e in losses)
    if has_actor:
        ts = np.array([e["timestep"] for e in losses if "actor_loss" in e])
        actor = np.array([e["actor_loss"] for e in losses if "actor_loss" in e])
        critic = np.array([e.get("critic_loss", np.nan)
                           for e in losses if "actor_loss" in e])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        ax1.plot(ts, smooth(actor, smooth_w)[:len(ts) - smooth_w + 1] if len(actor) > smooth_w else actor,
                 color=PALETTE["primary"])
        ax1.set_title("SAC Actor Loss")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Actor loss")
        ax2.plot(ts, smooth(critic, smooth_w)[:len(ts) - smooth_w + 1] if len(critic) > smooth_w else critic,
                 color=PALETTE["bad"])
        ax2.set_title("SAC Critic Loss")
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Critic loss")
        ax2.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)


def plot_entropy(metrics, out, smooth_w=200):
    losses = metrics.get("losses", []) if metrics else []
    if not losses:
        return
    ts = np.array([e["timestep"] for e in losses if "ent_coef" in e])
    ec = np.array([e["ent_coef"] for e in losses if "ent_coef" in e])
    if len(ec) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sm = smooth(ec, smooth_w)
    x = ts[smooth_w - 1:] if len(sm) < len(ts) else ts
    ax.plot(x, sm, color=PALETTE["purple"], linewidth=2.5)
    ax.set_xlabel("Total environment timesteps")
    ax.set_ylabel("Entropy coefficient α")
    ax.set_title("SAC Entropy Coefficient (Auto-Tuned)")
    ax.set_yscale("log")
    fig.savefig(out)
    plt.close(fig)


def plot_reward_distribution(df, out):
    if df is None:
        return
    r = df["r"].values
    n = len(r)
    early = r[:n // 4]
    late = r[3 * n // 4:]
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(min(r), max(r), 60)
    ax.hist(early, bins=bins, alpha=0.6, color=PALETTE["bad"],
            label=f"first 25% of episodes (n={len(early)})")
    ax.hist(late, bins=bins, alpha=0.6, color=PALETTE["good"],
            label=f"last 25% of episodes (n={len(late)})")
    ax.set_xlabel("Episode return")
    ax.set_ylabel("Frequency")
    ax.set_title("Episode Return Distribution — Early vs Late Training")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="path to run directory")
    parser.add_argument("--out", default=None,
                        help="output dir (default: <run>/presentation)")
    parser.add_argument("--smooth", type=int, default=50)
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.run, "presentation")
    os.makedirs(out_dir, exist_ok=True)

    df = load_monitor(args.run)
    eval_data = load_eval(args.run)
    metrics = load_metrics(args.run)

    plots = [
        ("01_learning_curve.png",       lambda p: plot_learning_curve(df, p, args.smooth)),
        ("02_episode_length.png",       lambda p: plot_episode_length(df, p, args.smooth)),
        ("03_eval_performance.png",     lambda p: plot_eval(eval_data, p)),
        ("04_forward_velocity.png",     lambda p: plot_forward_velocity(metrics, p, args.smooth)),
        ("05_x_distance.png",           lambda p: plot_x_distance(metrics, p, args.smooth)),
        ("06_reward_components.png",    lambda p: plot_reward_components(metrics, p, args.smooth)),
        ("07_component_pie.png",        lambda p: plot_component_pie(metrics, p)),
        ("08_sac_losses.png",           lambda p: plot_losses(metrics, p)),
        ("09_entropy_coef.png",         lambda p: plot_entropy(metrics, p)),
        ("10_reward_distribution.png",  lambda p: plot_reward_distribution(df, p)),
    ]

    for name, fn in plots:
        path = os.path.join(out_dir, name)
        try:
            fn(path)
            if os.path.exists(path):
                print(f"[ok]  {path}")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    print(f"\nDone. Plots in: {out_dir}")


if __name__ == "__main__":
    main()
