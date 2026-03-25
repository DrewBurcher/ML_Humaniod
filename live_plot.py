"""
Live training dashboard with plots and controls.

    python live_plot.py --run runs/sac_live_test

Controls:
    - Render ON/OFF: toggle PyBullet rendering (OFF = faster training)
    - Smoothing slider: adjust plot smoothing window
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def read_monitor_csv(run_dir):
    """Read all monitor CSVs (current + previous sessions) and concatenate."""
    import glob
    dfs = []
    # Load archived sessions first (sorted by name = chronological)
    archives = sorted(glob.glob(os.path.join(run_dir, "monitor_session_*.csv")))
    for path in archives:
        try:
            df = pd.read_csv(path, skiprows=1)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass
    # Load current session
    current = os.path.join(run_dir, "monitor.monitor.csv")
    if not os.path.exists(current):
        candidates = glob.glob(os.path.join(run_dir, "*.monitor.csv"))
        candidates = [c for c in candidates if "session_" not in c]
        if candidates:
            current = candidates[0]
    if os.path.exists(current):
        try:
            df = pd.read_csv(current, skiprows=1)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def load_eval(run_dir):
    path = os.path.join(run_dir, "eval_logs", "evaluations.npz")
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass
    return None


def load_metadata(run_dir):
    path = os.path.join(run_dir, "run_metadata.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def load_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def smooth(y, window):
    if len(y) < 2 or window < 2:
        return y
    return pd.Series(y).rolling(window, min_periods=1).mean().values


COMPONENT_LABELS = {
    "velocity_reward": "Velocity",
    "survival_reward": "Survival",
    "energy_penalty": "Energy",
    "orientation_penalty": "Orientation",
    "joint_limit_penalty": "Joint Limits",
    "height_reward": "Height",
    "z_fall_penalty": "Z Fall",
}

COMPONENT_COLORS = {
    "velocity_reward": "#2ecc71",
    "survival_reward": "#3498db",
    "energy_penalty": "#e74c3c",
    "orientation_penalty": "#e67e22",
    "joint_limit_penalty": "#9b59b6",
    "height_reward": "#1abc9c",
    "z_fall_penalty": "#c0392b",
}


class Dashboard:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.render_flag_path = os.path.join(run_dir, ".render_on")
        self.smooth_window = 20
        self.render_on = True
        self._last_timesteps = 0
        self._last_time = time.time()

        # ── Layout: 3 rows x 3 cols + control bar ──
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.canvas.manager.set_window_title(
            f"Training Dashboard - {os.path.basename(run_dir)}")

        gs = gridspec.GridSpec(4, 3, figure=self.fig,
                               height_ratios=[1, 1, 1, 0.08],
                               hspace=0.45, wspace=0.35)

        # Row 1: Episode Reward, Episode Length, Reward vs Timesteps
        self.ax_reward = self.fig.add_subplot(gs[0, 0])
        self.ax_length = self.fig.add_subplot(gs[0, 1])
        self.ax_reward_ts = self.fig.add_subplot(gs[0, 2])

        # Row 2: Eval Performance, Actor/Critic Loss, Reward Distribution
        self.ax_eval = self.fig.add_subplot(gs[1, 0])
        self.ax_loss = self.fig.add_subplot(gs[1, 1])
        self.ax_loss2 = self.ax_loss.twinx()  # create once, reuse
        self.ax_dist = self.fig.add_subplot(gs[1, 2])

        # Row 3: Pie chart, Reward components over time, Run Info
        self.ax_pie = self.fig.add_subplot(gs[2, 0])
        self.ax_components = self.fig.add_subplot(gs[2, 1])
        self.ax_info = self.fig.add_subplot(gs[2, 2])

        # ── Controls row ──
        ax_render_btn = self.fig.add_axes([0.05, 0.01, 0.10, 0.03])
        self.render_btn = Button(ax_render_btn, 'Render: ON',
                                  color='lightgreen', hovercolor='palegreen')
        self.render_btn.on_clicked(self._toggle_render)

        ax_smooth = self.fig.add_axes([0.25, 0.015, 0.20, 0.02])
        self.smooth_slider = Slider(ax_smooth, 'Smooth', 1, 100,
                                     valinit=20, valstep=1)
        self.smooth_slider.on_changed(self._on_smooth_change)

        self.ax_status = self.fig.add_axes([0.50, 0.005, 0.48, 0.035])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(
            0, 0.5, "", fontsize=8, fontfamily="monospace",
            verticalalignment="center")

        self.fig.suptitle(
            f"Training Dashboard: {os.path.basename(run_dir)}",
            fontsize=14, fontweight="bold")

    def _toggle_render(self, event):
        self.render_on = not self.render_on
        try:
            with open(self.render_flag_path, "w") as f:
                f.write("1" if self.render_on else "0")
        except Exception:
            pass
        label = "Render: ON" if self.render_on else "Render: OFF"
        color = "lightgreen" if self.render_on else "lightsalmon"
        self.render_btn.label.set_text(label)
        self.render_btn.color = color
        self.render_btn.ax.set_facecolor(color)
        self.fig.canvas.draw_idle()

    def _on_smooth_change(self, val):
        self.smooth_window = int(val)

    def update(self, frame=None):
        meta = load_metadata(self.run_dir)
        df = read_monitor_csv(self.run_dir)
        eval_data = load_eval(self.run_dir)
        metrics = load_metrics(self.run_dir)
        w = self.smooth_window
        has_data = df is not None and len(df) > 0

        # ── Episode Reward ──
        ax = self.ax_reward
        ax.clear()
        if has_data:
            r = df["r"].values
            eps = np.arange(1, len(r) + 1)
            ax.plot(eps, r, alpha=0.2, color="steelblue", linewidth=0.5)
            ax.plot(eps, smooth(r, w), color="steelblue", linewidth=2)
        ax.set_xlim(left=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.grid(True, alpha=0.3)

        # ── Episode Length ──
        ax = self.ax_length
        ax.clear()
        if has_data:
            l = df["l"].values
            eps = np.arange(1, len(l) + 1)
            ax.plot(eps, l, alpha=0.2, color="darkorange", linewidth=0.5)
            ax.plot(eps, smooth(l, w), color="darkorange", linewidth=2)
        ax.set_xlim(left=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Episode Length (survival)")
        ax.grid(True, alpha=0.3)

        # ── Reward vs Timesteps ──
        ax = self.ax_reward_ts
        ax.clear()
        if has_data:
            r = df["r"].values
            l = df["l"].values
            cum = np.cumsum(l)
            ax.plot(cum, r, alpha=0.2, color="green", linewidth=0.5)
            ax.plot(cum, smooth(r, w), color="green", linewidth=2)
        ax.set_xlim(left=0)
        ax.set_xlabel("Total Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title("Reward vs Timesteps")
        ax.grid(True, alpha=0.3)

        # ── Evaluation Performance ──
        ax = self.ax_eval
        ax.clear()
        if eval_data is not None:
            try:
                ts = eval_data["timesteps"]
                results = eval_data["results"]
                mean_r = np.mean(results, axis=1)
                std_r = np.std(results, axis=1)
                ax.plot(ts, mean_r, color="purple", linewidth=2,
                        marker="o", markersize=3)
                ax.fill_between(ts, mean_r - std_r, mean_r + std_r,
                                alpha=0.2, color="purple")
            except Exception:
                ax.text(0.5, 0.5, "Waiting for eval data...",
                        ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Waiting for eval data...",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(left=0)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Eval Reward")
        ax.set_title("Evaluation Performance")
        ax.grid(True, alpha=0.3)

        # ── Actor / Critic Loss ──
        ax = self.ax_loss
        ax.clear()
        self.ax_loss2.clear()
        if metrics and len(metrics.get("losses", [])) > 0:
            losses = metrics["losses"]
            ts = [e["timestep"] for e in losses]

            # SAC losses
            actor = [e.get("actor_loss", None) for e in losses]
            critic = [e.get("critic_loss", None) for e in losses]
            # PPO losses
            policy_grad = [e.get("policy_gradient_loss", None) for e in losses]
            value = [e.get("value_loss", None) for e in losses]

            has_actor = any(v is not None for v in actor)
            has_critic = any(v is not None for v in critic)
            has_pg = any(v is not None for v in policy_grad)
            has_vl = any(v is not None for v in value)

            if has_actor:
                vals = [v for v in actor if v is not None]
                t = [ts[i] for i, v in enumerate(actor) if v is not None]
                ax.plot(t, vals, color="crimson", linewidth=1.5, alpha=0.7, label="Actor")
            if has_pg:
                vals = [v for v in policy_grad if v is not None]
                t = [ts[i] for i, v in enumerate(policy_grad) if v is not None]
                ax.plot(t, vals, color="crimson", linewidth=1.5, alpha=0.7, label="Policy")
            if has_critic:
                vals = [v for v in critic if v is not None]
                t = [ts[i] for i, v in enumerate(critic) if v is not None]
                self.ax_loss2.plot(t, vals, color="royalblue", linewidth=1.5, alpha=0.7, label="Critic")
                self.ax_loss2.set_ylabel("Critic", color="royalblue", fontsize=8)
                self.ax_loss2.tick_params(axis='y', labelcolor="royalblue", labelsize=7)
            if has_vl:
                vals = [v for v in value if v is not None]
                t = [ts[i] for i, v in enumerate(value) if v is not None]
                self.ax_loss2.plot(t, vals, color="royalblue", linewidth=1.5, alpha=0.7, label="Value")
                self.ax_loss2.set_ylabel("Value", color="royalblue", fontsize=8)
                self.ax_loss2.tick_params(axis='y', labelcolor="royalblue", labelsize=7)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = self.ax_loss2.get_legend_handles_labels()
            if lines1 or lines2:
                ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
        else:
            ax.text(0.5, 0.5, "Waiting for loss data...",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(left=0)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Actor", color="crimson", fontsize=8)
        ax.tick_params(axis='y', labelcolor="crimson", labelsize=7)
        ax.set_title("Actor / Critic Loss")
        ax.grid(True, alpha=0.3)

        # ── Reward Distribution ──
        ax = self.ax_dist
        ax.clear()
        if has_data:
            r = df["r"].values
            n = len(r)
            if n > 50:
                ax.hist(r[:50], bins=20, alpha=0.5, color="salmon", label="First 50")
                ax.hist(r[-50:], bins=20, alpha=0.5, color="steelblue", label="Last 50")
                ax.legend(fontsize=7)
            elif n > 0:
                ax.hist(r, bins=max(5, n // 3), alpha=0.7, color="steelblue")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.set_title("Reward Distribution")
        ax.grid(True, alpha=0.3)

        # ── Pie Chart: Reward component contribution ──
        ax = self.ax_pie
        ax.clear()
        if metrics and len(metrics.get("reward_components", [])) > 0:
            rc = metrics["reward_components"]
            # Use last 50 episodes
            recent = rc[-50:] if len(rc) > 50 else rc
            keys = [k for k in COMPONENT_LABELS.keys()]
            avgs = {}
            for k in keys:
                vals = [e.get(k, 0) for e in recent]
                avgs[k] = np.mean(vals) if vals else 0

            # Use absolute values for pie chart sizing
            labels = []
            sizes = []
            colors = []
            for k in keys:
                v = abs(avgs[k])
                if v > 0.001:
                    labels.append(COMPONENT_LABELS[k])
                    sizes.append(v)
                    colors.append(COMPONENT_COLORS[k])

            if sizes:
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, colors=colors,
                    autopct='%1.1f%%', textprops={'fontsize': 7},
                    pctdistance=0.8)
                for t in autotexts:
                    t.set_fontsize(6)
                ax.set_title("Reward Breakdown\n(avg |magnitude| per step, last 50 eps)",
                             fontsize=9)
            else:
                ax.text(0.5, 0.5, "Waiting for data...",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Reward Breakdown")
        else:
            ax.text(0.5, 0.5, "Waiting for data...",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Reward Breakdown")

        # ── Reward components over time ──
        ax = self.ax_components
        ax.clear()
        if metrics and len(metrics.get("reward_components", [])) > 5:
            rc = metrics["reward_components"]
            eps = [e.get("episode", i) for i, e in enumerate(rc)]
            for k in COMPONENT_LABELS:
                vals = [e.get(k, 0) for e in rc]
                if any(v != 0 for v in vals):
                    ax.plot(eps, smooth(np.array(vals), w),
                            color=COMPONENT_COLORS[k], linewidth=1.5,
                            label=COMPONENT_LABELS[k])
            ax.legend(fontsize=6, loc="best")
        else:
            ax.text(0.5, 0.5, "Waiting for data...",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(left=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("|Reward| per step")
        ax.set_title("Reward Components Over Time")
        ax.grid(True, alpha=0.3)

        # ── Run Info ──
        ax = self.ax_info
        ax.clear()
        ax.axis("off")
        ax.set_title("Run Info", fontsize=11, fontweight="bold")

        lines = []
        if meta:
            lines.append(f"Run:    {meta.get('run_name', '?')}")
            lines.append(f"Algo:   {meta.get('algo', '?').upper()}")
            total_steps = meta.get('total_timesteps_all_sessions', '?')
            lines.append(f"Target: {total_steps:,} steps" if isinstance(total_steps, int)
                         else f"Target: {total_steps} steps")
            lines.append("")

            env_cfg = meta.get("env_config", {})
            lines.append(f"Policy freq:   {env_cfg.get('policy_freq', '?')} Hz")
            lines.append(f"Target speed:  {env_cfg.get('target_speed', '?')} m/s")
            lines.append(f"Max ep steps:  {env_cfg.get('max_episode_steps', '?')}")
            lines.append("")

            algo_cfg = meta.get("algo_config", {})
            lines.append(f"LR:     {algo_cfg.get('learning_rate', '?')}")
            lines.append(f"Gamma:  {algo_cfg.get('gamma', '?')}")
            lines.append(f"Batch:  {algo_cfg.get('batch_size', '?')}")
            lines.append("")

            rw = meta.get("reward_weights", {})
            lines.append("Rewards:")
            for k, v in rw.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

            hist = meta.get("training_history", [{}])
            latest = hist[-1] if hist else {}
            git = latest.get("git", {})
            lines.append(f"Git:      {git.get('commit_short', '?')} ({git.get('branch', '?')})")
            lines.append(f"Sessions: {len(hist)}")

        if has_data:
            r = df["r"].values
            l = df["l"].values
            cum = np.cumsum(l)
            recent = r[-50:] if len(r) >= 50 else r
            lines.append("")
            lines.append("--- Live ---")
            lines.append(f"Episodes:   {len(r)}")
            lines.append(f"Timesteps:  {cum[-1]:,}")
            lines.append(f"Best:       {r.max():.1f}")
            lines.append(f"Avg (50):   {np.mean(recent):.1f}")
            lines.append(f"Best len:   {l.max()}")

            if "t" in df.columns and len(df) > 1:
                total_time = df["t"].values[-1]
                if total_time > 0:
                    fps = cum[-1] / total_time
                    lines.append(f"Avg speed:  {fps:.0f} steps/sec")

        text = "\n".join(lines)
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=6.5, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))

        # ── Status bar ──
        status_parts = []
        if has_data:
            r = df["r"].values
            l = df["l"].values
            cur_ts = int(np.sum(l))
            now = time.time()
            dt = now - self._last_time
            if dt > 0 and self._last_timesteps > 0:
                speed = (cur_ts - self._last_timesteps) / dt
                status_parts.append(f"Speed: {speed:.0f} steps/sec")
            self._last_timesteps = cur_ts
            self._last_time = now
            status_parts.append(f"Eps: {len(r)}")
            status_parts.append(f"Steps: {cur_ts:,}")
            status_parts.append(f"Last reward: {r[-1]:.1f}")
        status_parts.append(f"Render: {'ON' if self.render_on else 'OFF'}")
        status_parts.append(f"Smooth: {self.smooth_window}")
        status_parts.append(f"Updated: {time.strftime('%H:%M:%S')}")
        self.status_text.set_text("  |  ".join(status_parts))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self):
        self.update()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.update)
        timer.start()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument("--run", required=True,
                        help="Path to run directory (e.g. runs/sac_test)")
    args = parser.parse_args()

    if not os.path.isdir(args.run):
        os.makedirs(args.run, exist_ok=True)

    dash = Dashboard(args.run)
    dash.run()
