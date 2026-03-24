"""
Live training dashboard with plots and controls.

    python live_plot.py --run runs/sac_live_test

Controls:
    - Render ON/OFF: toggle PyBullet rendering (OFF = faster training)
    - Smoothing slider: adjust plot smoothing window

Reads monitor.csv + eval logs + run_metadata.json, refreshes every 2 seconds.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def read_monitor_csv(run_dir):
    """Read SB3 Monitor CSV from the run directory."""
    path = os.path.join(run_dir, "monitor.monitor.csv")
    if not os.path.exists(path):
        # Try alternate names
        import glob
        candidates = glob.glob(os.path.join(run_dir, "*.monitor.csv"))
        if not candidates:
            return None
        path = candidates[0]
    try:
        return pd.read_csv(path, skiprows=1)
    except Exception:
        return None


def load_eval(run_dir):
    """Load eval results."""
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


def smooth(y, window):
    if len(y) < 2 or window < 2:
        return y
    return pd.Series(y).rolling(window, min_periods=1).mean().values


class Dashboard:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.render_flag_path = os.path.join(run_dir, ".render_on")
        self.smooth_window = 20
        self.render_on = True
        self._last_timesteps = 0
        self._last_time = time.time()

        # ── Layout ──
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title(
            f"Training Dashboard — {os.path.basename(run_dir)}")

        # Main grid: top row for plots, bottom row for plots + info + controls
        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               height_ratios=[1, 1, 0.15],
                               hspace=0.4, wspace=0.35)

        self.ax_reward = self.fig.add_subplot(gs[0, 0])
        self.ax_length = self.fig.add_subplot(gs[0, 1])
        self.ax_reward_ts = self.fig.add_subplot(gs[0, 2])
        self.ax_eval = self.fig.add_subplot(gs[1, 0])
        self.ax_dist = self.fig.add_subplot(gs[1, 1])
        self.ax_info = self.fig.add_subplot(gs[1, 2])

        # ── Controls row ──
        # Render toggle button
        ax_render_btn = self.fig.add_axes([0.05, 0.02, 0.12, 0.04])
        self.render_btn = Button(ax_render_btn, 'Render: ON',
                                  color='lightgreen', hovercolor='palegreen')
        self.render_btn.on_clicked(self._toggle_render)

        # Smoothing slider
        ax_smooth = self.fig.add_axes([0.30, 0.02, 0.25, 0.03])
        self.smooth_slider = Slider(ax_smooth, 'Smoothing', 1, 100,
                                     valinit=20, valstep=1)
        self.smooth_slider.on_changed(self._on_smooth_change)

        # Status text
        self.ax_status = self.fig.add_axes([0.62, 0.01, 0.35, 0.05])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(
            0, 0.5, "", fontsize=9, fontfamily="monospace",
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

            # Compute speed from time column if available
            if "t" in df.columns and len(df) > 1:
                total_time = df["t"].values[-1]
                if total_time > 0:
                    fps = cum[-1] / total_time
                    lines.append(f"Avg speed:  {fps:.0f} steps/sec")

        text = "\n".join(lines)
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))

        # ── Status bar with speed ──
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
        """Start the dashboard with a timer-based refresh."""
        self.update()
        timer = self.fig.canvas.new_timer(interval=2000)
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
