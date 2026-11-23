"""
render_env.py
=============
Grid-world style visualizer for BuffaloSnowEnv.
Produces:
  - Static frame (env_frame.png)
  - Multi-panel episode summary (env_summary.png)
  - Animated GIF of a full rollout (env_rollout.gif)
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx

from env.buffalo_snow_env import BuffaloSnowEnv
from env.road_graph import get_graph, get_edge_list, get_zone_partition
from agents.heuristic import GreedyHeuristicAgent

# ── Colour maps ────────────────────────────────────────────────────────
SNOW_CMAP   = LinearSegmentedColormap.from_list(
    "snow", ["#1a1a2e", "#16213e", "#0f3460", "#4fc3f7", "#e3f2fd", "#ffffff"])
ZONE_COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261",
               "#264653","#8ecae6","#a8dadc","#f1faee","#ffb703"]
TRUCK_COLOR = "#ff4444"
CLEARED_COLOR = "#00e676"

PRIORITY_COLORS = {1:"#90caf9", 2:"#42a5f5", 3:"#f59e0b", 4:"#ef4444", 5:"#7c3aed"}
PRIORITY_LABELS = {1:"Residential", 2:"Tertiary", 3:"Secondary", 4:"Primary", 5:"Highway/Hospital"}


def _get_pos(G):
    """Extract (x, y) position dict from graph."""
    return {n: (G.nodes[n].get("x", n % 10 * 200),
                G.nodes[n].get("y", n // 10 * 200)) for n in G.nodes()}


def render_frame(env, ax, step=0, title=None):
    """
    Draw one environment frame onto ax — grid-world style.
    Roads coloured by snow depth; trucks shown as red dots; cleared = green.
    """
    ax.set_facecolor("#0d0d0d")
    G    = env.G
    pos  = _get_pos(G)
    edges = env.edges

    max_snow = max(env.snow_depth.max(), 0.01)

    # ── Draw edges ─────────────────────────────────────────────────────
    for idx, (u, v) in enumerate(edges):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        snow = env.snow_depth[idx]
        priority = env.edge_priorities[idx]

        if snow <= 0:
            color  = CLEARED_COLOR
            lw     = 1.5 + priority * 0.3
            alpha  = 0.9
            zorder = 3
        else:
            norm   = snow / max_snow
            color  = SNOW_CMAP(0.2 + norm * 0.8)
            lw     = 1.0 + priority * 0.4
            alpha  = 0.7 + norm * 0.3
            zorder = 2

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", zorder=zorder)

    # ── Draw nodes (intersections) ─────────────────────────────────────
    xs = [pos[n][0] for n in G.nodes() if n in pos]
    ys = [pos[n][1] for n in G.nodes() if n in pos]
    ax.scatter(xs, ys, s=6, color="#37474f", zorder=4, linewidths=0)

    # ── Draw trucks ────────────────────────────────────────────────────
    for t_idx in range(env.n_trucks):
        node_idx = env.truck_positions[t_idx]
        if node_idx < len(env.nodes):
            node = env.nodes[node_idx]
            if node in pos:
                tx, ty = pos[node]
                ax.scatter(tx, ty, s=220, color=TRUCK_COLOR, marker="s",
                           zorder=6, edgecolors="white", linewidths=1.2)
                ax.text(tx, ty, f"T{t_idx+1}", ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold", zorder=7)

    # ── Stats overlay ──────────────────────────────────────────────────
    cleared   = int((env.snow_depth <= 0).sum())
    total     = len(env.snow_depth)
    mean_snow = env.snow_depth.mean()
    pct       = cleared / total * 100

    stats_text = (f"Step: {env.current_step}   "
                  f"Cleared: {cleared}/{total} ({pct:.0f}%)   "
                  f"Avg snow: {mean_snow:.2f} cm")
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=7.5, color="white", va="top",
            bbox=dict(facecolor="#1a1a2e", alpha=0.7, pad=3, edgecolor="none"))

    if title:
        ax.set_title(title, fontsize=9, color="white", pad=4)

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    return ax


def add_legend(ax):
    """Compact legend for snow depth and trucks."""
    handles = [
        mpatches.Patch(color="#ffffff", label="Heavy snow"),
        mpatches.Patch(color="#4fc3f7", label="Light snow"),
        mpatches.Patch(color="#1a1a2e", label="No snow (uncleared)"),
        mpatches.Patch(color=CLEARED_COLOR, label="Cleared ✓"),
        mpatches.Patch(color=TRUCK_COLOR,   label="Plow truck"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5,
              facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
              framealpha=0.85)


def render_static_frame(env, out_path, step_label="Initial State"):
    """Save a single environment frame."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d0d0d")
    render_frame(env, ax, title=f"SmartSnow Environment — {step_label}")
    add_legend(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"[render] Saved frame → {out_path}")


def render_summary(frames_data, rewards, cleared_pcts, out_path):
    """
    Multi-panel figure: 6 env snapshots + reward curve + clearance curve.
    frames_data: list of (env_snapshot_dict, step_label)
    """
    fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25)

    # ── 6 env snapshots ────────────────────────────────────────────────
    snapshot_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for i, (snap, label) in enumerate(frames_data[:6]):
        ax = snapshot_axes[i]
        _draw_snapshot(ax, snap, label)

    # ── Reward curve ───────────────────────────────────────────────────
    ax_reward = fig.add_subplot(gs[2, :2])
    ax_reward.set_facecolor("#0d1117")
    steps = list(range(len(rewards)))
    ax_reward.plot(steps, rewards, color="#4fc3f7", linewidth=1.5, alpha=0.9)
    ax_reward.fill_between(steps, rewards, alpha=0.15, color="#4fc3f7")
    ax_reward.axhline(0, color="#555", linewidth=0.7, linestyle="--")
    ax_reward.set_xlabel("Step", color="#aaa", fontsize=8)
    ax_reward.set_ylabel("Cumulative Reward", color="#aaa", fontsize=8)
    ax_reward.set_title("Reward Curve", color="white", fontsize=9)
    ax_reward.tick_params(colors="#888", labelsize=7)
    for sp in ax_reward.spines.values(): sp.set_edgecolor("#333")

    # ── Clearance curve ────────────────────────────────────────────────
    ax_clear = fig.add_subplot(gs[2, 2:])
    ax_clear.set_facecolor("#0d1117")
    ax_clear.plot(steps, cleared_pcts, color=CLEARED_COLOR, linewidth=1.5, alpha=0.9)
    ax_clear.fill_between(steps, cleared_pcts, alpha=0.15, color=CLEARED_COLOR)
    ax_clear.set_xlabel("Step", color="#aaa", fontsize=8)
    ax_clear.set_ylabel("% Roads Cleared", color="#aaa", fontsize=8)
    ax_clear.set_title("Clearance Progress", color="white", fontsize=9)
    ax_clear.set_ylim(0, 100)
    ax_clear.tick_params(colors="#888", labelsize=7)
    for sp in ax_clear.spines.values(): sp.set_edgecolor("#333")

    fig.suptitle("SmartSnow — Buffalo Snow Plow RL Environment",
                 fontsize=13, color="white", y=0.98, fontweight="bold")

    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"[render] Saved summary → {out_path}")


def _draw_snapshot(ax, snap, label):
    """Draw a pre-captured env snapshot dict onto ax."""
    ax.set_facecolor("#0d0d0d")
    pos    = snap["pos"]
    edges  = snap["edges"]
    snow   = snap["snow"]
    trucks = snap["trucks"]
    nodes_list = snap["nodes"]
    priorities = snap["priorities"]
    max_s  = max(snow.max(), 0.01)

    for idx, (u, v) in enumerate(edges):
        if u not in pos or v not in pos: continue
        x0,y0 = pos[u]; x1,y1 = pos[v]
        s = snow[idx]; pr = priorities[idx]
        if s <= 0:
            col, lw, al = CLEARED_COLOR, 1.5+pr*0.3, 0.9
        else:
            col = SNOW_CMAP(0.2 + (s/max_s)*0.8)
            lw  = 1.0+pr*0.4; al = 0.7+(s/max_s)*0.3
        ax.plot([x0,x1],[y0,y1], color=col, linewidth=lw, alpha=al,
                solid_capstyle="round")

    for t_idx, node_idx in enumerate(trucks):
        if node_idx < len(nodes_list):
            nd = nodes_list[node_idx]
            if nd in pos:
                tx,ty = pos[nd]
                ax.scatter(tx, ty, s=180, color=TRUCK_COLOR, marker="s",
                           zorder=6, edgecolors="white", linewidths=1.1)
                ax.text(tx,ty,f"T{t_idx+1}",ha="center",va="center",
                        fontsize=4.5,color="white",fontweight="bold",zorder=7)

    cleared = int((snow<=0).sum()); total=len(snow)
    ax.text(0.03, 0.96, f"Step {snap['step']}  |  {cleared}/{total} cleared",
            transform=ax.transAxes, fontsize=6, color="white", va="top",
            bbox=dict(facecolor="#1a1a2e", alpha=0.7, pad=2, edgecolor="none"))
    ax.set_title(label, fontsize=7.5, color="#ccc", pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#222")


def capture_snapshot(env):
    """Capture serializable snapshot of env state."""
    pos = _get_pos(env.G)
    return {
        "step":      env.current_step,
        "pos":       pos,
        "edges":     list(env.edges),
        "nodes":     list(env.nodes),
        "snow":      env.snow_depth.copy(),
        "trucks":    env.truck_positions.copy(),
        "priorities":env.edge_priorities.copy(),
    }


def render_gif(env, agent, out_path, n_steps=60, fps=4):
    """
    Animate a rollout as a GIF — grid world style.
    """
    frames = []

    obs, _ = env.reset(seed=42)
    frames.append(capture_snapshot(env))

    for _ in range(n_steps):
        action = agent.select_action()
        obs, _, done, trunc, _ = env.step(action)
        frames.append(capture_snapshot(env))
        if done or trunc:
            break

    print(f"[render] Captured {len(frames)} frames for GIF …")

    fig, ax = plt.subplots(figsize=(6, 5.2), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    def update(i):
        ax.clear()
        ax.set_facecolor("#0d0d0d")
        snap  = frames[i]
        label = f"Step {snap['step']}"
        _draw_snapshot(ax, snap, label)
        ax.set_title(f"SmartSnow — Buffalo Plow Routing  |  {label}",
                     fontsize=8.5, color="white", pad=4)
        # Mini legend
        legend_handles = [
            mpatches.Patch(color="#ffffff", label="Heavy snow"),
            mpatches.Patch(color="#4fc3f7", label="Light snow"),
            mpatches.Patch(color=CLEARED_COLOR, label="Cleared ✓"),
            mpatches.Patch(color=TRUCK_COLOR,   label="Plow truck"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=5.5,
                  facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
                  framealpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[render] Saved GIF → {out_path}")


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), "../results")
    os.makedirs(OUT, exist_ok=True)

    env   = BuffaloSnowEnv(n_trucks=3, n_zones=10, snowfall_intensity="medium",
                           max_steps=80, seed=42)
    agent = GreedyHeuristicAgent(env)

    # ── 1. Static initial frame ────────────────────────────────────────
    obs, _ = env.reset(seed=42)
    render_static_frame(env, f"{OUT}/env_initial_frame.png", "Initial State (t=0)")

    # ── 2. Collect rollout data for summary ────────────────────────────
    obs, _ = env.reset(seed=42)
    snapshots   = [(capture_snapshot(env), "t=0  (Reset)")]
    cum_rewards = [0.0]
    cleared_pcts= [0.0]
    cum_r = 0.0
    snap_steps = [10, 20, 30, 40, 60]

    for step in range(80):
        action = agent.select_action()
        obs, reward, done, trunc, info = env.step(action)
        cum_r += reward
        cum_rewards.append(cum_r)
        cleared_pcts.append(info.get("clearance_pct", 0.0))
        if (step+1) in snap_steps:
            snapshots.append((capture_snapshot(env),
                              f"t={step+1}  |  {info.get('clearance_pct',0):.0f}% cleared"))
        if done or trunc:
            break

    render_summary(snapshots[:6], cum_rewards, cleared_pcts,
                   f"{OUT}/env_episode_summary.png")

    # ── 3. Animated GIF ────────────────────────────────────────────────
    env_gif = BuffaloSnowEnv(n_trucks=3, n_zones=10, snowfall_intensity="medium",
                             max_steps=60, seed=42)
    agent_gif = GreedyHeuristicAgent(env_gif)
    render_gif(env_gif, agent_gif, f"{OUT}/env_rollout.gif", n_steps=60, fps=3)

    print("\n[render] All outputs saved:")
    for f in ["env_initial_frame.png", "env_episode_summary.png", "env_rollout.gif"]:
        path = f"{OUT}/{f}"
        size = os.path.getsize(path) // 1024
        print(f"  {f:35s}  {size} KB")
