"""
visualize.py
============
Visualisation utilities for SmartSnow results.

Functions:
    plot_learning_curves       — reward vs training steps
    plot_comparison_bar        — metric comparison bar chart
    plot_snow_heatmap          — snow depth heatmap over graph
    plot_clearance_progression — % cleared over episode timesteps
    plot_priority_clearance    — time-to-clear by priority class
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/notebook use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Optional, List, Dict


# ── Colour palette ────────────────────────────────────────────────────
AGENT_COLORS = {
    "Heuristic": "#E74C3C",
    "DQN":       "#3498DB",
    "PPO":       "#2ECC71",
    "A2C":       "#9B59B6",
    "Random":    "#95A5A6",
}

PRIORITY_COLORS = {
    1: "#BDC3C7",  # residential — light grey
    2: "#85C1E9",  # tertiary    — light blue
    3: "#F39C12",  # secondary   — orange
    4: "#E74C3C",  # primary     — red
    5: "#922B21",  # highway     — dark red
}


# ── 1. Learning curves ────────────────────────────────────────────────
def plot_learning_curves(
    reward_histories: Dict[str, List[float]],
    title: str = "Learning Curves — SmartSnow",
    save_path: Optional[str] = None,
    window: int = 20,
):
    """
    Plot smoothed reward curves for all agents.

    Parameters
    ----------
    reward_histories : {agent_name: [episode_rewards]}
    window : smoothing window size
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative Episode Reward", fontsize=12)

    for name, rewards in reward_histories.items():
        color = AGENT_COLORS.get(name, "#333333")
        rewards = np.array(rewards)
        x = np.arange(len(rewards))

        # Raw (faint)
        ax.plot(x, rewards, color=color, alpha=0.15, linewidth=0.8)

        # Smoothed
        if len(rewards) >= window:
            smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(len(smooth)) + window // 2, smooth,
                    label=name, color=color, linewidth=2.2)
        else:
            ax.plot(x, rewards, label=name, color=color, linewidth=2.2)

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved learning curves → {save_path}")
    return fig


# ── 2. Comparison bar chart ───────────────────────────────────────────
def plot_comparison_bar(
    results: Dict[str, Dict],
    metric: str = "total_reward",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    higher_is_better: bool = True,
):
    """
    Bar chart comparing agents on a single metric.
    """
    agents = list(results.keys())
    means  = [results[a].get(f"{metric}_mean", 0) for a in agents]
    stds   = [results[a].get(f"{metric}_std",  0) for a in agents]
    colors = [AGENT_COLORS.get(a, "#7F8C8D") for a in agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agents, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="white", linewidth=1.2, alpha=0.88)

    # Annotate values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + abs(max(means)) * 0.01,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    label_map = {
        "total_reward":       "Cumulative Episode Reward",
        "clearance_time":     "Steps to Complete Clearance",
        "clearance_pct":      "Final Clearance % of Segments",
        "high_pri_clear_step":"Steps to Clear High-Priority Roads",
        "total_distance":     "Total Distance Travelled (fuel proxy)",
    }
    ylabel = label_map.get(metric, metric.replace("_", " ").title())
    ax.set_ylabel(ylabel + (" (higher better)" if higher_is_better else " (lower better)"),
                  fontsize=11)
    ax.set_title(title or f"Agent Comparison — {ylabel}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved bar chart → {save_path}")
    return fig


# ── 3. Snow heatmap on graph ──────────────────────────────────────────
def plot_snow_heatmap(
    G, snow_depth: np.ndarray, edges: list,
    title: str = "Snow Depth Heatmap",
    save_path: Optional[str] = None,
    truck_positions: Optional[list] = None,
    nodes: Optional[list] = None,
):
    """
    Draw road graph coloured by snow depth.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    pos = {n: (G.nodes[n].get("x", 0), G.nodes[n].get("y", 0)) for n in G.nodes()}

    # Draw edges coloured by snow depth
    max_snow = max(snow_depth.max(), 1.0)
    for seg_idx, (u, v) in enumerate(edges):
        if u not in pos or v not in pos:
            continue
        depth = snow_depth[seg_idx]
        norm  = depth / max_snow
        # White → dark blue gradient
        r = int(255 * (1 - norm))
        g = int(255 * (1 - norm))
        b = 255
        color = f"#{r:02x}{g:02x}{b:02x}"
        x_vals = [pos[u][0], pos[v][0]]
        y_vals = [pos[u][1], pos[v][1]]
        ax.plot(x_vals, y_vals, color=color, linewidth=2.0, solid_capstyle="round")

    # Truck positions
    if truck_positions and nodes:
        for t_idx, node_idx in enumerate(truck_positions):
            if node_idx < len(nodes):
                n = nodes[node_idx]
                if n in pos:
                    ax.scatter(*pos[n], c="red", s=120, zorder=5,
                               marker="v", edgecolors="white", linewidths=1.5,
                               label=f"Truck {t_idx+1}" if t_idx == 0 else "")

    # Colorbar
    from matplotlib.cm import Blues
    from matplotlib.colors import Normalize
    sm = plt.cm.ScalarMappable(cmap=Blues, norm=Normalize(0, max_snow))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Snow Depth (inches)", shrink=0.7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved heatmap → {save_path}")
    return fig


# ── 4. Clearance progression ──────────────────────────────────────────
def plot_clearance_progression(
    progression_data: Dict[str, List[float]],
    title: str = "Clearance Progression Over Episode",
    save_path: Optional[str] = None,
):
    """
    Line plot of % segments cleared vs timestep for each agent.
    progression_data: {agent_name: [clearance_pct at each step]}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Segments Cleared (%)", fontsize=12)
    ax.set_ylim(0, 105)

    for name, prog in progression_data.items():
        color = AGENT_COLORS.get(name, "#333333")
        ax.plot(prog, label=name, color=color, linewidth=2.2)

    ax.axhline(100, color="green", linestyle="--", alpha=0.4, label="Full clearance")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved clearance progression → {save_path}")
    return fig


# ── 5. Priority-class clearance time ─────────────────────────────────
def plot_priority_clearance(
    priority_times: Dict[str, Dict[int, float]],
    title: str = "Time-to-Clear by Road Priority Class",
    save_path: Optional[str] = None,
):
    """
    Grouped bar chart: x=priority class, groups=agents.
    priority_times: {agent: {priority: mean_steps}}
    """
    priorities = sorted({p for agent_data in priority_times.values() for p in agent_data})
    agents     = list(priority_times.keys())
    x          = np.arange(len(priorities))
    width      = 0.8 / max(len(agents), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Road Priority Class", fontsize=12)
    ax.set_ylabel("Mean Steps to First Clear", fontsize=12)

    pri_labels = {1: "Residential\n(P1)", 2: "Tertiary\n(P2)",
                  3: "Secondary\n(P3)", 4: "Primary\n(P4)", 5: "Highway\n(P5)"}

    for i, agent in enumerate(agents):
        vals = [priority_times[agent].get(p, 0) for p in priorities]
        color = AGENT_COLORS.get(agent, "#7F8C8D")
        offset = (i - len(agents) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=agent, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([pri_labels.get(p, f"P{p}") for p in priorities], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved priority clearance → {save_path}")
    return fig
