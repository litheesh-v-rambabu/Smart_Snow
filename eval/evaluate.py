"""
evaluate.py
===========
Evaluation module for SmartSnow agents.

Collects 4 core metrics over n_episodes:
    1. total_reward         — cumulative episode reward
    2. clearance_time       — steps until episode ends
    3. total_distance       — proxy for fuel consumption
    4. high_pri_clear_step  — steps until all P4/P5 roads are first cleared

Also records:
    - clearance_pct at episode end
    - per-step reward traces (for convergence plots)
"""

import os
import sys
import json
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.buffalo_snow_env import BuffaloSnowEnv
from agents.heuristic import GreedyHeuristicAgent


def evaluate_agent(
    model,
    env: BuffaloSnowEnv,
    n_episodes: int = 20,
    seed_offset: int = 0,
    deterministic: bool = True,
) -> dict:
    """
    Evaluate an SB3 model over n_episodes.

    Parameters
    ----------
    model        : SB3 model with .predict() method
    env          : BuffaloSnowEnv instance
    n_episodes   : number of evaluation episodes
    seed_offset  : starting seed
    deterministic: use deterministic policy (True for eval)

    Returns
    -------
    dict with mean/std for each metric
    """
    records = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        total_reward   = 0.0
        total_distance = 0.0
        high_pri_step  = None

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if high_pri_step is None and info.get("high_pri_cleared", False):
                high_pri_step = info["step"]

            if terminated or truncated:
                break

        records.append({
            "total_reward":       total_reward,
            "clearance_time":     info["step"],
            "clearance_pct":      info["clearance_pct"],
            "high_pri_clear_step": high_pri_step or info["step"],
            "total_distance":     total_distance,
        })

    return _aggregate(records)


def evaluate_heuristic_agent(
    env: BuffaloSnowEnv,
    n_episodes: int = 20,
    lookahead: int = 8,
    seed_offset: int = 0,
) -> dict:
    """Evaluate the greedy heuristic baseline."""
    agent = GreedyHeuristicAgent(env, lookahead=lookahead)
    records = []
    for ep in range(n_episodes):
        result = agent.run_episode(seed=seed_offset + ep)
        records.append(result)
    return _aggregate(records)


def _aggregate(records: list) -> dict:
    """Compute mean ± std across episodes."""
    result = {}
    for key in records[0]:
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        if vals:
            result[f"{key}_mean"] = float(np.mean(vals))
            result[f"{key}_std"]  = float(np.std(vals))
    result["n_episodes"] = len(records)
    return result


def compare_agents(
    env: BuffaloSnowEnv,
    models: dict,            # {"DQN": model, "PPO": model, ...}
    n_episodes: int = 20,
    include_heuristic: bool = True,
) -> dict:
    """
    Run evaluation for all agents and return comparison dict.

    Parameters
    ----------
    models : dict mapping agent name → SB3 model (or None for heuristic)

    Returns
    -------
    results : dict  {agent_name: metrics_dict}
    """
    results = {}

    if include_heuristic:
        print("Evaluating: Heuristic Baseline …")
        results["Heuristic"] = evaluate_heuristic_agent(env, n_episodes=n_episodes)

    for name, model in models.items():
        if model is None:
            continue
        print(f"Evaluating: {name} …")
        results[name] = evaluate_agent(model, env, n_episodes=n_episodes)

    return results


def print_comparison_table(results: dict):
    """Pretty-print a comparison table to stdout."""
    metrics = [
        ("total_reward",       "Total Reward",         "higher"),
        ("clearance_time",     "Clearance Time (steps)","lower"),
        ("clearance_pct",      "Final Clearance %",    "higher"),
        ("high_pri_clear_step","High-Pri Clear (steps)","lower"),
    ]

    header = f"{'Metric':<30}" + "".join(f"{name:<20}" for name in results.keys())
    print("\n" + "=" * (30 + 20 * len(results)))
    print(header)
    print("=" * (30 + 20 * len(results)))

    for key, label, _ in metrics:
        row = f"{label:<30}"
        for agent_metrics in results.values():
            mean = agent_metrics.get(f"{key}_mean", float("nan"))
            std  = agent_metrics.get(f"{key}_std",  float("nan"))
            row += f"{mean:>8.1f}±{std:<8.1f}"
        print(row)

    print("=" * (30 + 20 * len(results)))


def save_results(results: dict, path: str = "results/evaluation.json"):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Results saved to {path}")


if __name__ == "__main__":
    env = BuffaloSnowEnv(n_trucks=3, n_zones=10, max_steps=100,
                         snowfall_intensity="medium", seed=42)
    results = compare_agents(env, models={}, n_episodes=5, include_heuristic=True)
    print_comparison_table(results)
    save_results(results, path="../results/heuristic_only.json")
