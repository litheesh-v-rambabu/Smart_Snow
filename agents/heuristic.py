"""
heuristic.py
============
Greedy Priority-Based Heuristic Baseline Agent.

Strategy:
    At each timestep, each truck is assigned to the reachable road segment
    with the highest score:
        score(seg) = priority_class(seg) * snow_depth(seg)
                     / (shortest_path_distance(truck, seg) + 1)

    Ties broken by priority_class first, then snow_depth.

This is our primary comparison baseline against which RL agents are evaluated.
"""

import numpy as np
import networkx as nx
from typing import List


class GreedyHeuristicAgent:
    """
    Greedy priority-weighted nearest-segment heuristic.

    Parameters
    ----------
    env : BuffaloSnowEnv
        The environment instance (read-only access).
    lookahead : int
        Maximum graph distance (hop count) considered for reachability.
        Lower = faster but myopic; higher = better but slower.
    """

    def __init__(self, env, lookahead: int = 8):
        self.env       = env
        self.lookahead = lookahead
        self._name     = "GreedyHeuristic"

    def select_action(self) -> np.ndarray:
        """
        Select an action for the current environment state.

        Returns
        -------
        action : np.ndarray of shape (n_trucks,)
            Segment index for each truck.
        """
        env = self.env
        action = np.zeros(env.n_trucks, dtype=np.int32)

        # Track which segments are already assigned this step
        # to avoid two trucks going to the same segment
        assigned = set()

        for t_idx in range(env.n_trucks):
            truck_node = env.nodes[env.truck_positions[t_idx]]

            # Get reachable nodes within lookahead hops
            try:
                dist_map = nx.single_source_shortest_path_length(
                    env.G, truck_node, cutoff=self.lookahead
                )
            except Exception:
                dist_map = {n: 1 for n in env.nodes}

            best_score = -np.inf
            best_seg   = 0

            for seg_idx, (u, v) in enumerate(env.edges):
                if seg_idx in assigned:
                    continue
                if env.snow_depth[seg_idx] <= 0:
                    continue   # Already clear

                # Minimum distance from truck to either endpoint
                d_u = dist_map.get(u, self.lookahead + 1)
                d_v = dist_map.get(v, self.lookahead + 1)
                dist = min(d_u, d_v)

                priority = env.edge_priorities[seg_idx]
                snow     = env.snow_depth[seg_idx]
                urgency  = env.steps_uncleared[seg_idx]

                # Score: higher priority + more snow + urgent + close
                score = (priority * 3.0 + snow + urgency * 0.5) / (dist + 1.0)

                if score > best_score:
                    best_score = score
                    best_seg   = seg_idx

            action[t_idx] = best_seg
            assigned.add(best_seg)

        return action

    def run_episode(self, seed: int = 0) -> dict:
        """
        Run a complete episode and return metrics.

        Returns
        -------
        dict with keys:
            total_reward, clearance_time, total_distance,
            high_pri_clear_step, clearance_pct_final
        """
        obs, info = self.env.reset(seed=seed)
        total_reward    = 0.0
        total_distance  = 0.0
        high_pri_step   = None

        while True:
            action = self.select_action()

            # Estimate distance this step
            for t_idx, seg_idx in enumerate(action):
                truck_node = self.env.nodes[self.env.truck_positions[t_idx]]
                u, v = self.env.edges[seg_idx]
                try:
                    d = nx.shortest_path_length(self.env.G, truck_node, u, weight="length")
                except Exception:
                    d = 0
                total_distance += d + self.env.edge_lengths[seg_idx]

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if high_pri_step is None and info.get("high_pri_cleared", False):
                high_pri_step = info["step"]

            if terminated or truncated:
                break

        return {
            "total_reward":      total_reward,
            "clearance_time":    info["step"],
            "total_distance":    total_distance,
            "high_pri_clear_step": high_pri_step or info["step"],
            "clearance_pct":     info["clearance_pct"],
            "segments_cleared":  info["segments_cleared"],
            "segments_total":    info["segments_total"],
        }


def evaluate_heuristic(env, n_episodes: int = 10, lookahead: int = 8) -> dict:
    """
    Run n_episodes and return mean ± std of all metrics.
    """
    agent = GreedyHeuristicAgent(env, lookahead=lookahead)
    records = [agent.run_episode(seed=i) for i in range(n_episodes)]

    metrics = {}
    for key in records[0].keys():
        vals = [r[key] for r in records]
        metrics[f"{key}_mean"] = float(np.mean(vals))
        metrics[f"{key}_std"]  = float(np.std(vals))

    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from env.buffalo_snow_env import BuffaloSnowEnv

    env = BuffaloSnowEnv(n_trucks=3, n_zones=10, max_steps=100,
                         snowfall_intensity="medium", seed=42)
    metrics = evaluate_heuristic(env, n_episodes=5)

    print("\nGreedy Heuristic Baseline Results (5 episodes)")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<35s}: {v:.3f}")
