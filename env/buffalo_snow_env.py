"""
buffalo_snow_env.py
===================
BuffaloSnowEnv — Custom Gymnasium environment for snow plow routing.

State Space (zone-aggregated, fixed-size):
    For each of N_ZONES zones:
        - mean snow depth (normalised 0–1)
        - fraction of segments cleared (0–1)
        - fraction of trucks in zone (0–1)
    Plus global features:
        - normalised timestep
        - normalised total snow remaining

Action Space:
    MultiDiscrete — each truck selects one road segment index.
    With action masking: only reachable segments are valid.

Reward:
    Urgency-decay shaped reward:
        + priority_class * 2.0   for each segment cleared this step
        - priority * steps_uncleared^1.5 * 0.01   per uncleared segment
        - 0.05 * distance_travelled                 fuel penalty
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Optional, Tuple, Dict, Any, List

from env.road_graph import get_graph, get_edge_list, get_zone_partition
from env.snowfall_model import SnowfallModel


# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_N_ZONES  = 10
DEFAULT_N_TRUCKS = 3
MAX_SNOW_DEPTH   = 30.0    # inches — used for normalisation
MAX_STEPS        = 200     # episode length


class BuffaloSnowEnv(gym.Env):
    """
    Gymnasium environment simulating snow plow operations over Buffalo's
    road network (synthetic graph by default; real OSMnx optional).

    Parameters
    ----------
    n_trucks     : number of plow trucks (agents)
    n_zones      : number of geographic zones for state aggregation
    max_steps    : episode time limit
    snowfall_intensity : 'none' | 'low' | 'medium' | 'high' | 'blizzard'
    use_real_graph     : use real OSMnx Buffalo graph (requires osmnx)
    n_nodes            : number of nodes for synthetic graph
    seed               : env random seed
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        n_trucks: int = DEFAULT_N_TRUCKS,
        n_zones: int = DEFAULT_N_ZONES,
        max_steps: int = MAX_STEPS,
        snowfall_intensity: str = "medium",
        use_real_graph: bool = False,
        n_nodes: int = 100,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        noaa_data_path: Optional[str] = None,
    ):
        super().__init__()

        self.n_trucks   = n_trucks
        self.n_zones    = n_zones
        self.max_steps  = max_steps
        self.intensity  = snowfall_intensity
        self.render_mode = render_mode
        self._seed      = seed

        # ── Build road graph ──────────────────────────────────────
        self.G = get_graph(use_real=use_real_graph, n_nodes=n_nodes, seed=seed or 0)
        self.edges      = get_edge_list(self.G)
        self.n_segments = len(self.edges)
        self.nodes      = list(self.G.nodes())
        self.n_nodes    = len(self.nodes)

        # Edge → index mapping
        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}

        # Zone partition {node: zone_id}
        self.zone_partition = get_zone_partition(self.G, n_zones=n_zones)

        # Map each edge to a zone (use first node's zone)
        self.edge_zones = np.array(
            [self.zone_partition.get(u, 0) for u, v in self.edges], dtype=np.int32
        )

        # Edge priority classes
        self.edge_priorities = np.array(
            [self.G[u][v].get("priority_class", 1) for u, v in self.edges],
            dtype=np.float32
        )

        # Edge lengths (for fuel penalty)
        self.edge_lengths = np.array(
            [self.G[u][v].get("length", 1.0) for u, v in self.edges],
            dtype=np.float32
        )

        # ── Snowfall model ────────────────────────────────────────
        self.snow_model = SnowfallModel(
            data_path=noaa_data_path,
            intensity=snowfall_intensity,
            seed=seed
        )

        # ── Spaces ───────────────────────────────────────────────
        # Observation: (n_zones * 3) + 2 global features
        obs_dim = n_zones * 3 + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: each truck independently picks a segment
        self.action_space = spaces.MultiDiscrete(
            [self.n_segments] * n_trucks
        )

        # ── Episode state (initialised in reset) ─────────────────
        self.snow_depth     = np.zeros(self.n_segments, dtype=np.float32)
        self.steps_uncleared = np.zeros(self.n_segments, dtype=np.int32)
        self.truck_positions = np.zeros(n_trucks, dtype=np.int32)  # node indices
        self.current_step   = 0
        self._rng           = np.random.default_rng(seed)

        # Logging
        self._episode_rewards: List[float] = []
        self._cleared_log: List[int] = []

    # ── Core API ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        # Always re-seed the internal RNG from gymnasium's np_random for determinism
        int_seed = int(self.np_random.integers(0, 2**31))
        self._rng = np.random.default_rng(int_seed)
        self.snow_model = SnowfallModel(
            intensity=self.intensity, seed=int_seed
        )

        # Initial snow: random light dusting on all segments
        self.snow_depth = self._rng.uniform(
            0.5, 4.0, size=self.n_segments
        ).astype(np.float32)

        self.steps_uncleared = np.ones(self.n_segments, dtype=np.int32)

        # Place trucks at random nodes
        self.truck_positions = self._rng.choice(
            self.n_nodes, size=self.n_trucks, replace=True
        ).astype(np.int32)

        self.current_step = 0
        self._episode_rewards = []
        self._cleared_log = []

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.

        action: array of shape (n_trucks,) — each value ∈ [0, n_segments)
        """
        self.current_step += 1
        reward = 0.0
        distance_this_step = 0.0
        n_cleared_this_step = 0

        # ── 1. Move each truck and attempt clearance ──────────────
        for t_idx, seg_idx in enumerate(action):
            seg_idx = int(seg_idx) % self.n_segments
            u, v = self.edges[seg_idx]

            # Distance cost: shortest path from current truck node to segment
            truck_node = self.nodes[self.truck_positions[t_idx]]
            try:
                path_len = nx.shortest_path_length(
                    self.G, truck_node, u, weight="length"
                )
            except nx.NetworkXNoPath:
                path_len = self.edge_lengths[seg_idx]

            distance_this_step += path_len + self.edge_lengths[seg_idx]

            # Move truck to end of target segment
            self.truck_positions[t_idx] = self.nodes.index(v) if v in self.nodes else self.truck_positions[t_idx]

            # Clear the segment
            if self.snow_depth[seg_idx] > 0:
                n_cleared_this_step += 1
                reward += float(self.edge_priorities[seg_idx]) * 2.0
                self.snow_depth[seg_idx] = 0.0
                self.steps_uncleared[seg_idx] = 0
            else:
                # Penalty for re-visiting already clear road
                reward -= 0.5

        # ── 2. Accumulate new snowfall ────────────────────────────
        new_snow = self.snow_model.sample(self.intensity)
        if new_snow > 0:
            # Snow falls proportionally more on high-priority roads (traffic)
            snow_per_seg = new_snow * (0.5 + 0.5 * self.edge_priorities / 5.0)
            self.snow_depth += snow_per_seg.astype(np.float32)
            self.snow_depth = np.clip(self.snow_depth, 0, MAX_SNOW_DEPTH)

        # ── 3. Update urgency counters ────────────────────────────
        uncleared_mask = self.snow_depth > 0
        self.steps_uncleared[uncleared_mask] += 1
        self.steps_uncleared[~uncleared_mask] = 0

        # ── 4. Urgency penalty ────────────────────────────────────
        urgency_penalty = np.sum(
            self.edge_priorities[uncleared_mask]
            * (self.steps_uncleared[uncleared_mask].astype(float) ** 1.5)
            * 0.005
        )
        reward -= urgency_penalty

        # ── 5. Fuel / distance penalty ───────────────────────────
        reward -= 0.05 * distance_this_step / (self.n_segments * 200)

        # ── 6. Bonus for fully clearing high-priority segments ────
        high_pri = self.edge_priorities >= 4
        if np.all(self.snow_depth[high_pri] == 0):
            reward += 10.0

        # ── 7. Termination ────────────────────────────────────────
        all_clear   = np.all(self.snow_depth == 0)
        time_limit  = self.current_step >= self.max_steps
        terminated  = all_clear
        truncated   = time_limit and not all_clear

        self._episode_rewards.append(reward)
        self._cleared_log.append(n_cleared_this_step)

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ── Observation ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Zone-aggregated observation vector (fixed size regardless of graph scale).

        For each zone (n_zones):
            [0] mean snow depth  (normalised 0–1)
            [1] fraction cleared (0–1)
            [2] fraction of trucks in zone (0–1)

        Global:
            [n_zones*3 + 0] normalised step (0–1)
            [n_zones*3 + 1] total snow remaining (0–1)
        """
        obs = np.zeros(self.n_zones * 3 + 2, dtype=np.float32)

        # Zone-level features
        zone_snow_sum   = np.zeros(self.n_zones, dtype=np.float32)
        zone_seg_count  = np.zeros(self.n_zones, dtype=np.float32)
        zone_clear_count = np.zeros(self.n_zones, dtype=np.float32)

        for seg_idx, zone in enumerate(self.edge_zones):
            if zone >= self.n_zones:
                continue
            zone_seg_count[zone]  += 1
            zone_snow_sum[zone]   += self.snow_depth[seg_idx]
            if self.snow_depth[seg_idx] == 0:
                zone_clear_count[zone] += 1

        for z in range(self.n_zones):
            count = max(zone_seg_count[z], 1)
            base  = z * 3
            obs[base + 0] = min(zone_snow_sum[z] / count / MAX_SNOW_DEPTH, 1.0)
            obs[base + 1] = zone_clear_count[z] / count
            # Truck density in zone
            trucks_in_zone = sum(
                1 for t_node_idx in self.truck_positions
                if self.zone_partition.get(self.nodes[t_node_idx], 0) == z
            )
            obs[base + 2] = trucks_in_zone / max(self.n_trucks, 1)

        # Global features
        obs[-2] = self.current_step / self.max_steps
        obs[-1] = min(np.sum(self.snow_depth) / (self.n_segments * MAX_SNOW_DEPTH), 1.0)

        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step":            self.current_step,
            "total_snow":      float(np.sum(self.snow_depth)),
            "segments_cleared": int(np.sum(self.snow_depth == 0)),
            "segments_total":  self.n_segments,
            "clearance_pct":   float(np.mean(self.snow_depth == 0)) * 100,
            "high_pri_cleared": bool(np.all(self.snow_depth[self.edge_priorities >= 4] == 0)),
        }

    # ── Action mask ───────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """
        Binary mask of valid actions for each truck (flattened).
        Used with sb3-contrib MaskablePPO.
        A segment is valid if it has snow > 0 OR is reachable from truck.
        """
        mask = np.zeros((self.n_trucks, self.n_segments), dtype=bool)
        for t_idx, node_idx in enumerate(self.truck_positions):
            truck_node = self.nodes[node_idx]
            try:
                reachable = set(nx.single_source_shortest_path_length(
                    self.G, truck_node, cutoff=5
                ).keys())
            except Exception:
                reachable = set(self.nodes)

            for seg_idx, (u, v) in enumerate(self.edges):
                if u in reachable or v in reachable:
                    mask[t_idx, seg_idx] = True

        return mask.flatten()

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "ansi":
            cleared = int(np.sum(self.snow_depth == 0))
            total   = self.n_segments
            snow    = float(np.sum(self.snow_depth))
            print(
                f"Step {self.current_step:>4d} | "
                f"Cleared {cleared}/{total} ({100*cleared/total:.0f}%) | "
                f"Snow remaining: {snow:.1f} in"
            )

    def close(self):
        pass


# ── Quick smoke test ──────────────────────────────────────────────────
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = BuffaloSnowEnv(n_trucks=2, n_zones=10, max_steps=50,
                         snowfall_intensity="medium", seed=42)
    print("Checking environment …")
    check_env(env, warn=True, skip_render_check=True)
    print("check_env passed!")

    obs, info = env.reset(seed=0)
    print(f"Obs shape : {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Obs sample: {obs[:6].round(3)} …")

    total_reward = 0
    for _ in range(50):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        total_reward += rew
        if terminated or truncated:
            break

    print(f"Episode reward (random): {total_reward:.2f}")
    print(f"Final info: {info}")
