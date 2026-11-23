"""
tests/test_env.py
=================
Unit tests for BuffaloSnowEnv, snowfall model, and road graph.
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.road_graph import get_graph, get_edge_list, get_zone_partition
from env.snowfall_model import SnowfallModel
from env.buffalo_snow_env import BuffaloSnowEnv


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def small_env():
    return BuffaloSnowEnv(n_trucks=2, n_zones=5, max_steps=30, n_nodes=36, seed=0)

@pytest.fixture
def medium_env():
    return BuffaloSnowEnv(n_trucks=3, n_zones=10, max_steps=100, n_nodes=100, seed=42)


# ── Road graph tests ──────────────────────────────────────────────────
class TestRoadGraph:
    def test_synthetic_graph_created(self):
        G = get_graph(use_real=False, n_nodes=100)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_edge_attributes_present(self):
        G = get_graph(use_real=False, n_nodes=36)
        for u, v, d in G.edges(data=True):
            assert "priority_class" in d
            assert "snow_depth" in d
            assert "length" in d
            assert 1 <= d["priority_class"] <= 5

    def test_zone_partition_covers_all_nodes(self):
        G = get_graph(use_real=False, n_nodes=36)
        partition = get_zone_partition(G, n_zones=5)
        for node in G.nodes():
            assert node in partition


# ── Snowfall model tests ──────────────────────────────────────────────
class TestSnowfallModel:
    def test_sample_non_negative(self):
        model = SnowfallModel(intensity="medium", seed=0)
        for _ in range(100):
            assert model.sample() >= 0.0

    def test_high_intensity_more_than_low(self):
        low  = SnowfallModel(intensity="low",  seed=0)
        high = SnowfallModel(intensity="high", seed=0)
        assert high.expected_daily_inches > low.expected_daily_inches

    def test_none_intensity_zero(self):
        model = SnowfallModel(intensity="none", seed=0)
        for _ in range(20):
            assert model.sample() == 0.0

    def test_batch_shape(self):
        model = SnowfallModel(intensity="medium", seed=42)
        batch = model.sample_batch(50)
        assert batch.shape == (50,)


# ── Environment core tests ────────────────────────────────────────────
class TestBuffaloSnowEnv:
    def test_env_checker_passes(self, small_env):
        check_env(small_env, skip_render_check=True)

    def test_reset_returns_valid_obs(self, small_env):
        obs, info = small_env.reset(seed=0)
        assert obs.shape == small_env.observation_space.shape
        assert small_env.observation_space.contains(obs)
        assert "clearance_pct" in info

    def test_step_returns_correct_types(self, small_env):
        small_env.reset(seed=0)
        action = small_env.action_space.sample()
        obs, rew, term, trunc, info = small_env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(rew, (float, np.floating))
        assert isinstance(term, (bool, np.bool_))
        assert isinstance(trunc, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_obs_in_bounds(self, medium_env):
        obs, _ = medium_env.reset(seed=1)
        for _ in range(20):
            action = medium_env.action_space.sample()
            obs, _, term, trunc, _ = medium_env.step(action)
            assert medium_env.observation_space.contains(obs), f"obs out of bounds: {obs.min():.3f}..{obs.max():.3f}"
            if term or trunc:
                obs, _ = medium_env.reset(seed=2)

    def test_episode_terminates(self, small_env):
        small_env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 200:
            action = small_env.action_space.sample()
            _, _, term, trunc, _ = small_env.step(action)
            done = term or trunc
            steps += 1
        assert done, "Episode should terminate within 200 steps"

    def test_reward_is_float(self, small_env):
        small_env.reset(seed=0)
        for _ in range(10):
            action = small_env.action_space.sample()
            _, reward, term, trunc, _ = small_env.step(action)
            assert isinstance(reward, float)
            if term or trunc:
                break

    def test_action_mask_shape(self, medium_env):
        medium_env.reset(seed=0)
        mask = medium_env.action_masks()
        expected_len = medium_env.n_trucks * medium_env.n_segments
        assert mask.shape == (expected_len,)
        assert mask.dtype == bool

    def test_clearance_increases_without_snow(self, small_env):
        """Clearance % should generally increase when snowfall_intensity='none'."""
        env = BuffaloSnowEnv(n_trucks=2, n_zones=5, max_steps=50, n_nodes=36,
                             seed=0, snowfall_intensity="none")
        obs, info = env.reset(seed=0)
        initial_cleared = info["segments_cleared"]
        for _ in range(20):
            action = env.action_space.sample()
            _, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        # With no new snow, we should clear at least as many segments
        assert info["segments_cleared"] >= initial_cleared


# ── Heuristic agent tests ─────────────────────────────────────────────
class TestHeuristicAgent:
    def test_heuristic_action_valid(self, medium_env):
        from agents.heuristic import GreedyHeuristicAgent
        medium_env.reset(seed=0)
        agent = GreedyHeuristicAgent(medium_env)
        action = agent.select_action()
        assert len(action) == medium_env.n_trucks
        for a in action:
            assert 0 <= a < medium_env.n_segments

    def test_heuristic_runs_full_episode(self, medium_env):
        from agents.heuristic import GreedyHeuristicAgent
        medium_env.reset(seed=0)
        agent = GreedyHeuristicAgent(medium_env)
        result = agent.run_episode(seed=0)
        assert "total_reward" in result
        assert "clearance_pct" in result
        assert 0 <= result["clearance_pct"] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
