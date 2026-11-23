"""
train.py
========
Training pipeline for SmartSnow RL agents.

Supports:
    - DQN  (stable-baselines3)
    - PPO  (stable-baselines3)
    - A2C  (stable-baselines3)
    - MaskablePPO (sb3-contrib, optional)

All agents are trained under identical environment settings for fair comparison.
Hyperparameters are loaded from configs/*.yaml.
"""

import os
import sys
import time
import json
import yaml
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.buffalo_snow_env import BuffaloSnowEnv

# ── Stable-Baselines3 imports (graceful fallback) ─────────────────────
SB3_AVAILABLE = False
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    print("[train] stable-baselines3 not found. Run: pip install stable-baselines3")

MASKABLE_AVAILABLE = False
try:
    from sb3_contrib import MaskablePPO
    MASKABLE_AVAILABLE = True
except ImportError:
    pass


# ── Default hyperparameters ───────────────────────────────────────────
DEFAULT_CONFIGS = {
    "dqn": {
        "learning_rate":     1e-4,
        "buffer_size":       50_000,
        "learning_starts":   1_000,
        "batch_size":        64,
        "gamma":             0.99,
        "exploration_fraction": 0.25,
        "exploration_final_eps": 0.05,
        "train_freq":        4,
        "target_update_interval": 500,
        "policy":            "MlpPolicy",
    },
    "ppo": {
        "learning_rate":     3e-4,
        "n_steps":           1024,
        "batch_size":        64,
        "n_epochs":          10,
        "gamma":             0.99,
        "gae_lambda":        0.95,
        "clip_range":        0.2,
        "ent_coef":          0.01,
        "policy":            "MlpPolicy",
    },
    "a2c": {
        "learning_rate":     7e-4,
        "n_steps":           5,
        "gamma":             0.99,
        "gae_lambda":        1.0,
        "ent_coef":          0.0,
        "vf_coef":           0.5,
        "max_grad_norm":     0.5,
        "policy":            "MlpPolicy",
    },
}

AGENT_CLASSES = {
    "dqn": "DQN",
    "ppo": "PPO",
    "a2c": "A2C",
    "maskableppo": "MaskablePPO",
}


def load_config(agent_name: str, config_dir: str = None) -> dict:
    """Load hyperparameters from YAML or use defaults."""
    agent_lower = agent_name.lower().replace("-", "")
    cfg = DEFAULT_CONFIGS.get(agent_lower, DEFAULT_CONFIGS["ppo"]).copy()

    if config_dir:
        path = os.path.join(config_dir, f"{agent_lower}.yaml")
        if os.path.exists(path):
            with open(path) as f:
                override = yaml.safe_load(f) or {}
            cfg.update(override)
            print(f"[train] Loaded config from {path}")

    return cfg


def make_env(intensity="medium", n_trucks=3, max_steps=200, seed=0):
    """Factory function for creating a monitored environment."""
    def _init():
        env = BuffaloSnowEnv(
            n_trucks=n_trucks,
            n_zones=10,
            max_steps=max_steps,
            snowfall_intensity=intensity,
            seed=seed,
        )
        return Monitor(env)
    return _init


def train_agent(
    agent_name: str,
    total_timesteps: int = 50_000,
    n_trucks: int = 3,
    snowfall_intensity: str = "medium",
    max_steps: int = 200,
    log_dir: str = "logs",
    model_dir: str = "models",
    config_dir: str = None,
    seed: int = 42,
    eval_freq: int = 5_000,
):
    """
    Train a single RL agent and save the model + logs.

    Returns
    -------
    model : trained SB3 model
    metrics : dict with training summary
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("stable-baselines3 is required. pip install stable-baselines3")

    agent_key = agent_name.lower().replace("-", "")
    cfg = load_config(agent_key, config_dir)
    policy = cfg.pop("policy", "MlpPolicy")

    # ── Create environments ───────────────────────────────────────
    train_env = make_vec_env(
        make_env(intensity=snowfall_intensity, n_trucks=n_trucks,
                 max_steps=max_steps, seed=seed),
        n_envs=1,
        seed=seed,
    )
    eval_env = make_env(intensity=snowfall_intensity, n_trucks=n_trucks,
                        max_steps=max_steps, seed=seed + 100)()

    # ── Log dirs ──────────────────────────────────────────────────
    tb_log   = os.path.join(log_dir, agent_key)
    save_dir = os.path.join(model_dir)
    os.makedirs(tb_log, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # ── Callbacks ─────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, f"best_{agent_key}"),
        log_path=os.path.join(log_dir, f"eval_{agent_key}"),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # ── Build model ───────────────────────────────────────────────
    model_cls_map = {
        "dqn": DQN, "ppo": PPO, "a2c": A2C,
    }

    if agent_key == "maskableppo" and MASKABLE_AVAILABLE:
        model = MaskablePPO(policy, train_env, verbose=0,
                            tensorboard_log=tb_log, seed=seed, **cfg)
    elif agent_key in model_cls_map:
        model = model_cls_map[agent_key](
            policy, train_env, verbose=0,
            tensorboard_log=tb_log, seed=seed, **cfg
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}. Choose from {list(AGENT_CLASSES.keys())}")

    # ── Train ─────────────────────────────────────────────────────
    start = time.time()
    print(f"[train] Training {agent_name.upper()} for {total_timesteps:,} timesteps …")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=False,
    )
    elapsed = time.time() - start

    # ── Save ──────────────────────────────────────────────────────
    save_path = os.path.join(save_dir, f"{agent_key}_final")
    model.save(save_path)
    print(f"[train] Model saved to {save_path}.zip  ({elapsed:.0f}s)")

    metrics = {
        "agent":         agent_name,
        "total_timesteps": total_timesteps,
        "train_time_s":  elapsed,
        "intensity":     snowfall_intensity,
        "n_trucks":      n_trucks,
        "seed":          seed,
    }

    return model, metrics


def train_all(
    agents=("dqn", "ppo", "a2c"),
    total_timesteps=50_000,
    **kwargs,
):
    """Train all specified agents and return results dict."""
    if not SB3_AVAILABLE:
        print("[train] stable-baselines3 not available. Skipping training.")
        return {}

    results = {}
    for agent in agents:
        try:
            model, metrics = train_agent(agent, total_timesteps=total_timesteps, **kwargs)
            results[agent] = {"model": model, "metrics": metrics}
        except Exception as e:
            print(f"[train] ERROR training {agent}: {e}")
            results[agent] = {"error": str(e)}
    return results


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartSnow agent trainer")
    parser.add_argument("--agent",      default="ppo",     help="Agent: dqn|ppo|a2c")
    parser.add_argument("--timesteps",  type=int, default=50_000)
    parser.add_argument("--intensity",  default="medium")
    parser.add_argument("--n_trucks",   type=int, default=3)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--log_dir",    default="logs")
    parser.add_argument("--model_dir",  default="models")
    args = parser.parse_args()

    model, metrics = train_agent(
        agent_name=args.agent,
        total_timesteps=args.timesteps,
        snowfall_intensity=args.intensity,
        n_trucks=args.n_trucks,
        seed=args.seed,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "model"}, indent=2))
