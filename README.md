# SmartSnow 🚛❄️
**Buffalo Snow Plow Routing with Deep Reinforcement Learning**

> CSE 4/546: Reinforcement Learning — Spring 2026 | University at Buffalo
> Instructor: Alina Vereshchaka

---

## Overview

SmartSnow is an Applied RL framework that trains autonomous plow truck routing agents over Buffalo's road network. The environment simulates real stochastic snowfall (calibrated to NOAA Buffalo data) on a prioritised road graph and compares three deep RL algorithms — DQN, PPO, and A2C — against a greedy heuristic baseline.

---

## Project Structure

```
SmartSnow/
├── env/
│   ├── buffalo_snow_env.py     # Custom Gymnasium environment
│   ├── road_graph.py           # Road graph builder (synthetic + OSMnx hook)
│   └── snowfall_model.py       # Gamma-distributed stochastic snowfall
├── agents/
│   └── heuristic.py            # Greedy priority-weighted baseline agent
├── training/
│   └── train.py                # DQN / PPO / A2C / MaskablePPO pipeline
├── eval/
│   ├── evaluate.py             # Multi-agent evaluation & comparison
│   ├── visualize.py            # Learning curves, heatmaps, bar charts
│   └── render_env.py           # Grid-world style environment renderer
├── tests/
│   └── test_env.py             # 17 unit tests (all passing)
├── configs/                    # YAML hyperparameter files
├── results/                    # Generated figures and metrics
├── data/                       # NOAA CSV + cached OSMnx graph (Phase 2)
├── models/                     # Saved model checkpoints
├── logs/                       # Training logs
├── requirements.txt
└── final_project_checkpoint_gandhars_litheesh_sindukur.ipynb
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-REPO-URL.git
cd SmartSnow
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `stable-baselines3` and `sb3-contrib` are required for RL training.
> `osmnx` is only needed for Phase 2 (real Buffalo road network).

---

## Reproducing Results

### Step 1 — Run the unit tests

Verify the full environment is working correctly:

```bash
cd SmartSnow
pytest tests/ -v
```

Expected output:
```
17 passed in ~1s
```

---

### Step 2 — Run the heuristic baseline

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from env.buffalo_snow_env import BuffaloSnowEnv
from agents.heuristic import GreedyHeuristicAgent

env = BuffaloSnowEnv(n_trucks=3, n_zones=10, snowfall_intensity='medium', max_steps=200, seed=42)
agent = GreedyHeuristicAgent(env)
metrics = agent.run_episode(seed=42)
print(metrics)
"
```

Expected output:
```
{'total_reward': ~-190000, 'clearance_pct': ~1.7, 'clearance_time': 200, ...}
```

---

### Step 3 — Generate all result figures

```bash
python3 eval/render_env.py
```

This generates the following in `results/`:

| File | Description |
|------|-------------|
| `env_initial_frame.png` | Environment at t=0 — snow depth heatmap + truck positions |
| `env_episode_summary.png` | 6-panel rollout snapshots + reward & clearance curves |
| `env_road_structure.png` | Road priority map + zone partition map |
| `env_rollout.gif` | Animated 60-step rollout (3 fps) |
| `priority_distribution.png` | Priority class bar chart + colour-coded road map |
| `snowfall_distribution.png` | Gamma distribution across 5 intensity regimes |
| `clearance_progression.png` | Clearance % over episode steps |
| `heuristic_metrics.json` | Heuristic baseline metrics (10 episodes) |

---

### Step 4 — Train RL agents (requires stable-baselines3)

```bash
pip install stable-baselines3 sb3-contrib
```

Train each agent for 200,000 timesteps:

```bash
# DQN
python3 training/train.py --agent dqn --timesteps 200000

# PPO
python3 training/train.py --agent ppo --timesteps 200000

# A2C
python3 training/train.py --agent a2c --timesteps 200000

# MaskablePPO (bonus — action masking)
python3 training/train.py --agent maskable_ppo --timesteps 200000
```

Trained models are saved to `models/` and training logs to `logs/`.

---

### Step 5 — Run the checkpoint notebook

```bash
jupyter notebook final_project_checkpoint_gandhars_litheesh_sindukur.ipynb
```

Run all cells top to bottom. The notebook covers:
1. Environment setup and validation
2. Road network + snowfall model
3. Heuristic baseline evaluation
4. RL training pipeline overview
5. Result visualisations
6. Next steps

---

## Environment Details

| Property | Value |
|----------|-------|
| Observation space | `Box(0, 1, shape=(32,))` — 10 zones × 3 features + 2 global |
| Action space | `MultiDiscrete([180, 180, 180])` — 3 trucks × 180 segments |
| Max steps per episode | 200 |
| Number of trucks | 3 |
| Number of road segments | 180 |
| Snowfall intensities | `none`, `low`, `medium`, `high`, `blizzard` |

### Road Priority System

| Priority | Road Type | Segments | % |
|----------|-----------|----------|---|
| P5 | Highway / Hospital | 18 | 10% |
| P4 | Primary | 18 | 10% |
| P3 | Secondary | 18 | 10% |
| P2 | Tertiary | 18 | 10% |
| P1 | Residential | 108 | 60% |

### Reward Function

```
reward = Σ (priority × 2.0)          # for each segment cleared
       - Σ (priority × steps_uncleared^1.5 × 0.005)  # urgency penalty
       - λ × total_distance           # distance cost
```

---

## Snowfall Model

Fitted to NOAA station **USW00014733** (Buffalo Niagara International Airport):

| Intensity | P(snow) | Mean (cm) |
|-----------|---------|-----------|
| none | 0% | 0.000 |
| low | 25% | 0.190 |
| medium | 45% | 0.897 |
| high | 70% | 2.396 |
| blizzard | 95% | 6.572 |

Distribution: `Gamma(shape=0.72, scale=2.8)` with zero-inflation gate.

To use real NOAA data, download the CSV from [ncdc.noaa.gov](https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00014733/detail) and place it at `data/noaa_buffalo_snowfall.csv`. The model will auto-fit on startup.

---

## Switching to Real Buffalo Road Network (Phase 2)

```python
from env.buffalo_snow_env import BuffaloSnowEnv

# Phase 1 (default) — synthetic 100-node grid
env = BuffaloSnowEnv(use_real_graph=False)

# Phase 2 — real OSMnx Buffalo road network (~8,000 nodes)
# Requires: pip install osmnx
env = BuffaloSnowEnv(use_real_graph=True)
# Downloads and caches to data/buffalo_graph.pkl on first run
```

---

## Baseline Performance (Checkpoint)

Greedy heuristic agent, 10 episodes, medium snowfall, 200 steps:

| Metric | Value |
|--------|-------|
| Mean cumulative reward | −203,880 |
| Std | ±46,136 |
| Mean % roads cleared | 1.3% |
| Mean episode length | 200 steps |

This is the **target floor** all RL agents must surpass.

---

## Dependencies

```
numpy>=1.24.0        scipy>=1.10.0       matplotlib>=3.7.0
networkx>=3.1        gymnasium>=0.29.0   stable-baselines3>=2.2.0
sb3-contrib>=2.2.0   osmnx>=1.9.0        pyyaml>=6.0
pytest>=7.4.0        jupyter>=1.0.0      folium>=0.14.0
```

Full list in `requirements.txt`.

---

## GitHub Project Board

Track progress: [GitHub Project Board](https://github.com/YOUR-BOARD-URL)

---

## References

- Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015
- Schulman et al., "Proximal Policy Optimization Algorithms," *arXiv*, 2017
- Raffin et al., "Stable-Baselines3," *JMLR*, 2021
- Boeing, "OSMnx," *Computers, Environment and Urban Systems*, 2017
- NOAA NCEI, Station USW00014733, Buffalo Niagara International Airport
