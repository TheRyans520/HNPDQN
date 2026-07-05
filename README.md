# HNPDQN — Hybrid Neural Polynomial Deep Q-Network

HNPDQN is a deep reinforcement learning agent for **anti-jamming spectrum access** in wireless communications. It enhances DQN with a **polynomial expansion layer** (high-order feature interactions) and a **dueling network architecture** to learn better Q-value approximations in highly dynamic jamming environments.

The environment uses real RF spectral scan data collected with QC9880 chips across the 2.4 GHz WiFi band under multiple jamming scenarios.

## Project Structure

```
├── agents/
│   ├── Base_Agent.py              # Base agent class
│   ├── Trainer.py                 # Training/testing loop
│   └── DQN_agents/
│       ├── HNP_DQN.py             # ★ Core: polynomial expansion + dueling
│       ├── DQN.py
│       ├── DDQN.py
│       ├── DQN_With_Fixed_Q_Targets.py
│       └── Dueling_DDQN.py
├── environments/
│   └── RF_spectrum.py             # Gym environment for RF anti-jamming
├── utilities/                     # Replay buffer, config, exploration strategies
├── data/                          # Dataset loader
├── datasets/                      # Raw & processed spectral scan data
├── exploration_strategies/        # Epsilon-greedy, Gaussian, OU Noise
├── results/
│   ├── Anti_Jam.py                # ★ Main entry: training & testing
│   ├── models/                    # Pre-trained model weights (.pt)
│   └── data_and_graphs/           # Saved results & figures
└── requirements.txt
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
cd results
python Anti_Jam.py
```

Set `RUN_MODE = "train"` in `Anti_Jam.py` (line 76). This trains all agents  for 100 episodes × 3 runs each.

### Test with pre-trained models

```bash
cd results
python Anti_Jam.py
```

Set `RUN_MODE = "test"` in `Anti_Jam.py`. Pre-trained model weights are already provided under:

- `results/models/exp_0.1_sweeping/` — sweep jammer scenario
- `results/models/exp_0.1_dynamic/` — dynamic jammer scenario

Each folder contains `.pt` files for all agents. Testing runs 50 episodes and generates reward curves under `results/data_and_graphs/`.

### Run both (train then test)

Set `RUN_MODE = "both"`.

## Key Configuration

In `results/Anti_Jam.py`:

| Setting | Description |
|---|---|
| `RUN_MODE` | `"train"`, `"test"`, or `"both"` |
| `jammer` | Jammer type: `"sweep"` or `"dynamic"` |
| `BAND` | Frequency band: `"2.4G"` |
| `config.num_episodes_to_run` | Episodes per run |
| `config.runs_per_agent` | Number of independent runs |
| `config.use_GPU` | Set to `True` if CUDA is available |