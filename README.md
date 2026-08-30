# HNP-DQN — Hybrid Neural–Polynomial Deep Q-Network

This repository provides an implementation of **HNP-DQN (Hybrid Neural–Polynomial Deep Q-Network)** for switching-aware adaptive spectrum selection in a controlled radio-frequency (RF) measurement-replay environment.

The work is associated with the paper:

> **Evaluation of a Hybrid Neural–Polynomial Deep Q-Network for Switching-Aware Spectrum Selection in a Controlled Radio-Frequency Measurement-Replay Testbed**

HNP-DQN is evaluated as a structured value-function approximator for an eight-channel spectrum-selection task. The architecture combines a learned neural representation with an explicit element-wise second-order feature expansion, LayerNorm, a trainable projection layer, and a dueling Double Deep Q-Network (DDQN) backbone.

The experimental environment uses measurements from the **RF Jamming Dataset** and focuses on a controlled spectrum-decision layer motivated by cognitive-radar spectrum agility. The repository and paper do **not** constitute an end-to-end radar hardware, detection, tracking, or waveform-validation system.

## Important Reproduction Note

This repository contains the implementation used as the basis of the HNP-DQN experiments. However, some configuration values, legacy experiment scripts, directory names, and default parameters in the current repository may not exactly match the frozen protocol reported in the final paper.

**Readers attempting to reproduce the published experiments should use the parameter values and data-split protocol reported in the paper as the authoritative configuration and manually adjust the corresponding settings in the repository where necessary.**

In particular, older scripts or folders may retain settings from earlier development experiments, such as different frequency bands, numbers of episodes, numbers of runs, evaluation counts, or jammer naming conventions.

## Paper-Aligned Experimental Setting

| Setting | Paper Configuration |
|---|---|
| Frequency band | 5 GHz |
| Candidate channels | 8 |
| Channel centers | 5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320 MHz |
| Channel spacing | 20 MHz |
| Window length | 32 samples |
| Window stride | 1 |
| RF observation features | 40 |
| Previous-channel encoding | 8 |
| Total observation dimension | 48 |
| Maximum steps per episode | 100 |
| Training episodes | 200 |
| Switching cost | 0.1 |
| Independent training seeds | 10 per learned method and jammer mode |
| Evaluation trajectories | 20 fixed trajectories per condition and jammer mode |
| Jammer modes | Periodic sweeping and independent random jamming |

The RF part of the observation contains five statistics for each of the eight channels:

- mean of the dataset-provided `snr` field;
- standard deviation of the dataset-provided `snr` field;
- mean RSSI;
- standard deviation of RSSI;
- mean noise level.

These 40 RF features are concatenated with an eight-dimensional one-hot encoding of the previously selected channel to produce the 48-dimensional agent observation.

The dataset-provided `snr` field is used as provided by the source dataset and should not be interpreted as a separately calibrated physical SNR or SINR measurement.

## Data Split Used in the Paper

Raw scan files are separated before sliding-window construction.

| Physical Condition | Scan IDs | Role |
|---|---:|---|
| 20 cm / 10 dBm | 0–6 | Training |
| 20 cm / 10 dBm | 7 | Validation |
| 20 cm / 10 dBm | 8–9 | Pre-inspected within-condition pilot |
| 40 cm / 10 dBm | 0–9 | Evaluation-only distance transfer |
| 20 cm / 5 dBm | 0–9 | Evaluation-only power transfer |

Windows are generated independently within each raw source file and never cross file boundaries.

The 40 cm / 10 dBm and 20 cm / 5 dBm conditions are treated as the two declared cross-configuration evaluation settings in the final paper.

## HNP-DQN Architecture

The paper-aligned HNP-DQN architecture is:

```text
48-dimensional observation
        │
        ▼
Linear 48 → 128
BatchNorm + ReLU
        │
        ▼
Linear 128 → 64
BatchNorm + ReLU
        │
        ▼
Latent representation h ∈ R^64
        │
        ▼
Element-wise second-order expansion
[h, h²] ∈ R^128
        │
        ▼
LayerNorm
        │
        ▼
Linear projection 128 → 128
ReLU
        │
        ▼
Dueling value / advantage heads
        │
        ▼
8 channel-action Q-values
```

The polynomial transformation contains the first- and second-order element-wise terms:

```text
[h1, ..., h64, h1², ..., h64²]
```

It does **not** include a constant term or pairwise cross-feature products such as `hi * hj`.

The HNP-DQN model reported in the paper contains **32,841 trainable parameters**.

## Learning Configuration Reported in the Paper

| Parameter | Paper Value |
|---|---:|
| Learning rate | 0.001 |
| Optimizer | Adam |
| TD objective | Mean-squared TD error |
| Discount factor `gamma` | 0.95 |
| Soft target-update rate `tau` | 0.005 |
| Replay-buffer capacity | 100,000 |
| Batch size | 64 |
| Learning start | 64 steps |
| Update frequency | Every step |
| Gradient clipping threshold | 0.7 |
| Adam weight decay | `1e-4` |
| Minimum epsilon | 0.01 |
| Epsilon schedule | `max(0.01, 1 / (1 + episode / 10))` |
| HNP hidden widths | `[128, 64]` |
| Polynomial order | 2, element-wise only |

HNP-DQN uses a DDQN bootstrap target, uniform experience replay, Polyak soft target updates, and epsilon-greedy exploration during training.

Evaluation is greedy with exploration disabled.

## Project Structure

```text
├── agents/
│   ├── Base_Agent.py
│   ├── Trainer.py
│   └── DQN_agents/
│       ├── HNP_DQN.py
│       ├── DQN.py
│       ├── DDQN.py
│       ├── DQN_With_Fixed_Q_Targets.py
│       └── Dueling_DDQN.py
│
├── environments/
│   └── RF_spectrum.py
│
├── utilities/
│
├── data/
│
├── datasets/
│
├── exploration_strategies/
│
├── results/
│   ├── Anti_Jam.py
│   ├── models/
│   └── data_and_graphs/
│
└── requirements.txt
```

Some directories, scripts, or pretrained-model folders may retain legacy names originating from earlier development versions. These names should not be interpreted as defining the final experimental protocol.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Main Experiment Script

```bash
cd results
python Anti_Jam.py
```

The repository currently uses configuration options in `Anti_Jam.py` and related configuration classes to control training and evaluation.

Typical run modes include:

```python
RUN_MODE = "train"
```

```python
RUN_MODE = "test"
```

or:

```python
RUN_MODE = "both"
```

### 3. Adjust the Configuration Before Reproduction

Before attempting to reproduce the results reported in the paper, check the configuration used by the current script.

Some repository defaults may correspond to earlier development experiments and therefore may need to be changed manually.

For paper-aligned experiments, ensure that the relevant configuration is consistent with:

```text
Band: 5 GHz
Channels: 5180–5320 MHz, 8 channels
Training episodes: 200
Maximum steps per episode: 100
Independent training seeds: 10
Evaluation trajectories: 20 fixed trajectories
Window size: 32
Window stride: 1
Switching cost: 0.1
```

Also ensure that the correct raw scan files are assigned to the training, validation, pilot, distance-transfer, and power-transfer roles described above.

**The final published manuscript should be treated as the authoritative reference if a repository default differs from the reported experimental protocol.**

## Jammer Settings

The final paper considers two controlled jammer-label schedules.

### Periodic Sweeping Jammer

The jammer advances by one candidate channel per decision step and wraps around after the eighth channel.

### Independent Random Jammer

At each decision step, the jammer channel is selected independently and uniformly from the eight candidate channels.

Some older code or experiment-folder names may use different terminology. Readers should verify the actual jammer implementation rather than relying only on legacy directory names.

## Evaluation Protocol

For each learned method and jammer mode, the final evaluation uses:

- 10 independently trained seeds;
- 20 fixed trajectories for each physical condition and jammer mode;
- identical trajectory seeds and starting conditions across compared methods;
- greedy learned policies during evaluation;
- seed-level statistical inference rather than treating individual episodes or time steps as independent replicates.

The paper also compares HNP-DQN against deterministic rules and an approximately parameter-matched MLP-DDQN control.

These additional evaluation and analysis components may not all be exposed through the original `Anti_Jam.py` entry point and may require the corresponding experiment or analysis scripts.

## Dataset

The experiments use the **RF Jamming Dataset** described in:

> Ali, A. S.; Lunardi, W. T.; Singh, G.; Bariah, L.; Baddeley, M.; Lopez, M. A.; Giacalone, J.-P.; Muhaidat, S.  
> *RF Jamming Dataset: A Wireless Spectral Scan Approach for Malicious Interference Detection.*  
> IEEE Communications Magazine, 2024, 62, 114–120.

The raw measurement files are not originally produced by this repository. Please obtain the dataset from its original source and follow the file-selection protocol reported in the paper.

## Scope

This repository should be interpreted as a research implementation for a controlled **RF measurement-replay spectrum-selection problem**.

Although the work is motivated by cognitive-radar spectrum agility, the experimental hardware is a wireless spectrum-scanning platform rather than radar hardware.

Therefore, the reported experiments do not directly validate:

- radar detection performance;
- target tracking;
- radar waveform adaptation;
- radar-band operation;
- hardware-level anti-jamming performance.

These remain separate validation directions.

## Citation

If you use this repository in academic work, please cite the associated paper.

The final bibliographic information and DOI can be added here after the article is formally published.

```bibtex
@article{HNP_DQN_2026,
  title   = {Evaluation of a Hybrid Neural--Polynomial Deep Q-Network for Switching-Aware Spectrum Selection in a Controlled Radio-Frequency Measurement-Replay Testbed},
  author  = {Pan, Yuxuan and Yan, Ying and Sun, Dingyi and Li, Zhenyu and Zhang, Zhixuan and Cai, Jun and Chen, Dapeng and Wu, Qi and Shen, Zongyuan},
  journal = {Sensors},
  year    = {2026}
}
```

## Reproducibility Notice

The repository is being synchronized with the final published experimental description.

The source code provides the implementation framework, but some legacy defaults and experiment scripts may require manual adjustment to reproduce the exact frozen protocol reported in the paper.

For exact experimental settings, data roles, hyperparameters, evaluation design, and statistical analysis, please refer to the final paper.
