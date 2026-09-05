# HNP-DQN: Hybrid Neural-Polynomial Deep Q-Network

This repository accompanies the study **“Evaluation of a Hybrid Neural-Polynomial Deep Q-Network for Switching-Aware Spectrum Selection in a Controlled Radio-Frequency Measurement-Replay Testbed.”** It evaluates HNP-DQN as a structured value-function approximator for an eight-channel, switching-aware spectrum-selection problem built from RF spectral scans.

> **Important — use the final-paper-aligned workflow.**
>
> - [`experiment_v2/`](experiment_v2/) is the primary implementation of the frozen experimental protocol reported in the final paper.
> - [`paper_aligned_results_and_audit/`](paper_aligned_results_and_audit/) is the primary location for the corresponding checkpoints, evaluation outputs, statistics, figures, manifests, and audit materials.
> - The other top-level experiment code and result folders are retained as legacy development implementation and supporting context. They may contain earlier defaults that differ from the final protocol and should not be used as the authoritative reproduction entry point.

## Paper and citation

The associated manuscript is published in *Sensors*:

> Yuxuan Pan, Ying Yan, Dingyi Sun, Zhenyu Li, Zhixuan Zhang, Jun Cai, Dapeng Chen, Qi Wu, and Zongyuan Shen. “Evaluation of a Hybrid Neural–Polynomial Deep Q-Network for Switching-Aware Spectrum Selection in a Controlled Radio-Frequency Measurement-Replay Testbed.” *Sensors* 2026, *26*(17), 5501. https://doi.org/10.3390/s26175501

You can cite this work using the following formats:

**MDPI and ACS Style**

Pan, Y.; Yan, Y.; Sun, D.; Li, Z.; Zhang, Z.; Cai, J.; Chen, D.; Wu, Q.; Shen, Z. Evaluation of a Hybrid Neural–Polynomial Deep Q-Network for Switching-Aware Spectrum Selection in a Controlled Radio-Frequency Measurement-Replay Testbed. *Sensors* **2026**, *26*, 5501. https://doi.org/10.3390/s26175501

**AMA Style**

Pan Y, Yan Y, Sun D, Li Z, Zhang Z, Cai J, Chen D, Wu Q, Shen Z. Evaluation of a Hybrid Neural–Polynomial Deep Q-Network for Switching-Aware Spectrum Selection in a Controlled Radio-Frequency Measurement-Replay Testbed. *Sensors*. 2026;26(17):5501. https://doi.org/10.3390/s26175501

**Chicago/Turabian Style**

Pan, Yuxuan, Ying Yan, Dingyi Sun, Zhenyu Li, Zhixuan Zhang, Jun Cai, Dapeng Chen, Qi Wu, and Zongyuan Shen. 2026. "Evaluation of a Hybrid Neural–Polynomial Deep Q-Network for Switching-Aware Spectrum Selection in a Controlled Radio-Frequency Measurement-Replay Testbed" *Sensors* 26, no. 17: 5501. https://doi.org/10.3390/s26175501

**APA Style**

Pan, Y., Yan, Y., Sun, D., Li, Z., Zhang, Z., Cai, J., Chen, D., Wu, Q., & Shen, Z. (2026). Evaluation of a Hybrid Neural–Polynomial Deep Q-Network for Switching-Aware Spectrum Selection in a Controlled Radio-Frequency Measurement-Replay Testbed. *Sensors*, *26*(17), 5501. https://doi.org/10.3390/s26175501


## Repository structure

```text
HNPDQN/
├── experiment_v2/                         # final-paper-aligned experiment code
│   ├── run_experiment.py                   # smoke/formal training and evaluation CLI
│   ├── audit_data.py                       # source-data and split audit
│   ├── profile_model_costs.py              # isolated model-cost profiler
│   ├── PROTOCOL.md                         # frozen protocol
│   ├── PROTOCOL_AMENDMENT_2026-08-10.md    # pre-outcome protocol amendment
│   ├── src/                                # config, data, environment, models, runner, statistics
│   ├── tests/                              # protocol and implementation tests
│   └── results/data_audit/                 # audited file manifest and data summary
├── paper_aligned_results_and_audit/        # final results and reproducibility evidence
│   ├── formal_artifacts/
│   │   ├── primary/                        # HNP-DQN and matched MLP-DDQN
│   │   ├── gamma0/                         # myopic gamma=0 control
│   │   ├── no_polynomial/                  # HNP component ablation
│   │   ├── no_layernorm/                   # HNP component ablation
│   │   └── no_dueling/                     # HNP component ablation
│   ├── analysis/                           # summaries, audits, figure data, figures, scripts
│   └── MANUAL_TEXT_AUDIT.csv
├── datasets/                               # source corpus and legacy processed arrays
│   └── raw/spectral_scans_QC9880_ht20_background/
├── agents/                                 # legacy development agents
├── environments/                           # legacy environment and exploratory visualizations
├── results/                                # legacy entry script and development checkpoints
├── data/, exploration_strategies/, utilities/
├── figures/, Jammer_Visual/                # legacy/manuscript-supporting figures
├── requirements.txt                        # legacy dependency set
└── README.md
```

The two paper-aligned directories are self-contained with respect to code and reported artifacts, but the experiment runner reads the raw CSVs from a user-supplied data directory. In this checkout, the required files are under the `datasets/` path shown above.

## Final paper experimental protocol

| Setting | Frozen value |
|---|---|
| Band | 5 GHz |
| Candidate channel centers | 5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320 MHz |
| Channel spacing | 20 MHz |
| Window length / stride | 32 samples / 1 |
| RF features | 5 summaries × 8 channels = 40 |
| Switching-state encoding | 8-dimensional previous-channel one-hot vector |
| Agent observation | 48 dimensions |
| Episode horizon | Maximum 100 steps |
| Training budget | 200 episodes |
| Switching cost | 0.1 |
| Learned repetitions | 10 independent training seeds per method and jammer mode |
| Evaluation endpoint | 20 fixed trajectories per condition and jammer mode |
| Jammer modes | Periodic sweeping and independent random |

Each channel contributes `snr_mean`, `snr_std`, `rssi_mean`, `rssi_std`, and `noise_mean`. The first 40 values are standardized using training scans only; the previous-action one-hot block is preserved. The dataset-provided `snr` field is used as supplied and is not a separately calibrated physical SNR/SINR measurement.

The primary HNP network is:

```text
48 → Linear(128) + BatchNorm + ReLU
   → Linear(64)  + BatchNorm + ReLU
   → element-wise [h, h²] (128 values)
   → LayerNorm → Linear(128) + ReLU
   → dueling value/advantage heads → 8 Q-values
```

The expansion has no constant term and no pairwise cross-products. HNP-DQN has 32,841 trainable parameters. The primary control is a 32,691-parameter, capacity-matched MLP-DDQN. Both use Double-DQN targets, uniform replay, Adam, mean-squared TD error, Polyak target updates, and the same training budget.

Key learning values are `gamma=0.95`, learning rate `0.001`, weight decay `1e-4`, replay capacity `100000`, batch size `64`, learning start at 64 stored transitions, one update per environment step, `tau=0.005`, gradient-norm clipping at `0.7`, and `epsilon_e = max(0.01, 1 / (1 + e / 10))`. Evaluation is greedy.

### Data split

Raw scan files are assigned before any overlapping window construction; windows never cross source-file boundaries.

| Physical condition | Scan IDs | Role |
|---|---:|---|
| 20 cm / 10 dBm | 0–6 | Training |
| 20 cm / 10 dBm | 7 | Validation |
| 20 cm / 10 dBm | 8–9 | Transparent, pre-inspected within-condition pilot |
| 40 cm / 10 dBm | 0–9 | Evaluation-only distance transfer |
| 20 cm / 5 dBm | 0–9 | Evaluation-only power transfer |

Each ten-scan transfer condition contributes two fixed trajectories per scan. The two-scan pilot contributes ten trajectories per scan. All compared methods reuse the same forced scan, start index, trajectory seed, and jammer realization. Statistical inference treats the ten independent training seeds—not episodes or time steps—as the experimental units.

## Reproducing the final experiments

Run these commands from `experiment_v2/`. Its dependency file, not the legacy root `requirements.txt`, defines the paper-aligned environment.

```powershell
cd experiment_v2
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The runner defaults to `experiment_v2/data/raw`, which is not populated in this repository layout. The checked-in source corpus contains all 240 required final-protocol CSVs under `../datasets/raw/spectral_scans_QC9880_ht20_background`, so pass that path explicitly.

First run the validation-only engineering check. Smoke mode does not load the pilot or cross-configuration outcomes:

```powershell
.\.venv\Scripts\python.exe run_experiment.py smoke `
  --data-dir ..\datasets\raw\spectral_scans_QC9880_ht20_background `
  --model hnp --train-seeds 1 --eval-seeds 1001 1002 --jammer sweeping
```

Then run the frozen primary protocol. The `formal` preset supplies `hnp matched`, training seeds 1–10, evaluation seeds 1001–1020, both jammer modes, 200 episodes, and the remaining frozen hyperparameters:

```powershell
.\.venv\Scripts\python.exe run_experiment.py formal `
  --data-dir ..\datasets\raw\spectral_scans_QC9880_ht20_background `
  --device cpu
```

Each invocation creates a timestamped directory under `experiment_v2/results/` and refuses to overwrite a non-empty output directory. Formal training checkpoints every selected model first, materializes and hashes the evaluation schedule, writes `FROZEN_BEFORE_EVALUATION.json`, and only then computes policy-performance outcomes.

Optional secondary runs use the same formal seed and budget controls:

```powershell
# Run one command at a time.
.\.venv\Scripts\python.exe run_experiment.py formal --data-dir ..\datasets\raw\spectral_scans_QC9880_ht20_background --device cpu --model no_polynomial
.\.venv\Scripts\python.exe run_experiment.py formal --data-dir ..\datasets\raw\spectral_scans_QC9880_ht20_background --device cpu --model no_layernorm
.\.venv\Scripts\python.exe run_experiment.py formal --data-dir ..\datasets\raw\spectral_scans_QC9880_ht20_background --device cpu --model no_dueling
.\.venv\Scripts\python.exe run_experiment.py formal --data-dir ..\datasets\raw\spectral_scans_QC9880_ht20_background --device cpu --model hnp --gamma 0
```

To regenerate the data audit in a separate directory:

```powershell
.\.venv\Scripts\python.exe audit_data.py `
  --raw-dir ..\datasets\raw\spectral_scans_QC9880_ht20_background `
  --output-dir results\data_audit_reproduced
```

The canonical audit is [`experiment_v2/results/data_audit/data_audit.json`](experiment_v2/results/data_audit/data_audit.json), with per-file hashes in [`source_file_manifest.csv`](experiment_v2/results/data_audit/source_file_manifest.csv). The manifest hashes use LF line endings; a Windows checkout that converts the CSVs to CRLF changes their raw byte hashes, although all 240 files in this checkout match after LF normalization. For protocol details and the pre-outcome design correction, read [`PROTOCOL.md`](experiment_v2/PROTOCOL.md) and [`PROTOCOL_AMENDMENT_2026-08-10.md`](experiment_v2/PROTOCOL_AMENDMENT_2026-08-10.md).

## Experiment code (`experiment_v2`)

| File | Role |
|---|---|
| `run_experiment.py` | Supported `smoke` and `formal` command-line entry point |
| `src/config.py` | Frozen presets, seeds, hyperparameters, supported primary models and ablations |
| `src/data.py` | File discovery, scan-disjoint splitting, within-file windowing, 40-feature construction, OOD loaders |
| `src/env.py` | 48-dimensional measurement-replay environment, switching reward, sweeping/random schedules |
| `src/models.py` | HNP-DQN, matched MLP-DDQN, no-polynomial, no-LayerNorm, and no-dueling networks |
| `src/agent.py` | Training-only normalization, replay, Double-DQN update, soft target update, checkpointing |
| `src/baselines.py` | Random, stay, max-quality/min-interference, threshold/hysteresis, schedule-aware sweep, and privileged references |
| `src/runner.py` | Training, fixed scan-balanced evaluation, freeze records, output generation, model-cost reporting |
| `src/statistics.py` | Paired seed-level intervals, effect sizes, exact sign-flip tests, Holm adjustment |
| `tests/` | Data/environment, model, runner, model-cost, and polynomial-branch audit tests |

The schedule-aware rule is evaluated only for deterministic sweeping. `jammer_greedy` observes the current jammer label and is a privileged descriptive reference; `clairvoyant_oracle` sees the full jammer sequence and supplies a finite-horizon dynamic-programming upper bound. Neither is an ordinary deployable baseline.

## Results and audit (`paper_aligned_results_and_audit`)

`formal_artifacts/primary/` is the canonical primary run. It contains 40 checkpoints (HNP and matched MLP × two jammer modes × ten seeds), 120 predeclared schedule rows, 3,180 episode-level evaluation rows, 159 seed-summary rows, 12 primary comparisons, 40 model-cost rows, and 8,000 training-history rows.

Every formal artifact directory follows the same conceptual layout:

- `config.json`, `software.json`, and `normalizer.json` record the run environment and training-fitted preprocessing.
- `PREDECLARED_EVALUATION_SCHEDULE.json` and `evaluation_schedule.csv` preserve the fixed scan/start/seed schedule.
- `FROZEN_BEFORE_EVALUATION.json` records configuration, checkpoint, schedule, and core-code hashes before policy evaluation.
- `checkpoints/`, `training_history.csv`, and `model_costs.csv` contain learned states, episode histories, and cost measurements.
- `evaluation_episodes.csv` and `seed_summary.csv` provide trajectory-level and independent-seed-level outcomes.
- `primary_comparisons.csv/json` contain the predeclared paired comparisons where applicable. Ablation-only runs intentionally leave this primary comparison table empty.

The remaining formal directories are secondary analyses: `gamma0/` keeps the HNP architecture with `gamma=0`; `no_polynomial/`, `no_layernorm/`, and `no_dueling/` each remove one named component while retaining the formal seeds, jammer modes, and training/evaluation budget.

The `analysis/` directory provides a validated cross-run `result_manifest.json`, exploratory `ablation_comparisons.csv`, independent QA and source-manifest notes, a descriptive checkpoint-weight audit under `branch_weight_audit_v1/`, sequential idle-system cost measurements under `isolated_cost_profile_v1/`, generation/summarization scripts, and both rendered figures and their source CSV data. `formal_figures/` contains the primary and ablation performance plots; `figures/` contains the architecture and training-SNR-direction figures.

## Legacy development code

The root `agents/`, `environments/`, `data/`, `exploration_strategies/`, `utilities/`, `results/`, and `run_experiments.sh` are retained for historical/development context. They document earlier implementations and exploratory workflows, but they are not the frozen final-paper pipeline.

For example, the current legacy `results/Anti_Jam.py` is configured for an 11-channel 2.4 GHz setting, 100 training episodes, and three runs; the legacy environment exposes 40 RF features without the final previous-channel observation block. The legacy shell script also describes earlier band and switching-cost experiment combinations. These files are related to the development of HNP-DQN, not errors to be discarded, but their defaults should not be substituted for `experiment_v2/src/config.py` or the formal artifact `config.json` files.

Likewise, the root `requirements.txt` pins the historical stack. Use `experiment_v2/requirements.txt` for the paper-aligned workflow. Legacy checkpoints and plots under `results/`, `figures/`, and `Jammer_Visual/` are not confirmatory evidence for the final protocol.

## Dataset

The experiments use the **RF Jamming Dataset**:

> A. S. Ali, W. T. Lunardi, G. Singh, L. Bariah, M. Baddeley, M. A. Lopez, J.-P. Giacalone, and S. Muhaidat, “RF Jamming Dataset: A Wireless Spectral Scan Approach for Malicious Interference Detection,” *IEEE Communications Magazine*, vol. 62, pp. 114–120, 2024.

The larger source corpus is retained under `datasets/raw/spectral_scans_QC9880_ht20_background/`. The final protocol uses 240 files: eight jammer-center frequencies × ten scans × three physical conditions (20 cm/10 dBm, 40 cm/10 dBm, and 20 cm/5 dBm). Verify any restored or relocated copy against the paper-aligned source-file manifest before training.

## Scope and limitations

This is a controlled RF measurement-replay spectrum-decision benchmark, not an end-to-end radar system. The raw files do not provide verified timestamps that synchronize the eight frequency blocks within a file or align files recorded under different fixed jammer labels. The simulator therefore constructs reproducible ordinal-index replay states and synthetic sweeping/random transitions; it does not establish continuously measured multichannel jammer dynamics or acquisition-session transfer.

The eight `noise_mean` fields are retained for schema compatibility but are constant at −102 in the audited development data and become zero after standardization. The ten-seed inference is conditional on the 20 pre-fixed trajectories per condition and jammer mode; it does not estimate new-site or new-session population variability. The matched MLP controls parameter count and dueling-head type, but it is not a one-factor isolation of the polynomial branch, so interpretation should be considered together with the component ablations.

The reported experiments do not directly validate radar detection, target tracking, waveform adaptation, radar-band operation, or hardware-level anti-jamming performance.
