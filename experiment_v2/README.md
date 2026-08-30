# Round-2 reproducible anti-jamming experiment

This directory is a clean replacement for the legacy scripts. It implements
scan-file-disjoint preprocessing, a 48-dimensional observation whose previous-
action augmentation makes switching-dependent reward observable,
fixed non-learning baselines, HNP-DQN trained with a Double-DQN target, a
capacity-matched MLP-DDQN, seed-level
statistics, and model-cost reporting. Original source files and old result
folders are not modified or treated as confirmatory evidence.

## Leakage controls

- Development condition: 20 cm / 10 dBm; scans 0--6 train, scan 7 validation,
  scans 8--9 transparent within-condition pilot.
- The normalizer is mechanically restricted to the split named `train`. Only
  the first 40 RF features are z-scored; the 8-value previous-action one-hot
  block is preserved exactly.
- `smoke` loads only train and validation content for execution and does not
  load pilot or cross-configuration outcomes.
- `formal` trains and checkpoints every selected seed first, writes
  the forced `PREDECLARED_EVALUATION_SCHEDULE.json` and
  `FROZEN_BEFORE_EVALUATION.json`, and only then computes policy-performance
  outcomes on 40 cm / 10 dBm and 20 cm / 5 dBm. The freeze record contains the
  schedule hash, checkpoint hashes, and core-code hashes. Prior file-format/
  finiteness auditing is allowed and is not misrepresented as outcome blindness.
- Threshold/hysteresis is fitted once on train+validation. Test/OOD fitting is
  rejected by the baseline implementation.
- The twenty formal trajectory slots use forced scan IDs and recorded start
  indices. Every ten-scan transfer condition contributes exactly two
  trajectories per scan; the two-scan transparent pilot contributes ten per
  scan. Every method and training seed reuses the identical trajectory seeds.

## Reference policies and evaluation semantics

- `schedule_sweep` uses only the public sweeping phase and step count. It is
  defined only for `sweeping`; over 100 steps it has zero collisions, fourteen
  switches, and the deterministic return 98.6.
- `jammer_greedy` is the former legacy “Oracle”: it sees only the current true
  jammer and is now accurately labelled a jammer-aware greedy reference.
- `clairvoyant_oracle` receives the complete action-independent jammer sequence
  and solves a finite-horizon dynamic program. It is a descriptive reward upper
  bound, not an ordinary deployable baseline.
- Primary comparisons are predeclared by jammer: HNP versus `schedule_sweep`
  under sweeping, HNP versus threshold/hysteresis under random jamming, and HNP
  versus matched MLP under both. Privileged references do not enter the primary
  Holm family.
- Episode/data horizons are truncations rather than absorbing terminal states.
  Collection stops at the limit, but the DDQN target retains its bootstrap term.

## Verified environment

Python 3.10, NumPy 2.2, pandas 2.3, SciPy 1.15, and PyTorch 2.11.0+cu130 were
used locally. On an RTX 5090, use an official CUDA 12.8-or-newer PyTorch wheel.

The final delivery intentionally excludes the large raw measurement CSV files.
It is therefore not a data-self-contained package: restore `data/raw` from the
original `jam_shield` ZIP and verify the restored files against
`results/data_audit/source_file_manifest.csv` before rerunning preprocessing or
training. The accompanying `data_audit.json` records the format/finiteness audit.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Commands

Minimal end-to-end engineering check (validation only):

```powershell
.\.venv\Scripts\python.exe run_experiment.py smoke --model hnp --train-seeds 1 --eval-seeds 1001 1002 --jammer sweeping
```

Frozen formal protocol (both learned methods, ten training seeds, twenty fixed
evaluation seeds, sweeping and random jammer):

```powershell
.\.venv\Scripts\python.exe run_experiment.py formal --model hnp matched --train-seeds 1 2 3 4 5 6 7 8 9 10 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 --jammer sweeping random
```

The formal default remains exactly `hnp matched`. The three single-component
ablations are secondary, opt-in runs and do not enter the primary Holm family.
Each may be run independently with the same formal seed/budget controls, for
example:

```powershell
.\.venv\Scripts\python.exe run_experiment.py formal --model no_polynomial --train-seeds 1 2 3 4 5 6 7 8 9 10 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 --jammer sweeping random
.\.venv\Scripts\python.exe run_experiment.py formal --model no_layernorm --train-seeds 1 2 3 4 5 6 7 8 9 10 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 --jammer sweeping random
.\.venv\Scripts\python.exe run_experiment.py formal --model no_dueling --train-seeds 1 2 3 4 5 6 7 8 9 10 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 --jammer sweeping random
```

An ablation-only run deliberately emits an empty `primary_comparisons` table;
its seed-level values remain in `seed_summary.csv` for the separately labelled
exploratory ablation analysis.

The formal defaults are 200 episodes x 100 steps, gamma 0.95, learning rate
0.001, Adam weight decay 0.0001, replay capacity 100000, batch size 64, one
update per environment step after 64 stored samples, target soft-update
coefficient 0.005, gradient-norm clipping at 0.7, and
`epsilon_e = max(0.01, 1/(1+e/10))`. HNP and matched MLP share every training
setting. `--gamma 0` runs the predeclared myopic control without changing the
architecture or budget.

## Validated isolated model-cost profile

The isolated model-cost profile was completed after the formal jobs ended and
the system was otherwise idle. Its validated outputs are supplied with the
delivery copy under `04_正式结果与审计/analysis/isolated_cost_profile_v1`.
The measurements must not be replaced by timings recorded while several jobs
share the CPU/GPU. To reproduce the profile in a fresh output directory, run:

```powershell
.\.venv\Scripts\python.exe profile_model_costs.py `
  --checkpoint-root ..\..\04_正式结果与审计\formal_artifacts\primary `
  --output-dir results\isolated_cost_profile_reproduced `
  --acknowledge-idle-system
```

The supplied frozen primary and ablation artifacts were moved out of this code
directory to avoid duplicating the 120 checkpoints. From this README's working
directory they are available under
`..\..\04_正式结果与审计\formal_artifacts\` as `primary`, `gamma0`,
`no_polynomial`, `no_layernorm`, and `no_dueling`.

The profiler mechanically refuses to start without the idle-system
acknowledgement or with a missing/duplicate/noncanonical primary checkpoint.
It performs no policy evaluation and constructs only development train scans
0--6 plus validation scan 7; pilot scans 8--9 and both transfer configurations
are never constructed. HNP and matched MLP are trained sequentially on one CPU
thread with the complete 200x100 budget and the same dedicated engineering
seeds `91001, 91002, 91003` in a fixed counterbalanced order. These profiling
seeds measure computational cost only and are not units for policy-performance
inference.

It then processes all canonical primary checkpoints sequentially, using 200
warm-up forwards and 1000 recorded batch-1 forwards per CPU/GPU measurement.
The outputs are `isolated_model_costs.csv` and `isolated_model_costs.json` in
the analysis directory. Exact parameter, registered-buffer, and combined
persistent-tensor bytes exclude activations, optimizer state, framework/runtime
objects, temporary workspaces, and allocator overhead. Serialized state and
checkpoint bytes are explicitly labelled storage sizes, not memory.

## Outputs

Each run gets a new directory and refuses to overwrite a non-empty directory.
Important files are:

- `config.json`, `software.json`, `normalizer.json`: complete audit metadata.
- `FROZEN_BEFORE_EVALUATION.json`: configuration/checkpoint hashes and the
  fixed train+validation threshold, plus the predeclared-schedule and core-code
  hashes, written before formal OOD policy outcomes.
- `PREDECLARED_EVALUATION_SCHEDULE.json`: immutable condition, jammer mode,
  evaluation index/seed, forced scan, per-scan replicate, start index, and
  trajectory seed, materialized before the first evaluation-policy step.
- `evaluation_schedule.csv`: tabular copy of the same fixed schedule.
- `training_history.csv`: every training episode and its TD MSE.
- `evaluation_episodes.csv`: each policy x evaluation-seed return, collision,
  and switching result, including scan ID, start index, and trajectory seed.
- `seed_summary.csv`: learned results averaged over fixed evaluation seeds for
  each independent training seed; baselines are reported separately.
- `primary_comparisons.csv/json`: paired mean difference, Student-t 95% CI,
  Cohen's dz, exact two-sided sign-flip p-value, and Holm-adjusted p-value.
- `model_costs.csv`: trainable parameters, exact parameter bytes, registered
  buffer bytes, their sum as inference-persistent tensor bytes, serialized
  state/checkpoint storage sizes, training time, and median batch-1 CPU/GPU
  inference latency. Persistent tensor bytes exclude activations, framework/
  runtime objects, allocator reservations, optimizer state, and workspaces;
  checkpoint/serialized byte counts are storage sizes and are not memory claims.

Do not select seed subsets, baselines, thresholds, or hyperparameters after
reading pilot/OOD results. Mixed or non-significant outcomes remain reportable
results and require narrowing the manuscript claim rather than altering the
comparison.
