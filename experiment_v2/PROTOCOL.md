# Round-2 confirmatory experiment protocol (v1.1)

This document freezes the evaluation design before inspecting the declared
cross-configuration performance outcomes. Any later change must be recorded in
a dated amendment and cannot be selected because it improves HNP-DQN's
evaluation performance.

Version 1.1 incorporates the outcome-blinded quality-control changes recorded
in `PROTOCOL_AMENDMENT_2026-08-10.md`: a schedule-aware sweep baseline, a true
finite-horizon oracle, and scan-stratified evaluation. The amendment was made
before either declared transfer performance outcome was computed.

## Research question

Does HNP-DQN provide a practically meaningful improvement over (i) a strong
switching-aware deterministic rule and (ii) a capacity-matched conventional
MLP-DDQN under a file-disjoint measurement evaluation?

## Data and split

- Channels: 5180--5320 MHz in 20 MHz increments (eight actions).
- Development condition: chamber, 20 cm, 10 dBm.
- Window: 32 samples, stride 1, generated **within** each raw CSV file.
- Train files: scan IDs 0--6.
- Validation file: scan ID 7.
- Within-condition pilot files: scan IDs 8--9.
- Windows never cross file boundaries. Pilot and cross-configuration files are
  never used for threshold fitting, hyperparameter selection, early stopping,
  or architecture changes.
- A pre-protocol engineering diagnostic already inspected rule-based performance
  on 20 cm / 10 dBm scan IDs 8--9. These files therefore remain a transparent
  within-condition pilot and are not labelled an untouched confirmatory test.
- Two content-untouched cross-configuration evaluations were selected from file
  names before loading their measurements: 40 cm / 10 dBm (distance shift) and
  20 cm / 5 dBm (power shift). All ten scan files in each condition are reserved
  for evaluation only. Code, thresholds, hyperparameters, and claims must be
  frozen before their outcomes are inspected.
- The supported claims are file-disjoint within-condition performance and
  cross-configuration transfer to the two declared distance/power shifts. The
  data are not described as acquisition-session-disjoint unless independent
  session metadata is supplied.

## State, transition, action, and reward

- RF observation: five summaries for each channel: SNR mean, SNR standard
  deviation, RSSI mean, RSSI standard deviation, and noise mean (40 values).
  The development audit found the eight `noise_mean` fields constant at -102;
  after training-fitted standardization they contribute zeros and are retained
  for schema compatibility, not as independent information.
- Switching state: one-hot encoding of the previously selected channel
  (8 values).
- Agent-observation dimension: 48. Adding the previous action makes the
  switching-dependent reward observable; the rolling RF summaries are not
  claimed to be a proven Markov-sufficient statistic of the physical process.
- Action: select one of eight channels.
- RF evolution is exogenous. The action affects the next state through the
  previous-channel component and therefore affects future switching cost.
- Each raw source file was recorded under one fixed jammer label. The replay
  simulator constructs sweeping or random episodes by selecting the source
  file for the current label and advancing a shared ordinal window index.
  Within a source file, the eight frequency records also occur as consecutive
  blocks without a timestamp proving that equal ordinal indices are concurrent.
  Because neither within-file channel synchronization nor cross-file temporal
  alignment is verified, the resulting vectors and transitions are offline
  synthetic measurement-replay constructions, not continuously measured or
  time-synchronised multichannel jammer trajectories.
- Reward: collision `-1`; successful retention `+1`; successful switch
  `1 - C_switch`, with primary `C_switch = 0.1`.
- `reset()` resets the previous channel, selected frequency, collision count,
  and step count. No episode inherits hidden state from the preceding episode.
- The 100-step boundary is a time-limit truncation, not an absorbing terminal
  state. Collection stops, while the stored DDQN transition keeps bootstrapping.

## Methods

Primary learned methods:

1. HNP-DQN with Double-DQN targets, element-wise `[h, h^2]` expansion,
   LayerNorm, projection, and dueling heads.
2. Capacity-matched MLP-DDQN with the same optimizer, replay, update budget,
   target update, exploration schedule, and parameter count within 5%.

Secondary learned analyses are the three component deletions and an otherwise
matched HNP-DDQN with `gamma=0`. No uncompleted conventional DDQN result or cost
row is promised.

Non-learning references:

1. Switching-aware threshold/hysteresis rule fitted on train/validation only.
2. Maximum-quality / minimum-interference rule. The direction of the raw
   dataset `snr` feature is documented because higher values empirically track
   the jammed channel in this dataset.
3. Stay-on-current-channel.
4. Random selection.
5. Schedule-aware sweeping rule, evaluated only for the deterministic sweep.
   It uses the declared starting phase and period but no RF vector or privileged
   runtime label; it is not presented as deployable when the schedule is
   unknown.
6. Jammer-aware greedy reference, allowed to observe the current simulated
   jammer label and clearly marked as privileged but not an upper bound.
7. Clairvoyant finite-horizon oracle, which receives the complete realized
   jammer-label sequence and uses dynamic programming to maximize episode
   reward including switching cost. It is a non-deployable mathematical upper
   bound.

All policies are evaluated on the same fixed test trajectories. Ordinary
state-based heuristics may use only the 48-value state. The schedule-aware rule
and two privileged references are identified separately with their exact
information sets.

## Repetitions and endpoints

- Training seeds: 1--10, subject to available compute. Five seeds are a pilot,
  not the target confirmatory sample.
- For each evaluation-only transfer condition and jammer mode, the formal
  endpoint contains ten declared scan IDs times two fixed replicates per scan,
  for 20 equally weighted trajectories. Derived trajectory seeds are never
  reused as training seeds, and every method receives identical
  `(scan ID, replicate)` cases. The two-scan within-condition pilot uses its own
  explicitly balanced fixed trajectory list and is reported separately.
- Primary endpoint: mean 100-step test return for each independently trained
  seed, averaged over the fixed evaluation trajectories.
- Supporting endpoints: collision count/rate, switch count/rate, training
  time, batch-1 CPU/GPU inference latency, trainable parameters, and serialized
  model size.
- Primary comparisons are fixed by jammer mode before outcome inspection:
  under sweeping, HNP-DQN vs the schedule-aware rule; under random jamming,
  HNP-DQN vs the threshold/hysteresis rule. HNP-DQN vs the capacity-matched
  MLP-DDQN is primary under both modes. The clairvoyant oracle and
  jammer-aware greedy controller are descriptive privileged references and do
  not enter the primary Holm family.
- Other comparisons and component ablations are secondary/exploratory.

## Statistical analysis

- The independent experimental unit is the training seed, not an episode or
  time step.
- Report every seed, mean and standard deviation, paired mean difference,
  95% confidence interval for the paired difference, and paired standardized
  effect size.
- Use a two-sided exact sign-flip/permutation test for paired seed differences
  when computationally feasible.
- Apply Holm correction across the declared comparison family.
- Non-significant findings are reported as inconclusive; they are not converted
  into significance by changing the endpoint, seed subset, or test direction.

## Ablations

Each ablation changes one component only. Hidden widths, optimizer, replay,
seeds, training budget, evaluation trajectories, and all unrelated modules stay
fixed. A capacity-matched no-polynomial control is required to separate the
effect of the explicit expansion from model capacity. A gamma=0 HNP-DDQN run
uses the same architecture and budget as the primary model and tests whether
long-horizon bootstrapping adds value beyond myopic state-conditioned selection.

## Go/no-go rule

After code and validation checks, run a development pilot without consulting
the cross-configuration performance outcomes. If either declared transfer
evaluation later shows that a deterministic rule or the capacity-matched MLP is
materially better than HNP-DQN, report that result and narrow the paper's claim.
Baselines, splits, or seed sets will not be weakened after observing the result.
