# Protocol amendment before confirmatory outcome inspection

**Date:** 2026-08-10 (Asia/Shanghai)  
**Status:** adopted before any policy-performance outcome was computed on the
40 cm / 10 dBm or 20 cm / 5 dBm evaluation conditions.

## Reason for the amendment

Independent static review of the frozen simulator and evaluation runner found
two design problems without consulting either declared transfer outcome.

1. The periodic sweeping schedule has a declared initial phase and advances by
   one channel per step. A deterministic schedule-aware controller can exploit
   this information and is stronger than a controller that only reacts to the
   current RF vector. Omitting it would understate the strength of simple
   non-learning alternatives. In addition, the previously named `Oracle`
   observed only the current jammer label; it was a privileged greedy reference,
   not a mathematical upper bound over the full episode.
2. Drawing 20 episodes by randomly sampling among ten evaluation scan IDs did
   not guarantee that every scan was represented and produced unequal scan
   weights. This conflicted with the intended claim that the reported endpoint
   covered all ten evaluation scans.

## Adopted changes

- Add a **schedule-aware sweeping heuristic** that uses only the declared sweep
  phase, period, and its own action history. It is evaluated only under the
  deterministic sweeping condition and is labelled non-deployable when the
  phase or schedule is unknown.
- Relabel the former current-label controller as the **jammer-aware greedy
  reference**. It may read the current simulated jammer label and is therefore
  privileged, but it is not called an upper bound.
- Add a **clairvoyant dynamic-programming oracle**. It receives the complete
  realized jammer-label sequence for the episode and maximizes the stated
  finite-horizon reward, including the switching penalty. It is an
  information-privileged, non-deployable mathematical upper bound.
- Replace random scan selection for the formal endpoint with a fixed stratified
  design: each of the ten evaluation scan IDs contributes two predeclared
  trajectory replicates per jammer mode and physical condition. All methods
  use exactly the same `(scan ID, replicate)` trajectories. A trained model's
  endpoint remains the mean over these 20 fixed trajectories; the independent
  inferential unit remains the training seed.
- Add the selected scan ID to every episode-level output row and test that every
  formal condition contains exactly two trajectories from each scan.
- Freeze the simple-rule primary comparator by jammer mode rather than choosing
  it after seeing results: HNP-DQN versus the schedule-aware rule for sweeping,
  and HNP-DQN versus threshold/hysteresis for random jamming. HNP-DQN versus
  the capacity-matched MLP remains primary in both modes; privileged references
  remain descriptive and outside the Holm family.

## Outcome blinding and superseded runs

The following directories were stopped before evaluation outcomes were
produced and are permanently excluded from analysis:

- `results/formal_primary_v1`
- `results/formal_primary_v2`
- `results/formal_gamma0_v1`
- `results/formal_no_polynomial_v1`

Each directory contains an `ABORTED_RUN.md` marker. No checkpoint or partial
training artifact from these runs may be selected or reused. Formal runs after
this amendment use new versioned output directories and the amended code and
tests.

## Scope that this amendment does not repair

The source CSV files do not provide verified timestamps aligning the eight
frequency blocks within a file, nor timestamps aligning files recorded under
different fixed jammer labels. Consequently, grouping observations by ordinal
index and splicing jammer-labelled files creates offline synthetic replay
states and transitions. File-disjoint evaluation prevents same-file window
leakage, but it does not establish synchronized multichannel dynamics, a
continuously measured sweeping jammer, or real-world deployment transfer.
