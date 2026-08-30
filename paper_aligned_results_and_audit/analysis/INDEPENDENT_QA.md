# Independent pre-outcome QA record

Date: 2026-08-10  
Scope: static review, unit/integration tests, and development-validation smoke
only. No formal pilot or cross-configuration policy outcome was opened during
this audit.

## Go decision

The amended runner received a **GO** for new formal runs after 44 tests passed.
The audit verified:

- one BatchNorm running-statistics update per learning step;
- time limits are truncations and retain DDQN bootstrapping;
- ten-scan transfer conditions contain two fixed trajectories per scan;
- every method receives the same forced scan, start index, trajectory seed, and
  jammer realization;
- the schedule-aware rule appears only under deterministic sweeping;
- the finite-horizon DP oracle uses the exact environment reward and is clearly
  labelled clairvoyant;
- the evaluation schedule is written and hashed before the first evaluation
  policy step; and
- the primary family contains 12 predeclared rows: HNP versus matched under
  both jammer modes, HNP versus schedule-aware under sweeping, and HNP versus
  threshold/hysteresis under random jamming, for all three physical conditions.

## Limitations that must remain in the paper

1. The raw frequency blocks and fixed-jammer files lack verified synchronization
   timestamps. The benchmark uses reproducible ordinal-index replay states and
   transitions; it is not a continuously measured moving-jammer trace.
2. Statistical inference uses ten independent training seeds and is conditional
   on 20 pre-fixed evaluation trajectories per physical condition and jammer
   mode. It does not estimate new-session or new-site population variability.
3. The matched MLP controls total parameter count and dueling-head type, but is
   not a one-factor isolation of the squared branch because widths and the
   LayerNorm path differ. Interpretation must be triangulated with the narrow
   no-polynomial and no-LayerNorm ablations.
4. The eight development `noise_mean` entries are constant at -102 and become
   zero after standardization; the field is retained for schema compatibility,
   not claimed as independent information.

## Canonical primary-run acceptance checks

- 40 checkpoints;
- 120 predeclared schedule rows;
- 3,180 evaluation-episode rows;
- 159 seed-summary rows;
- 12 primary comparisons, each with `n_pairs=10`;
- 40 model-cost rows;
- two trajectories per scan for every transfer condition and jammer mode;
- every 100-step episode has `terminated=false` and `truncated=true`; and
- schedule, checkpoint, configuration, and core-code SHA-256 values verify.
