# Figure captions

## Main performance

**Main performance under the frozen scan-stratified protocol.** Columns show
the transparent 20 cm/10 dBm pilot, the 40 cm/10 dBm distance shift, and the
20 cm/5 dBm power shift; rows show sweeping and random jammer modes. For
HNP-DQN and the capacity-matched MLP-DDQN, faint symbols are the 10 independent
training-seed estimates after averaging 20 fixed trajectories per seed, and
the larger symbol/error bar is the seed mean with a two-sided 95% Student-t
confidence interval. Ordinary deterministic references are shown only as
their means over the same 20 fixed trajectories and are not assigned a
training-seed confidence interval. The schedule-aware rule appears only under
sweeping. Hollow diamond/star symbols identify references with current-label
or full-sequence privileged jammer information; the latter is the
clairvoyant finite-horizon dynamic-programming upper bound. Pilot scans 8 and
9 occur ten times each; every transfer scan 0--9 occurs twice. Colors use the
Okabe-Ito palette, with redundant marker shapes for grayscale reproduction.

Source: `main_performance_source_data.csv`.

## Primary comparison forest

**Predeclared primary paired comparisons.** Points show the mean difference in
100-step return, HNP-DQN minus comparator, across 10 paired training seeds;
horizontal bars are two-sided 95% Student-t confidence intervals for the seed
differences. The simple comparator is the schedule-aware rule under sweeping
and threshold/hysteresis under random jamming; the capacity-matched MLP-DDQN
comparison is shown under both modes. The dashed line denotes no difference.
Numeric labels are Holm-adjusted p-values for the single 12-comparison primary
family; asterisks are not used. The pilot is transparently distinguished from
the two cross-configuration evaluations.

Source: `primary_forest_source_data.csv`.

## Exploratory ablation forest

**Exploratory paired ablation comparisons.** Points show the mean difference
in 100-step return, full HNP-DQN minus variant, across 10 paired training
seeds; horizontal bars are two-sided 95% Student-t confidence intervals. The
dashed line denotes no difference. Numeric labels are Holm-adjusted p-values
for the separate 24-comparison exploratory family spanning four variants,
three physical conditions, and two jammer modes. This family is not mixed
with the primary HNP-versus-simple-rule/capacity-matched family, and
nonsignificant intervals are interpreted as inconclusive rather than as
equivalence.

Source: `ablation_forest_source_data.csv`.
