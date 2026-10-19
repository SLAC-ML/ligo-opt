# Run 4 surrogate diagnostics

Run 4 improves substantially over run 3 on matched initial validation designs and
finds better Finesse-verified PSO designs. Its best saved full cost is
**0.0102834 at round 10**,
39.3% lower than run 3's best.
Signed-difference MAE on the same valid initial validation designs falls by
**53.2%**. However, the model still misses symmetric
response shape, often understates sensitivity, and does not improve difference
accuracy consistently with each feedback update. Low difference MAE alone is
misleading when the true changes are very small.

This analysis uses saved Finesse labels, production checkpoints and captured online
evaluations. Separate local experiments use diagnostic checkpoints. The notebook
can be refreshed from these artifacts without training or optical power solves.

## Data, features, and the actual intervention

The full run 4 audit contains **140 groups / 700 rows**:
500 initial rows and 200 feedback rows. There are 118 valid fixed-q groups.
The initial data has 78 valid groups and 22 invalid groups, 104 zero-power rows,
and 230 rows in the saved 282.09 ± 0.01 W center band. All 40 feedback groups are
valid fixed-q groups. Zero labels alone cannot identify instability or solver failure.

Run 3 and run 4 have identical parsed initial features and labels, initial design
splits (80 training / 20 validation), normalization, architecture, and seed 9302.
Both use 1,000 base epochs, batch size 32, 20 feedback epochs, and automatic newest-row
weight 100. Matching these settings makes this comparison more controlled than run 2
versus run 3. Both manifests record a dirty working tree; byte-identical initial
weights and execution determinism were not independently recorded.

Run 4 uses production **`nominal_delta`** supervision. It directly supervises nominal
power and signed partner-minus-nominal differences. This also changes which rows
receive the direct power criterion: partner rows reference their nominal design,
rather than retaining the old direct power penalty on every partner label. It is a
combined objective change, rather than simply adding one term to the old row loss.
The fixed training-derived difference scale is approximately **8.9252 W**, reused
for feedback. Sub-0.1 W responses therefore have small quadratic loss contributions.
Checkpoint selection still uses all-row validation **power loss**, without an explicit
signed-difference selection term. Invalid-q training transitions remain in production
training and in the scale calculation.
Invalid-q partners contribute 27.34% of squared initial training differences;
valid-only RMS is still 8.57323 W, so removing invalid groups alone would change
this scale only modestly.

All 25 selected feature reconstructions from rounds
1, 10, 20, 30, and 40 passed the saved `rtol=2e-4, atol=0.05` tolerances. These checks
use beam tracing and support feature plumbing; they do not validate optical labels.

Evidence: [full data audit](outputs/run4_audit/audit.json),
[feature audit](outputs/run4_selected_feature_audit/audit.json),
[training audit and provenance](outputs/run4_training_audit/training_audit.json).

## Matched initial validation improves, but remains imperfect

Both base models are evaluated on exactly the same **15 valid held-out designs**
(75 rows / 60 directional responses). These validation designs were excluded from
weight updates but used for production checkpoint selection; this is not an independent test.
The five invalid-q validation designs are excluded here and retained in the separate
all-row production metrics. Only 10 directional labels exceed the 0.1 W sign threshold.

| Base checkpoint | Power MAE (W) | ΔPower MAE (W) | Perturbation-cost MAE (W/m) | Significant signs correct |
| --- | ---: | ---: | ---: | ---: |
| Run 3 base | 7.12385 | 0.642019 | 0.23489 | 2/10 |
| Run 4 base | 6.25405 | 0.30074 | 0.110748 | 6/10 |

The error magnitudes and significant signs improve. The symmetric response problem
remains: on these 15 designs, Finesse decreases in both ITM directions for 11 designs
and both ETM directions for 13; the run 4 base model predicts **zero** both-decrease
pairs for either mirror. Most initial true differences are tiny: only 52 of the 312
valid initial directional labels exceed 0.1 W. The baseline comparison below helps
distinguish a small error average from learning those changes.

Production selected run 4 epoch 780 using the 20-design all-row validation power
loss. Its recorded all-row power MAE is 4.74142 W and difference MAE 0.225926 W;
those values use a different cohort from the valid-only table above. Loss values
and selection criteria should not be compared as if the objectives were unchanged.

## Accuracy on the same 40 run 4 feedback designs

All checkpoints below see the same saved geometries and labels. The final run 4
checkpoint has trained on all these feedback labels, so its result measures replay
reconstruction. Earlier models did not train on these labels, but the geometries
were selected adaptively by run 4's search.

| Checkpoint | Power MAE (W) | ΔPower MAE (W) | Perturbation-cost MAE (W/m) | Significant signs correct |
| --- | ---: | ---: | ---: | ---: |
| Run 3 base | 2.13289 | 0.935642 | 0.273561 | 7/47 |
| Run 3 round 40 | 6.76059 | 0.898612 | 0.258706 | 7/47 |
| Run 4 base | 4.87073 | 0.103184 | 0.00807261 | 9/47 |
| Run 4 round 40 (replay) | 1.18294 | 0.0934732 | 0.0178189 | 40/47 |

Run 4's base model is much better than run 3's on difference magnitudes at these
points. Final replay improves nominal power and significant signs, yet its
perturbation-cost MAE is higher than the run 4 base model's. In final replay,
all 40 significant ETM+ signs are correct and all seven significant ETM− signs are
wrong; neither ITM direction has a significant true difference in this cohort.
Finesse decreases in both ETM directions for **32/40** feedback designs;
the final checkpoint predicts **0/40** such pairs.
The improved ETM+ signs therefore coexist with missing symmetric curvature.
The search visits a region dominated by much smaller responses than run 3's trajectory.
[Final replay response shape](outputs/run4_comparison/final_response_shape.json).

| Evaluation | Model ΔPower MAE (W) | Zero-response ΔPower MAE (W) |
| --- | ---: | ---: |
| Run3 base / valid initial validation | 0.642019 | 0.039697 |
| Run4 base / valid initial validation | 0.30074 | 0.039697 |
| Run4 base / run4 feedback | 0.103184 | 0.0725293 |
| Run4 final / run4 feedback replay | 0.0934732 | 0.0725293 |
| Run4 original online | 0.0723349 | 0.0725293 |

The baseline predicts ΔPower = 0 everywhere. Its error is mean absolute true
ΔPower; it says nothing about nominal-power accuracy. These values show why the
observed magnitude and sign improvements still do not establish an accurate local
response model. [Response baseline records](outputs/run4_comparison/response_baselines.csv).
Both base checkpoints on valid initial validation and the final checkpoint on
feedback replay have higher difference MAE than simply predicting zero response.

At round 40, replay through the final checkpoint gives:

| Direction | Finesse power (W) | Final MLP power (W) | Finesse ΔPower (W) | Final MLP ΔPower (W) |
| --- | ---: | ---: | ---: | ---: |
| nominal | 282.014 | 283.069 | 0 | 0 |
| ITM_plus | 282.003 | 283.073 | -0.0109718 | 0.00392821 |
| ITM_minus | 282.018 | 283.069 | 0.00407539 | 0.000541819 |
| ETM_plus | 281.751 | 283.063 | -0.262587 | -0.00582449 |
| ETM_minus | 281.899 | 283.072 | -0.115191 | 0.00298002 |

The true perturbation cost is 0.031198 W/m, while final replay predicts about
0.00114383 W/m, approximately 27 times too small. Across the 40 matching feedback
updates, nominal-power MAE changes **2.35016 → 2.2145 W**, while
signed-difference MAE changes **0.0723571 → 0.0731025 W**. Individual
updates therefore offer little average difference improvement.

Original online predictions are distinct from replay reconstructed from float32
saved features. Tiny rounded differences can change signs: original online has
32/47 significant signs correct, versus 30/47 in matching pre-update replay.

Evidence: [checkpoint comparison](outputs/run4_comparison/comparison.json),
[per-update feedback records](outputs/run4_training_audit/feedback_updates.csv).

## The actual PSO outcome is substantially better

Costs use each saved full objective:

`0.5 / nominal_power + 0.5 * sum(abs(ΔPower) / ROC_step)`.

| Run | Best true cost | Mean true cost | Last true cost | Online nominal MAE (W) | Online ΔPower MAE (W) | Significant signs correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 3 | 0.0169407 (round 4) | 2.9962 | 4.77661 | 11.3952 | 9.92912 | 48/131 |
| Run 4 | 0.0102834 (round 10) | 0.0136565 | 0.0173719 | 2.35021 | 0.0723349 | 32/47 |

Run 4's mean true cost is about 219 times lower,
and its best cost is 39.3% lower.
Each run chose its own trajectory, so this measures the actual optimization outcome
and cannot by itself isolate the loss change. All captured costs were reconstructed
from original powers, exact ROC steps, and weights and checked against saved costs.

Run 4 searches ITM ROC −3169 to −2419 m and ETM ROC 3774 to 3868 m, away from the
upper ITM bound repeatedly selected by run 3. Its best saved design (round 10) has
nominal power 281.940 W at ITM ROC −2575.356 m / ETM ROC 3788.523 m. The original
surrogate cost there is 0.00196104 versus true 0.0102834. At round 40, original
online cost is 0.00225026 versus true 0.0173719 (about 7.72 times larger).
The median true/predicted cost ratio remains about 6.08. Better verified designs
coexist with systematic optimism about sensitivity.

![Original online costs and nominal powers](outputs/run4_comparison/online_cost_and_power.png)

[All original PSO selections and cost components](outputs/run4_comparison/online_rounds.csv).

## Controlled local fits and transfer limitations

The same local diagnostic as run 3 fits rounds 1, 10, 20, 30, and 40 (25 rows),
one seed 9302, for up to 2,000 epochs. Loss variants within an initialization share
initial weights, scaling and optimizer settings. The diagnostic `paired` objective
is row log-power plus signed-difference supervision; **it differs from production
`nominal_delta`**. Its difference scale is derived from the selected training set.
Checkpoint monitoring uses training power MAE / power RMS plus difference MAE /
floored difference RMS. These are local fitting tests, not a global replacement.

| Initialization / scaling | Diagnostic loss | Selected epoch | Power MAE (W) | ΔPower MAE (W) |
| --- | --- | ---: | ---: | ---: |
| Fresh / local scaling | center_focused | 0 | 280.94 | 0.0700343 |
| Fresh / local scaling | log_smooth_l1 | 0 | 280.94 | 0.0700343 |
| Fresh / local scaling | paired | 1850 | 4.17671 | 0.0147322 |
| Round 40 / original scaling | center_focused | 0 | 1.25328 | 0.0946254 |
| Round 40 / original scaling | log_smooth_l1 | 0 | 1.25328 | 0.0946254 |
| Round 40 / original scaling | paired | 1750 | 0.426908 | 0.015795 |

The row-power variants select **epoch 0** for both initializations. The fresh
initial model has extremely poor absolute powers but nearly flat differences;
the combined score can favor it over later power fits because the true differences
are tiny. Those rows describe retained initial checkpoints, not successful row-power
training. Paired supervision improves both local power and difference accuracy.

Fresh diagnostic checkpoints were then evaluated without retraining on the other
35 feedback designs (175 rows):

| Fresh diagnostic checkpoint | Power MAE (W) | Max power error (W) | ΔPower MAE (W) | Max ΔPower error (W) |
| --- | ---: | ---: | ---: | ---: |
| Fresh row-power | 280.921 | 281.234 | 0.0716717 | 0.401067 |
| Fresh paired | 8.43388 | 44.9436 | 0.236029 | 3.73229 |

Transfer groups remain correlated with the saved PSO collection policy. Good local
or nearby-transfer scores do not establish coverage across the global search domain.
Here the paired model's difference MAE increases from 0.0147 W on fitted groups to
0.236 W on transfer, with a 3.73 W maximum difference error. The retained row-power
initial model stays almost flat and has roughly 281 W absolute-power error.

Evidence: [fresh fit results](outputs/run4_five_groups_fresh/summary.csv),
[warm fit results](outputs/run4_five_groups_warm/summary.csv),
[row-power transfer](outputs/run4_transfer_center_focused/metrics.json),
[paired transfer](outputs/run4_transfer_paired/metrics.json).

## Interpretation and next experiment

The nominal/difference objective change is supported by better matched-validation
metrics and markedly better Finesse-verified optimization results. Remaining evidence
points to weak small-response supervision, missing symmetric curvature, and a mismatch
between the training objective and power-only checkpoint selection. These mechanisms
are plausible explanations, rather than independently isolated causes.

A useful next controlled experiment would preserve the complete initial split and
global normalization, compare response scaling and a validation power/difference
selection criterion, and report constant-response baselines alongside signed errors,
sign counts and verified full costs. Validate on new whole designs spanning the
search domain before interpreting nearby replay or local fits as global accuracy.
Keep the best Finesse-verified design alongside the latest round.
