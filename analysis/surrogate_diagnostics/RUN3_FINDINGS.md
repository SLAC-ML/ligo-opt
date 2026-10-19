# Run 3 surrogate diagnostics

Run 3 contains the intended 100 nominal designs and 400 single-mirror
perturbations. This change improves prediction magnitudes on the initial
validation designs, but the model still learns the wrong local responses at
many points selected by PSO. Feedback mostly corrects nominal power while
leaving perturbation sensitivity inaccurate. Run 3 also found a substantially
better Finesse design early in the search, then repeatedly selected worse ones.

These results use saved Finesse labels, saved production checkpoints, and the
original captured PSO evaluations. The new fitting experiments write separate
diagnostic checkpoints. No production training, checkpoint, data file, or
optical power simulation was changed.

## What was checked

The diagnostic loader now reads the initial HDF5 `design_group_size` attribute.
Run 3 therefore has 100 five-row initial groups, while historical run 2 retains
100 two-row nominal/joint groups. Five-row direction ordering is checked for
both initial and feedback data. Treating run 3 as consecutive pairs would have
silently produced incorrect nominal/partner associations.

The full run 3 audit contains 140 groups and 700 rows: 500 initial rows and
200 feedback rows. Of the initial groups, 78 have valid fixed q throughout;
22 do not. Initial data contains 104 zero-power rows and 230 rows in the saved
282.09 +/- 0.01 W center band. Production training includes these rows; valid-q
subsets below are explicitly identified. Zero labels do not establish whether
a cavity was unstable or a solver/lock failed.

All 25 feature reconstructions from feedback rounds 1, 10, 20, 30, and 40
passed the documented tolerances. This supports the sampled feature path;
it does not validate optical power solves. Artifacts:
[full audit](outputs/run3_audit/audit.json),
[feature audit](outputs/run3_selected_feature_audit/audit.json).

## The initial sampling change helped, but did not fix response signs

The two runs share identical nominal features and identical initial training
and validation design IDs. Both base checkpoints were evaluated on the same
15 valid held-out run 3 designs, containing 75 rows and 60 directional responses.
This avoids comparing different response types or different geometries.

| Base checkpoint | Power MAE (W) | DeltaPower MAE (W) | Perturbation-cost MAE (W/m) | Correct significant signs |
| --- | ---: | ---: | ---: | ---: |
| Run 2 | 8.123 | 1.213 | 0.478 | 2/10 |
| Run 3 | 7.124 | 0.642 | 0.235 | 2/10 |

Run 3 reduces difference error by about 47% and perturbation-cost error by
about 51% on this cohort. The sample count and significance threshold matter:
only 10 of these 60 responses have absolute true differences above 0.1 W.

The response shape is still wrong. Across all 78 valid initial designs,
Finesse power decreases in both ITM directions for 59 designs and in both ETM
directions for 62 designs. The run 3 base model predicts opposite signs for
the two directions in every one of the 78 designs, for both mirrors. On the
held-out subset, Finesse has 11/15 ITM and 13/15 ETM pairs with both directions
decreasing; the model has zero such pairs.

Most initial responses are small: only 52 of 312 valid directional labels
exceed 0.1 W. Typical signed changes are a few thousandths of a watt. The
ROC steps average about 0.012-0.013 input standard deviations. Matching the
sampling pattern therefore does not by itself force accurate local slopes
and curvature. See [training audit](outputs/run3_training_audit/training_audit.json).

## Accuracy where run 3 actually searches

The following checkpoints were evaluated on exactly the same 40 run 3
feedback designs. The final run 3 checkpoint has already trained on these
feedback labels, so its result measures replay reconstruction.

| Checkpoint | DeltaPower MAE (W) | Correct significant signs |
| --- | ---: | ---: |
| Run 2 base | 10.250 | 2/131 |
| Run 3 base | 10.001 | 2/131 |
| Run 3 after round 40 | 9.935 | 48/131 |

Both run 3 base and final checkpoints get every significant ETM response
wrong: 0/67 correct. The final checkpoint improves nominal powers and some
ITM signs, but differences remain much too flat.

At round 40, the final checkpoint gives this concrete example:

| Sample | Finesse power (W) | Final MLP power (W) | Finesse DeltaPower (W) | MLP DeltaPower (W) |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 39.974 | 38.797 | 0 | 0 |
| ITM+ | 29.276 | 38.797 | -10.698 | approximately 0 |
| ITM- | 53.213 | 38.796 | +13.239 | approximately 0 |
| ETM+ | 62.251 | 38.586 | +22.277 | -0.211 |
| ETM- | 23.766 | 39.008 | -16.208 | +0.211 |

The true perturbation cost is 9.528 W/m; the final model predicts about
0.0500 W/m, roughly 190 times too small. The nominal power is quite close,
yet the sensitivity driving the optimization objective is badly wrong.

Across all 40 feedback updates, evaluating the matching checkpoint immediately
before and after its update gives nominal-power MAE 11.395 -> 7.151 W, while
DeltaPower MAE stays essentially unchanged at 9.929 -> 9.937 W. Those
reconstructed predictions use stored float32 features; original captured
online records are kept separately, since tiny predicted differences can
change sign under feature rounding.

Artifacts: [checkpoint comparison](outputs/run3_comparison/comparison.json),
[feedback updates](outputs/run3_training_audit/feedback_updates.csv).

## The actual PSO outcome is mixed

Costs use the saved full objective:

`0.5 / nominal_power + 0.5 * sum(abs(DeltaPower) / ROC_step)`.

Run 3's best saved Finesse cost was approximately **0.01694 at round 4**,
compared with **0.59014** for run 2's best saved design. Lower is better:
run 3 found a design about 35 times better under this objective.

Later it repeatedly chose poor designs that the surrogate scored as nearly
flat. At round 40, the original online prediction was cost 0.0380 versus
Finesse cost 4.7766, about 126 times larger. The original round-40 evaluation
used checkpoint 39; this is distinct from the final-checkpoint replay above.
The saved records retain every earlier good design, so the final round alone
does not describe the best result achieved by the run.

Online mean true cost was 0.975 for run 2 and 2.996 for run 3. This describes
each run's own selected trajectory; it is not a controlled same-point model
comparison. All captured costs were reconstructed from their original powers,
steps, and weights and checked against the saved values.

![Saved online full costs and nominal powers](outputs/run3_comparison/online_cost_and_power.png)

## Directly supervising differences helps local fitting

Separate models fitted the 25 saved rows from rounds 1, 10, 20, 30, and 40 for
up to 2,000 epochs. Within each experiment, loss variants shared identical
initial weights, normalization, and training settings (one seed, 9302). Checkpoints were
selected using the same training power/difference score.

| Initialization and scaling | Current row loss: DeltaPower MAE (W) | Paired loss: DeltaPower MAE (W) |
| --- | ---: | ---: |
| Fresh weights, scaling from these 25 rows | 3.276 | 0.290 |
| Production final checkpoint, original scaling | 4.945 | 1.603 |

The paired objective adds an explicit signed-difference penalty. It improves
these fits by about 91% and 68%, respectively. The center-focused and ordinary
log-power variants give identical results here because the selected labels
lie outside the narrow center band. Most selected checkpoints reach the epoch
budget, so the residual errors do not establish a model capacity limit.

Local fitting does not establish a usable global surrogate. The fresh models
were also evaluated on the other 35 feedback groups without further training.
Both produced extreme powers at the distant round-4 design, whose ITM ROC is
at the opposite end of the global range from the five training designs.
The ordinary and paired maximum difference errors reached roughly 6.6e161 W
and 1.4e136 W. Their median difference errors were 0.628 W and 0.284 W, but
the extreme failures dominate their mean errors. Small-set normalization and
coverage make these checkpoints unsuitable for global PSO use.

Artifacts: [fresh fits](outputs/run3_five_groups_fresh/summary.csv),
[final-checkpoint fits](outputs/run3_five_groups_warm/summary.csv),
[ordinary transfer evaluation](outputs/run3_transfer_center_focused/metrics.json),
[paired transfer evaluation](outputs/run3_transfer_paired/metrics.json).

## Interpretation and next experiment

The change to four-direction initial sampling was useful. The remaining
failure is that row-wise log-power training and checkpoint selection do not
directly require the differences consumed by PSO to be correct. The observed
flat feedback predictions, wrong symmetric response shape, and improvement
under paired supervision support investigating this training objective.
They do not isolate loss, scaling, coverage, optimization budget, and feedback
selection as independent causes. With 400 rather than 160 initial training
rows, the same batch size gives 13 rather than 5 optimizer updates per epoch.
The automatic newest-feedback weight also changes from 40 to 100. The run
comparison therefore measures the actual combined intervention, rather than
an isolated sampling-layout effect.

A controlled next experiment should keep the full initial design split,
global training-only scaling, model initialization, batching, and budget
fixed; compare current and paired supervision; select checkpoints using
validation power and signed-difference errors; and then assess held-out
designs and actual Finesse-verified optimization results. Dense response or
independent-q sampling could address coverage separately. The best
Finesse-verified design should also be tracked alongside the latest round.

No global retraining comparison or production change was made here.
