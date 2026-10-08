# Initial run 2 diagnostic results

These are saved-data diagnostics, not a new production model or an independent
accuracy claim. Exact configs and artifact hashes are in the output directories.

## Data and feature audit

The base dataset plus all 40 feedback files contain 140 groups / 400 samples.
Of these, 119 groups have valid fixed input q throughout: 79 initial pairs and
40 feedback groups. There are 41 zero-output rows and 86 rows within the
configured 282.09 ± 0.01 W center band.

Production beam-trace feature extraction was checked for the 25 samples from
rounds 1, 10, 20, 30, and 40. All passed the audit's documented tolerances.
This supports correct fixed-q feature plumbing for those samples; it does not
validate optical power labels, lock convergence, or mode-order convergence.

Artifacts: `outputs/run2_audit/` and `outputs/run2_selected_feature_audit/`.

## Can one group be learned?

Round 10's nominal plus four fixed-q perturbations were fitted for up to 2,000
full-batch AdamW epochs (learning rate 0.001, no weight decay). Every loss
variant shared the same starting state within each experiment. Selection used
the combined normalized training power/difference MAE defined in the README.

| Initialization and normalization | Loss | Power MAE (W) | ΔPower MAE (W) | Maximum ΔPower error (W) |
| --- | --- | ---: | ---: | ---: |
| Fresh weights, local training scaling | Current center-focused | 0.00000170 | 0.00000138 | 0.00000260 |
| Fresh weights, local training scaling | Ordinary log-power | 0.00000170 | 0.00000138 | 0.00000260 |
| Fresh weights, local training scaling | Log-power + paired | 0.00000170 | 0.00000138 | 0.00000260 |
| Run 2 base weights and original scaling | Current center-focused | 1.20285 | 1.43671 | 1.59728 |
| Run 2 base weights and original scaling | Ordinary log-power | 1.20285 | 1.43671 | 1.59728 |
| Run 2 base weights and original scaling | Log-power + paired | 0.27716 | 0.34694 | 0.75625 |

The plateau term is negligible at this group's powers, explaining why the
center-focused and ordinary variants match here. This isolated experiment
cannot assess how plateau examples compete for gradients during full training.

Fresh/local scaling reached its best monitored checkpoint at epoch 1,100 for
ordinary/center-focused loss and epoch 450 for paired loss. All three original-
checkpoint variants selected epoch 2,000, the budget limit; they may still be
improving. Do not interpret the remaining error as a proven capacity limit.

Artifacts and plots: `outputs/round10_tiny/` and
`outputs/round10_warm_start/`. Each has `summary.csv`, `loss_comparison.png`,
and per-variant checkpoint, prediction, history, and figure files.

## What this establishes

- The existing MLP architecture can represent this group's power differences.
- Paired supervision improves this warm-start fit under the same budget.
- Changing both initialization and normalization substantially changes fitting
  difficulty. Their individual effects have not been isolated.
- q is constant within one group, so this does not prove learned q dependence.
- These runs omit the production replay distribution, newest-row weighting,
  regularization, and short fine-tuning budget. They do not diagnose those
  factors individually or establish a production-ready correction.

The next substantive experiment is several-group fitting under both configs,
followed by validation-based comparison on the grouped generalization split.
That config has 78 training, 20 validation, and 21 test groups after excluding
invalid fixed-q groups. A five-epoch smoke run checks the workflow only; it is
not a meaningful model-quality experiment. No full generalization training has
been run as part of this initial verification.
