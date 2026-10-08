# Surrogate diagnostics

Diagnose whether the MLP can reproduce fixed-q perturbation responses before
changing the production training loop. These scripts read existing HDF5 samples
and reuse `MLP.data`, `MLP.model`, `MLP.losses`, and `MLP.checkpoint`. They do not
run PSO, generate optical power simulations, or modify run 2 artifacts.

See [FINDINGS.md](FINDINGS.md) for the first run 2 audit and one-group experiments.

Run the commands below from the repository root. Paths in configs and CLI
arguments are relative to the repository root, not the config file. Use the
repository's Python environment exactly as shown. All output directories must
be new; choose another name when repeating an experiment. Generated outputs
are ignored by Git through this folder's `.gitignore`.

## 1. Audit the available examples

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.surrogate_diagnostics.audit_data \
  --all-data --output analysis/surrogate_diagnostics/outputs/run2_audit
```

`audit.json` reports fixed-q coverage, zero outputs, power ranges, plateau
membership, and perturbation directions. `samples.csv` records source files,
HDF5 keys, group identities, all four features, power, nominal power, and signed
differences. Source hashes make the audit traceable to exact input artifacts.

Add `--check-inference` to reconstruct features through the production beam-trace
helper. This checks that an unforced nominal trace followed by fixed-q
perturbations agrees with stored features. It runs no optical power solves.
Reconstructed ROCs come from float32 training inputs, so inspect the reported
errors and tolerances near stability boundaries before concluding that a
discrepancy is a data bug. To check only a handful of examples, omit `--all-data`.

The initial dataset contains consecutive nominal/joint-ROC-perturbation pairs.
Feedback files contain nominal, ITM+, ITM−, ETM+, ETM− groups. The loader checks
feedback direction ordering and never treats joint initial perturbations as
individual-mirror responses. Invalid/zero-q groups are included in audits but
excluded from fitting by default, with their IDs recorded in `provenance.json`.
Zero output alone cannot distinguish instability from lock/solver failure in
the existing files. A fixed-q ROC perturbation combines geometry and mismatch
effects; these are not pure q-mismatch labels.

## 2. Can the model memorize five known samples?

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.surrogate_diagnostics.tiny_fit \
  --rounds 10 --seeds 9302 \
  --output analysis/surrogate_diagnostics/outputs/round10_tiny

/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.surrogate_diagnostics.plot_results \
  analysis/surrogate_diagnostics/outputs/round10_tiny
```

The default starts a fresh `[64, 64]` MLP, fits input scaling on the selected
training rows, and uses full-batch AdamW for 2,000 epochs with no weight decay.
Within each seed, every loss starts with identical weights and normalization,
verified by an initial-state hash. Epoch zero is included in training history
and checkpoint selection, so an experiment can retain the initial model if
training makes the monitored result worse. `--epochs` overrides the budget.

The three variants are:

| Variant | Objective |
| --- | --- |
| `center_focused` | Existing log-power loss plus narrow plateau emphasis |
| `log_smooth_l1` | Ordinary Smooth-L1 on log-power, without plateau emphasis |
| `paired` | Ordinary log-power loss plus Smooth-L1 on signed ΔPower error divided by a training-only scale |

For paired supervision, ΔPower is `P_perturbed - P_nominal`, computed **after**
`expm1`, with gradients through both predictions. Its scale is the RMS of true
training differences, floored at 1 W. The pair-term weight defaults to 1.
Each partner contributes a pair term, so feedback groups contribute four and
initial groups one. Absolute loss weights each row equally. The loss does not
assume that perturbation always lowers power.

All variants select their checkpoint using the same dimensionless score:

`power_MAE / training_power_RMS + delta_MAE / training_delta_RMS`

Both scales have configurable floors and are saved. Tiny-fit monitors the
training set. This common score avoids selecting one variant using a different
notion of success. It is deliberately a diagnostic, not an exact reproduction
of production checkpoint selection or the 20-epoch feedback procedure.

For round 10, the known ETM− response is approximately 37.20 → 50.61 W,
or +13.41 W. Check both absolute powers and ΔPower. An initial useful target
is less than 0.1 W error on multi-watt differences; this is a diagnostic target,
not a guaranteed result. A single group holds q constant, so success proves
local ROC-response fitting, not learned dependence on q.

## 3. Expand to several groups and test existing scaling

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.surrogate_diagnostics.tiny_fit \
  --output analysis/surrogate_diagnostics/outputs/run2_five_groups

/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.surrogate_diagnostics.tiny_fit \
  --config analysis/surrogate_diagnostics/configs/tiny_fit_checkpoint.json \
  --output analysis/surrogate_diagnostics/outputs/run2_warm_start
```

The first command compares rounds 1, 10, 20, 30, and 40 over three random
initializations. The second starts every variant from the run 2 base checkpoint
and preserves its normalization. It uses one seed because full-batch training
from identical pretrained weights has no seed-dependent randomness here.

Small-set normalization can amplify ROC displacements compared with production
scaling. Compare scratch/local-scaling and warm-start/original-scaling results
before attributing improvement solely to a loss. To separate initialization
from scaling, make a copy of the warm-start config and change `normalization`
to `training`; this intentionally changes the initial function as well as the
coordinates. Record that intervention rather than calling it equivalent.

## 4. Check generalization only after the tiny-fit diagnostic

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.surrogate_diagnostics.tiny_fit \
  --config analysis/surrogate_diagnostics/configs/generalization.json \
  --output analysis/surrogate_diagnostics/outputs/run2_generalization
```

This uses fresh weights, complete disjoint groups, and training-only input/loss
scales. The config explicitly lists 70/15/15 randomly partitioned initial
designs, plus feedback rounds 1–24 / 25–32 / 33–40. Invalid fixed-q groups are
excluded and reported; actual counts may be smaller. Checkpoints are selected
on validation, and test data is evaluated only after selection. The loader
rejects cross-split reuse of a group or an exact nominal geometry.

Nearby PSO geometries remain correlated even when group IDs differ. This split
measures performance on saved held-out designs; it is not a spatial extrapolation
benchmark. Repeatedly choosing settings or loss variants from test results
invalidates the holdout. Use validation for choices and reserve a separate final
test set when moving from diagnosis to a production claim. Generalization mode
rejects pretrained initialization because its earlier exposure cannot be assumed
independent of the new split.

## Outputs and interpretation

Each experiment saves the resolved config, source hashes, selected/excluded
groups, training-only scales, and a cross-variant `summary.csv`. Each loss/seed
subdirectory contains:

- `checkpoint.pt`: diagnostic checkpoint, separate from production models.
- `metrics.json`: power MAE/max error; signed-difference MAE/max error;
  sign accuracy per direction excluding true |ΔPower| ≤ 0.1 W.
- `predictions.csv`: per-sample predictions and errors with source identities.
- `costs.json`: the four-direction sum of |ΔPower/ΔROC| for complete feedback
  groups. Joint initial pairs cannot define this cost and are not scored as such.
- `training_history.csv`: epoch-zero baseline and monitored training progress.
- `figures/`: power/ΔPower parity, training curves, and three-point fixed-q
  responses for up to eight feedback groups per split, after plotting.

The root `loss_comparison.png` compares ΔPower MAE across variants/seeds.
Sample-response lines connect known points; they are not dense Finesse sweeps.
Tiny-fit has no independent validation accuracy. A low nominal error alone is
not success. Compare maximum difference error as well as average error, since
an average can hide one poorly learned direction.

For standalone evaluation of an existing model:

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.surrogate_diagnostics.evaluate \
  --config analysis/surrogate_diagnostics/configs/tiny_fit.json \
  --checkpoint MLP/models/run2/base_200_mlp.pt \
  --output analysis/surrogate_diagnostics/outputs/base_on_selected_rounds
```

This measures predictions without asserting independence from prior training.

If ordinary loss memorizes well but held-out differences remain wrong, investigate
coverage and representation. If the paired objective improves tiny-fit and
validation, consider adding it to production training. If neither fits, inspect
feature consistency, scaling, and optimization before collecting more data.
Later experiments can add explicit cavity-relative q features, independent q
sweeps at fixed geometry, and dense Finesse sweeps comparing matched and fixed-q
beams. Those require new physics evaluations and are outside this saved-data
diagnostic.

Focused verification:

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m unittest discover \
  -s analysis/surrogate_diagnostics/tests -v
```
