"""Rebuild run4's written findings from saved diagnostics, without retraining."""

import csv
import json
from pathlib import Path

from .render_findings import render


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fmt(value):
    return f"{float(value):.6g}"


def metric_row(label, metrics):
    return (f"| {label} | {fmt(metrics['power_mae'])} | {fmt(metrics['delta_mae'])} | "
            f"{fmt(metrics['perturbation_cost_mae'])} | "
            f"{metrics['sign_correct']}/{metrics['sign_count']} |")


def main():
    comparison = read_json(OUTPUTS / "run4_comparison/comparison.json")
    audit = read_json(OUTPUTS / "run4_audit/audit.json")
    feature = read_json(OUTPUTS / "run4_selected_feature_audit/audit.json")
    training = read_json(OUTPUTS / "run4_training_audit/training_audit.json")
    initial = comparison["own_initial"]
    same = comparison["same_geometry"]
    old = initial["run3_base_on_initial"]["validation_valid_fixed_q"]
    new = initial["run4_base_on_initial"]["validation_valid_fixed_q"]
    online_old, online_new = comparison["online"]["run3"], comparison["online"]["run4"]
    initial_table = "\n".join(metric_row(label, stats) for label, stats in (("Run 3 base", old), ("Run 4 base", new)))
    feedback_table = "\n".join(metric_row(label, same[key]["feedback"]) for label, key in (
        ("Run 3 base", "run3_base_on_run4_feedback"),
        ("Run 3 round 40", "run3_round40_on_run4_feedback"),
        ("Run 4 base", "run4_base_on_run4_feedback"),
        ("Run 4 round 40 (replay)", "run4_round40_on_run4_feedback"),
    ))
    online_table = "\n".join(
        f"| {label} | {fmt(stats['best_true_full_cost'])} (round {stats['best_true_full_cost_round']}) | "
        f"{fmt(stats['true_full_cost_mean'])} | {fmt(stats['last_true_full_cost'])} | "
        f"{fmt(stats['nominal_power_mae'])} | {fmt(stats['delta_mae'])} | "
        f"{stats['sign_correct']}/{stats['sign_count']} |"
        for label, stats in (("Run 3", online_old), ("Run 4", online_new))
    )
    replay_rows = read_csv(OUTPUTS / "run4_comparison/run4_round40_on_run4_feedback/predictions.csv")
    r40 = [row for row in replay_rows if row["group_id"] == "round:40:0"]
    shape = {}
    for mirror in ("ITM", "ETM"):
        pairs = {}
        for row in replay_rows:
            if row["direction"].startswith(mirror):
                pairs.setdefault(row["group_id"], []).append(row)
        shape[mirror] = {"groups": len(pairs),
                         "true_both_decrease": sum(all(float(r["delta_power"]) < 0 for r in pair) for pair in pairs.values()),
                         "predicted_both_decrease": sum(all(float(r["predicted_delta_power"]) < 0 for r in pair) for pair in pairs.values())}
    shape_path = OUTPUTS / "run4_comparison/final_response_shape.json"
    shape_path.write_text(json.dumps({"checkpoint": "run4_round40", "split": "feedback_replay", "mirrors": shape}, indent=2) + "\n")
    example = "\n".join(
        f"| {row['direction']} | {fmt(row['power'])} | {fmt(row['predicted_power'])} | "
        f"{fmt(row['delta_power'])} | {fmt(row['predicted_delta_power'])} |" for row in r40
    )
    fit_rows = []
    for folder, label in (("run4_five_groups_fresh", "Fresh / local scaling"),
                          ("run4_five_groups_warm", "Round 40 / original scaling")):
        for row in read_csv(OUTPUTS / folder / "summary.csv"):
            fit_rows.append(f"| {label} | {row['variant']} | {row['best_epoch']} | "
                            f"{fmt(row['power_mae'])} | {fmt(row['delta_mae'])} |")
    transfers = []
    for folder, label in (("run4_transfer_center_focused", "Fresh row-power"),
                          ("run4_transfer_paired", "Fresh paired")):
        stats = read_json(OUTPUTS / folder / "metrics.json")["splits"]["transfer"]
        transfers.append(f"| {label} | {fmt(stats['power_mae'])} | {fmt(stats['power_max_abs_error'])} | "
                         f"{fmt(stats['delta_mae'])} | {fmt(stats['delta_max_abs_error'])} |")
    baseline_rows = []
    for label, stats in (("Run3 base / valid initial validation", old),
                         ("Run4 base / valid initial validation", new),
                         ("Run4 base / run4 feedback", same["run4_base_on_run4_feedback"]["feedback"]),
                         ("Run4 final / run4 feedback replay", same["run4_round40_on_run4_feedback"]["feedback"]),
                         ("Run4 original online", online_new)):
        baseline_rows.append({"evaluation": label, "response_count": stats["count"],
                              "model_delta_mae_w": stats["delta_mae"],
                              "zero_response_delta_mae_w": stats["true_mean_abs_delta"],
                              "predicted_mean_abs_delta_w": stats["predicted_mean_abs_delta"],
                              "note": "Zero-response baseline predicts delta=0; it does not predict nominal power."})
    baseline_path = OUTPUTS / "run4_comparison/response_baselines.csv"
    with baseline_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)
    baseline_table = "\n".join(f"| {row['evaluation']} | {fmt(row['model_delta_mae_w'])} | "
                               f"{fmt(row['zero_response_delta_mae_w'])} |" for row in baseline_rows)
    details = training["feedback_updates"]["details"]
    pre_nominal = sum(abs(row["pre"]["predicted_powers"][0] - row["true_powers"][0]) for row in details) / len(details)
    post_nominal = sum(abs(row["post"]["predicted_powers"][0] - row["true_powers"][0]) for row in details) / len(details)
    pre_delta = sum(row["pre"]["metrics"]["delta_mae"] for row in details) / len(details)
    post_delta = sum(row["post"]["metrics"]["delta_mae"] for row in details) / len(details)
    report = f"""# Run 4 surrogate diagnostics

Run 4 improves substantially over run 3 on matched initial validation designs and
finds better Finesse-verified PSO designs. Its best saved full cost is
**{fmt(online_new['best_true_full_cost'])} at round {online_new['best_true_full_cost_round']}**,
{100 * (1 - online_new['best_true_full_cost'] / online_old['best_true_full_cost']):.1f}% lower than run 3's best.
Signed-difference MAE on the same valid initial validation designs falls by
**{100 * (1 - new['delta_mae'] / old['delta_mae']):.1f}%**. However, the model still misses symmetric
response shape, often understates sensitivity, and does not improve difference
accuracy consistently with each feedback update. Low difference MAE alone is
misleading when the true changes are very small.

This analysis uses saved Finesse labels, production checkpoints and captured online
evaluations. Separate local experiments use diagnostic checkpoints. The notebook
can be refreshed from these artifacts without training or optical power solves.

## Data, features, and the actual intervention

The full run 4 audit contains **{audit['groups']} groups / {audit['rows']} rows**:
500 initial rows and 200 feedback rows. There are {audit['valid_fixed_q_groups']} valid fixed-q groups.
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

All {feature['inference_check']['rows']} selected feature reconstructions from rounds
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
{initial_table}

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
{feedback_table}

Run 4's base model is much better than run 3's on difference magnitudes at these
points. Final replay improves nominal power and significant signs, yet its
perturbation-cost MAE is higher than the run 4 base model's. In final replay,
all 40 significant ETM+ signs are correct and all seven significant ETM− signs are
wrong; neither ITM direction has a significant true difference in this cohort.
Finesse decreases in both ETM directions for **{shape['ETM']['true_both_decrease']}/40** feedback designs;
the final checkpoint predicts **{shape['ETM']['predicted_both_decrease']}/40** such pairs.
The improved ETM+ signs therefore coexist with missing symmetric curvature.
The search visits a region dominated by much smaller responses than run 3's trajectory.
[Final replay response shape](outputs/run4_comparison/final_response_shape.json).

| Evaluation | Model ΔPower MAE (W) | Zero-response ΔPower MAE (W) |
| --- | ---: | ---: |
{baseline_table}

The baseline predicts ΔPower = 0 everywhere. Its error is mean absolute true
ΔPower; it says nothing about nominal-power accuracy. These values show why the
observed magnitude and sign improvements still do not establish an accurate local
response model. [Response baseline records](outputs/run4_comparison/response_baselines.csv).
Both base checkpoints on valid initial validation and the final checkpoint on
feedback replay have higher difference MAE than simply predicting zero response.

At round 40, replay through the final checkpoint gives:

| Direction | Finesse power (W) | Final MLP power (W) | Finesse ΔPower (W) | Final MLP ΔPower (W) |
| --- | ---: | ---: | ---: | ---: |
{example}

The true perturbation cost is 0.031198 W/m, while final replay predicts about
0.00114383 W/m, approximately 27 times too small. Across the 40 matching feedback
updates, nominal-power MAE changes **{fmt(pre_nominal)} → {fmt(post_nominal)} W**, while
signed-difference MAE changes **{fmt(pre_delta)} → {fmt(post_delta)} W**. Individual
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
{online_table}

Run 4's mean true cost is about {online_old['true_full_cost_mean']/online_new['true_full_cost_mean']:.0f} times lower,
and its best cost is {100*(1-online_new['best_true_full_cost']/online_old['best_true_full_cost']):.1f}% lower.
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
{chr(10).join(fit_rows)}

The row-power variants select **epoch 0** for both initializations. The fresh
initial model has extremely poor absolute powers but nearly flat differences;
the combined score can favor it over later power fits because the true differences
are tiny. Those rows describe retained initial checkpoints, not successful row-power
training. Paired supervision improves both local power and difference accuracy.

Fresh diagnostic checkpoints were then evaluated without retraining on the other
35 feedback designs (175 rows):

| Fresh diagnostic checkpoint | Power MAE (W) | Max power error (W) | ΔPower MAE (W) | Max ΔPower error (W) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(transfers)}

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
"""
    path = ROOT / "RUN4_FINDINGS.md"
    path.write_text(report)
    render(path, ROOT / "RUN4_FINDINGS.html")


if __name__ == "__main__":
    main()
