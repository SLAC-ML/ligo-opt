"""Audit saved base training and feedback updates on matching design groups.

This compares base checkpoints on the same target run's complete held-out
initial designs. Feedback pre/post comparisons are fits to adaptively selected
saved examples; they are not independent generalization estimates. No training
or optical power simulation is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from MLP.checkpoint import load_checkpoint
from MLP.finetune import discover_rounds
from .dataset import DIRECTIONS, fingerprints, load_groups, repo_path, write_csv, write_json
from .evaluate import predictions


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    return {"count": len(values), "min": float(values.min()),
            "median": float(np.median(values)), "max": float(values.max()),
            "mean": float(values.mean())}


def _selection_ids(metadata, group_size, key):
    indices = metadata[key]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Checkpoint {key} contains duplicate row indices.")
    groups = {index // group_size for index in indices}
    expected = {index for group in groups
                for index in range(group * group_size, (group + 1) * group_size)}
    if set(indices) != expected:
        raise ValueError(f"Checkpoint {key} separates a design group.")
    return sorted(groups)


def _load_run(number):
    root = repo_path(f"MLP/data/run{number}")
    manifest_path = root / "metadata.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = repo_path(manifest["files"]["base_checkpoint"])
    model, payload = load_checkpoint(checkpoint, device="cpu")
    groups = load_groups({"data_dir": str(root), "detector": payload["detector"]},
                         {"base_ids": "all"})
    group_size = len(groups[0].x)
    metadata = payload["training_metadata"]
    if metadata.get("base_group_size", 2) != group_size:
        raise ValueError("Checkpoint base group size disagrees with its dataset.")
    base_path = Path(groups[0].source)
    if fingerprints([base_path])[str(base_path)] != metadata.get("base_data_sha256"):
        raise ValueError("Original base dataset changed after training: checkpoint hash disagrees.")
    train_ids = _selection_ids(metadata, group_size, "train_indices")
    validation_ids = _selection_ids(metadata, group_size, "validation_indices")
    if metadata["base_data_count"] != sum(len(g.x) for g in groups):
        raise ValueError("Checkpoint base row count disagrees with its dataset.")
    if set(train_ids) & set(validation_ids):
        raise ValueError("This audit requires an independent initial validation split.")
    if set(train_ids) | set(validation_ids) != set(range(len(groups))):
        raise ValueError("Checkpoint split does not cover its base design groups.")
    return {"run": number, "root": root, "manifest_path": manifest_path,
            "checkpoint": checkpoint, "model": model, "payload": payload,
            "groups": groups, "train_ids": train_ids, "validation_ids": validation_ids}


def _initial_delta_scale_summary(run):
    """Describe the recorded response scale using its original training labels."""
    metadata = run["payload"]["training_metadata"]
    if metadata.get("delta_scale_w") is None:
        return None
    by_validity = {True: [], False: []}
    for index in run["train_ids"]:
        group = run["groups"][index]
        powers = group.y.double().expm1().flatten().numpy()
        by_validity[group.valid_fixed_q].extend((powers[1:] - powers[0]).tolist())
    all_deltas = np.array(by_validity[True] + by_validity[False], dtype=float)
    total_squared = float(np.square(all_deltas).sum())
    return {
        "recorded_scale_w": metadata["delta_scale_w"],
        "training_partner_rows": len(all_deltas),
        "training_delta_rms_w": float(np.sqrt(np.square(all_deltas).mean())),
        "floor_w": metadata["loss_config"]["delta_scale_floor_w"],
        "by_fixed_q_validity": {
            "valid" if valid else "invalid": {
                "partner_rows": len(values),
                "delta_rms_w": float(np.sqrt(np.square(values).mean())) if values else None,
                "squared_delta_share": float(np.square(values).sum()) / total_squared
                if total_squared else None,
            } for valid, values in by_validity.items()
        },
        "note": "Scale is fixed from all original training partners, including invalid fixed-q groups; validation and feedback labels are excluded.",
    }


def _base_summary(run):
    metadata = run["payload"]["training_metadata"]
    history = metadata["history"]
    return {"checkpoint": str(run["checkpoint"]),
            "model_config": run["payload"]["model_config"],
            "group_size": len(run["groups"][0].x),
            "base_row_count": metadata["base_data_count"],
            "train_design_ids": run["train_ids"],
            "validation_design_ids": run["validation_ids"],
            "validation_mode": metadata["validation_mode"],
            "loss_config": metadata.get("loss_config"),
            "base_data_sha256": metadata["base_data_sha256"],
            "split_seed": metadata["split_seed"],
            "delta_scale_w": metadata.get("delta_scale_w"),
            "delta_scale_audit": _initial_delta_scale_summary(run),
            "checkpoint_selection": metadata.get("checkpoint_selection", "all_row_power_loss"),
            "hyperparams": metadata["hyperparams"],
            "best_epoch": metadata["best_epoch"],
            "best_validation_loss": metadata["best_validation_loss"],
            "best_power_metrics": metadata.get("best_power_metrics"),
            "first_history_row": history[0],
            "best_history_row": next(row for row in history
                                     if row["epoch"] == metadata["best_epoch"]),
            "last_history_row": history[-1],
            "input_mean": run["model"].input_mean.tolist(),
            "input_scale": run["model"].input_scale.tolist()}


def _data_summary(groups, loss_config, sign_threshold):
    records = [record for group in groups for record in group.records()]
    powers = np.array([r["power"] for r in records])
    center_power = loss_config["center_power"]
    center_width = loss_config["center_width"]
    partial = []
    for group in groups:
        power = group.y.expm1().flatten().numpy()
        count = int((power == 0).sum())
        if 0 < count < len(power):
            partial.append({"group_id": group.group_id,
                            "valid_fixed_q": group.valid_fixed_q,
                            "features": group.x.tolist(), "powers": power.tolist()})
    result = {"groups": len(groups), "rows": len(records),
              "valid_fixed_q_groups": sum(g.valid_fixed_q for g in groups),
              "invalid_fixed_q_ids": [g.group_id for g in groups if not g.valid_fixed_q],
              "zero_power_rows": int((powers == 0).sum()),
              "nominal_zero_groups": sum(float(g.y[0].expm1()) == 0 for g in groups),
              "center_power": center_power, "center_width": center_width,
              "center_rows": int((abs(powers - center_power) <= center_width).sum()),
              "partial_zero_groups": partial, "power_distribution": _summary(powers),
              "by_direction": {}}
    for direction in sorted({r["direction"] for r in records}):
        selected = [r for r in records if r["direction"] == direction]
        valid = [r for r in selected if r["valid_fixed_q_group"]]
        delta = np.array([r["delta_power"] for r in valid])
        result["by_direction"][direction] = {
            "rows": len(selected), "valid_fixed_q_rows": len(valid),
            "zero_power_rows": sum(r["zero_output"] for r in selected),
            "center_rows": sum(abs(r["power"] - center_power) <= center_width for r in selected),
            "valid_delta_distribution": _summary(delta) if len(delta) else None,
            "valid_negative_delta_count": int((delta < 0).sum()),
            "valid_positive_delta_count": int((delta > 0).sum()),
            "valid_significant_delta_count": int((abs(delta) > sign_threshold).sum()),
            "valid_mean_abs_delta": float(abs(delta).mean()) if len(delta) else None,
        }
    return result


@torch.no_grad()
def _symmetric_summary(model, groups):
    if not groups or any(g.directions != DIRECTIONS for g in groups):
        return None
    true = np.array([g.y.expm1().flatten().numpy() for g in groups])
    predicted = np.array([model(g.x).double().expm1().flatten().numpy() for g in groups])
    result = {}
    for axis, index in (("ITM", 1), ("ETM", 3)):
        actual_delta = true[:, index:index + 2] - true[:, 0, None]
        predicted_delta = predicted[:, index:index + 2] - predicted[:, 0, None]
        actual_even = actual_delta.mean(1)
        predicted_even = predicted_delta.mean(1)
        actual_odd = (actual_delta[:, 0] - actual_delta[:, 1]) / 2
        predicted_odd = (predicted_delta[:, 0] - predicted_delta[:, 1]) / 2
        result[axis] = {
            "groups": len(groups),
            "true_even_mean": float(actual_even.mean()),
            "predicted_even_mean": float(predicted_even.mean()),
            "true_mean_abs_odd": float(abs(actual_odd).mean()),
            "predicted_mean_abs_odd": float(abs(predicted_odd).mean()),
            "even_mae": float(abs(predicted_even - actual_even).mean()),
            "odd_mae": float(abs(predicted_odd - actual_odd).mean()),
            "true_both_decrease_count": int((actual_delta < 0).all(1).sum()),
            "predicted_both_decrease_count": int((predicted_delta < 0).all(1).sum()),
            "predicted_opposite_sign_count": int((predicted_delta[:, 0] * predicted_delta[:, 1] < 0).sum()),
        }
    return result


def _feedback_summary(run, sign_threshold, source_paths):
    rounds = discover_rounds(run["root"])
    if not rounds:
        return {"rounds": 0}, [], []
    if set(rounds) != set(range(1, max(rounds) + 1)):
        raise ValueError("Feedback data must be contiguous from round 1.")
    prior = run["model"]
    details, rows, all_groups = [], [], []
    for number in sorted(rounds):
        groups = load_groups({"data_dir": str(run["root"]),
                              "detector": run["payload"]["detector"]}, {"rounds": [number]})
        if len(groups) != 1:
            raise ValueError("Feedback update audit expects one complete design per round.")
        group = groups[0]
        checkpoint = repo_path(
            f"MLP/models/run{run['run']}/power_predictor_ligoParams_finetuned_round{number}.pt")
        after, payload = load_checkpoint(checkpoint, device="cpu")
        source_paths.extend([rounds[number], checkpoint])
        metadata = payload["training_metadata"]
        if metadata["current_round"] != number:
            raise ValueError("Feedback checkpoint round disagrees with its filename.")
        detail = {"round": number, "group_id": group.group_id,
                  "valid_fixed_q": group.valid_fixed_q,
                  "nominal_roc_m": group.x[0, :2].tolist(),
                  "true_powers": group.y.expm1().flatten().tolist(),
                  "best_epoch": metadata["best_epoch"],
                  "newest_row_weight": metadata["newest_round_weight"],
                  "best_validation_loss": metadata["best_validation_loss"],
                  "training_row_count": metadata["training_row_count"]}
        for stage, model in (("pre", prior), ("post", after)):
            records, metrics, costs = predictions(model, [group], f"round{number}_{stage}", sign_threshold)
            detail[stage] = {"metrics": metrics, "cost": costs[0],
                             "predicted_powers": [r["predicted_power"] for r in records]}
            rows.append({"round": number, "stage": stage,
                         "valid_fixed_q": group.valid_fixed_q,
                         "nominal_itm_roc_m": group.x[0, 0].item(),
                         "nominal_etm_roc_m": group.x[0, 1].item(),
                         "true_nominal_power": records[0]["power"],
                         "predicted_nominal_power": records[0]["predicted_power"],
                         "nominal_power_abs_error": abs(records[0]["power_error"]),
                         "power_mae": metrics["power_mae"], "delta_mae": metrics["delta_mae"],
                         "sign_count": metrics["sign_count"], "sign_accuracy": metrics["sign_accuracy"],
                         **costs[0],
                         "valid_perturbation_cost": all(costs[0][key] is not None for key in
                             ("true_perturbation_cost", "predicted_perturbation_cost")),
                         "best_epoch": metadata["best_epoch"],
                         "newest_row_weight": metadata["newest_round_weight"]})
        details.append(detail)
        all_groups.extend(groups)
        prior = after
    valid = [g for g in all_groups if g.valid_fixed_q]
    aggregate = {"rounds": len(rounds), "valid_fixed_q_groups": len(valid),
                 "has_valid_fixed_q_groups": bool(valid),
                 "excluded_fixed_q_group_ids": [g.group_id for g in all_groups if not g.valid_fixed_q],
                 "details": details, "pre_post": {}}

    def mean_or_none(values):
        return float(np.mean(values)) if values else None
    for stage in ("pre", "post"):
        current = [r for r in rows if r["stage"] == stage and r["valid_fixed_q"]]
        count = sum(r["sign_count"] for r in current)
        valid_costs = [r for r in current if r["valid_perturbation_cost"]]
        aggregate["pre_post"][stage] = {
            "groups": len(current), "nominal_power_mae": mean_or_none([r["nominal_power_abs_error"] for r in current]),
            "power_mae": mean_or_none([r["power_mae"] for r in current]),
            "delta_mae": mean_or_none([r["delta_mae"] for r in current]),
            "sign_count": count,
            "sign_accuracy": sum(r["sign_count"] * (r["sign_accuracy"] or 0) for r in current) / count if count else None,
            "valid_cost_groups": len(valid_costs),
            "invalid_cost_groups": len(current) - len(valid_costs),
            "cost_excluded_group_ids": [r["group_id"] for r in current if not r["valid_perturbation_cost"]],
            "perturbation_cost_mae": mean_or_none([
                abs(r["predicted_perturbation_cost"] - r["true_perturbation_cost"]) for r in valid_costs]),
        }
    aggregate["rounds_with_lower_post_delta_mae"] = sum(
        d["post"]["metrics"]["delta_mae"] < d["pre"]["metrics"]["delta_mae"]
        for d in details if d["valid_fixed_q"])
    aggregate["base_on_all_feedback"] = predictions(run["model"], valid, "feedback", sign_threshold)[1] if valid else None
    aggregate["final_on_all_feedback"] = predictions(prior, valid, "feedback", sign_threshold)[1] if valid else None
    return aggregate, rows, all_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=3)
    parser.add_argument("--reference-run", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sign-threshold-w", type=float, default=0.1)
    args = parser.parse_args()
    if args.run == args.reference_run:
        raise ValueError("Select different target and reference runs.")
    if args.sign_threshold_w < 0 or not np.isfinite(args.sign_threshold_w):
        raise ValueError("Sign threshold must be finite and nonnegative.")
    output = repo_path(args.output)
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    torch.set_num_threads(1)
    target, reference = _load_run(args.run), _load_run(args.reference_run)
    source_paths = [Path(__file__).resolve(), *(r["manifest_path"] for r in (target, reference))]
    source_paths.extend(r["checkpoint"] for r in (target, reference))
    source_paths.extend(Path(g.source) for r in (target, reference) for g in r["groups"])
    if target["payload"]["detector"] != reference["payload"]["detector"]:
        raise ValueError("Run checkpoints use different power detectors.")
    same_count = len(target["groups"]) == len(reference["groups"])
    same_nominal = same_count and all(torch.equal(a.x[0], b.x[0])
                                    for a, b in zip(target["groups"], reference["groups"]))
    result = {
        "target_run": args.run, "reference_run": args.reference_run,
        "detector": target["payload"]["detector"], "sign_threshold_w": args.sign_threshold_w,
        "same_nominal_features": same_nominal,
        "same_initial_features": same_count and all(torch.equal(a.x, b.x)
            for a, b in zip(target["groups"], reference["groups"])),
        "same_initial_labels": same_count and all(torch.equal(a.y, b.y)
            for a, b in zip(target["groups"], reference["groups"])),
        "same_train_design_ids": target["train_ids"] == reference["train_ids"],
        "same_validation_design_ids": target["validation_ids"] == reference["validation_ids"],
        "base_training": {}, "initial_data": {}, "same_target_cohort_metrics": {},
        "symmetric_response_note": "Even=(Delta_plus+Delta_minus)/2; odd=(Delta_plus-Delta_minus)/2, in W.",
        "notes": ["Base checkpoints are compared on identical target-run groups.",
                  "Whole initial designs in validation remain excluded from production training.",
                  "Feedback pre/post and final replay measure fit to adaptively selected samples, not independent accuracy.",
                  "Invalid fixed-q groups remain in label counts and are excluded from directional model comparisons."]}
    history_rows, prediction_rows = [], []
    for run in (reference, target):
        label = f"run{run['run']}"
        result["base_training"][label] = _base_summary(run)
        config = run["payload"]["training_metadata"]["loss_config"]
        result["initial_data"][label] = _data_summary(run["groups"], config, args.sign_threshold_w)
        history_rows.extend({"run": run["run"], **row} for row in run["payload"]["training_metadata"]["history"])
    for split, ids in (("train", target["train_ids"]), ("validation", target["validation_ids"]),
                       ("all", list(range(len(target["groups"]))))):
        groups = [target["groups"][i] for i in ids if target["groups"][i].valid_fixed_q]
        result["same_target_cohort_metrics"][split] = {"valid_design_ids": [g.group_id for g in groups]}
        for run in (reference, target):
            label = f"run{run['run']}"
            if not groups:
                result["same_target_cohort_metrics"][split][label] = None
                continue
            records, metrics, _ = predictions(run["model"], groups, split, args.sign_threshold_w)
            result["same_target_cohort_metrics"][split][label] = {
                **metrics, "symmetric_response": _symmetric_summary(run["model"], groups)}
            prediction_rows.extend({"checkpoint_run": run["run"], **r} for r in records)
    valid = [g for g in target["groups"] if g.valid_fixed_q]
    if all(g.directions == DIRECTIONS for g in valid):
        result["normalized_roc_steps"] = {
            axis: _summary([abs(g.x[row, column] - g.x[0, column]).item() /
                            target["model"].input_scale[column].item() for g in valid])
            for axis, row, column in (("ITM", 1, 0), ("ETM", 3, 1))}
    result["feedback_updates"], feedback_rows, _ = _feedback_summary(target, args.sign_threshold_w, source_paths)
    result["source_sha256"] = fingerprints(source_paths)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "training_audit.json", result)
    write_csv(output / "training_history.csv", history_rows)
    write_csv(output / "initial_predictions.csv", prediction_rows)
    write_csv(output / "feedback_updates.csv", feedback_rows)
    print(json.dumps({"output": str(output),
                      "validation": result["same_target_cohort_metrics"]["validation"],
                      "feedback_pre_post": result["feedback_updates"].get("pre_post")}, indent=2))


if __name__ == "__main__":
    main()
