"""Compare saved MLP runs, their original online records, and common geometries.

No optical power simulations or training are performed. The local final_result.pkl
files are trusted project artifacts; do not point this command at untrusted pickle
files. Existing output directories are rejected.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from MLP.checkpoint import load_checkpoint
from utils.training_groups import read_design_group_size
from .dataset import (
    DIRECTIONS, REPO_ROOT, fingerprints, load_groups, repo_path, write_csv, write_json,
)


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _max(values):
    return float(np.max(values)) if len(values) else None


def _cost_components(powers, delta, cost_mode="full", power_weight=0.5,
                     perturbation_weight=0.5):
    """Pure arithmetic equivalent of the saved objective, with null invalid costs."""
    nominal = float(powers["nominal"])
    power_cost = 1.0 / nominal if np.isfinite(nominal) and nominal > 0 else None
    perturbation_cost = None
    if all(powers.get(key) is not None for key in DIRECTIONS[1:]):
        selected = np.array([powers[key] for key in DIRECTIONS], dtype=float)
        steps = np.array([delta[0], delta[0], delta[1], delta[1]], dtype=float)
        if (np.isfinite(selected).all() and (selected >= 0).all() and nominal > 0
                and np.isfinite(steps).all() and (steps != 0).all()):
            # Preserve production summation order: ITM+, ETM+, ITM-, ETM-.
            terms = np.abs((selected[1:] - nominal) / steps)
            perturbation_cost = float(terms[0] + terms[2] + terms[1] + terms[3])
    if cost_mode == "design_power":
        total = power_cost
    elif cost_mode == "perturbation":
        total = perturbation_cost
    elif cost_mode == "full":
        total = (power_weight * power_cost + perturbation_weight * perturbation_cost
                 if power_cost is not None and perturbation_cost is not None else None)
    else:
        raise ValueError(f"Unknown saved cost mode {cost_mode!r}")
    return {
        "power_cost": power_cost, "perturbation_cost": perturbation_cost,
        "weighted_power_cost": power_weight * power_cost if power_cost is not None else None,
        "weighted_perturbation_cost": (perturbation_weight * perturbation_cost
                                       if perturbation_cost is not None else None),
        "total_cost": total,
    }


def _assert_saved_cost(record, components):
    saved = float(record["cost"])
    actual = components["total_cost"]
    if actual is None:
        if not np.isposinf(saved):
            raise ValueError("Invalid captured powers do not reproduce the saved cost.")
    elif not np.isclose(actual, saved, rtol=1e-6, atol=1e-6):
        raise ValueError(f"Reconstructed cost {actual} differs from saved cost {saved}.")


def _response_metrics(rows, threshold):
    errors = [abs(row["delta_error"]) for row in rows]
    significant = [row for row in rows if abs(row["delta_power"]) > threshold]
    correct = sum(np.sign(row["predicted_delta_power"]) == np.sign(row["delta_power"])
                  for row in significant)
    return {
        "count": len(rows), "delta_mae": _mean(errors),
        "delta_max_abs_error": _max(errors), "sign_count": len(significant),
        "sign_correct": int(correct),
        "sign_accuracy": correct / len(significant) if significant else None,
        "true_mean_abs_delta": _mean([abs(row["delta_power"]) for row in rows]),
        "predicted_mean_abs_delta": _mean([abs(row["predicted_delta_power"]) for row in rows]),
    }


def _summarize(rows, costs, threshold, center_config=None):
    center_config = center_config or {}
    center_power = float(center_config.get("center_power", 282.09))
    center_width = float(center_config.get("center_width", 0.01))
    nominal = [row for row in rows if row["direction"] == "nominal"]
    perturbed = [row for row in rows if row["direction"] != "nominal"]
    valid_costs = [row for row in costs if row["true_perturbation_cost"] is not None
                   and row["predicted_perturbation_cost"] is not None]
    full_costs = [row for row in costs if row["true_total_cost"] is not None
                  and row["predicted_total_cost"] is not None]
    center = [row for row in rows if abs(row["power"] - center_power) <= center_width]
    return {
        "groups": len(nominal), "rows": len(rows),
        "valid_fixed_q_groups": (sum(bool(row.get("valid_fixed_q_group")) for row in nominal)
                                 if any(row.get("valid_fixed_q_group") is not None for row in nominal) else None),
        "power_mae": _mean([abs(row["power_error"]) for row in rows]),
        "power_max_abs_error": _max([abs(row["power_error"]) for row in rows]),
        "nominal_power_mae": _mean([abs(row["power_error"]) for row in nominal]),
        "center_rows": len(center), "center_power": center_power, "center_width": center_width,
        "center_power_mae": _mean([abs(row["power_error"]) for row in center]),
        **_response_metrics(perturbed, threshold), "sign_threshold_w": threshold,
        "by_direction": {
            direction: _response_metrics([row for row in perturbed if row["direction"] == direction], threshold)
            for direction in sorted({row["direction"] for row in perturbed})
        },
        "valid_cost_groups": len(valid_costs), "invalid_cost_groups": len(costs) - len(valid_costs),
        "perturbation_cost_mae": _mean([
            abs(row["predicted_perturbation_cost"] - row["true_perturbation_cost"])
            for row in valid_costs]),
        "full_cost_mae": _mean([abs(row["predicted_total_cost"] - row["true_total_cost"])
                               for row in full_costs]),
        "predicted_full_cost_mean": _mean([row["predicted_total_cost"] for row in full_costs]),
        "true_full_cost_mean": _mean([row["true_total_cost"] for row in full_costs]),
        "optimistic_cost_groups": sum(row["predicted_total_cost"] < row["true_total_cost"]
                                      for row in full_costs),
    }


def _paired_cost_record(group_id, predicted, true, delta, settings):
    pc = _cost_components(predicted, delta, **settings)
    tc = _cost_components(true, delta, **settings)
    return {"group_id": group_id, "delta_itm": float(delta[0]), "delta_etm": float(delta[1]),
            **{f"predicted_{key}": value for key, value in pc.items()},
            **{f"true_{key}": value for key, value in tc.items()}}


@torch.inference_mode()
def _evaluate(model, groups, split, threshold, objective, recorded_deltas=None,
              training_exposure="", evaluation_role="", center_config=None):
    rows, costs = [], []
    for group in groups:
        predicted = model(group.x).double().expm1().flatten().cpu().numpy()
        if not np.isfinite(predicted).all() or (predicted < 0).any():
            raise FloatingPointError("Checkpoint produced nonfinite or negative power.")
        true = group.y.double().expm1().flatten().numpy()
        for i, record in enumerate(group.records()):
            rows.append({"split": split, **record,
                         "predicted_power": float(predicted[i]),
                         "power_error": float(predicted[i] - true[i]),
                         "predicted_delta_power": float(predicted[i] - predicted[0]),
                         "delta_error": float((predicted[i] - predicted[0]) - (true[i] - true[0])),
                         "training_exposure": training_exposure,
                         "evaluation_role": evaluation_role})
        if group.directions == DIRECTIONS:
            # Saved exact ROC steps take precedence over float32 feature coordinates.
            delta = ((recorded_deltas or {}).get(group.group_id)
                     or (abs(float(group.x[1, 0]) - float(group.x[0, 0])),
                         abs(float(group.x[3, 1]) - float(group.x[0, 1]))))
            costs.append(_paired_cost_record(
                group.group_id, dict(zip(DIRECTIONS, predicted)),
                dict(zip(DIRECTIONS, true)), delta, objective))
    return rows, _summarize(rows, costs, threshold, center_config), costs


def _read_run(number):
    root = REPO_ROOT / "MLP" / "data" / f"run{number}"
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["settings"]["surrogate_type"] != "MLP":
        raise ValueError(f"Expected MLP metadata in {root}")
    result_path = repo_path(metadata["files"]["final_result"])
    # Local experiment files contain numpy arrays, so safe tensor loading cannot read them.
    with result_path.open("rb") as stream:
        result = pickle.load(stream)
    for name in ("data", "evaluations_surrogate", "evaluations_finesse"):
        if name not in result:
            raise ValueError(f"{result_path} lacks original online field {name}")
    counts = [len(result[name]) for name in ("data", "evaluations_surrogate", "evaluations_finesse")]
    if len(set(counts)) != 1 or not counts[0]:
        raise ValueError(f"Misaligned or empty online records in {result_path}")
    return {"number": number, "root": root, "metadata": metadata,
            "result": result, "metadata_path": metadata_path, "result_path": result_path}


def _objective(run):
    values = run["metadata"]["settings"]["objective"]
    return {key: values[key] for key in ("cost_mode", "power_weight", "perturbation_weight")}


def _online(run, threshold):
    rows, costs, rounds = [], [], []
    result = run["result"]
    center_config = run["metadata"]["settings"]["training"]["base"].get("loss", {})
    saved_groups = None
    for number, (params, predicted_record, true_record) in enumerate(zip(
            result["data"], result["evaluations_surrogate"], result["evaluations_finesse"]), 1):
        if not np.array_equal(params, predicted_record["params"]) or not np.array_equal(params, true_record["params"]):
            raise ValueError(f"Run {run['number']} round {number} has misassociated powers and coordinates.")
        if saved_groups is None:
            loaded = load_groups({"data_dir": str(run["root"]), "detector": true_record["pd_name"]},
                                 {"rounds": list(range(1, len(result["data"]) + 1))})
            saved_groups = {group.group_id: group for group in loaded}
            if len(saved_groups) != len(result["data"]):
                raise ValueError("Expected one saved feedback group per online round.")
        saved_group = saved_groups[f"round:{number}:0"]
        if not np.allclose(saved_group.x[0, :2].numpy(), params, rtol=1e-7, atol=1e-4):
            raise ValueError("Saved feedback nominal geometry differs from original online parameters.")
        record_settings = {key: predicted_record[key]
                           for key in ("cost_mode", "power_weight", "perturbation_weight")}
        if any(predicted_record[key] != true_record[key] for key in record_settings):
            raise ValueError("Predicted and true online objective settings differ.")
        if predicted_record["delta"] != true_record["delta"]:
            raise ValueError("Predicted and true online ROC steps differ.")
        pc = _cost_components(predicted_record["powers"], predicted_record["delta"], **record_settings)
        tc = _cost_components(true_record["powers"], true_record["delta"], **record_settings)
        _assert_saved_cost(predicted_record, pc)
        _assert_saved_cost(true_record, tc)
        pred = predicted_record["powers"]
        true = true_record["powers"]
        group_id = f"round:{number}:0"
        common = {"run": run["number"], "round": number, "group_id": group_id,
                  "itm_roc": float(params[0]), "etm_roc": float(params[1]),
                  "model_path": predicted_record["model_path"],
                  "checkpoint_round": number - 1, "evaluation_role": "original_online_before_feedback",
                  "training_exposure": f"Own initial training rows and feedback rounds 1-{number-1}",
                  "cost_mode": predicted_record["cost_mode"],
                  "power_weight": predicted_record["power_weight"],
                  "perturbation_weight": predicted_record["perturbation_weight"]}
        current = []
        for direction in DIRECTIONS:
            if pred[direction] is None or true[direction] is None:
                continue
            dp, dt = pred[direction] - pred["nominal"], true[direction] - true["nominal"]
            current.append({**common, "split": "online", "direction": direction,
                            "power": true[direction], "nominal_power": true["nominal"],
                            "predicted_power": pred[direction], "power_error": pred[direction] - true[direction],
                            "delta_power": dt, "predicted_delta_power": dp, "delta_error": dp - dt,
                            "sign_significant": direction != "nominal" and abs(dt) > threshold,
                            "sign_correct": bool(np.sign(dp) == np.sign(dt)),
                            "valid_fixed_q_group": saved_group.valid_fixed_q})
        cost = _paired_cost_record(group_id, pred, true, predicted_record["delta"], record_settings)
        costs.append(cost)
        metrics = _summarize(current, [cost], threshold, center_config)
        rounds.append({**common, **cost,
                       "nominal_power": true["nominal"], "predicted_nominal_power": pred["nominal"],
                       "nominal_power_error": pred["nominal"] - true["nominal"],
                       "delta_mae": metrics["delta_mae"],
                       "sign_count": metrics["sign_count"], "sign_correct": metrics["sign_correct"],
                       "true_to_predicted_cost_ratio": (tc["total_cost"] / pc["total_cost"]
                                                       if pc["total_cost"] and tc["total_cost"] is not None else None),
                       "true_weighted_power_fraction": (tc["weighted_power_cost"] / tc["total_cost"]
                                                        if tc["total_cost"] else None)})
        rows.extend(current)
    summary = _summarize(rows, costs, threshold, center_config)
    valid_rounds = [row for row in rounds if row["true_total_cost"] is not None]
    best = min(valid_rounds, key=lambda row: row["true_total_cost"])
    summary.update({"best_true_full_cost": best["true_total_cost"],
                    "best_true_full_cost_round": best["round"],
                    "last_true_full_cost": rounds[-1]["true_total_cost"],
                    "last_predicted_full_cost": rounds[-1]["predicted_total_cost"],
                    "median_true_to_predicted_cost_ratio": float(np.median([
                        row["true_to_predicted_cost_ratio"] for row in valid_rounds
                        if row["true_to_predicted_cost_ratio"] is not None])),
                    "median_true_weighted_power_fraction": float(np.median([
                        row["true_weighted_power_fraction"] for row in valid_rounds])),
                    "upper_itm_bound_rounds": int(sum(np.isclose(row["itm_roc"], run["metadata"]["settings"]["sampling"]["itm_roc_bounds_m"][1]) for row in rounds)),
                    "itm_range_m": [min(row["itm_roc"] for row in rounds), max(row["itm_roc"] for row in rounds)],
                    "etm_range_m": [min(row["etm_roc"] for row in rounds), max(row["etm_roc"] for row in rounds)],
                    "source": str(run["result_path"]),
                    "note": "Each run chose its own geometries. These are original predictions before that round's Finesse feedback; not a controlled comparison of architectures.",
                    "segments": {}})
    for start in range(1, len(rounds) + 1, 10):
        end = min(start + 9, len(rounds))
        segment_rows = [row for row in rows if start <= row["round"] <= end]
        segment_costs = costs[start-1:end]
        summary["segments"][f"rounds_{start}_{end}"] = _summarize(segment_rows, segment_costs, threshold, center_config)
    return rows, rounds, summary


def _base_split_groups(run, payload):
    metadata = payload["training_metadata"]
    source = repo_path(run["metadata"]["files"]["base_data"])
    group_size = read_design_group_size(source)
    if metadata.get("base_group_size", 2) != group_size:
        raise ValueError("Checkpoint and saved initial data disagree on group size.")
    config = {"data_dir": str(run["root"]), "detector": payload["detector"]}
    all_groups = load_groups(config, {"base_ids": "all"})
    all_rows = set(range(sum(len(group.x) for group in all_groups)))
    training_rows = set(metadata["train_indices"])
    validation_rows = set(metadata["validation_indices"])
    if training_rows & validation_rows:
        raise ValueError("Initial training and validation splits overlap.")
    if training_rows | validation_rows != all_rows:
        raise ValueError("Initial training and validation splits do not cover all base rows.")
    if len(all_rows) != metadata["base_data_count"]:
        raise ValueError("Initial data row count differs from checkpoint provenance.")
    splits = {}
    for name, key in (("training", "train_indices"), ("validation", "validation_indices")):
        indices = metadata[key]
        wanted = set(indices)
        group_ids = sorted({index // group_size for index in wanted})
        expanded = {group_id * group_size + offset for group_id in group_ids for offset in range(group_size)}
        if wanted != expanded or len(indices) != len(wanted):
            raise ValueError("Checkpoint split does not contain complete distinct initial groups.")
        groups = load_groups(config, {"base_ids": group_ids})
        if sum(len(group.x) for group in groups) != len(indices):
            raise ValueError("Loaded initial split differs from checkpoint row count.")
        splits[name] = groups
        splits[f"{name}_valid_fixed_q"] = [group for group in groups if group.valid_fixed_q]
    if fingerprints([source])[str(source)] != metadata["base_data_sha256"]:
        raise ValueError("Initial data hash differs from the checkpoint's training provenance.")
    return config, splits


def _write_evaluation(output, label, run, checkpoint, splits, config, threshold,
                      objective, notes, exposure, recorded_deltas=None):
    model, payload = load_checkpoint(checkpoint, device="cpu")
    if payload["detector"] != config["detector"]:
        raise ValueError("Checkpoint and diagnostic detector differ.")
    destination = output / label
    destination.mkdir()
    records, metrics, costs = [], {}, {}
    for split, groups in splits.items():
        current, metrics[split], costs[split] = _evaluate(
            model, groups, split, threshold, objective, recorded_deltas,
            exposure[split], notes[split],
            center_config=payload["training_metadata"].get("loss_config", {}))
        records.extend(current)
    sources = [checkpoint] + [group.source for groups in splits.values() for group in groups]
    checkpoint_metadata = {key: value for key, value in payload["training_metadata"].items() if key != "history"}
    write_json(destination / "config.json", {**config, "sign_threshold_w": threshold,
               "valid_fixed_q_only": False, "objective": objective,
               "evaluation_notes": notes, "checkpoint": str(checkpoint),
               "model_config": payload["model_config"]})
    write_json(destination / "metrics.json", {"checkpoint": str(checkpoint), "splits": metrics,
               "excluded": {split: [] for split in splits}, "evaluation_notes": notes,
               "training_exposure": exposure, "checkpoint_training_metadata": checkpoint_metadata,
               "source_sha256": fingerprints(sources),
               "note": "Initial all-row splits preserve production training rows, including invalid q; predictions call the raw network without runtime beam-trace zero masking. Fixed-q subsets overlap the all-row splits. Final own-feedback accuracy measures replay fit after feedback, not independent generalization."})
    write_json(destination / "costs.json", costs)
    write_csv(destination / "predictions.csv", records)
    return metrics, records


def _flat_summary(kind, run, checkpoint, split, metrics, note):
    return {"comparison": kind, "run": run, "checkpoint": checkpoint, "split": split,
            **{key: metrics[key] for key in (
                "groups", "rows", "power_mae", "nominal_power_mae", "delta_mae",
                "sign_count", "sign_correct", "sign_accuracy", "perturbation_cost_mae",
                "full_cost_mae", "predicted_full_cost_mean", "true_full_cost_mean")},
            "interpretation": note}


def _plots(output, online_rounds, model_records, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(runs), figsize=(12, 7), squeeze=False, sharex="col")
    for column, run in enumerate(runs):
        rows = [row for row in online_rounds if row["run"] == run["number"]]
        rounds = [row["round"] for row in rows]
        axes[0, column].plot(rounds, [row["predicted_total_cost"] for row in rows], label="MLP saved winning cost")
        axes[0, column].plot(rounds, [row["true_total_cost"] for row in rows], label="Finesse at same point")
        axes[0, column].set_yscale("log")
        axes[0, column].set_title(f"Run {run['number']}: original online evaluations")
        axes[0, column].set_ylabel("Full cost (log scale)")
        axes[0, column].legend(fontsize=8)
        axes[1, column].plot(rounds, [row["predicted_nominal_power"] for row in rows], label="MLP nominal power")
        axes[1, column].plot(rounds, [row["nominal_power"] for row in rows], label="Finesse nominal power")
        axes[1, column].set_ylabel("Nominal power (W)")
        axes[1, column].set_xlabel("Feedback round")
        axes[1, column].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "online_cost_and_power.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(runs), figsize=(12, 4.5), squeeze=False)
    for column, run in enumerate(runs):
        groups = load_groups({"data_dir": str(run["root"]), "detector": run["metadata"]["settings"]["objective"]["detector"]}, {"base_ids": "all"})
        xy = np.array([group.x[0, :2].numpy() for group in groups])
        axes[0, column].scatter(xy[:, 0], xy[:, 1], marker=".", alpha=0.4, color="gray", label="100 initial nominal designs")
        rows = [row for row in online_rounds if row["run"] == run["number"]]
        points = axes[0, column].scatter([row["itm_roc"] for row in rows], [row["etm_roc"] for row in rows],
                                       c=[row["round"] for row in rows], cmap="viridis", edgecolors="black", s=40)
        axes[0, column].set_title(f"Run {run['number']}: nominal geometries")
        axes[0, column].set_xlabel("ITM ROC (m)")
        axes[0, column].set_ylabel("ETM ROC (m)")
        axes[0, column].legend(fontsize=8)
        fig.colorbar(points, ax=axes[0, column], label="Feedback round")
    fig.tight_layout()
    fig.savefig(output / "online_roc_coverage.png", dpi=180)
    plt.close(fig)

    labels = list(model_records)
    positions = np.arange(4)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.18
    for index, label in enumerate(labels):
        records = [row for row in model_records[label] if row["split"] == "feedback"]
        errors = [_mean([abs(row["delta_error"]) for row in records if row["direction"] == direction]) for direction in DIRECTIONS[1:]]
        ax.bar(positions + width * (index - (len(labels)-1)/2), errors, width=width,
               label=label.replace("_on_", " on ").replace("_", " "))
    ax.set_xticks(positions, DIRECTIONS[1:])
    ax.set_ylabel("Mean absolute signed ΔPower error (W)")
    ax.set_title(f"Identical saved run {runs[-1]['number']} feedback geometries for all checkpoints")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "same_geometry_delta_errors.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=3, help="Target MLP run, default 3")
    parser.add_argument("--reference-run", type=int, default=2, help="Reference MLP run, default 2")
    parser.add_argument("--output", required=True, help="New diagnostic directory; existing directories are rejected")
    parser.add_argument("--sign-threshold", type=float, default=0.1)
    args = parser.parse_args()
    if args.run < 1 or args.reference_run < 1 or args.run == args.reference_run:
        raise ValueError("Choose two distinct positive run numbers.")
    if not np.isfinite(args.sign_threshold) or args.sign_threshold < 0:
        raise ValueError("Sign threshold must be finite and nonnegative.")
    torch.set_num_threads(1)
    reference, target = _read_run(args.reference_run), _read_run(args.run)
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    online_samples, online_rounds, online_summaries, summary = [], [], {}, []
    for run in (reference, target):
        samples, rounds, metrics = _online(run, args.sign_threshold)
        online_samples.extend(samples)
        online_rounds.extend(rounds)
        online_summaries[f"run{run['number']}"] = metrics
        summary.append(_flat_summary("original_online_own_selected_points", run["number"],
                       "base for round1; roundN-1 for roundN", "online", metrics, metrics["note"]))
    write_csv(output / "online_samples.csv", online_samples)
    write_csv(output / "online_rounds.csv", online_rounds)

    count = len(target["result"]["data"])
    selection = {"rounds": list(range(1, count + 1))}
    config = {"data_dir": str(target["root"]), "detector": target["metadata"]["settings"]["objective"]["detector"],
              "splits": {"feedback": selection}}
    groups = load_groups(config, selection)
    if len(groups) != count:
        raise ValueError("Expected one saved five-row group for every target feedback round.")
    recorded_deltas = {f"round:{i}:0": tuple(record["delta"])
                       for i, record in enumerate(target["result"]["evaluations_finesse"], 1)}
    # Verify data used for common-geometry evaluation matches original recorded labels.
    for group, record in zip(groups, target["result"]["evaluations_finesse"]):
        if not np.allclose(group.y.double().expm1().flatten().numpy(),
                           [record["powers"][key] for key in DIRECTIONS], rtol=1e-10, atol=1e-8):
            raise ValueError("Saved feedback data and original Finesse records disagree.")
    checkpoints, shared_metrics, initial_metrics, model_records = [], {}, {}, {}
    for run in (reference, target):
        base = repo_path(run["metadata"]["files"]["base_checkpoint"])
        last = len(run["result"]["data"])
        template = run["metadata"]["files"]["finetune_checkpoint_template"]
        final = repo_path(template.format(last))
        for stage, checkpoint in (("base", base), (f"round{last}", final)):
            label = f"run{run['number']}_{stage}_on_run{target['number']}_feedback"
            exposure = ("All target feedback labels were replayed during training; measures fit, not generalization."
                        if run["number"] == target["number"] and stage != "base" else
                        "Target feedback labels were not used to train this checkpoint; target geometry selection can still depend on its own models.")
            note = "Controlled checkpoint comparison at identical target feedback geometries. " + exposure
            shared_metrics[label], model_records[label] = _write_evaluation(
                output, label, run, checkpoint, {"feedback": groups}, config,
                args.sign_threshold, _objective(target), {"feedback": note},
                {"feedback": exposure}, recorded_deltas)
            checkpoints.append(checkpoint)
            summary.append(_flat_summary("same_target_feedback_geometries", run["number"], stage,
                           "feedback", shared_metrics[label]["feedback"], note))
        model, payload = load_checkpoint(base, device="cpu")
        initial_config, splits = _base_split_groups(run, payload)
        label = f"run{run['number']}_base_on_initial"
        initial_config["splits"] = {name: {"base_ids": [int(group.group_id.split(":")[1]) for group in current]}
                                    for name, current in splits.items()}
        notes, exposure = {}, {}
        for name in splits:
            training = name.startswith("training")
            notes[name] = ("Training fit, including all rows production trained." if training else
                           "Whole-design validation labels were held out from weight updates but used for checkpoint selection; not an independent test.")
            if name.endswith("valid_fixed_q"):
                notes[name] += " Only valid fixed-q groups; overlaps corresponding all-row split."
            exposure[name] = "used for weight updates" if training else "used for checkpoint selection only"
        initial_metrics[label], _ = _write_evaluation(
            output, label, run, base, splits, initial_config, args.sign_threshold,
            _objective(run), notes, exposure)
        for split, metrics in initial_metrics[label].items():
            summary.append(_flat_summary("own_initial_checkpoint_split", run["number"], "base",
                           split, metrics, notes[split]))

    write_csv(output / "summary.csv", summary)
    source_files = [Path(__file__), REPO_ROOT / "analysis/surrogate_diagnostics/dataset.py",
                    REPO_ROOT / "MLP/checkpoint.py", REPO_ROOT / "MLP/model.py",
                    REPO_ROOT / "MLP/data.py", REPO_ROOT / "MLP/training.py",
                    REPO_ROOT / "MLP/losses.py", REPO_ROOT / "MLP/finetune.py",
                    REPO_ROOT / "utils/calc.py", REPO_ROOT / "utils/power_results.py"]
    for run in (reference, target):
        source_files.extend([run["metadata_path"], run["result_path"],
                             repo_path(run["metadata"]["files"]["base_data"])])
    source_files.extend(checkpoints)
    source_files.extend(group.source for group in groups)
    report = {"target_run": target["number"], "reference_run": reference["number"],
              "sign_threshold_w": args.sign_threshold, "torch_threads": 1,
              "online": online_summaries, "same_geometry": shared_metrics,
              "own_initial": initial_metrics,
              "run_metadata": {f"run{run['number']}": run["metadata"] for run in (reference, target)},
              "source_sha256": fingerprints(source_files),
              "notes": ["No Finesse optical power solve, model training, or production output mutation was performed.",
                        "Online metrics compare original predictions before feedback; each run samples different selected points.",
                        "All common-geometry checkpoint metrics use target-run saved feedback rows and target-run objective weights.",
                        "Final target checkpoint has seen all target feedback labels, so its accuracy is replay fit.",
                        "Each own-initial set preserves its checkpoint's split and all production training rows. Valid-q subsets overlap the corresponding full split.",
                        "Initial costs use ROC differences reconstructed from float32 inputs; feedback costs use exact saved original ROC steps.",
                        "Full online objective costs were reconstructed from saved components and checked against original costs with rtol=atol=1e-6."]}
    write_json(output / "comparison.json", report)
    write_json(output / "config.json", {"target_run": target["number"], "reference_run": reference["number"],
               "sign_threshold_w": args.sign_threshold, "output": str(output),
               "command": f"/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.surrogate_diagnostics.compare_runs --run {target['number']} --reference-run {reference['number']} --output {output}"})
    _plots(output, online_rounds, model_records, (reference, target))
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
