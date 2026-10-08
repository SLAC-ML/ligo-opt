"""Power and signed-difference metrics, with optional standalone checkpoint evaluation."""

import argparse
import json
import numpy as np
import torch

from MLP.checkpoint import load_checkpoint
from .dataset import DIRECTIONS, load_splits, repo_path, write_csv, write_json


@torch.no_grad()
def predictions(model, groups, split, sign_threshold=0.1):
    device = next(model.parameters()).device
    model.eval()
    records, costs = [], []
    for group in groups:
        predicted = model(group.x.to(device)).double().expm1().flatten().cpu().numpy()
        if not np.isfinite(predicted).all():
            raise FloatingPointError("Nonfinite predicted powers.")
        true = group.y.expm1().flatten().numpy()
        for i, record in enumerate(group.records()):
            delta = predicted[i] - predicted[0]
            records.append({"split": split, **record, "predicted_power": float(predicted[i]),
                            "power_error": float(predicted[i] - true[i]),
                            "predicted_delta_power": float(delta),
                            "delta_error": float(delta - (true[i] - true[0]))})
        if group.directions == DIRECTIONS:
            # Matches calc_gain_cost's sum of four |delta power / delta ROC|.
            # No call to calc_cost: it writes production CSV logs as a side effect.
            steps = (group.x[1:, :2] - group.x[0, :2]).abs().sum(1).double().numpy()
            valid = true[0] > 0 and predicted[0] > 0
            true_cost = float((np.abs(true[1:] - true[0]) / steps).sum()) if valid else None
            pred_cost = float((np.abs(predicted[1:] - predicted[0]) / steps).sum()) if valid else None
            costs.append({"group_id": group.group_id, "true_perturbation_cost": true_cost,
                          "predicted_perturbation_cost": pred_cost})
    errors = np.array([r["power_error"] for r in records])
    paired = [r for r in records if r["direction"] != "nominal"]

    def delta_metrics(rows):
        error = np.array([r["delta_error"] for r in rows])
        significant = [r for r in rows if abs(r["delta_power"]) > sign_threshold]
        return {"count": len(rows), "delta_mae": float(np.abs(error).mean()),
                "delta_max_abs_error": float(np.abs(error).max()),
                "sign_count": len(significant),
                "sign_accuracy": float(np.mean([
                    np.sign(r["predicted_delta_power"]) == np.sign(r["delta_power"])
                    for r in significant])) if significant else None}

    valid_costs = [r for r in costs if r["true_perturbation_cost"] is not None]
    metrics = {"groups": len(groups), "rows": len(records),
               "power_mae": float(np.abs(errors).mean()),
               "power_max_abs_error": float(np.abs(errors).max()),
               **delta_metrics(paired), "sign_threshold_w": sign_threshold,
               "by_direction": {direction: delta_metrics([r for r in paired if r["direction"] == direction])
                                for direction in sorted({r["direction"] for r in paired})},
               "valid_cost_groups": len(valid_costs), "invalid_cost_groups": len(costs) - len(valid_costs),
               "perturbation_cost_mae": float(np.mean([
                   abs(r["predicted_perturbation_cost"] - r["true_perturbation_cost"])
                   for r in valid_costs])) if valid_costs else None}
    return records, metrics, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(repo_path(args.config).read_text())
    splits, excluded = load_splits(config)
    model, payload = load_checkpoint(repo_path(args.checkpoint), device="cpu")
    if payload["detector"] != config["detector"]:
        raise ValueError("Checkpoint detector differs from the experiment detector.")
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows, metrics, costs = [], {}, {}
    for split, groups in splits.items():
        current, metrics[split], costs[split] = predictions(model, groups, split, config["sign_threshold_w"])
        rows.extend(current)
    write_json(output / "config.json", config)
    write_json(output / "metrics.json", {"splits": metrics, "excluded": excluded,
                                        "checkpoint": str(repo_path(args.checkpoint)),
                                        "note": "Independent accuracy requires a checkpoint that has not trained on these groups."})
    write_json(output / "costs.json", costs)
    write_csv(output / "predictions.csv", rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
