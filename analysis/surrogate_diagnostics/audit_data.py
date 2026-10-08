"""Audit saved labels, perturbation coverage, and optional inference feature consistency."""

import argparse
from collections import Counter
import json

import torch

from .dataset import fingerprints, load_groups, repo_path, write_csv, write_json


def check_inference(groups):
    # Imported only on request. This uses beam tracing, never a power solve.
    from GNN.GNN_utils import kat_manipulation
    from MLP.data import kat_to_features
    from MLP.predictor import MLPPowerPredictor
    from utils.finesse_base import gnn_base_kat

    rows = []
    for group in groups:
        if not group.valid_fixed_q:
            continue
        nominal_q = None
        for i, x in enumerate(group.x):
            kat = kat_manipulation(float(x[0]), float(x[1]), gnn_base_kat,
                                   nominal_q_value=nominal_q, include_aperture_maps=False)
            names, values = MLPPowerPredictor._beam_q_values(kat)
            q = values[names.index("q_ITM_p1_i")]
            if i == 0:
                nominal_q = q
            inferred = kat_to_features(kat, q)
            rows.append({"group_id": group.group_id, "h5_key": group.keys[i],
                         "max_feature_abs_error": float((inferred - x).abs().max()),
                         "q_real_error": float(inferred[2] - x[2]),
                         "q_imag_error": float(inferred[3] - x[3]),
                         "matches_tolerance": bool(torch.allclose(inferred, x, rtol=2e-4, atol=0.05)),
                         "trace_valid": q != 0})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="analysis/surrogate_diagnostics/configs/tiny_fit.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-data", action="store_true", help="Audit all base pairs and feedback rounds in data_dir.")
    parser.add_argument("--check-inference", action="store_true", help="Rebuild features using inexpensive Finesse beam tracing.")
    args = parser.parse_args()
    config = json.loads(repo_path(args.config).read_text())
    if args.all_data:
        rounds = sorted(int(p.stem[5:]) for p in repo_path(config["data_dir"]).glob("round*.h5")
                        if p.stem[5:].isdigit())
        selections = {"all": {"base_ids": "all", "rounds": rounds}}
    else:
        selections = config["splits"]
    all_groups = []
    records = []
    for split, selection in selections.items():
        groups = load_groups(config, selection)
        all_groups.extend(groups)
        records.extend({"split": split, **record} for group in groups for record in group.records())
    x = torch.cat([g.x for g in all_groups]).double()
    y = torch.cat([g.y for g in all_groups]).expm1().flatten()
    deltas = torch.tensor([r["delta_power"] for r in records if r["direction"] != "nominal"], dtype=torch.float64)
    inference = check_inference(all_groups) if args.check_inference else None
    center = config["center_loss"]
    summary = {
        "groups": len(all_groups), "rows": len(records),
        "valid_fixed_q_groups": sum(g.valid_fixed_q for g in all_groups),
        "zero_output_rows": int((y == 0).sum()),
        "center_rows": int(((y - center["center_power"]).abs() <= center["center_width"]).sum()),
        "power_range_w": [float(y.min()), float(y.max())],
        "delta_range_w": [float(deltas.min()), float(deltas.max())],
        "directions": dict(Counter(r["direction"] for r in records)),
        "feature_min": x.min(0).values.tolist(), "feature_max": x.max(0).values.tolist(),
        "feature_std": x.std(0, unbiased=False).tolist(),
        "source_sha256": fingerprints([g.source for g in all_groups]),
        "inference_check": None if inference is None else {
            "rows": len(inference), "mismatches": sum(not r["matches_tolerance"] for r in inference),
            "note": "ROCs reconstructed from float32 training features; tolerance rtol=2e-4, atol=0.05. Inspect discrepancies near stability boundaries.",
        },
        "notes": ["Zero labels cannot distinguish instability from solver/lock failure in these saved artifacts.",
                  "A fixed-q ROC perturbation combines geometry and mismatch effects; it is not a pure mismatch label.",
                  "Coverage statistics include zero-output rows; the trainer defaults to valid fixed-q groups only."],
    }
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "audit.json", summary)
    write_csv(output / "samples.csv", records)
    if inference is not None:
        write_csv(output / "inference_features.csv", inference)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
