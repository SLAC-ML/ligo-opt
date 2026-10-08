"""Compare losses with identical initialization; also run grouped generalization experiments."""

import argparse
import copy
import hashlib
import json
import math
import platform

import torch
from torch.nn import functional as F

from MLP.checkpoint import load_checkpoint, save_checkpoint
from MLP.losses import PowerPredictionLoss
from MLP.model import build_power_predictor
from .dataset import fingerprints, load_splits, pack, repo_path, write_csv, write_json
from .evaluate import predictions


def paired_loss(log_prediction, log_target, nominal, partners, delta_scale):
    """Huber loss on signed power differences, retaining gradients to both rows."""
    predicted = log_prediction.double().expm1().flatten()
    target = log_target.double().expm1().flatten()
    error = (predicted[partners] - predicted[nominal]) - (target[partners] - target[nominal])
    return F.smooth_l1_loss(error / delta_scale, torch.zeros_like(error))


def model_fingerprint(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def run_experiment(config, output):
    mode = config["mode"]
    if mode not in ("tiny_fit", "generalization"):
        raise ValueError("mode must be tiny_fit or generalization.")
    if mode == "generalization" and config.get("initial_checkpoint"):
        raise ValueError("Generalization requires fresh weights to avoid pretrained holdout leakage.")
    required = {"train"} if mode == "tiny_fit" else {"train", "validation", "test"}
    if set(config["splits"]) != required:
        raise ValueError(f"Expected splits: {sorted(required)}")
    if config["normalization"] not in ("training", "checkpoint"):
        raise ValueError("normalization must be training or checkpoint.")
    if config["normalization"] == "checkpoint" and not config.get("initial_checkpoint"):
        raise ValueError("Checkpoint normalization requires initial_checkpoint.")
    for key in ("epochs", "record_every", "threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("learning_rate", "delta_scale_floor_w", "power_scale_floor_w"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    for key in ("weight_decay", "delta_weight", "sign_threshold_w"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    for key in ("seeds", "losses"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must be nonempty and unique.")
    if not set(config["losses"]) <= {"center_focused", "log_smooth_l1", "paired"}:
        raise ValueError("Unknown loss variant.")
    torch.set_num_threads(config["threads"])
    splits, excluded = load_splits(config)
    x, y, nominal, partners = pack(splits["train"])
    powers = y.expm1().flatten()
    delta_scale = max(config["delta_scale_floor_w"],
                      float((powers[partners] - powers[nominal]).square().mean().sqrt()))
    power_scale = max(config["power_scale_floor_w"], float(powers.square().mean().sqrt()))
    monitor_split = "train" if mode == "tiny_fit" else "validation"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    sources = [g.source for groups in splits.values() for g in groups]
    if config.get("initial_checkpoint"):
        sources.append(repo_path(config["initial_checkpoint"]))
    write_json(output / "provenance.json", {
        "sha256": fingerprints(sources), "excluded_invalid_fixed_q_groups": excluded,
        "groups": {name: [g.group_id for g in groups] for name, groups in splits.items()},
        "delta_scale_w": delta_scale, "power_scale_w": power_scale,
        "monitor_split": monitor_split,
        "selection_score": "power_mae / training_power_rms + delta_mae / training_delta_rms (floored scales)",
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "batching": "full batch, complete groups",
    })
    summary = []
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        if config.get("initial_checkpoint"):
            initial, payload = load_checkpoint(repo_path(config["initial_checkpoint"]), device="cpu")
            if payload["detector"] != config["detector"]:
                raise ValueError("Initial checkpoint detector differs from requested detector.")
        else:
            initial = build_power_predictor(config["model_config"])
        if config["normalization"] == "training":
            initial.set_normalization(x)
        initial_hash = model_fingerprint(initial)
        for variant in config["losses"]:
            model = copy.deepcopy(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])
            criterion = PowerPredictionLoss({**config["center_loss"],
                                             "mode": "center_focused" if variant == "center_focused" else "log_smooth_l1"})
            directory = output / f"{variant}_seed{seed}"
            directory.mkdir()
            resolved = {**config, "variant": variant, "seed": seed,
                        "initial_state_sha256": initial_hash, "delta_scale_w": delta_scale,
                        "power_scale_w": power_scale}
            write_json(directory / "config.json", resolved)
            best_score, best_epoch, best_state = math.inf, None, None
            history = []
            for epoch in range(config["epochs"] + 1):
                if epoch:
                    model.train()
                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y).mean()
                    if variant == "paired":
                        loss = loss + config["delta_weight"] * paired_loss(out, y, nominal, partners, delta_scale)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss: {variant}, epoch {epoch}")
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError(f"Nonfinite gradient: {variant}, epoch {epoch}")
                    optimizer.step()
                if epoch % config["record_every"] == 0 or epoch == config["epochs"]:
                    _, train_metrics, _ = predictions(model, splits["train"], "train", config["sign_threshold_w"])
                    if monitor_split == "train":
                        monitor = train_metrics
                    else:
                        _, monitor, _ = predictions(model, splits[monitor_split], monitor_split,
                                                     config["sign_threshold_w"])
                    score = monitor["power_mae"] / power_scale + monitor["delta_mae"] / delta_scale
                    with torch.no_grad():
                        out = model(x)
                        absolute = criterion(out, y).mean().item()
                        paired = paired_loss(out, y, nominal, partners, delta_scale).item()
                    history.append({"epoch": epoch, "absolute_loss": absolute, "paired_loss": paired,
                                    "training_loss": absolute + (config["delta_weight"] * paired if variant == "paired" else 0),
                                    "train_power_mae": train_metrics["power_mae"],
                                    "train_delta_mae": train_metrics["delta_mae"],
                                    "monitor_power_mae": monitor["power_mae"],
                                    "monitor_delta_mae": monitor["delta_mae"], "selection_score": score})
                    if score < best_score:
                        best_score, best_epoch = score, epoch
                        best_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            save_checkpoint(directory / "checkpoint.pt", model, config["detector"], {
                "diagnostic_only": True, "mode": mode, "variant": variant,
                "best_epoch": best_epoch, "monitor_split": monitor_split,
                "initial_state_sha256": initial_hash,
            })
            rows, metrics, costs = [], {}, {}
            for split, groups in splits.items():
                current, metrics[split], costs[split] = predictions(model, groups, split, config["sign_threshold_w"])
                rows.extend(current)
                summary.append({"variant": variant, "seed": seed, "split": split,
                                "best_epoch": best_epoch,
                                **{key: value for key, value in metrics[split].items() if not isinstance(value, dict)}})
            write_json(directory / "metrics.json", {"mode": mode, "best_epoch": best_epoch,
                       "monitor_split": monitor_split, "splits": metrics,
                       "interpretation": "Training-set diagnostic only." if mode == "tiny_fit" else
                       "Test used only after checkpoint selection; repeated tuning on it invalidates independence."})
            write_json(directory / "costs.json", costs)
            write_csv(directory / "predictions.csv", rows)
            write_csv(directory / "training_history.csv", history)
            result = metrics[monitor_split]
            print(f"{variant}, seed {seed}, best epoch {best_epoch}: "
                  f"{monitor_split} power MAE={result['power_mae']:.5g} W; "
                  f"delta MAE={result['delta_mae']:.5g} W", flush=True)
    write_csv(output / "summary.csv", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="analysis/surrogate_diagnostics/configs/tiny_fit.json")
    parser.add_argument("--output", required=True, help="New directory; existing results are never overwritten.")
    parser.add_argument("--rounds", nargs="+", type=int, help="Tiny-fit only: replace selection with these feedback rounds.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    config = json.loads(repo_path(args.config).read_text())
    if args.rounds is not None:
        if config["mode"] != "tiny_fit":
            raise ValueError("--rounds only applies to tiny-fit diagnostics.")
        config["splits"] = {"train": {"rounds": args.rounds}}
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seeds is not None:
        config["seeds"] = args.seeds
    run_experiment(config, repo_path(args.output))


if __name__ == "__main__":
    main()
