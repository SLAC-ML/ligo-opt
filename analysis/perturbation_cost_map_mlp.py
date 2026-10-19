"""Generate a resumable 50x50 MLP cost map for subspace_plot.ipynb.

Run from the repository root:
    /home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.perturbation_cost_map_mlp

Use --model-path to select another MLP checkpoint, --device cpu to use the CPU,
or --output to save a separate map. Rerun to resume; --overwrite recomputes it.
In full mode, cost details are recorded beside the map in <map-name>_records.csv,
instead of the shared full_cost_records.csv. Resuming appends new evaluations;
--overwrite also resets this map's cost records.
The objective follows COST_MODE and the weights in utils/calc.py, just like
the Finesse script. Each point predicts nominal power and four perturbations with fixed
nominal q. No full Finesse power simulations are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]

# Checkpoint selected in the notebook; edit here or pass --model-path.
# MLP_CHECKPOINT_PATH = REPO_ROOT / "MLP/models/run1/power_predictor_ligoParams_finetuned_round40.pt"
MLP_CHECKPOINT_PATH = REPO_ROOT / "MLP/models/run3/power_predictor_ligoParams_finetuned_round40.pt"

ITM_ROC_NOMINAL = -1934.0
ETM_ROC_NOMINAL = 2245.0
GRID_SIZE = 50
ITM_ROC_LIST = np.linspace(ITM_ROC_NOMINAL - 2200.0, ITM_ROC_NOMINAL + 500.0, GRID_SIZE)
ETM_ROC_LIST = np.linspace(ETM_ROC_NOMINAL - 500.0, ETM_ROC_NOMINAL + 2000.0, GRID_SIZE)

from utils.detectors import POWER_DETECTOR

DEFAULT_PD_NAME = POWER_DETECTOR  # Reject a checkpoint for another default objective.
OUTPUT_DIRECTORY = REPO_ROOT / "MLP/data/subspace"
DEFAULT_SAVE_EVERY = 10


def full_mlp_cost(itm_roc, etm_roc, pd_name, predictor, *, full_cost_csv=None):
    """Compute the COST_MODE objective from nominal and four perturbed geometries."""
    from GNN.GNN_utils import kat_manipulation
    from perturbation import calc_perturbed_params
    from utils.finesse_base import base_kat
    from utils import calc

    if calc.COST_MODE not in {"design_power", "perturbation", "full"}:
        raise ValueError(f"Unknown COST_MODE in utils/calc.py: {calc.COST_MODE!r}")

    nominal_kat = kat_manipulation(
        itm_roc, etm_roc, base_kat, include_aperture_maps=False
    )
    d_names, d_powers, nominal_q = predictor.run_batch([nominal_kat])[0]
    (itm_pos, itm_neg), (etm_pos, etm_neg), delta = calc_perturbed_params(
        itm_roc, etm_roc
    )
    perturbed_kats = [
        kat_manipulation(itm, etm, base_kat, nominal_q_value=nominal_q,
                         include_aperture_maps=False)
        for itm, etm in [(itm_pos, etm_roc), (itm_neg, etm_roc),
                         (itm_roc, etm_pos), (itm_roc, etm_neg)]
    ]
    itm_pos_out, itm_neg_out, etm_pos_out, etm_neg_out = predictor.run_batch(perturbed_kats)
    return calc.calc_cost(
        None,
        [itm_pos_out[1], d_powers, itm_neg_out[1]],
        [etm_pos_out[1], d_powers, etm_neg_out[1]],
        [itm_pos_out[0], d_names, itm_neg_out[0]],
        [etm_pos_out[0], d_names, etm_neg_out[0]],
        delta, pd_name,
        full_cost_csv=full_cost_csv,
    )


def _cost_settings():
    """Identify objective settings so resumed maps cannot mix different costs."""
    from utils import calc
    from perturbation import roc_perturbation_percent

    settings = {
        "cost_mode": calc.COST_MODE,
        "roc_perturbation_percent": roc_perturbation_percent,
    }
    if calc.COST_MODE == "full":
        settings.update(
            full_power_weight=calc.FULL_POWER_WEIGHT,
            full_perturbation_weight=calc.FULL_PERTURBATION_WEIGHT,
        )
    return json.dumps(settings, sort_keys=True)


def _save_checkpoint(
    output_path: Path,
    cost_grid: np.ndarray,
    completed_grid: np.ndarray,
    *,
    pd_name: str,
    model_metadata: dict,
) -> None:
    """Atomically save map progress and metadata."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez(
        temporary_path,
        ITM_ROC_List=ITM_ROC_LIST,
        ETM_ROC_List=ETM_ROC_LIST,
        pd_name=pd_name,
        cost_settings=_cost_settings(),
        **model_metadata,
        cost_grid=cost_grid,
        completed_grid=completed_grid,
    )
    temporary_path.replace(output_path)


def _load_or_initialize(output_path: Path, pd_name: str, overwrite: bool, model_metadata: dict):
    """Load a compatible checkpoint or create empty 50x50 result arrays."""
    output_path = output_path.expanduser().resolve()

    shape = (GRID_SIZE, GRID_SIZE)
    if overwrite or not output_path.exists():
        return np.full(shape, np.nan, dtype=float), np.zeros(shape, dtype=bool)

    with np.load(output_path) as saved:
        if ("mlp_checkpoint_sha256" not in saved
                or str(saved["mlp_checkpoint_sha256"]) != model_metadata["mlp_checkpoint_sha256"]):
            raise ValueError("Checkpoint uses a different MLP model. Choose a new --output or use --overwrite.")
        if "cost_settings" not in saved or str(saved["cost_settings"]) != _cost_settings():
            raise ValueError(
                f"Checkpoint cost settings are missing or differ from utils/calc.py: {output_path}. "
                "Choose a new --output or use --overwrite to recompute."
            )
        saved_itm = saved["ITM_ROC_List"]
        saved_etm = saved["ETM_ROC_List"]
        saved_pd_name = str(saved["pd_name"])
        cost_grid = saved["cost_grid"].copy()
        completed_grid = saved["completed_grid"].astype(bool, copy=True)

    if not np.array_equal(saved_itm, ITM_ROC_LIST):
        raise ValueError(f"Checkpoint ITM grid does not match the required 50x50 grid: {output_path}")
    if not np.array_equal(saved_etm, ETM_ROC_LIST):
        raise ValueError(f"Checkpoint ETM grid does not match the required 50x50 grid: {output_path}")
    if saved_pd_name != pd_name:
        raise ValueError(
            f"Checkpoint detector is {saved_pd_name!r}, not requested {pd_name!r}: {output_path}"
        )
    if cost_grid.shape != shape or completed_grid.shape != shape:
        raise ValueError(f"Checkpoint arrays are not 50x50: {output_path}")
    return cost_grid, completed_grid


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MLP_CHECKPOINT_PATH)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="result .npz file (default: MLP/data/subspace/<COST_MODE>_cost_map_mlp_50x50.npz)",
    )
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, cuda:0 (default: automatic)")
    parser.add_argument("--pd-name", default=DEFAULT_PD_NAME)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    if args.output is not None and args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    args.model_path = args.model_path.expanduser().resolve()
    if not args.model_path.is_file():
        parser.error(f"MLP checkpoint not found: {args.model_path}. Set MLP_CHECKPOINT_PATH or --model-path.")
    return args


def main(argv=None):
    args = parse_args(argv)
    from utils import calc
    from MLP.predictor import MLPPowerPredictor

    if calc.COST_MODE not in {"design_power", "perturbation", "full"}:
        raise ValueError(f"Unknown COST_MODE in utils/calc.py: {calc.COST_MODE!r}")
    predictor = MLPPowerPredictor(str(args.model_path), device=args.device)
    if args.pd_name is None:
        args.pd_name = predictor.detector
    elif args.pd_name != predictor.detector:
        raise ValueError(
            f"Requested detector {args.pd_name!r} differs from checkpoint detector {predictor.detector!r}."
        )
    model_metadata = {
        "mlp_checkpoint_path": str(args.model_path),
        "mlp_checkpoint_sha256": hashlib.sha256(args.model_path.read_bytes()).hexdigest(),
    }
    output_path = (
        args.output or OUTPUT_DIRECTORY / f"{calc.COST_MODE}_cost_map_mlp_50x50.npz"
    ).expanduser().resolve()
    if output_path == args.model_path:
        raise ValueError("The result path must differ from the model checkpoint path.")
    full_cost_csv = output_path.with_name(f"{output_path.stem}_records.csv").resolve()
    record_full_cost = calc.COST_MODE == "full" and calc.RECORD_FULL_COST
    if record_full_cost and full_cost_csv == Path(calc.FULL_COST_CSV).expanduser().resolve():
        raise ValueError(
            "The map's cost-record path would use the shared FULL_COST_CSV. "
            "Choose a different --output."
        )
    cost_grid, completed_grid = _load_or_initialize(
        output_path, args.pd_name, args.overwrite, model_metadata
    )
    if args.overwrite and record_full_cost:
        full_cost_csv.unlink(missing_ok=True)
    if record_full_cost:
        print(f"Cost records: {full_cost_csv}")
    tasks = np.argwhere(~completed_grid)
    if len(tasks):
        print(f"Cost mode: {calc.COST_MODE}")
        if calc.COST_MODE == "full":
            print(f"Full cost: {calc.FULL_POWER_WEIGHT} * inverse nominal power + "
                  f"{calc.FULL_PERTURBATION_WEIGHT} * perturbation cost")
        print(f"Model: {args.model_path}")
        print(f"Resuming {completed_grid.sum()}/{completed_grid.size} completed points.")
        completed_since_save = 0
        try:
            with tqdm(total=len(tasks), desc=f"MLP {calc.COST_MODE} cost map") as pbar:
                for i, j in tasks:
                    cost_grid[i, j] = full_mlp_cost(
                        float(ITM_ROC_LIST[i]), float(ETM_ROC_LIST[j]), args.pd_name, predictor,
                        full_cost_csv=full_cost_csv,
                    )
                    completed_grid[i, j] = True
                    completed_since_save += 1
                    pbar.update(1)
                    if completed_since_save >= args.save_every:
                        _save_checkpoint(output_path, cost_grid, completed_grid,
                                         pd_name=args.pd_name, model_metadata=model_metadata)
                        completed_since_save = 0
        finally:
            _save_checkpoint(output_path, cost_grid, completed_grid,
                             pd_name=args.pd_name, model_metadata=model_metadata)
    print(f"Saved {completed_grid.sum()}/{completed_grid.size} completed points: {output_path}")
    finite = np.isfinite(cost_grid)
    if finite.any():
        best = np.unravel_index(np.argmin(np.where(finite, cost_grid, np.inf)), cost_grid.shape)
        print(f"Minimum finite {calc.COST_MODE} cost: {cost_grid[best]}")
        print(f"ITM ROC, ETM ROC: {ITM_ROC_LIST[best[0]]}, {ETM_ROC_LIST[best[1]]}")
    else:
        print("No finite costs found in the completed map.")
    return output_path


if __name__ == "__main__":
    main()
