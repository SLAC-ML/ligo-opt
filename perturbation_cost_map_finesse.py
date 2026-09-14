"""Compute a 50x50 Finesse cost map using COST_MODE in utils/calc.py.

Each grid point performs the same five simulations as the perturbed path in
PSOoptimizer.py: nominal D, ITM +/- 0.33%, and ETM +/- 0.33%. The perturbed
runs use the nominal point's q at ITM.p1.i, and the final value is calculated
with utils.calc.calc_cost, including its full-mode weights and CSV recording.

The parent process owns the result arrays and writes atomic checkpoints, so an
interrupted job can be resumed by running the same command again.

Examples
--------
Run with the default maximum of 60 worker processes::

    python perturbation_cost_map_finesse.py

Use fewer workers::

    python perturbation_cost_map_finesse.py --workers 20

Start over intentionally instead of resuming the existing checkpoint::

    python perturbation_cost_map_finesse.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
import traceback

# Keep every Finesse worker single-threaded. Otherwise numerical libraries may
# create their own thread pools and heavily oversubscribe the machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from tqdm import tqdm


ITM_ROC_NOMINAL = -1934.0
ETM_ROC_NOMINAL = 2245.0
GRID_SIZE = 50
ITM_ROC_LIST = np.linspace(ITM_ROC_NOMINAL - 2200.0, ITM_ROC_NOMINAL + 500.0, GRID_SIZE)
ETM_ROC_LIST = np.linspace(ETM_ROC_NOMINAL - 500.0, ETM_ROC_NOMINAL + 2000.0, GRID_SIZE)

MAX_WORKERS = 60
DEFAULT_PD_NAME = "p_ETM_p1_i"
OUTPUT_DIRECTORY = Path("GNN/data/subspace")
DEFAULT_SAVE_EVERY = 10


def _full_finesse_output(itm_roc: float, etm_roc: float, fixed_q=None):
    """Run one full Finesse simulation and return names, powers, and ITM q."""
    # Imports are local so --help and argument errors do not initialize Finesse.
    # In each persistent worker Python caches these modules after its first task.
    from GNN.GNN_utils import kat_manipulation
    from utils.finesse_base import base_kat
    from utils.sim import finesse_sim

    run_kat = kat_manipulation(
        itm_roc, etm_roc, base_kat, nominal_q_value=fixed_q
    )
    names, powers, q_names, q_values = finesse_sim(run_kat)
    q_at_itm = q_values[q_names.index("q_ITM_p1_i")]
    return names, powers, q_at_itm


def full_finesse_cost(
    itm_roc: float, etm_roc: float, pd_name: str
) -> float:
    """Calculate one grid point using five full Finesse simulations."""
    from perturbation import calc_perturbed_params
    from utils.calc import calc_cost

    itm_perturbed, etm_perturbed, delta = calc_perturbed_params(itm_roc, etm_roc)
    itm_pos, itm_neg = itm_perturbed
    etm_pos, etm_neg = etm_perturbed

    d_names, d_powers, nominal_q = _full_finesse_output(itm_roc, etm_roc)
    itm_pos_names, itm_pos_powers, _ = _full_finesse_output(
        itm_pos, etm_roc, fixed_q=nominal_q
    )
    itm_neg_names, itm_neg_powers, _ = _full_finesse_output(
        itm_neg, etm_roc, fixed_q=nominal_q
    )
    etm_pos_names, etm_pos_powers, _ = _full_finesse_output(
        itm_roc, etm_pos, fixed_q=nominal_q
    )
    etm_neg_names, etm_neg_powers, _ = _full_finesse_output(
        itm_roc, etm_neg, fixed_q=nominal_q
    )

    itm_powers = [itm_pos_powers, d_powers, itm_neg_powers]
    itm_names = [itm_pos_names, d_names, itm_neg_names]
    etm_powers = [etm_pos_powers, d_powers, etm_neg_powers]
    etm_names = [etm_pos_names, d_names, etm_neg_names]
    # calc_cost currently does not use kat; all objective inputs are below.
    return calc_cost(
        None, itm_powers, etm_powers, itm_names, etm_names, delta, pd_name
    )


def full_finesse_perturbation_cost(itm_roc: float, etm_roc: float, pd_name: str) -> float:
    """Backward-compatible name; follows COST_MODE in utils/calc.py."""
    return full_finesse_cost(itm_roc, etm_roc, pd_name)


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


def _evaluate_task(task):
    """Worker entry point; unexpected failures remain resumable in the parent."""
    i, j, itm_roc, etm_roc, pd_name = task
    try:
        cost = full_finesse_cost(itm_roc, etm_roc, pd_name)
        return i, j, float(cost), None
    except BaseException:
        return i, j, np.nan, traceback.format_exc()


def _save_checkpoint(
    output_path: Path,
    cost_grid: np.ndarray,
    completed_grid: np.ndarray,
    *,
    pd_name: str,
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
        cost_grid=cost_grid,
        completed_grid=completed_grid,
    )
    temporary_path.replace(output_path)


def _load_or_initialize(output_path: Path, pd_name: str, overwrite: bool):
    """Load a compatible checkpoint or create empty 50x50 result arrays."""
    output_path = output_path.expanduser().resolve()
    if overwrite and output_path.exists():
        output_path.unlink()

    shape = (GRID_SIZE, GRID_SIZE)
    if not output_path.exists():
        return np.full(shape, np.nan, dtype=float), np.zeros(shape, dtype=bool)

    with np.load(output_path) as saved:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"worker processes (default and hard maximum: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="checkpoint/result .npz file (default: GNN/data/subspace/<COST_MODE>_cost_map_finesse_50x50.npz)",
    )
    parser.add_argument(
        "--pd-name",
        default=DEFAULT_PD_NAME,
        help=f"power detector used by calc_cost (default: {DEFAULT_PD_NAME})",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help=f"checkpoint after this many completed points (default: {DEFAULT_SAVE_EVERY})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="discard an existing compatible checkpoint and start from zero",
    )
    args = parser.parse_args(argv)

    available_cpus = os.cpu_count() or 1
    worker_limit = min(MAX_WORKERS, available_cpus)
    if not 1 <= args.workers <= worker_limit:
        parser.error(f"--workers must be between 1 and {worker_limit} on this machine")
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    if args.output is not None and args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    return args


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    from utils.calc import COST_MODE

    if COST_MODE not in {"design_power", "perturbation", "full"}:
        raise ValueError(f"Unknown COST_MODE in utils/calc.py: {COST_MODE!r}")
    output_path = (
        args.output or OUTPUT_DIRECTORY / f"{COST_MODE}_cost_map_finesse_50x50.npz"
    ).expanduser().resolve()
    cost_grid, completed_grid = _load_or_initialize(
        output_path, args.pd_name, args.overwrite
    )

    tasks = [
        (i, j, float(itm_roc), float(etm_roc), args.pd_name)
        for i, itm_roc in enumerate(ITM_ROC_LIST)
        for j, etm_roc in enumerate(ETM_ROC_LIST)
        if not completed_grid[i, j]
    ]
    total_points = GRID_SIZE * GRID_SIZE
    already_complete = total_points - len(tasks)
    if not tasks:
        print(f"All {total_points} points are already complete: {output_path}")
        return output_path

    effective_workers = min(args.workers, len(tasks))
    print(
        f"Computing 50x50 {COST_MODE} cost map for {args.pd_name} with "
        f"{effective_workers} workers."
    )
    print(f"Each point runs 5 Finesse simulations; total full grid = {total_points * 5} simulations.")
    print(f"Resuming with {already_complete}/{total_points} points complete.")
    print(f"Checkpoint: {output_path}")

    completed_since_save = 0
    failures = []
    start_time = perf_counter()
    pool = mp.get_context("fork").Pool(
        processes=effective_workers,
        # Periodic worker recycling limits accumulation from repeated Finesse models.
        maxtasksperchild=25,
    )
    try:
        results = pool.imap_unordered(_evaluate_task, tasks, chunksize=1)
        with tqdm(total=len(tasks), desc=f"Finesse {COST_MODE} cost map") as pbar:
            for i, j, cost, error in results:
                if error is None:
                    cost_grid[i, j] = cost
                    completed_grid[i, j] = True
                    completed_since_save += 1
                else:
                    failures.append((i, j, error))
                pbar.update(1)
                pbar.set_postfix(failed=len(failures))

                if completed_since_save >= args.save_every:
                    _save_checkpoint(
                        output_path,
                        cost_grid,
                        completed_grid,
                        pd_name=args.pd_name,
                    )
                    completed_since_save = 0
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
        _save_checkpoint(
            output_path,
            cost_grid,
            completed_grid,
            pd_name=args.pd_name,
        )

    elapsed = perf_counter() - start_time
    complete_count = int(completed_grid.sum())
    print(f"Saved {complete_count}/{total_points} completed points in {elapsed:.1f} s.")
    print(f"Result: {output_path}")

    if failures:
        print(f"{len(failures)} points failed unexpectedly and were left incomplete for retry:")
        for i, j, error in failures[:10]:
            print(
                f"  ITM={ITM_ROC_LIST[i]:.6g}, ETM={ETM_ROC_LIST[j]:.6g}\n{error}"
            )
        if len(failures) > 10:
            print(f"  ... {len(failures) - 10} additional failures omitted")
        raise RuntimeError(
            "Some grid points failed. Rerun the same command to retry only incomplete points."
        )

    return output_path


if __name__ == "__main__":
    main()