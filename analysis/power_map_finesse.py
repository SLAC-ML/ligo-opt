"""Generate the nominal Finesse power, beam-size, and loss maps for subspace_plot.

Run from the repository root:
    /home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.power_map_finesse

Rerun to resume interrupted work. Use --workers to limit parallel simulations,
--output to choose another NPZ file, or --overwrite to recompute the map.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
from tqdm import tqdm

from utils.detectors import POWER_DETECTOR

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / f"analysis/results/power_map_finesse_{POWER_DETECTOR}_50x50.npz"
ITM_ROC_LIST = np.linspace(-1934.0 - 2200.0, -1934.0 + 500.0, 50)
ETM_ROC_LIST = np.linspace(2245.0 - 500.0, 2245.0 + 2000.0, 50)
PD_NAME = POWER_DETECTOR
MAP_POINTS = 501
APERTURE_RADIUS = 0.17
FIELDS = ("Power", "w_ITM", "w_ETM", "Loss")
MAX_WORKERS = min(60, os.cpu_count() or 1)


def simulate_point(base_kat, itm_roc, etm_roc):
    """Return nominal power, both spot radii, and total round-trip loss.

    Preserve the notebook's zero sentinel for unstable or failed-lock points.
    Unexpected errors outside the lock solve propagate and remain retryable.
    """
    from finesse.knm import Map
    from finesse.utilities.maps import circular_aperture
    from finesse.analysis.actions.operator import Eigenmodes

    kat = base_kat.deepcopy()
    kat.ITM.Rc = itm_roc
    kat.ETM.Rc = etm_roc
    x = y = np.linspace(-APERTURE_RADIUS, APERTURE_RADIUS, MAP_POINTS)
    kat.ITM.surface_map = Map(x, y, amplitude=circular_aperture(x, y, APERTURE_RADIUS))
    kat.ETM.surface_map = Map(x, y, amplitude=circular_aperture(x, y, APERTURE_RADIUS))
    kat.parse("bp w_ITM ITM.p2.o prop=w")
    kat.parse("bp w_ETM ETM.p1.i prop=w")
    if not kat.cavArm.is_stable:
        return 0.0, 0.0, 0.0, 0.0
    try:
        result = kat.run("Series(run_locks(max_iterations=100000,), noxaxis(),)")["noxaxis"]
    except Exception as exc:
        print(f"Lock failed at ITM={itm_roc}, ETM={etm_roc}: {exc}", flush=True)
        return 0.0, 0.0, 0.0, 0.0
    modes = kat.run(Eigenmodes(kat.cavArm, frequency=0.0))
    indices, losses = modes.loss(remove_planewave_loss=False)
    return (float(result[PD_NAME]), float(result["w_ITM"]),
            float(result["w_ETM"]), float(losses[indices[0]]))


def _evaluate_task(task):
    i, j, itm, etm = task
    try:
        from utils.finesse_base import base_kat
        values = simulate_point(base_kat, itm, etm)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite simulation result: {values}")
        return i, j, values, None
    except Exception:
        return i, j, None, traceback.format_exc()


def _save_checkpoint(path, grids, completed):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, ITM_ROC_List=ITM_ROC_LIST, ETM_ROC_List=ETM_ROC_LIST,
             pd_name=PD_NAME, map_points=MAP_POINTS, aperture_radius=APERTURE_RADIUS,
             completed_grid=completed, **grids)
    temporary.replace(path)


def _read_checkpoint(path):
    with np.load(path) as saved:
        if (str(saved["pd_name"]) != PD_NAME
                or int(saved["map_points"]) != MAP_POINTS
                or float(saved["aperture_radius"]) != APERTURE_RADIUS):
            raise ValueError("Saved detector/aperture settings differ from the requested detector "
                             f"{PD_NAME!r}; choose another --output or use --overwrite.")
        itm = saved["ITM_ROC_List"].copy()
        etm = saved["ETM_ROC_List"].copy()
        grids = {key: saved[key].copy() for key in FIELDS}
        completed = saved["completed_grid"].astype(bool, copy=True)
    shape = (len(ITM_ROC_LIST), len(ETM_ROC_LIST))
    if not np.array_equal(itm, ITM_ROC_LIST) or not np.array_equal(etm, ETM_ROC_LIST):
        raise ValueError("Saved ROC grid differs; choose another --output or use --overwrite.")
    if completed.shape != shape or any(grid.shape != shape for grid in grids.values()):
        raise ValueError(f"Expected all saved arrays to have shape {shape}.")
    if any(not np.isfinite(grid[completed]).all() for grid in grids.values()):
        raise ValueError("Completed map points contain non-finite values.")
    return itm, etm, grids, completed


def load_power_map(path=DEFAULT_OUTPUT):
    """Read a completed map without importing Finesse or running simulations.

    Return ROC axes and flattened power_list, including its pd_name provenance.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Power map not found: {path}. Run python -m analysis.power_map_finesse first."
        )
    itm, etm, grids, completed = _read_checkpoint(path)
    if not completed.all():
        raise RuntimeError(
            f"Power map is incomplete ({completed.sum()}/{completed.size} points). "
            "Rerun analysis.power_map_finesse with the same --output to resume."
        )
    x, y = np.meshgrid(itm, etm, indexing="ij")
    values = {"ITM_ROC": x.ravel().tolist(), "ETM_ROC": y.ravel().tolist(),
              "pd_name": PD_NAME}
    values.update({key: grid.ravel().tolist() for key, grid in grids.items()})
    return itm, etm, values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    args.output = args.output.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists() and not args.overwrite:
        _, _, grids, completed = _read_checkpoint(args.output)
    else:
        shape = (len(ITM_ROC_LIST), len(ETM_ROC_LIST))
        grids = {key: np.full(shape, np.nan) for key in FIELDS}
        completed = np.zeros(shape, dtype=bool)
    tasks = [(i, j, float(ITM_ROC_LIST[i]), float(ETM_ROC_LIST[j]))
             for i, j in np.argwhere(~completed)]
    print(f"Completed {completed.sum()}/{completed.size} points. Result: {args.output}")
    if not tasks:
        return args.output
    failures = []
    since_save = 0
    pool = mp.get_context("fork").Pool(
        processes=min(args.workers, len(tasks)), maxtasksperchild=25,
    )
    try:
        results = pool.imap_unordered(_evaluate_task, tasks, chunksize=1)
        for i, j, values, error in tqdm(results, total=len(tasks), desc="Finesse power map"):
            if error is None:
                for key, value in zip(FIELDS, values):
                    grids[key][i, j] = value
                completed[i, j] = True
                since_save += 1
            else:
                failures.append((i, j, error))
            if since_save >= args.save_every:
                _save_checkpoint(args.output, grids, completed)
                since_save = 0
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
        _save_checkpoint(args.output, grids, completed)
    print(f"Saved {completed.sum()}/{completed.size} points: {args.output}")
    if failures:
        for i, j, error in failures[:10]:
            print(f"ITM={ITM_ROC_LIST[i]}, ETM={ETM_ROC_LIST[j]}:\n{error}")
        raise RuntimeError(f"{len(failures)} points failed; rerun to retry incomplete points.")
    return args.output


if __name__ == "__main__":
    main()
