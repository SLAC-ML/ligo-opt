"""Run one Finesse-only PSO using the objective selected in utils/calc.py.

Each particle evaluation performs five Finesse simulations: the nominal design
and ITM/ETM +/- 0.33% ROC perturbations with the nominal input q held fixed.
The objective uses calc_cost at ``p_ITM_p2_i``, following COST_MODE and weights
in utils/calc.py. No GNN training or inference is performed.
Particle evaluations use a bounded process pool with up to 60 CPU cores.

Examples
--------
Run the full optimization using 60 workers and a 60-particle swarm::

    /home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.run_finesse_pso

Run a smaller job and choose another result file::

    /home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.run_finesse_pso --workers 8 --population 16 --iterations 20 \
        --output quick_results.txt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

# Prevent numerical libraries in each worker from starting additional thread
# pools and oversubscribing the machine. Users can override these before launch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sko.PSO import PSO
from tqdm import tqdm

ITM_ROC = -1934.0
ETM_ROC = 2245.0
PORT = "p_ITM_p2_i"

LOWER_BOUNDS = [ITM_ROC - 2200.0, ETM_ROC - 500.0]
UPPER_BOUNDS = [ITM_ROC + 500.0, ETM_ROC + 2000.0]

MAX_WORKERS = 60
DEFAULT_POPULATION = 60
DEFAULT_ITERATIONS = 100
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "Testing_Ground_results_perturbed.txt"


def cost_fn_finesse_perturbed(params: np.ndarray) -> float:
    """Return the cost selected in utils/calc.py from five Finesse simulations."""
    # Reuse the Finesse-only implementation; no surrogate is constructed.
    from analysis.perturbation_cost_map_finesse import full_finesse_cost

    itm_roc, etm_roc = params
    return float(full_finesse_cost(itm_roc, etm_roc, PORT))


class ParallelPSO(PSO):
    """A steppable scikit-opt PSO with a persistent, bounded process pool."""

    def __init__(self, objective, *, workers: int, **kwargs):
        self._objective = objective
        self.workers = workers
        # This project runs on Linux. Fork avoids repeatedly serializing the
        # sizeable Finesse base model and imported numerical modules.
        self._pool = mp.get_context("fork").Pool(processes=workers)
        self._closed = False
        try:
            super().__init__(func=objective, **kwargs)
            # Preserve candidates from the initial swarm, as run.py does.
            self.update_pbest()
            self.update_gbest()
        except BaseException:
            self.close(terminate=True)
            raise

    def cal_y(self):
        """Evaluate all particles concurrently while preserving swarm order."""
        values = self._pool.map(self._objective, self.X)
        self.Y = np.asarray(values, dtype=float).reshape(-1, 1)
        return self.Y

    def update_gbest(self):
        """Keep the historical best position paired with its historical cost."""
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.pbest_x[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min].copy()

    def step(self):
        """Run exactly one PSO iteration."""
        self.update_V()
        self.recorder()
        self.update_X()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()
        self.gbest_y_hist.append(self.gbest_y)
        return self.gbest_x, self.gbest_y

    def close(self, *, terminate: bool = False) -> None:
        """Shut down all worker processes."""
        if self._closed:
            return
        if terminate:
            self._pool.terminate()
        else:
            self._pool.close()
        self._pool.join()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(terminate=exc_type is not None)


def _scalar(value) -> float:
    """Convert scikit-opt's scalar or one-element array values to float."""
    return float(np.asarray(value).reshape(-1)[0])


def save_results(
    output_path: Path,
    *,
    pso: ParallelPSO,
    workers: int,
    population: int,
    iterations: int,
    elapsed_seconds: float,
) -> None:
    """Atomically save the optimization summary and complete best-cost history."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_x = np.asarray(pso.gbest_x, dtype=float).reshape(-1)
    best_cost = _scalar(pso.gbest_y)
    history = [_scalar(value) for value in pso.gbest_y_hist]

    lines = [
        "Single PSO: Finesse fixed-q optimization",
        f"workers: {workers}",
        f"population: {population}",
        f"iterations: {iterations}",
        f"total_objective_evaluations: {population * (iterations + 1)}",
        f"total_finesse_evaluations: {5 * population * (iterations + 1)}",
        f"elapsed_seconds: {elapsed_seconds:.6f}",
        f"port: {PORT}",
        f"ITM_ROC_bounds_m: [{LOWER_BOUNDS[0]:.16g}, {UPPER_BOUNDS[0]:.16g}]",
        f"ETM_ROC_bounds_m: [{LOWER_BOUNDS[1]:.16g}, {UPPER_BOUNDS[1]:.16g}]",
        f"best_ITM_ROC_m: {best_x[0]:.16g}",
        f"best_ETM_ROC_m: {best_x[1]:.16g}",
        f"best_cost: {best_cost:.16g}",
        "gbest_cost_history:",
    ]
    lines.extend(f"{index}\t{cost:.16g}" for index, cost in enumerate(history, 1))

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"parallel worker processes (default: {MAX_WORKERS}, maximum: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION,
        help=f"PSO swarm size (default: {DEFAULT_POPULATION})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"PSO iterations after initial swarm evaluation (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"result text file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--seed", type=int, default=None, help="optional NumPy random seed")
    args = parser.parse_args(argv)

    available_cpus = os.cpu_count() or 1
    if not 1 <= args.workers <= min(MAX_WORKERS, available_cpus):
        parser.error(
            f"--workers must be between 1 and {min(MAX_WORKERS, available_cpus)} "
            f"on this machine"
        )
    if args.population < 1:
        parser.error("--population must be at least 1")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    return args


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    effective_workers = min(args.workers, args.population)
    if effective_workers < args.workers:
        print(
            f"Swarm has {args.population} particles; using {effective_workers} "
            "workers because additional workers would be idle."
        )

    print(
        f"Running one Finesse PSO with {args.population} particles, {args.iterations} "
        f"iterations, and {effective_workers} workers."
    )
    start_time = perf_counter()
    with ParallelPSO(
        cost_fn_finesse_perturbed,
        workers=effective_workers,
        n_dim=2,
        pop=args.population,
        max_iter=args.iterations,
        lb=LOWER_BOUNDS,
        ub=UPPER_BOUNDS,
        w=0.9,
        c1=2.0,
        c2=0.4,
        verbose=True,
    ) as pso:
        for _ in tqdm(
            range(args.iterations),
            desc="PSO (Finesse, 5-sim cost)",
        ):
            pso.step()

    elapsed_seconds = perf_counter() - start_time
    save_results(
        args.output,
        pso=pso,
        workers=effective_workers,
        population=args.population,
        iterations=args.iterations,
        elapsed_seconds=elapsed_seconds,
    )

    best_x = np.asarray(pso.gbest_x).reshape(-1)
    best_cost = _scalar(pso.gbest_y)
    print(f"Best parameters (ITM_ROC, ETM_ROC): {best_x}")
    print(f"Best cost: {best_cost:.16g}")
    print(f"Results saved to: {args.output.expanduser().resolve()}")
    return args.output.expanduser().resolve()


if __name__ == "__main__":
    main()