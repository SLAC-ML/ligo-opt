"""
Fabry-Perot Cavity Simulations using Finesse

This module provides a framework for generating simulations of Fabry-Perot cavities using the Finesse software. It includes functions for building a KAT model in an ITM, ETM ROC space, converting the model to a GNN model, and collecting graph-based representations of the optical network.

The results are saved to an HDF5 file, which can be used to train a GNN model.
"""


from pathlib import Path

import numpy as np
import finesse
from scipy.stats import qmc

from GNN.GNN_utils import model_to_nx_port, append_graph_to_h5, kat_manipulation
from tqdm import tqdm
from utils.finesse_base import base_kat
from utils.sim import finesse_sim


finesse.init_plotting(fmts=["png"])


ROC_PERTURBATION_PERCENT = 0.0033


def _validate_bounds(name, bounds):
    """Return finite, increasing ``(lower, upper)`` bounds as floats."""
    if len(bounds) != 2:
        raise ValueError(
            f'{name} bounds must contain exactly two values; got {bounds!r}.'
        )
    lower, upper = map(float, bounds)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError(f'{name} bounds must be finite and increasing; got {bounds!r}.')
    return lower, upper


def optimized_lhs_roc_samples(num_design_points, itm_bounds, etm_bounds, seed=9302):
    """Return a reproducible, discrepancy-optimized LHS over the ROC rectangle."""
    if (
        not isinstance(num_design_points, (int, np.integer))
        or isinstance(num_design_points, (bool, np.bool_))
        or num_design_points < 1
    ):
        raise ValueError(
            'num_design_points must be a positive integer; '
            f'got {num_design_points!r}.'
        )
    itm_bounds = _validate_bounds('ITM ROC', itm_bounds)
    etm_bounds = _validate_bounds('ETM ROC', etm_bounds)

    sampler = qmc.LatinHypercube(
        d=2,
        optimization='random-cd',
        seed=seed,
    )
    unit_samples = sampler.random(n=int(num_design_points))
    return qmc.scale(
        unit_samples,
        [itm_bounds[0], etm_bounds[0]],
        [itm_bounds[1], etm_bounds[1]],
    )


def _perturb_pair(
    ITM_ROC,
    ETM_ROC,
    rng=None,
    itm_bounds=None,
    etm_bounds=None,
    percentage=ROC_PERTURBATION_PERCENT,
):
    """Perturb both ROCs by ``percentage`` while remaining inside any bounds."""
    if percentage <= 0:
        raise ValueError(f'percentage must be positive; got {percentage!r}.')
    rng = rng or np.random.default_rng()

    def perturb(value, bounds):
        candidates = [value * (1 - percentage), value * (1 + percentage)]
        if bounds is not None:
            lower, upper = _validate_bounds('ROC', bounds)
            candidates = [
                candidate for candidate in candidates if lower <= candidate <= upper
            ]
        if not candidates:
            raise ValueError(
                f'Cannot perturb ROC {value!r} by {percentage:.4%} within {bounds!r}.'
            )
        return candidates[int(rng.integers(len(candidates)))]

    return perturb(ITM_ROC, itm_bounds), perturb(ETM_ROC, etm_bounds)


def generate_initial_data(
    output_path,
    num_design_points=5,
    itm_bounds=(-4134, -1434),
    etm_bounds=(1745, 4245),
    overwrite=False,
    seed=9302,
):
    """Generate optimized-LHS design graphs and their fixed-q perturbations.

    Every LHS design point contributes two consecutive HDF5 graphs: the design
    cavity with its self-consistent eigenmode, followed by a nearby geometry with
    the design point's q held fixed. Consequently, the output contains exactly
    ``2 * num_design_points`` graphs.
    """
    roc_samples = optimized_lhs_roc_samples(
        num_design_points,
        itm_bounds,
        etm_bounds,
        seed=seed,
    )
    itm_bounds = _validate_bounds('ITM ROC', itm_bounds)
    etm_bounds = _validate_bounds('ETM ROC', etm_bounds)
    total_graphs = 2 * int(num_design_points)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(
                f'{output_path} already exists. Pass overwrite=True to regenerate it.'
            )

    print(
        f'Generating {num_design_points} optimized-LHS design points across the '
        f'full ROC space. Each point adds one design graph and one fixed-q '
        f'perturbation graph ({total_graphs} total graphs, seed={seed}).'
    )

    rng = np.random.default_rng(seed)
    valid_q_pairs = 0
    with tqdm(total=total_graphs, position=0, desc="initial sims") as pbar:
        for ITM_ROC, ETM_ROC in roc_samples:
            design_kat = kat_manipulation(
                ITM_ROC,
                ETM_ROC,
                base_kat,
                nominal_q_value=None,
            )
            pd_names, powers, q_names, q_values = finesse_sim(design_kat)
            graph = model_to_nx_port(
                design_kat, pd_names, powers, q_names, q_values
            )
            append_graph_to_h5(graph, output_path)
            pbar.update(1)

            design_q_value = q_values[q_names.index('q_ITM_p1_i')]
            fixed_q = None if design_q_value == 0 else design_q_value
            valid_q_pairs += fixed_q is not None
            perturbed_ITM_ROC, perturbed_ETM_ROC = _perturb_pair(
                ITM_ROC,
                ETM_ROC,
                rng=rng,
                itm_bounds=itm_bounds,
                etm_bounds=etm_bounds,
            )
            perturbed_kat = kat_manipulation(
                perturbed_ITM_ROC,
                perturbed_ETM_ROC,
                base_kat,
                nominal_q_value=fixed_q,
            )
            pd_names, powers, q_names, q_values = finesse_sim(perturbed_kat)
            graph = model_to_nx_port(
                perturbed_kat, pd_names, powers, q_names, q_values
            )
            append_graph_to_h5(graph, output_path)
            pbar.update(1)

    print(
        f'Generated {total_graphs} graphs. {valid_q_pairs}/{num_design_points} '
        'perturbation graphs used a valid fixed design-point q; pairs whose '
        'design cavity was unstable/unsolvable were retained as zero-power data.'
    )

    return output_path


if __name__ == '__main__':
    generate_initial_data('GNN/data/run6/base_10.h5', num_design_points=5)