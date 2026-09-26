"""Sample cavity design parameters for initial training data.

Sampling and bounds validation are independent of Finesse simulation and graph
storage, which are coordinated by fabry_perot.generate_initial_data.
"""

import numpy as np
from scipy.stats import qmc


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
