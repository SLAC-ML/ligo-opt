"""Mockup objective calculation for locking-parameter optimization.

Computes objectives from design parameters D and locking parameters L.
In reality, this would evaluate physics performance metrics based on
the locking configuration achieved for a given design.
"""

import numpy as np


def calc_objective(D, L):
    """Compute scalar objective from design params and locking params.

    Mockup: minimize a combined metric that depends on both D and L.
    Lower is better.

    Args:
        D: Design parameters, shape (n, d)
        L: Locking parameters, shape (n, 2)

    Returns:
        obj: Objective values, shape (n,)
    """
    # Distance from target in design space
    d_cost = np.sum((D - 0.3) ** 2, axis=1)

    # Distance from target in locking space
    l_cost = np.sum((L - 0.5) ** 2, axis=1)

    return d_cost + 2.0 * l_cost
