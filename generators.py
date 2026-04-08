"""Initial data generators for locking-parameter optimization."""

import numpy as np


def gen_locking_initial(n, d):
    """Generate initial design parameters using uniform random sampling.

    Args:
        n: Number of samples
        d: Design parameter dimensionality

    Returns:
        D: Design parameters, shape (n, d) in [0, 1]^d
    """
    return np.random.rand(n, d)
