"""Mockup simulation for locking-parameter refinement.

In reality, this would be an expensive physics simulation (e.g., particle
tracking in an accelerator lattice) that finds accurate locking parameters
given a design configuration.

The mockup demonstrates the key property: with a good initial guess from
the surrogate model, the simulation converges faster and more accurately.
"""

import numpy as np


def true_locking_params(D):
    """Ground truth locking parameters as a nonlinear function of design params.

    This represents the "true" relationship that the surrogate model
    tries to learn, and the simulation tries to find.

    Args:
        D: Design parameters, shape (n, d)

    Returns:
        L_true: True locking parameters, shape (n, 2)
    """
    d = D.shape[1]
    # Nonlinear mixing of design parameters
    L1 = np.sin(np.pi * D[:, 0]) * D[:, 1 % d] + 0.5 * D[:, 2 % d]
    L2 = np.cos(np.pi * D[:, 1 % d]) * D[:, 3 % d] + 0.3 * D[:, 0]
    return np.column_stack([L1, L2])


def simulate(D, L_init=None, noise_scale=0.01):
    """Mockup simulation that finds accurate locking parameters.

    Simulates the behavior of an expensive physics simulation:
    - With good initial guess (from surrogate): small residual noise
    - Without initial guess: larger noise (convergence difficulties)

    Args:
        D: Design parameters, shape (n, d)
        L_init: Initial guess for locking params from surrogate, shape (n, 2).
                None if no surrogate prediction available.
        noise_scale: Base noise level (default 0.01)

    Returns:
        L_star: Refined locking parameters, shape (n, 2)
    """
    L_true = true_locking_params(D)

    if L_init is not None:
        # Good initial guess: simulation converges accurately
        noise = noise_scale * np.random.randn(*L_true.shape)
    else:
        # No initial guess: larger residual (simulation struggles)
        noise = 5 * noise_scale * np.random.randn(*L_true.shape)

    L_star = L_true + noise
    return L_star
