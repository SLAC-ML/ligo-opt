"""Oracle functions for locking-parameter optimization.

The oracle wraps the simulation: given design parameters D, it runs the
(expensive) simulation to obtain accurate locking parameters L*.

The oracle can optionally use the trained surrogate model to provide
an initial guess for L, which helps the simulation converge faster.
"""

import numpy as np

from utils.sim import simulate


class LockingOracle:
    """Oracle that runs locking simulation.

    Takes design params D, optionally uses surrogate model for initial
    guess, runs simulation to get accurate locking params L*.

    The oracle returns L* (not objectives). The mpBAX model trains on
    (D, L*) pairs, learning the D -> L mapping. The algorithm computes
    objectives internally from (D, L_predicted).

    Attributes:
        engine: Reference to the Engine instance (set after creation).
                Used to access the trained surrogate model for initial guesses.
        use_surrogate_init: Whether to use surrogate predictions as initial guess.
        noise_scale: Simulation noise level.
    """

    def __init__(self, use_surrogate_init=True, noise_scale=0.01):
        """Initialize LockingOracle.

        Args:
            use_surrogate_init: Use trained surrogate for initial guess (default True)
            noise_scale: Simulation noise level (default 0.01)
        """
        self.engine = None  # Set after engine creation via: oracle.engine = engine
        self.use_surrogate_init = use_surrogate_init
        self.noise_scale = noise_scale

    def __call__(self, D):
        """Run locking simulation on design parameters.

        Args:
            D: Design parameters, shape (n, d)

        Returns:
            L_star: Accurate locking parameters, shape (n, 2)
        """
        # Get initial guess from surrogate if available
        L_init = None
        if self.use_surrogate_init and self.engine is not None:
            if self.engine.models and self.engine.models[0].is_trained:
                L_init = self.engine.models[0].predict(D)

        # Run simulation (expensive in reality, mockup here)
        L_star = simulate(D, L_init=L_init, noise_scale=self.noise_scale)

        return L_star
