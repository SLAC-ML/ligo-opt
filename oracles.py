"""Oracle functions for locking-parameter optimization.

The oracle wraps the simulation: given design parameters D, it runs the
(expensive) simulation to obtain accurate locking parameters L*.

The oracle can optionally use the trained surrogate model to provide
an initial guess for L, which helps the simulation converge faster.
"""

import numpy as np

from utils.sim import finesse_sim


class FPOracle:
    """Oracle that runs fabry-perot simulation.

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

    def __init__(self):
        """Initialize FPOracle.

        Args:
            use_surrogate_init: Use trained surrogate for initial guess (default True)
            noise_scale: Simulation noise level (default 0.01)
        """
        self.engine = None  # Set after engine creation via: oracle.engine = engine

    def __call__(self, D):
        """Run fabry-perot simulation on design parameters.

        Args:
            D: Design parameters, shape (d)

        Returns:
            cavity_power: Cavity power, float
        """

        # Run expansive simulation
        cavity_power = finesse_sim(D)

        return cavity_power
