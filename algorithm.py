"""Custom algorithm for locking-parameter optimization.

Supports two approaches controlled via config:

Approach 1 (F=always true):
    Every engine loop runs the simulation. The algorithm uses the surrogate
    to evaluate candidates and proposes D_opt. R switch (retrain frequency)
    is handled by the custom engine.

Approach 2 (R=always true):
    The algorithm runs an inner optimization loop using only the surrogate
    (no simulation). After inner_steps fast iterations, it returns D_opt
    to the engine for simulation + retraining.
"""

import numpy as np
from typing import List, Callable
from mpbax.core.algorithm import BaseAlgorithm

from utils.calc import calc_objective


class LockingAlgorithm(BaseAlgorithm):
    """Algorithm for locking-parameter optimization.

    Uses surrogate model predictions (D -> L) to evaluate candidate
    designs, computing objectives from (D, L_predicted). Proposes
    the best designs for simulation evaluation.

    Approach 1: Standard propose — evaluate candidates, return best batch.
    Approach 2: Inner loop — run many surrogate-only iterations, return
                single best D_opt for simulation.
    """

    def __init__(self, input_dims, n_propose=5, n_candidates=1000,
                 approach=1, inner_steps=10, seed=42):
        """Initialize LockingAlgorithm.

        Args:
            input_dims: List with [d] — input dimensionality for the single oracle
            n_propose: Number of candidates to propose (approach 1)
            n_candidates: Number of random candidates to evaluate per step
            approach: 1 or 2 (see module docstring)
            inner_steps: Number of surrogate-only iterations (approach 2 only)
            seed: Random seed
        """
        self.input_dims = input_dims
        self.input_dim = input_dims[0]
        self.n_propose = n_propose
        self.n_candidates = n_candidates
        self.approach = approach
        self.inner_steps = inner_steps
        self.rng = np.random.RandomState(seed)

    def propose(self, fn_pred_list):
        """Propose design parameters for evaluation.

        Args:
            fn_pred_list: [fn_pred_L] — single predict function mapping D -> L

        Returns:
            [D_proposed] — list with one array of proposed design parameters
        """
        fn_pred_L = fn_pred_list[0]

        if self.approach == 2:
            return self._propose_approach2(fn_pred_L)
        else:
            return self._propose_approach1(fn_pred_L)

    def _propose_approach1(self, fn_pred_L):
        """Approach 1: Evaluate candidates, return best batch.

        Standard greedy selection using surrogate predictions.
        """
        D_candidates = self.rng.rand(self.n_candidates, self.input_dim)
        L_predicted = fn_pred_L(D_candidates)
        objectives = calc_objective(D_candidates, L_predicted)

        best_idx = np.argsort(objectives)[:self.n_propose]
        D_proposed = D_candidates[best_idx]

        print(f"  [Algorithm] Approach 1: evaluated {self.n_candidates} candidates, "
              f"proposing {len(D_proposed)} (best obj={objectives[best_idx[0]]:.4f})")

        return [D_proposed]

    def _propose_approach2(self, fn_pred_L):
        """Approach 2: Inner optimization loop, return best D_opt.

        Runs multiple surrogate-only iterations (fast, no simulation),
        then returns the single best design for simulation evaluation.
        """
        best_D = None
        best_obj = float('inf')

        for step in range(self.inner_steps):
            D_candidates = self.rng.rand(self.n_candidates, self.input_dim)
            L_predicted = fn_pred_L(D_candidates)
            objectives = calc_objective(D_candidates, L_predicted)

            min_idx = np.argmin(objectives)
            if objectives[min_idx] < best_obj:
                best_obj = objectives[min_idx]
                best_D = D_candidates[min_idx:min_idx + 1]

        print(f"  [Algorithm] Approach 2: {self.inner_steps} inner steps, "
              f"best obj={best_obj:.4f}")

        # Return small batch: the best point (+ a few nearby for diversity)
        n_extra = min(self.n_propose - 1, 4)
        if n_extra > 0:
            # Add small perturbations around best for diversity
            noise = 0.05 * self.rng.randn(n_extra, self.input_dim)
            D_extra = np.clip(best_D + noise, 0, 1)
            D_proposed = np.vstack([best_D, D_extra])
        else:
            D_proposed = best_D

        return [D_proposed]
