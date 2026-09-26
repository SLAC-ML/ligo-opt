"""Steppable particle swarm optimization for the main experiment.

Objective construction lives in PSOoptimizer.py. The Finesse-only baseline keeps
its separate ParallelPSO implementation in analysis/run_finesse_pso.py.
"""

from sko.PSO import PSO
from sko.tools import func_transformer


class SteppablePSO(PSO):
    def update_gbest(self):
        """Keep the best historical cost paired with its historical position.

        The installed scikit-opt implementation selects the index from ``pbest_y``
        but copies ``X[idx]``. Once that particle has moved, ``gbest_x`` therefore
        no longer corresponds to ``gbest_y``. The real-Finesse feedback point must
        be the same geometry whose surrogate cost won, so copy ``pbest_x`` here.
        """
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.pbest_x[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min].copy()

    def step(self):
        """Run exactly one PSO iteration (equivalent to one loop of run())."""
        self.update_V()
        self.recorder()
        self.update_X()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()
        self.gbest_y_hist.append(self.gbest_y)
        return self.gbest_x, self.gbest_y

    def set_func(self, func):
        """Swap the objective function between steps."""
        self.func = func_transformer(func)
