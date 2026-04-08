"""Custom engine for locking-parameter optimization.

Extends the base Engine with:
- R switch: conditional retraining (approach 1)
- Approach-aware behavior via config
"""

from mpbax.core.engine import Engine


class LockingEngine(Engine):
    """Engine with configurable retrain logic (R switch).

    For Approach 1 (F=always true):
        The R switch controls how often the model is retrained.
        Configured via locking.retrain_every in config.
        When skipping retrain, data still accumulates.

    For Approach 2 (R=always true):
        Training happens every loop (default behavior).
        The F switch is handled inside the algorithm's inner loop.
    """

    def _train_models(self, accumulated_data):
        """Train models with R switch logic.

        For approach 1: retrain every N loops (configurable).
        For approach 2: always retrain (R=always true).
        """
        locking_config = self.config.get('locking', {})
        approach = locking_config.get('approach', 1)
        retrain_every = locking_config.get('retrain_every', 1)

        if approach == 1 and retrain_every > 1:
            # Approach 1: R switch — retrain every N loops
            if self.current_loop % retrain_every == 0:
                print(f"  [R switch] Retraining at loop {self.current_loop}")
                super()._train_models(accumulated_data)
            else:
                print(f"  [R switch] Skipping retrain "
                      f"(next at loop {self.current_loop + retrain_every - self.current_loop % retrain_every})")
        else:
            # Approach 2 or retrain_every=1: always retrain
            super()._train_models(accumulated_data)
