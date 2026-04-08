"""Run locking-parameter optimization with mpBAX.

Usage:
    # Approach 1 (F=always, R=configurable):
    python run.py --approach 1

    # Approach 2 (R=always, inner loop):
    python run.py --approach 2

    # Custom retrain frequency (approach 1):
    python run.py --approach 1 --retrain-every 5
"""

import argparse
import tempfile
import numpy as np

from mpbax.core.model import DummyModel

from engine import LockingEngine
from algorithm import LockingAlgorithm
from oracles import LockingOracle


def main():
    parser = argparse.ArgumentParser(description='Locking-parameter optimization')
    parser.add_argument('--approach', type=int, default=1, choices=[1, 2],
                        help='Optimization approach (1 or 2)')
    parser.add_argument('--retrain-every', type=int, default=3,
                        help='Retrain frequency for approach 1 (default: 3)')
    parser.add_argument('--inner-steps', type=int, default=10,
                        help='Inner loop steps for approach 2 (default: 10)')
    parser.add_argument('--max-loops', type=int, default=10,
                        help='Maximum optimization loops (default: 10)')
    parser.add_argument('--input-dim', type=int, default=4,
                        help='Design parameter dimensionality (default: 4)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint directory (default: temp dir)')
    args = parser.parse_args()

    # Create oracle and algorithm instances
    oracle = LockingOracle(use_surrogate_init=True, noise_scale=0.01)
    algo = LockingAlgorithm

    # Build config programmatically (instance-based for oracle, class for algorithm)
    checkpoint_dir = args.checkpoint_dir or tempfile.mkdtemp(prefix='ligo_')

    config = {
        'seed': 42,
        'max_loops': args.max_loops,
        'locking': {
            'approach': args.approach,
            'retrain_every': args.retrain_every,
        },
        'checkpoint': {
            'dir': checkpoint_dir,
            'freq': 1,
        },
        'training': {
            'mode': 'finetune',
            'checkpoint_mode': 'final',
        },
        'oracles': [{
            'name': 'locking',
            'input_dim': args.input_dim,
            'n_initial': 200,
            'function': {'class': oracle},  # Callable class instance
            'model': {'class': DummyModel},
        }],
        'algorithm': {
            'class': LockingAlgorithm,
            'params': {
                'input_dims': [args.input_dim],
                'n_propose': 5,
                'n_candidates': 500,
                'approach': args.approach,
                'inner_steps': args.inner_steps,
            },
        },
    }

    # Print config summary
    print("=" * 60)
    print("Locking-Parameter Optimization")
    print("=" * 60)
    print(f"  Approach: {args.approach}")
    if args.approach == 1:
        print(f"  Retrain every: {args.retrain_every} loops (R switch)")
        print(f"  Finetune: always (F=always true)")
    else:
        print(f"  Inner steps: {args.inner_steps} (F switch in algorithm)")
        print(f"  Retrain: always (R=always true)")
    print(f"  Design dim: {args.input_dim}")
    print(f"  Max loops: {args.max_loops}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print("=" * 60)

    # Create engine and link oracle to it
    engine = LockingEngine(config)
    oracle.engine = engine  # Give oracle access to trained model

    # Run optimization
    engine.run()

    # Print final results
    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)

    # Show best result from accumulated data
    X, Y = engine.data_handlers[0].get_data()
    if X is not None and Y is not None:
        # Y contains L* (locking params). Compute objectives.
        from utils.calc import calc_objective
        objectives = calc_objective(X, Y)
        best_idx = np.argmin(objectives)
        print(f"\n  Best objective: {objectives[best_idx]:.6f}")
        print(f"  Best D: {X[best_idx]}")
        print(f"  Best L*: {Y[best_idx]}")
        print(f"  Total simulations: {engine.evaluators[0].get_eval_count()}")

    print("=" * 60)


if __name__ == '__main__':
    main()
