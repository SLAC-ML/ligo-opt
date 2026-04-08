# Locking-Parameter Optimization Template

Surrogate-assisted optimization where a neural network predicts locking parameters from design parameters, and a simulation refines them. Built on the [mpBAX](https://github.com/anthropics/mpBAX) framework.

## Setup

```bash
# Install mpBAX (required dependency)
pip install -e /path/to/mpBAX

# Or with DANetModel support (recommended for real problems)
pip install -e "/path/to/mpBAX[torch]"
```

## Workflow

```
Optimizer proposes D  -->  NN predicts L from D  -->  Sim refines L to L*
                                                          |
Optimizer  <--  obj(D, L*)  <--  calc objectives  <-------+
```

**Mapping to mpBAX:**

| Concept | mpBAX component | File |
|---------|----------------|------|
| Design params D | X (input to oracle) | - |
| Locking params L* | Y (oracle output) | - |
| Simulation (D,L) -> L* | Oracle function | `oracles.py` |
| NN surrogate D -> L | Model (DANetModel) | via config |
| Objective calc (D,L) -> obj | Used inside Algorithm | `utils/calc.py` |
| Optimizer | Algorithm | `algorithm.py` |

The model learns `D -> L` (not `D -> objectives`). The algorithm computes objectives internally from `(D, L_predicted)` to rank candidates.

## What You Need to Implement

### 1. `utils/sim.py` — Simulation

Replace the mockup `simulate()` with your real simulation.

```python
def simulate(D, L_init=None, noise_scale=0.01):
    """
    Args:
        D: Design parameters, shape (n, d)
        L_init: Initial guess from surrogate, shape (n, k) or None
    Returns:
        L_star: Accurate locking parameters, shape (n, k)
    """
```

- `L_init` is the surrogate's prediction — use it to speed up convergence
- When `L_init is None` (loop 0, no trained model yet), handle gracefully
- Output shape `(n, k)` must be consistent across all calls

### 2. `utils/calc.py` — Objective Calculation

Replace the mockup `calc_objective()` with your real objective function.

```python
def calc_objective(D, L):
    """
    Args:
        D: Design parameters, shape (n, d)
        L: Locking parameters, shape (n, k)
    Returns:
        obj: Objective values, shape (n,) — lower is better
    """
```

This is called by the algorithm to rank candidates using surrogate predictions. It is never called by the engine directly.

### 3. `generators.py` — Initial Sampling (optional)

Replace if your design space is not `[0,1]^d` or you need structured initial sampling (e.g., Latin Hypercube).

```python
def gen_locking_initial(n, d):
    """Returns: D with shape (n, d)"""
```

## Files You Probably Don't Need to Change

| File | Role | When to change |
|------|------|----------------|
| `engine.py` | R switch logic (conditional retraining) | Only if R switch needs custom criteria beyond "every N loops" |
| `algorithm.py` | Surrogate-based candidate selection | Only if you need a different optimizer (e.g., GA, Bayesian) |
| `oracles.py` | Wraps simulation + model initial guess | Only if the oracle interface differs |

## Two Approaches

Controlled by `--approach` flag (or `locking.approach` in config):

**Approach 1** (`--approach 1`): F=always, R=configurable
- Every loop: simulation runs on proposed D -> L*
- Model retrains every N loops (`--retrain-every N`)
- Between retrains: data accumulates, model unchanged

**Approach 2** (`--approach 2`): R=always, inner loop
- Algorithm runs N fast surrogate-only iterations internally (`--inner-steps N`)
- Then returns best D for one simulation call
- Model retrains every loop

## Running

```bash
# Approach 1: simulate every loop, retrain every 3
python run.py --approach 1 --retrain-every 3 --max-loops 20

# Approach 2: 10 surrogate-only iterations per simulation
python run.py --approach 2 --inner-steps 10 --max-loops 20

# All options
python run.py --help
```

## Using DANetModel (recommended for real problems)

Replace `DummyModel` in `run.py` with:

```python
from mpbax.plugins.models.da_net_model import DANetModel

# In config:
'model': {
    'class': DANetModel,
    'params': {
        'epochs': 100,        # Initial training
        'epochs_iter': 10,    # Per-loop finetuning
        'n_neur': 400,        # Hidden layer width
        'dropout': 0.1,
        'weight_new_data': 10.0,
    }
}
```

Requires PyTorch: `pip install -e ".[torch]"`

## Output

Checkpoints saved to `--checkpoint-dir` (or temp dir):
```
checkpoints/locking/
  data_0.pkl ... data_N.pkl    # Saved every loop (D, L* pairs)
  model_0_final.pkl ...        # Saved only when model is retrained
```

Load archived data:
```python
from mpbax.core.data_handler import DataHandler
dh = DataHandler.load('checkpoints/locking/data_5.pkl')
D, L_star = dh.get_data()
```

## Quick Test

Run the notebook `test_locking.ipynb` to verify everything works with mockup functions before plugging in real simulation.
