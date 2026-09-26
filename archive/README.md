# Archive

This directory preserves retired experiments and implementations for reference.
Archived files are not part of the current optimization workflow and are not
maintained as runnable examples.

## Locking-parameter template

`locking_template/` contains the original files moved from the repository root:

| File | Original role |
| --- | --- |
| `algorithm.py` | mpBAX candidate selection through `LockingAlgorithm` |
| `generators.py` | Uniform initial sampling through `gen_locking_initial` |
| `oracles.py` | Early Finesse wrapper through `FPOracle` |
| `test_locking.ipynb` | Demonstration of the earlier locking workflow |

The files were moved without changing their contents. Their original imports
were preserved; moving them does not make the old workflow runnable. In
particular, the notebook expects the missing `engine.py`, `LockingOracle`, and
`calc_objective` interfaces, and `algorithm.py` also expects `calc_objective`.

The active workflow starts in `../run.py`. `SteppablePSO` now lives in
`../algorithm.py`; `ParallelPSO` remains in `../analysis/run_finesse_pso.py` for the
Finesse-only baseline. Finesse execution now lives in
`../oracles.py`. The root `../generators.py` has been repurposed for optimized-LHS
sampling and bounds validation; `../fabry_perot.py` coordinates initial simulations
and graph storage. The archived `locking_template/generators.py` is the original
uniform sampler and remains unchanged. The root `../oracles.py` now contains the
simulation functions formerly in `../utils/sim.py`; the archived oracle is unchanged.
The root `../algorithm.py` contains the steppable PSO class formerly in
`../PSOoptimizer.py`; the archived locking algorithm remains unchanged.

## Neural-network playground

`notebooks/NN_Playground.ipynb` was moved unchanged from the repository root.
It preserves an earlier inference experiment that imports the removed `run_GNN`
interface and references missing dataset/checkpoint paths. Its original imports,
paths, and outputs are retained for reference; it is not a runnable example of
the current workflow.
