# Fabry–Perot Cavity Optimization

Optimize the radii of curvature (ROC) of the input test mass (ITM) and end test mass (ETM) in a Fabry–Perot cavity. A selectable graph neural network (GNN) or multilayer perceptron (MLP) predicts optical powers, particle swarm optimization (PSO) searches for a low-cost geometry, and Finesse simulations provide feedback to improve the selected surrogate.

The objective combines nominal optical power and sensitivity to small changes in mirror curvature. Lower cost is better. Finesse supplies the training labels and evaluates the geometry selected by each PSO search.

## Workflow

```text
Sample ITM/ETM radii using optimized Latin hypercube sampling
    ↓
Run Finesse and save design/perturbation graph pairs
    ↓
Train the selected surrogate
    ↓
Search with PSO using surrogate-predicted powers ←────────┐
    ↓                                                   │
Evaluate the selected geometry with Finesse              │
    ↓                                                   │
Save the new graphs and fine-tune the surrogate ──────────┘
    ↓ after the final round
Save selected geometries, costs, powers, and evaluation metadata
```

Each initial design contributes two training graphs: its nominal cavity and a nearby geometry with the nominal beam parameter `q` held fixed. During optimization, perturbed evaluation uses five geometries: the nominal design and ITM/ETM positive and negative perturbations. Holding the nominal `q` fixed captures the effect of beam mismatch when the mirror geometry changes.

The GNN consumes graph features describing mirror properties, beam parameters, and optical connections. It learns `log1p(power)` per node; inference converts predictions back to powers before calculating the objective. The surrogate path still uses Finesse model construction and beam tracing, but avoids full Finesse power simulations during PSO searches.

## Choose GNN or MLP

Set `SURROGATE_TYPE = "MLP"` in [run.py](run.py) to select the MLP for base training,
PSO inference, and every fine-tuning round. `"GNN"` remains the default. MLP runs
write to `MLP/data/runN/` and `MLP/models/runN/`; GNN runs keep their existing paths.
Choose an unused `RUN_NUMBER` within the selected model's directories.

The MLP uses four inputs (both mirror curvatures and the real/imaginary input q),
two 64-neuron hidden layers, and one output for `DETECTOR`. It reuses saved Finesse
graphs as tabular training examples and still uses beam tracing for q during PSO.
Its learning rates and batch sizes are configured separately through
`MLP_BASE_TRAINING_PARAMS` and `MLP_FINETUNE_PARAMS` in `run.py`.

Start with the [MLP tutorial](MLP/README.md) for an explanation of each module,
input scaling, layers, training, checkpoints, and feedback. The MLP's base
validation pairs remain held out throughout fine-tuning, and its initial
training-only input scaling stays fixed. `MLP_LOSS_CONFIG` adds extra loss sensitivity
near the 282.09 power plateau, with center-only validation error recorded in each
checkpoint and training log. See the [loss tutorial](MLP/README.md#6-learn-the-central-power-plateau-more-accurately)
for the formula and settings. GNN training behavior is unchanged.

## Environment

Run commands from the repository root. On the project machine, use the existing environment explicitly:

```bash
cd /home/xuesi.ma/ligo-opt
/home/xuesi.ma/.conda/envs/ligoopt/bin/python --version
```

The main dependencies are Finesse, PyTorch, PyTorch Geometric, scikit-opt (`sko`), NumPy, SciPy, NetworkX, h5py, tqdm, and Matplotlib. Notebook work also needs a Jupyter environment using the same Python interpreter. Training and inference select CUDA when available and otherwise use the CPU. The multiprocessing baseline and Finesse cost-map scripts target Linux.

This repository does not yet provide a dependency lockfile or environment specification. Use the existing `ligoopt` environment; for a new machine, obtain its environment specification from the project maintainer. If installing packages into this environment, invoke pip as `/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m pip`.

## Start an optimization experiment

Edit [run.py](run.py) before launching. Its experiment settings are grouped as Python constants at the top; it has no command-line argument parser or YAML configuration loader.

| Setting | Location | Current value or behavior |
| --- | --- | --- |
| Surrogate | `run.py`: `SURROGATE_TYPE` | `"GNN"` or `"MLP"`; default `"GNN"` |
| Experiment ID | `run.py`: `RUN_NUMBER` | `12`; choose a new, unused positive integer |
| Initial sampling seed | `run.py`: `INITIAL_DATA_SEED` | 9302 |
| Initial design count | `run.py`: `INITIAL_DESIGN_POINTS` | 100 designs, producing 200 graphs |
| ITM ROC bounds | `run.py`: `ITM_ROC_BOUNDS` | −4134 to −1434 m |
| ETM ROC bounds | `run.py`: `ETM_ROC_BOUNDS` | 1745 to 4245 m |
| Feedback rounds | `run.py`: `TOTAL_ROUNDS` | 40 |
| PSO settings | `run.py`: `PSO_SETTINGS` | 20 particles, 100 iterations per round |
| Selected detector | `run.py`: `DETECTOR` | `p_ITM_p2_i` |
| GNN base training | `run.py`: `GNN_BASE_TRAINING_PARAMS` | 200 epochs, learning rate `1e-5` |
| GNN fine-tuning | `run.py`: `GNN_FINETUNE_PARAMS` | 20 epochs per round, learning rate `5e-7` |
| MLP base training | `run.py`: `MLP_BASE_TRAINING_PARAMS` | 1,000 epochs, learning rate `1e-3` |
| MLP fine-tuning | `run.py`: `MLP_FINETUNE_PARAMS` | 20 epochs per round, learning rate `1e-4` |
| Newest-data loss weight | `run.py`: `FINETUNE_WEIGHT` | `None` selects automatic weighting |
| MLP architecture | `MLP/model.py`: `DEFAULT_MODEL_CONFIG` | 4 → 64 → 64 → 1, with saved input standardization |
| GNN architecture | `GNN/power_predictor.py`: `DEFAULT_MODEL_CONFIG` | Shared by base training, fine-tuning, and inference |
| Objective | `utils/calc.py`: `COST_MODE` | `"full"`, with both weights set to `0.5` |
| Evaluation mode | `PSOoptimizer.py`: `EVALUATION_MODE` | `"perturbed"` |
| Optimization perturbation | `perturbation.py`: `roc_perturbation_percent` | `0.0033`, or 0.33% of each ROC magnitude |

Data and model paths in `run.py` are anchored to the repository, so launching it from a different working directory uses the same locations.

Initial-data perturbations have a separate `ROC_PERTURBATION_PERCENT` setting in `fabry_perot.py`, also currently `0.0033`.

Launch the complete experiment after choosing the settings:

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python run.py
```

Startup rejects a run if its data or model directory contains artifacts. The script starts a fresh experiment; it does not resume an interrupted optimization. Initial training and per-round files are written along the way, while `final_result.pkl` is written only after all rounds finish.

Round 1 searches with the base checkpoint. Round N searches with the checkpoint produced by fine-tuning round N−1. Every graph from the newest Finesse round participates in fine-tuning; historical graphs supply replay and validation data. The default newest-data loss weight is `max(1, initial_graph_count / newest_round_graph_count)`.

### Objective and evaluation modes

Configure the objective in [utils/calc.py](utils/calc.py):

- `design_power`: inverse nominal power, `1 / P_nominal`.
- `perturbation`: sum of the four absolute power changes divided by their corresponding ROC step magnitudes.
- `full`: weighted sum of the two terms using `FULL_POWER_WEIGHT` and `FULL_PERTURBATION_WEIGHT`. These terms have different units; the weights control their relative contribution.

Invalid nominal power receives infinite cost. Perturbation cost also rejects nonfinite or negative selected powers and invalid step magnitudes.

In [PSOoptimizer.py](PSOoptimizer.py), `EVALUATION_MODE="perturbed"` supplies all five evaluations. `"nominal"` supplies only the design point and requires `COST_MODE="design_power"`. In perturbed mode, either surrogate evaluates a swarm in two batches: nominal geometries followed by their fixed-q perturbations. Finesse feedback writes five graphs per selected design; nominal mode writes one.

`RECORD_FULL_COST=True` in `utils/calc.py` currently appends full-cost evaluations to `analysis/results/full_cost_records.csv`. This log combines calls across runs and scripts. Set it to `False` to disable that diagnostic output.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| [run.py](run.py) | Configure and orchestrate the complete experiment |
| [generators.py](generators.py) | Sample initial ITM/ETM geometries using optimized LHS and validate bounds |
| [fabry_perot.py](fabry_perot.py) | Coordinate sampling, simulate designs and perturbation partners, and write paired training graphs |
| [train_power_predictor.py](train_power_predictor.py) | Load graph datasets, split initial pairs, and train the base GNN |
| [finetune_power_predictor.py](finetune_power_predictor.py) | Load prior/new round data and fine-tune the previous checkpoint |
| [algorithm.py](algorithm.py) | Implement `SteppablePSO` for the main optimization workflow |
| [PSOoptimizer.py](PSOoptimizer.py) | Build GNN, MLP, or Finesse objective functions |
| [perturbation.py](perturbation.py) | Calculate the four ROC perturbations and their step magnitudes |
| [utils/finesse_base.py](utils/finesse_base.py) | Define the cavity, lock, detectors, and model templates |
| [oracles.py](oracles.py) | Run Finesse and extract node powers and beam parameters |
| [utils/calc.py](utils/calc.py) | Calculate objectives from nominal and perturbed powers |
| [utils/run_metadata.py](utils/run_metadata.py) | Save startup settings, code revision, software versions, and artifact paths |
| [utils/power_results.py](utils/power_results.py) | Capture winning evaluations and reconstruct their costs |
| [GNN/power_predictor.py](GNN/power_predictor.py) | Define `LinGNN` and the shared `build_power_predictor()` factory |
| [GNN/GNN_run.py](GNN/GNN_run.py) | Load checkpoints and perform scalar or batched power prediction |
| [GNN/GNN_utils.py](GNN/GNN_utils.py) | Construct modified cavity models, convert models to graphs, and append HDF5 data |

Start reading with `run.py`, then follow the relevant module for sampling, training, or evaluation.

The shared model factory uses graph transformer layers followed by ordinary linear layers. Change `DEFAULT_MODEL_CONFIG` to keep all workflow stages consistent. Training and fine-tuning also accept `graph_layers` and `linear_layers` overrides; use the same overrides when loading those checkpoints with `GNNPowerPredictor`. The older `gat_layers` and `kan_layers` training keywords remain accepted as aliases. Existing `gat10_kan5` checkpoint filenames are retained for compatibility.

## Experiment outputs

For run N with G initial graphs:

```text
GNN/data/runN/
    metadata.json
    README.md
    base_G.h5
    round1.h5 ... round40.h5
    final_result.pkl
GNN/models/runN/
    base_G_gat10_kan5.pt
    base_G_gat10_kan5.txt
    power_predictor_ligoParams_finetuned_round1.pt
    power_predictor_ligoParams_finetuned_round1.txt
    ... matching checkpoint/log pairs through round40
```

The round count follows `run.py`. HDF5 files store serialized NetworkX graphs under `sim_0`, `sim_1`, and subsequent keys. Model `.pt` files store weights, and matching `.txt` files contain training logs. Experiment data in `GNN/data/`, model checkpoints, and `Figures/` are ignored by Git. Keep local copies or share these artifacts separately; existing historical commits may still contain data. Generated results in `analysis/results/` are also ignored by Git and kept locally.

Every new run writes `README.md` alongside `metadata.json` in `<SURROGATE_TYPE>/data/runN/`, before data generation. The README provides a readable snapshot of the run's purpose, settings, output paths, code revision, and software versions. Set the optional `RUN_NOTES` string in `run.py` to describe what you are trying or what changed from an earlier run. These notes are saved in both files. Existing runs are not modified.

`metadata.json` is written once after the fresh-run check, before data generation. It records the UTC start time, run number, planned rounds, sampling bounds/count/seed, both ROC perturbation fractions (0.0033 means 0.33%), PSO settings, training parameters, fine-tuning weight setting, selected model architecture, detector and objective settings, Git commit and dirty flag, Python/package versions, and repository-relative artifact paths. A `null` fine-tuning weight with mode `automatic` means the weight will be calculated from actual graph counts during each round.

The file stays available if the experiment stops early; its presence does not indicate completion. Existing metadata is never overwritten. Unavailable Git information or package versions are recorded as `null`. The dirty flag includes tracked edits and untracked files, excluding ignored outputs; it does not preserve the edits themselves. This startup record does not capture every random state or enable resume/identical replay. It is ignored with the rest of `GNN/data/`; copy it along with the experiment artifacts when sharing a run.

Both models now also save the model-neutral `costs_surrogate`, `powers_surrogate`,
and `evaluations_surrogate` lists. MLP uses `*_mlp` aliases; GNN retains `*_gnn`
aliases for existing analysis. Finesse entries retain their names. Startup metadata
records `surrogate_type` and the selected model's architecture and training settings.
The example tree above and older GNN-specific analysis scripts use GNN filenames.

The GNN `final_result.pkl` contains round-aligned lists:

- `data`: selected `(ITM_ROC, ETM_ROC)` geometries.
- `costs_gnn`, `costs_finesse`: predicted and simulated costs at those geometries.
- `powers_gnn`, `powers_finesse`: detector powers named `nominal`, `ITM_plus`, `ITM_minus`, `ETM_plus`, and `ETM_minus`. Unevaluated perturbations are `None` in nominal mode.
- `evaluations_gnn`, `evaluations_finesse`: powers together with geometry, cost, detector, checkpoint path, perturbation steps, modes, and weights.

For example, inspect a completed run in a notebook using the project interpreter:

```python
import pickle
from pathlib import Path
from utils.power_results import reconstruct_cost

result_path = Path("GNN/data/run12/final_result.pkl")  # select your completed run
with result_path.open("rb") as stream:
    result = pickle.load(stream)

print(result["data"][-1])
print(result["powers_finesse"][-1])
print(reconstruct_cost(result["evaluations_finesse"][-1]))
```

Cost reconstruction uses saved numbers without rerunning inference or simulation. Older result files may contain only `data`, `costs_gnn`, and `costs_finesse`.

## Baselines and analysis

The `analysis/` folder contains baseline scripts, cost-map tools, and exploratory notebooks. Run its scripts as modules from the repository root. Baseline text results and the shared full-cost CSV live in `analysis/results/`. The baseline defaults to `analysis/results/Testing_Ground_results_perturbed.txt`.

[run_finesse_pso.py](analysis/run_finesse_pso.py) runs a Finesse-only PSO baseline with a bounded worker pool. It evaluates the nominal design and all four perturbations using the objective selected in `utils/calc.py`:

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.run_finesse_pso --help
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.run_finesse_pso --workers 8 --population 16 --iterations 20 --output analysis/results/baseline_results.txt
```

[perturbation_cost_map_finesse.py](analysis/perturbation_cost_map_finesse.py) and [perturbation_cost_map_gnn.py](analysis/perturbation_cost_map_gnn.py) compute resumable 50×50 cost maps. Both always evaluate all five geometries and follow `COST_MODE` and its weights. Their default detector is `p_ITM_p2_i`, shared with `run.py` through `utils/detectors.py`. New Finesse/GNN map filenames include the detector name; older `p_ETM_p1_i` caches are preserved and must be recomputed for comparisons at the workflow detector.

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.perturbation_cost_map_finesse --help
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m analysis.perturbation_cost_map_gnn --help
```

Default script output and checkpoint paths are anchored to the repository location. Explicit `--output` paths are used as supplied (relative paths resolve from the working directory). Maps are saved under `GNN/data/subspace/` by default. Rerunning resumes compatible output; `--overwrite` starts it again. For GNN maps, select the checkpoint with `--model-path`; the script currently defaults to run 11, round 40. Checkpoint and objective metadata are checked before resuming.

Open notebooks with the `ligoopt` kernel and run their setup cells first. Their setup resolves the repository root for project imports and existing `GNN/` and `Figures/` paths. The runtime notebook already supplies this setup; the other three notebooks now include it. Scientific cells and saved outputs were preserved. `finesse_test.ipynb` still contains a legacy tuple argument to `finesse_sim()` that needs repair before running all its cells.

- [subspace_plot.ipynb](analysis/subspace_plot.ipynb): cost maps, optimization trajectories, and GNN/Finesse comparisons. Select the intended run, checkpoints, detector, and map files before executing.
- [PSO_runtime_analysis.ipynb](analysis/PSO_runtime_analysis.ipynb): profile the surrogate PSO workflow; review its run/checkpoint settings before executing.
- [Testing_Ground.ipynb](analysis/Testing_Ground.ipynb) and [finesse_test.ipynb](analysis/finesse_test.ipynb): exploratory simulation notebooks.

## Tests

Local regression tests live in `tests/`, which is ignored by Git and is not included in a fresh clone. When that folder is available, run the tests from the repository root:

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

The current tests cover initial sampling and paired graph splits (`test_initial_data.py`), fine-tuning weights (`test_finetune_weight.py`), GNN model reuse and batching (`test_gnn_model_reuse.py`), and captured powers and result alignment (`test_power_results.py`). `test_analysis_paths.py` also checks default output locations and CSV append behavior. `test_workflow_cleanup.py` compares model predictions and training updates against the pre-cleanup implementation. These tests exercise focused behavior rather than launching a full optimization experiment.

The new MLP checks are tracked under `MLP/tests/` and run separately:

```bash
/home/xuesi.ma/.conda/envs/ligoopt/bin/python -m unittest discover -s MLP/tests -p 'test_*.py' -v
```

## Legacy files

The earlier locking-parameter template is preserved in [archive/locking_template/](archive/locking_template/): `algorithm.py`, `generators.py`, `oracles.py`, and `test_locking.ipynb`. See the [archive notes](archive/README.md) for their status. These files are outside the current `run.py` workflow; mpBAX is not required by the current experiment.

[NN_Playground.ipynb](archive/notebooks/NN_Playground.ipynb) is archived under `archive/notebooks/`. It references the removed `run_GNN` interface and old dataset/checkpoint paths, and needs repair before being used as a runnable example.
