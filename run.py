"""Run a complete surrogate-assisted ITM/ETM ROC optimization experiment.

The script performs the complete workflow:
  1. Generates space-filling design graphs plus fixed-q perturbation partners.
  2. Trains the base GNN checkpoint from those graphs.
  3. Runs PSO (using the current GNN checkpoint as the surrogate cost function) to
     find the best (ITM_Roc, ETM_Roc).
  4. Evaluates real Finesse at that point and writes round{N}.h5.
  5. Fine-tunes the GNN and repeats PSO -> Finesse -> fine-tune.

Set RUN_NUMBER below to a new experiment ID before launch. If its data or model
directory contains anything, startup fails rather than overwriting/mixing results.

Usage:
    python run.py
"""

import pickle

from fabry_perot import generate_initial_data
from train_power_predictor import train_base_power_predictor
from utils.finesse_base import base_kat, gnn_base_kat
from PSOoptimizer import main_wrapper_PSO, SteppablePSO
from finetune_power_predictor import run_finetune
from tqdm import tqdm
from pathlib import Path

# Change only this value to start a new experiment. For example, RUN_NUMBER = 7
# writes exclusively under GNN/data/run7 and GNN/models/run7.
RUN_NUMBER = 11

# Each optimized-LHS design point produces two graphs: the self-consistent design
# and one nearby geometry with the design q held fixed to teach q mismatch.
INITIAL_DESIGN_POINTS = 15
INITIAL_GRAPH_COUNT = 2 * INITIAL_DESIGN_POINTS

ITM_ROC_NOMINAL = -1934
ETM_ROC_NOMINAL = 2245
ITM_ROC_BOUNDS = (ITM_ROC_NOMINAL - 2200, ITM_ROC_NOMINAL + 500)
ETM_ROC_BOUNDS = (ETM_ROC_NOMINAL - 500, ETM_ROC_NOMINAL + 2000)

RUN_NAME = f'run{RUN_NUMBER}'
DATA_DIR = Path('GNN/data') / RUN_NAME
MODEL_DIR = Path('GNN/models') / RUN_NAME
FINETUNE_DATA_DIR = DATA_DIR
BASE_DATA_PATH = DATA_DIR / f'base_{INITIAL_GRAPH_COUNT}.h5'
BASE_CHECKPOINT_PATH = MODEL_DIR / f'base_{INITIAL_GRAPH_COUNT}_gat10_kan5.pt'
CHECKPOINT_TEMPLATE = str(MODEL_DIR / 'power_predictor_ligoParams_finetuned_round{}')


def _existing_run_artifacts(*run_dirs):
    """Return every existing artifact below the selected run directories."""
    artifacts = []
    for run_dir in run_dirs:
        if run_dir.exists() and not run_dir.is_dir():
            artifacts.append(run_dir)
        elif run_dir.is_dir():
            artifacts.extend(sorted(run_dir.rglob('*')))
    return artifacts


def _prepare_new_run():
    """Validate a fresh experiment ID, then generate data and train its base GNN."""
    if not isinstance(RUN_NUMBER, int) or isinstance(RUN_NUMBER, bool) or RUN_NUMBER < 1:
        raise ValueError(f'RUN_NUMBER must be a positive integer; got {RUN_NUMBER!r}.')

    conflicts = _existing_run_artifacts(DATA_DIR, MODEL_DIR)
    if conflicts:
        formatted = '\n'.join(f'  - {path.resolve()}' for path in conflicts)
        raise FileExistsError(
            f'{RUN_NAME} conflicts with an existing experiment. Choose a new '
            f'RUN_NUMBER; existing artifacts:\n{formatted}'
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f'[{RUN_NAME}] Generating {INITIAL_DESIGN_POINTS} optimized-LHS design '
        f'pairs ({INITIAL_GRAPH_COUNT} graphs)...'
    )
    generate_initial_data(
        BASE_DATA_PATH,
        num_design_points=INITIAL_DESIGN_POINTS,
        itm_bounds=ITM_ROC_BOUNDS,
        etm_bounds=ETM_ROC_BOUNDS,
        seed=9302,
    )
    print(f'[{RUN_NAME}] Training base GNN checkpoint...')
    train_base_power_predictor(
        dataset_path=BASE_DATA_PATH,
        checkpoint_path=BASE_CHECKPOINT_PATH,
        gat_layers=10,
        kan_layers=5,
        hyperparams={
            'batch_size': 100,
            'save_loss_interval': 5,
            'print_interval': 1,
            'n_epochs': 200,
            'learning_rate': 1e-5,
        },
    )
def _full_gnn_loop(loop_num, port, start_ITM_Roc, end_ITM_Roc, start_ETM_Roc, end_ETM_Roc, model_path=None, finetune_data_path=None):

    cost_fn_finesse = main_wrapper_PSO(base_kat, port, model_path=None, finetune_data_path=finetune_data_path)
    cost_fn_gnn     = main_wrapper_PSO(gnn_base_kat, port, model_path=model_path)

    # print("start building pso")
    max_iter = 100
    pso = SteppablePSO(func=cost_fn_gnn, n_dim=2, pop=20, max_iter=max_iter, lb=[start_ITM_Roc, start_ETM_Roc], ub=[end_ITM_Roc, end_ETM_Roc], w=0.9, c1=2.0, c2=0.4, verbose=True)

    # sko.PSO evaluates its initial swarm in __init__, but does not copy those
    # results into pbest/gbest. Preserve good initial candidates before moving the
    # particles on the first explicit step.
    pso.update_pbest()
    pso.update_gbest()

    # print("start initla run")

    for _ in tqdm(range(max_iter)):
        pso.step()

    print("Best parameters found: ", pso.gbest_x)

    finesse_cost = cost_fn_finesse(pso.gbest_x)

    return pso.gbest_x, pso.gbest_y, finesse_cost

def main():
    _prepare_new_run()

    total_loop_num = 40

    start_ITM_Roc, end_ITM_Roc = ITM_ROC_BOUNDS
    start_ETM_Roc, end_ETM_Roc = ETM_ROC_BOUNDS

    port = 'p_ITM_p2_i'

    data = []
    costs_gnn = []
    costs_finesse = []

    # Round 1 has no prior fine-tuned checkpoint yet, so start from the same base
    # checkpoint run_finetune itself would fall back to for round 1.
    model_path = BASE_CHECKPOINT_PATH

    for loop_num in tqdm(range(1, total_loop_num + 1)):
        finetune_data_path = FINETUNE_DATA_DIR / f"round{loop_num}.h5"
        datum, cost_gnn, cost_finesse = _full_gnn_loop(loop_num, port, start_ITM_Roc, end_ITM_Roc, start_ETM_Roc, end_ETM_Roc, model_path=model_path, finetune_data_path=finetune_data_path)
        save_path = run_finetune(data_dir=DATA_DIR,
            base_checkpoint_path=BASE_CHECKPOINT_PATH,
            checkpoint_template=CHECKPOINT_TEMPLATE, current_round=loop_num,
            finetune_weight=None, hyperparams={'batch_size': 100,
                'save_loss_interval': 5, 'print_interval': 1, 'n_epochs': 20,
                'learning_rate': 5e-7})
        # Next iteration's PSO search uses the checkpoint just produced by fine-tuning,
        # so the model actually improves across loop iterations instead of every
        # iteration searching against the same static starting model.
        model_path = Path(save_path + '.pt')
        if not model_path.is_file():
            raise RuntimeError(f'Fine-tuning did not produce checkpoint: {model_path.resolve()}')
        data.append(datum)
        costs_gnn.append(cost_gnn)
        costs_finesse.append(cost_finesse)
    
    return data, costs_gnn, costs_finesse
if __name__ == '__main__':
    data, costs_gnn, costs_finesse = main()

    with open(DATA_DIR / 'final_result.pkl', 'wb') as f:
        pickle.dump({'data': data, 'costs_gnn': costs_gnn, 'costs_finesse': costs_finesse}, f)