"""Run a complete surrogate-assisted ITM/ETM ROC optimization experiment.

Choose SURROGATE_TYPE = "GNN" or "MLP" below, then an unused RUN_NUMBER.
Both choices follow the same experiment:
  1. Generate Finesse design graphs and fixed-q perturbation partners.
  2. Train a base surrogate (MLP reads a numeric row from each saved graph).
  3. Search for mirror curvatures with PSO using predicted optical powers.
  4. Evaluate the selected design with Finesse and save its feedback graphs.
  5. Fine-tune the selected surrogate and repeat.

Outputs live under <SURROGATE_TYPE>/data/runN and <SURROGATE_TYPE>/models/runN.
Startup rejects existing artifacts instead of overwriting an earlier experiment.
The final result uses costs_surrogate/powers_surrogate/evaluations_surrogate for
both networks, plus matching *_gnn or *_mlp aliases. Finesse keys are unchanged.
Power records come from original evaluations; no second prediction is needed.

Usage:
    /home/xuesi.ma/.conda/envs/ligoopt/bin/python run.py
"""


###############################
# Add additional note to README
###############################

RUN_NOTES = "More perturbation data helped, but not enough. Many local response signs is wrong. Add explicit signed perturbation in model loss function as well as nominal-power loss. Use a fixed training-derived scale and keep everything else the same. Hope this can work better. GL"  # Optional: describe this experiment or what changed from the last run.

###############################
# END
###############################

import pickle

import perturbation
import PSOoptimizer
from GNN import power_predictor
from utils import calc
from utils.detectors import POWER_DETECTOR
from utils.run_metadata import write_run_metadata

from fabry_perot import generate_initial_data
from train_power_predictor import train_base_power_predictor
from utils.finesse_base import base_kat, gnn_base_kat
from utils.power_results import PowerEvaluationRecorder
from utils.training_groups import INITIAL_GROUP_SIZE, INITIAL_SAMPLE_TYPES
from PSOoptimizer import main_wrapper_PSO
from algorithm import SteppablePSO
from finetune_power_predictor import run_finetune
from tqdm import tqdm
from pathlib import Path

# This is the model switch. MLP = a small fully connected neural network;
# GNN = the existing network that passes information along optical graph edges.
# See MLP/README.md for a walkthrough of the new model and its training process.
SURROGATE_TYPE = "MLP"  # Change to "MLP" to run the MLP workflow.

# Choose an unused ID within the selected model's folder. For example, MLP/run7
# uses MLP/data/run7 and MLP/models/run7, independently of any GNN run7.
RUN_NUMBER = 4


# Each optimized-LHS point produces five graphs: its self-consistent design and
# ITM +/-, ETM +/- perturbations with the design q held fixed, matching the cost.
INITIAL_DESIGN_POINTS = 100
INITIAL_GRAPH_COUNT = INITIAL_GROUP_SIZE * INITIAL_DESIGN_POINTS

ITM_ROC_NOMINAL = -1934
ETM_ROC_NOMINAL = 2245
ITM_ROC_BOUNDS = (ITM_ROC_NOMINAL - 2200, ITM_ROC_NOMINAL + 500)
ETM_ROC_BOUNDS = (ETM_ROC_NOMINAL - 500, ETM_ROC_NOMINAL + 2000)

# Experiment controls. Objective settings remain in utils/calc.py, and
# architectures live in GNN/power_predictor.py and MLP/model.py.
INITIAL_DATA_SEED = 9302
TOTAL_ROUNDS = 40
DETECTOR = POWER_DETECTOR  # Shared with analysis; configured in utils/detectors.py.
PSO_SETTINGS = {
    'pop': 20,
    'max_iter': 100,
    'w': 0.9,
    'c1': 2.0,
    'c2': 0.4,
    'verbose': True,
}
# These two dictionaries apply only when SURROGATE_TYPE = "GNN".
# For MLP runs, edit MLP_BASE_TRAINING_PARAMS and MLP_FINETUNE_PARAMS below.
GNN_BASE_TRAINING_PARAMS = {
    'batch_size': 100,
    'save_loss_interval': 5,
    'print_interval': 1,
    'n_epochs': 200,
    'learning_rate': 1e-5,
}
GNN_FINETUNE_PARAMS = {
    'batch_size': 100,
    'save_loss_interval': 5,
    'print_interval': 1,
    'n_epochs': 20,
    'learning_rate': 5e-7,
}
# Supervise nominal power and the signed power responses used by the objective.
# The existing center-focused criterion supplies the nominal power term.
# Set supervision to "row_power" for the previous all-row training loss.
MLP_LOSS_CONFIG = {
    'mode': 'center_focused',
    'supervision': 'nominal_delta',
    'delta_weight': 1.0,
    'delta_scale_floor_w': 1.0,  # Fixed scale: max(floor, initial-training delta RMS).
    'center_power': 282.09,
    'center_width': 0.01,
    'center_error_scale': 0.001,
    'center_weight': 10.0,
}
MLP_BASE_TRAINING_PARAMS = {
    'loss': MLP_LOSS_CONFIG,
    'batch_size': 32,       # Number of examples used for one weight update.
    'n_epochs': 1000,       # One epoch visits every training example once.
    'learning_rate': 1e-3,  # Controls the size of each optimizer update.
    'weight_decay': 1e-4,   # Discourages unnecessarily large weights (overfitting).
    'save_loss_interval': 5,
    'print_interval': 10,
}
MLP_FINETUNE_PARAMS = {
    'loss': MLP_LOSS_CONFIG,
    'batch_size': 32,
    'n_epochs': 20,
    'learning_rate': 1e-4,  # Smaller updates when adapting an already trained MLP.
    'weight_decay': 1e-4,
    'save_loss_interval': 5,
    'print_interval': 5,
}
FINETUNE_WEIGHT = None  # Automatically balance initial and newest sample counts.


def _active_training_settings():
    """Select settings without changing the established GNN defaults."""
    if SURROGATE_TYPE == "GNN":
        return GNN_BASE_TRAINING_PARAMS, GNN_FINETUNE_PARAMS, power_predictor.DEFAULT_MODEL_CONFIG
    if SURROGATE_TYPE == "MLP":
        from MLP.model import DEFAULT_MODEL_CONFIG
        return MLP_BASE_TRAINING_PARAMS, MLP_FINETUNE_PARAMS, DEFAULT_MODEL_CONFIG
    raise ValueError(f'Unknown SURROGATE_TYPE {SURROGATE_TYPE!r}; choose "GNN" or "MLP".')


# Derived paths are anchored to this repository, independently of the launch directory.
REPO_ROOT = Path(__file__).resolve().parent
RUN_NAME = f'run{RUN_NUMBER}'
DATA_DIR = REPO_ROOT / SURROGATE_TYPE / 'data' / RUN_NAME
MODEL_DIR = REPO_ROOT / SURROGATE_TYPE / 'models' / RUN_NAME
FINETUNE_DATA_DIR = DATA_DIR
BASE_DATA_PATH = DATA_DIR / f'base_{INITIAL_GRAPH_COUNT}.h5'
# Preserve old GNN filenames; MLP checkpoints contain a different model and also
# store input scaling, detector name, and training/validation split information.
BASE_CHECKPOINT_PATH = MODEL_DIR / (
    f'base_{INITIAL_GRAPH_COUNT}_gat10_kan5.pt' if SURROGATE_TYPE == "GNN"
    else f'base_{INITIAL_GRAPH_COUNT}_mlp.pt'
)
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
    """Validate a fresh experiment ID, then generate data and train its surrogate."""
    base_params, finetune_params, model_config = _active_training_settings()
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

    write_run_metadata(
        DATA_DIR / 'metadata.json',
        repo_root=REPO_ROOT,
        settings={
            'experiment': {
                'run_number': RUN_NUMBER, 'planned_rounds': TOTAL_ROUNDS,
                'notes': RUN_NOTES,
            },
            'sampling': {
                'method': 'optimized_lhs',
                'itm_roc_bounds_m': ITM_ROC_BOUNDS,
                'etm_roc_bounds_m': ETM_ROC_BOUNDS,
                'design_points': INITIAL_DESIGN_POINTS,
                'initial_graph_count': INITIAL_GRAPH_COUNT,
                'design_group_size': INITIAL_GROUP_SIZE,
                'sample_types': INITIAL_SAMPLE_TYPES,
                'seed': INITIAL_DATA_SEED,
                'roc_perturbation_fraction': perturbation.roc_perturbation_percent,
            },
            'pso': PSO_SETTINGS,
            'training': {
                'base': base_params,
                'finetune': finetune_params,
                'finetune_weight': FINETUNE_WEIGHT,
                'finetune_weight_mode': 'automatic' if FINETUNE_WEIGHT is None else 'explicit',
            },
            'surrogate_type': SURROGATE_TYPE,
            SURROGATE_TYPE.lower(): model_config,
            'objective': {
                'detector': DETECTOR,
                'evaluation_mode': PSOoptimizer.EVALUATION_MODE,
                'cost_mode': calc.COST_MODE,
                'power_weight': calc.FULL_POWER_WEIGHT,
                'perturbation_weight': calc.FULL_PERTURBATION_WEIGHT,
                'roc_perturbation_fraction': perturbation.roc_perturbation_percent,
            },
        },
        paths={
            'data_directory': DATA_DIR,
            'model_directory': MODEL_DIR,
            'base_data': BASE_DATA_PATH,
            'initial_samples': BASE_DATA_PATH.with_name('initial_samples.csv'),
            'base_checkpoint': BASE_CHECKPOINT_PATH,
            'round_data_template': FINETUNE_DATA_DIR / 'round{}.h5',
            'finetune_checkpoint_template': CHECKPOINT_TEMPLATE + '.pt',
            'final_result': DATA_DIR / 'final_result.pkl',
        },
    )

    print(
        f'[{RUN_NAME}] Generating {INITIAL_DESIGN_POINTS} optimized-LHS '
        f'designs with four perturbations each ({INITIAL_GRAPH_COUNT} graphs)...'
    )
    generate_initial_data(
        BASE_DATA_PATH,
        num_design_points=INITIAL_DESIGN_POINTS,
        itm_bounds=ITM_ROC_BOUNDS,
        etm_bounds=ETM_ROC_BOUNDS,
        seed=INITIAL_DATA_SEED,
    )
    print(f'[{RUN_NAME}] Training base {SURROGATE_TYPE} checkpoint...')
    if SURROGATE_TYPE == "MLP":
        from MLP.training import train_base_power_predictor as train_base_mlp
        train_base_mlp(
            dataset_path=BASE_DATA_PATH,
            checkpoint_path=BASE_CHECKPOINT_PATH,
            hyperparams=base_params,
            detector=DETECTOR,
            model_config=model_config,
            seed=INITIAL_DATA_SEED,
        )
    else:
        train_base_power_predictor(
            dataset_path=BASE_DATA_PATH,
            checkpoint_path=BASE_CHECKPOINT_PATH,
            hyperparams=base_params,
        )


def _full_gnn_loop(
    loop_num, port, start_ITM_Roc, end_ITM_Roc, start_ETM_Roc, end_ETM_Roc,
    model_path=None, finetune_data_path=None,
):
    """Run one search/feedback round with either model (legacy function name)."""
    finesse_recorder = PowerEvaluationRecorder()
    surrogate_recorder = PowerEvaluationRecorder()
    cost_fn_finesse = main_wrapper_PSO(
        base_kat, port, model_path=None, finetune_data_path=finetune_data_path,
        evaluation_recorder=finesse_recorder,
    )
    cost_fn_surrogate = main_wrapper_PSO(
        gnn_base_kat, port, model_path=model_path, evaluation_recorder=surrogate_recorder,
        surrogate_type=SURROGATE_TYPE,
    )

    pso = SteppablePSO(
        func=cost_fn_surrogate,
        n_dim=2,
        lb=[start_ITM_Roc, start_ETM_Roc],
        ub=[end_ITM_Roc, end_ETM_Roc],
        **PSO_SETTINGS,
    )

    # sko.PSO evaluates its initial swarm in __init__, but does not copy those
    # results into pbest/gbest. Preserve good initial candidates before moving the
    # particles on the first explicit step.
    pso.update_pbest()
    pso.update_gbest()

    for _ in tqdm(range(PSO_SETTINGS['max_iter'])):
        pso.step()

    print("Best parameters found: ", pso.gbest_x)

    # Fail before running Finesse if no valid surrogate optimum was found. Verification
    # uses only captured numbers, never a second prediction at the winning point.
    surrogate_record = surrogate_recorder.verified_record(pso.gbest_x, pso.gbest_y)
    finesse_cost = cost_fn_finesse(pso.gbest_x)
    finesse_record = finesse_recorder.verified_record(pso.gbest_x, finesse_cost)

    return pso.gbest_x, pso.gbest_y, finesse_cost, surrogate_record, finesse_record


def main():
    _prepare_new_run()
    _, finetune_params, _ = _active_training_settings()
    if SURROGATE_TYPE == "MLP":
        from MLP.finetune import run_finetune as finetune_surrogate
    else:
        finetune_surrogate = run_finetune

    start_ITM_Roc, end_ITM_Roc = ITM_ROC_BOUNDS
    start_ETM_Roc, end_ETM_Roc = ETM_ROC_BOUNDS

    port = DETECTOR

    data = []
    costs_surrogate = []
    costs_finesse = []
    evaluations_surrogate = []
    evaluations_finesse = []

    # Round 1 has no prior fine-tuned checkpoint yet, so start from the same base
    # checkpoint run_finetune itself would fall back to for round 1.
    model_path = BASE_CHECKPOINT_PATH

    for loop_num in tqdm(range(1, TOTAL_ROUNDS + 1)):
        finetune_data_path = FINETUNE_DATA_DIR / f"round{loop_num}.h5"
        datum, cost_surrogate, cost_finesse, surrogate_record, finesse_record = _full_gnn_loop(
            loop_num, port, start_ITM_Roc, end_ITM_Roc, start_ETM_Roc, end_ETM_Roc,
            model_path=model_path, finetune_data_path=finetune_data_path,
        )
        save_path = finetune_surrogate(
            data_dir=DATA_DIR,
            base_checkpoint_path=BASE_CHECKPOINT_PATH,
            checkpoint_template=CHECKPOINT_TEMPLATE, current_round=loop_num,
            finetune_weight=FINETUNE_WEIGHT, hyperparams=finetune_params,
        )
        # Next iteration's PSO search uses the checkpoint just produced by fine-tuning,
        # so the model actually improves across loop iterations instead of every
        # iteration searching against the same static starting model.
        model_path = Path(save_path + '.pt')
        if not model_path.is_file():
            raise RuntimeError(f'Fine-tuning did not produce checkpoint: {model_path.resolve()}')
        data.append(datum)
        costs_surrogate.append(cost_surrogate)
        costs_finesse.append(cost_finesse)
        evaluations_surrogate.append(surrogate_record)
        evaluations_finesse.append(finesse_record)
    
    # All lists are aligned by round. Named power entries avoid ambiguous column
    # ordering; full records retain checkpoint and settings for cost reconstruction.
    result = {
        'data': data,
        'costs_surrogate': costs_surrogate,
        'costs_finesse': costs_finesse,
        'powers_surrogate': [record['powers'] for record in evaluations_surrogate],
        'powers_finesse': [record['powers'] for record in evaluations_finesse],
        'evaluations_surrogate': evaluations_surrogate,
        'evaluations_finesse': evaluations_finesse,
    }

    # Generic keys make comparisons easy; model-specific aliases preserve existing
    # GNN notebooks and label MLP predictions honestly when that model is selected.
    suffix = SURROGATE_TYPE.lower()
    for field in ('costs', 'powers', 'evaluations'):
        result[f'{field}_{suffix}'] = result[f'{field}_surrogate']
    return result


if __name__ == '__main__':
    result = main()
    with open(DATA_DIR / 'final_result.pkl', 'wb') as f:
        pickle.dump(result, f)
