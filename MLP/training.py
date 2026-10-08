"""Train a small MLP to predict log optical power from four cavity inputs.

Think of training as repeatedly showing the network examples and correcting its
predictions. ``data.py`` turns saved graphs into a table; ``model.py`` defines
how a row becomes a prediction; this module handles the learning process.
"""

import hashlib
import math
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import save_checkpoint
from .data import PowerDataset, DEFAULT_DETECTOR
from .model import build_power_predictor
from .losses import DEFAULT_LOSS_CONFIG, PowerPredictionLoss, resolve_loss_config


DEFAULT_TRAINING_PARAMS = {
    'batch_size': 32,
    'n_epochs': 200,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'save_loss_interval': 5,
    'print_interval': 10,
    'loss': dict(DEFAULT_LOSS_CONFIG),
}


def _training_params(overrides=None, defaults=None):
    """Merge partial settings, then catch mistakes before an expensive run."""
    params = dict(DEFAULT_TRAINING_PARAMS if defaults is None else defaults)
    inherited_loss = resolve_loss_config(params['loss'])
    if overrides is not None:
        unknown = set(overrides) - set(params)
        if unknown:
            raise ValueError(f'Unknown MLP training settings: {sorted(unknown)}')
        params.update(overrides)
    for name in ('batch_size', 'n_epochs', 'save_loss_interval', 'print_interval'):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer; got {value!r}.')
    for name in ('learning_rate', 'weight_decay'):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a finite number; got {value!r}.')
        if not math.isfinite(value) or value < 0 or (name == 'learning_rate' and value == 0):
            raise ValueError(f'{name} has an invalid value: {value!r}.')
    if not isinstance(params['loss'], dict):
        raise ValueError('MLP loss settings must be a dictionary.')
    # Partial overrides preserve a fine-tuned checkpoint's other loss settings.
    params['loss'] = resolve_loss_config({**inherited_loss, **params['loss']})
    return params


def _file_sha256(path):
    """Remember the exact base data, so fine-tuning cannot silently change it."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_output_paths(checkpoint_path):
    """A new experiment must not replace another experiment's saved work."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != '.pt':
        raise ValueError('MLP checkpoint paths must end in .pt.')
    for path in (checkpoint_path, checkpoint_path.with_suffix('.txt')):
        if path.exists():
            raise FileExistsError(f'MLP training output already exists: {path}')
    return checkpoint_path


def split_pair_indices(sample_count, train_fraction=0.8, seed=9302):
    """Keep each nominal example and its perturbation in the same split.

    Initial data is saved in consecutive pairs: rows 0/1 belong together, as do
    rows 2/3, and so on. Splitting individual rows would let validation contain
    near-copies of training examples and make accuracy look too optimistic.

    With just one pair there is no independent holdout. Both rows are used for
    training and monitoring, with an explicit warning about that limitation.
    """
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be strictly between 0 and 1.')
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError('sample_count must be an integer.')
    if sample_count < 2 or sample_count % 2:
        raise ValueError('Base training needs an even number of at least 2 paired samples.')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer.')
    if sample_count == 2:
        warnings.warn(
            'Two-sample bootstrap: both samples are used for training and checkpoint '
            'monitoring. The reported loss is not independent validation accuracy.',
            UserWarning,
            stacklevel=2,
        )
        return [0, 1], [0, 1]
    pair_count = sample_count // 2
    training_pairs = min(pair_count - 1, max(1, int(train_fraction * pair_count)))
    pair_order = torch.randperm(
        pair_count, generator=torch.Generator().manual_seed(seed)
    ).tolist()

    def expand(pairs):
        return [index for pair in pairs for index in (2 * pair, 2 * pair + 1)]

    return expand(pair_order[:training_pairs]), expand(pair_order[training_pairs:])


def _fit_model(
    model, train_x, train_y, train_weights, validation_x, validation_y,
    checkpoint_path, detector, params, metadata, *, seed,
):
    """Shared learning loop for initial training and later Finesse feedback.

    Each row has one weight. Base training uses weight 1 everywhere. Fine-tuning
    gives the newest rows extra influence while replaying older rows to reduce
    forgetting. Validation rows always have equal weight.
    """
    checkpoint_path = _check_output_paths(checkpoint_path)
    if len(train_x) == 0 or len(validation_x) == 0:
        raise ValueError('MLP training and checkpoint-monitoring sets must be non-empty.')
    if len(train_x) != len(train_y) or len(train_x) != len(train_weights):
        raise ValueError('Training inputs, targets, and weights must have matching lengths.')
    if len(validation_x) != len(validation_y):
        raise ValueError('Validation inputs and targets must have matching lengths.')
    if not torch.isfinite(train_weights).all() or (train_weights <= 0).any():
        raise ValueError('Every training weight must be finite and positive.')

    criterion = PowerPredictionLoss(params['loss'])
    device = next(model.parameters()).device
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_weights.reshape(-1)),
        batch_size=params['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        TensorDataset(validation_x, validation_y), batch_size=params['batch_size']
    )
    # AdamW adjusts the learning step separately for each model parameter.
    # Weight decay mildly discourages overly large weights on a small dataset.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay']
    )

    def validation_loss():
        model.eval()
        total = 0.0
        predictions_all, targets_all = [], []
        with torch.no_grad():  # Validation measures accuracy; it never updates weights.
            for inputs, targets in validation_loader:
                predictions = model(inputs.to(device))
                targets = targets.to(device)
                # Checkpoint selection must use the same center-focused objective
                # as training, but without the latest-round feedback weights.
                total += criterion(predictions, targets).sum().item()
                predictions_all.append(predictions.cpu())
                targets_all.append(targets.cpu())
        result = total / len(validation_x)
        if not math.isfinite(result):
            raise FloatingPointError('MLP validation loss became non-finite.')
        metrics = criterion.power_metrics(torch.cat(predictions_all), torch.cat(targets_all))
        return result, metrics

    def copy_state():
        # A clone is essential: a reference to model.state_dict() would continue
        # changing as later epochs update the network.
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    best_loss = math.inf
    best_state = None
    best_epoch = None
    best_metrics = None
    history = []
    monitor_label = (
        'training monitor' if metadata['validation_mode'] == 'training_monitor_only'
        else 'validation'
    )
    for epoch in range(1, params['n_epochs'] + 1):
        model.train()
        total_training_loss = 0.0
        for inputs, targets, weights in loader:
            inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            # The loss emphasizes fine power differences near the plateau.
            # Feedback weights still apply independently to the entire row loss.
            row_losses = criterion(predictions, targets)
            loss = (row_losses * weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'MLP training loss became non-finite at epoch {epoch}.')
            loss.backward()  # Backpropagation computes how each weight caused error.
            optimizer.step()  # Apply a small correction to the network's weights.
            total_training_loss += loss.item() * len(inputs)

        # Always check the first and last epochs, including one-epoch smoke runs.
        if epoch == 1 or epoch % params['save_loss_interval'] == 0 or epoch == params['n_epochs']:
            current_loss, metrics = validation_loss()
            training_loss = total_training_loss / len(train_x)
            history.append({
                'epoch': epoch, 'training_loss': training_loss,
                'validation_loss': current_loss, **metrics,
            })
            if current_loss < best_loss:
                best_loss, best_epoch, best_state = current_loss, epoch, copy_state()
                best_metrics = dict(metrics)
            if epoch == 1 or epoch % params['print_interval'] == 0 or epoch == params['n_epochs']:
                print(
                    f'MLP epoch {epoch}/{params["n_epochs"]}: '
                    f'training loss {training_loss:.4e}; {monitor_label} loss {current_loss:.4e}; '
                    f'power MAE {metrics["power_mae"]:.4e}; '
                    f'center MAE {metrics["center_power_mae"]} '
                    f'({metrics["center_sample_count"]} center rows)'
                )

    # Save the best measured epoch, which can precede the final epoch. Keeping
    # its state in memory also means an existing checkpoint is never overwritten.
    model.load_state_dict(best_state)
    model.eval()
    saved_metadata = dict(metadata)
    saved_metadata.update({
        'hyperparams': dict(params), 'best_epoch': best_epoch,
        'best_validation_loss': best_loss, 'training_row_count': len(train_x),
        'validation_row_count': len(validation_x), 'history': history,
        'loss_config': dict(criterion.config),
        'best_power_metrics': best_metrics,
    })
    save_checkpoint(checkpoint_path, model, detector, training_metadata=saved_metadata)
    with checkpoint_path.with_suffix('.txt').open('x', encoding='utf-8') as log:
        log.write(f'MLP loss: {criterion.config}; checkpoint selection: {monitor_label}.\n')
        log.write('epoch\tweighted_training_loss\tmonitor_loss\tpower_mae\tcenter_power_mae\tcenter_sample_count\n')
        for row in history:
            log.write(
                f'{row["epoch"]}\t{row["training_loss"]:.8e}\t{row["validation_loss"]:.8e}\t'
                f'{row["power_mae"]:.8e}\t{row["center_power_mae"]}\t{row["center_sample_count"]}\n'
            )
        log.write(f'Best epoch: {best_epoch}; best {monitor_label} loss: {best_loss:.8e}\n')
    return checkpoint_path


def train_base_power_predictor(
    dataset_path, checkpoint_path, hyperparams=None, *,
    detector=DEFAULT_DETECTOR, model_config=None, seed=9302,
):
    """Learn the initial MLP and return the path of its self-contained checkpoint.

    Input normalization uses training rows only. Validation examples must not
    influence even the input mean and scale: they represent unseen designs.
    The checkpoint retains both those statistics and the exact row split so
    later fine-tuning can keep the same independent validation examples.
    """
    params = _training_params(hyperparams)
    checkpoint_path = _check_output_paths(checkpoint_path)
    dataset_path = Path(dataset_path)
    dataset = PowerDataset([dataset_path], detector=detector)
    train_indices, validation_indices = split_pair_indices(len(dataset), seed=seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_power_predictor(model_config)
    model.set_normalization(dataset.x[train_indices])
    model = model.to(device)
    metadata = {
        'base_data_filename': dataset_path.name,
        'base_data_count': len(dataset),
        'base_data_sha256': _file_sha256(dataset_path),
        'split_seed': seed,
        'train_indices': train_indices,
        'validation_indices': validation_indices,
        'validation_mode': 'training_monitor_only' if len(dataset) == 2 else 'held_out_pairs',
        'current_round': 0,
    }
    print(f'MLP base data: {len(train_indices)} training rows, {len(validation_indices)} monitoring rows.')
    return _fit_model(
        model, dataset.x[train_indices], dataset.y[train_indices],
        torch.ones(len(train_indices)), dataset.x[validation_indices], dataset.y[validation_indices],
        checkpoint_path, detector, params, metadata, seed=seed,
    )
