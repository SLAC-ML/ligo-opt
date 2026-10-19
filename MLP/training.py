"""Train a small MLP to predict log optical power from four cavity inputs.

Think of training as repeatedly showing the network examples and correcting its
predictions. ``data.py`` turns saved graphs into a table; ``model.py`` defines
how a row becomes a prediction; this module handles the learning process.
"""

import hashlib
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import save_checkpoint
from .data import PowerDataset, DEFAULT_DETECTOR
from .model import build_power_predictor
from .losses import DEFAULT_LOSS_CONFIG, PowerPredictionLoss, resolve_loss_config, signed_delta_losses
from .groups import design_row_metadata, initial_delta_scale, response_metrics
from utils.training_groups import read_design_group_size, split_design_indices


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


def split_pair_indices(sample_count, train_fraction=0.8, seed=9302, *, group_size=2):
    """Keep complete designs together; the legacy function name defaults to pairs."""
    return split_design_indices(sample_count, train_fraction, seed, group_size=group_size)


def _fit_model(
    model, train_x, train_y, train_weights, validation_x, validation_y,
    checkpoint_path, detector, params, metadata, *, seed,
    train_nominal_indices=None, train_group_sizes=None,
    validation_nominal_indices=None, validation_group_sizes=None, delta_scale_w=None,
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
    paired = criterion.config['supervision'] == 'nominal_delta'
    if paired:
        if any(value is None for value in (
                train_nominal_indices, train_group_sizes,
                validation_nominal_indices, validation_group_sizes)):
            raise ValueError('Nominal/delta training requires explicit design references.')
        if (isinstance(delta_scale_w, bool) or not isinstance(delta_scale_w, (int, float))
                or not math.isfinite(delta_scale_w) or delta_scale_w <= 0):
            raise ValueError('Nominal/delta training requires a fixed positive delta_scale_w.')
    device = next(model.parameters()).device
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_weights.reshape(-1), torch.arange(len(train_x))),
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
                # Preserve the existing all-row power criterion for checkpoint
                # selection, isolating this experiment's training-loss change.
                total += criterion(predictions, targets).sum().item()
                predictions_all.append(predictions.cpu())
                targets_all.append(targets.cpu())
        result = total / len(validation_x)
        if not math.isfinite(result):
            raise FloatingPointError('MLP validation loss became non-finite.')
        predictions_all, targets_all = torch.cat(predictions_all), torch.cat(targets_all)
        metrics = criterion.power_metrics(predictions_all, targets_all)
        if paired:
            metrics.update(response_metrics(
                predictions_all, targets_all, validation_nominal_indices,
                validation_group_sizes, validation_x,
            ))
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
        total_nominal_loss, total_delta_loss = 0.0, 0.0
        for inputs, targets, weights, row_indices in loader:
            inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
            optimizer.zero_grad()
            if paired:
                nominal_indices = train_nominal_indices[row_indices]
                # Preserve the sampled rows and optimizer schedule. Add nominal
                # references only for this forward pass, keeping both sides
                # attached to autograd and deduplicating shared inputs.
                references, positions = torch.unique(
                    torch.cat((row_indices, nominal_indices)), return_inverse=True,
                )
                all_predictions = model(train_x[references].to(device))
                positions = positions.to(device)
                predictions = all_predictions[positions[:len(inputs)]]
                nominal_predictions = all_predictions[positions[len(inputs):]]
                nominal_targets = train_y[nominal_indices].to(device)
                nominal_losses = criterion(nominal_predictions, nominal_targets)
                delta_losses = torch.zeros_like(nominal_losses)
                partner_mask = (row_indices != nominal_indices).to(device)
                if partner_mask.any() and criterion.config['delta_weight']:
                    sizes = train_group_sizes[row_indices].to(device=device, dtype=torch.float64)
                    # Four of five rows supply deltas: 5/4 makes the row mean
                    # equal nominal loss + lambda * mean(four delta losses).
                    # This also handles legacy pairs and nominal-only feedback.
                    delta_losses[partner_mask] = signed_delta_losses(
                        predictions[partner_mask], targets[partner_mask],
                        nominal_predictions[partner_mask], nominal_targets[partner_mask],
                        delta_scale_w,
                    ) * sizes[partner_mask] / (sizes[partner_mask] - 1)
                row_losses = nominal_losses + criterion.config['delta_weight'] * delta_losses
                total_nominal_loss += (nominal_losses * weights).sum().item()
                total_delta_loss += (delta_losses * weights).sum().item()
            else:
                predictions = model(inputs)
                row_losses = criterion(predictions, targets)
            loss = (row_losses * weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'MLP training loss became non-finite at epoch {epoch}.')
            loss.backward()  # Backpropagation computes how each weight caused error.
            if paired and any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                              for parameter in model.parameters()):
                raise FloatingPointError(f'MLP training gradients became non-finite at epoch {epoch}.')
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
            if paired:
                history[-1].update({
                    'training_nominal_loss': total_nominal_loss / len(train_x),
                    'training_delta_loss': total_delta_loss / len(train_x),
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
                    + (f'; nominal MAE {metrics["nominal_power_mae"]:.4e}; '
                       f'delta MAE {metrics["delta_mae"]}' if paired else '')
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
        'checkpoint_selection': 'all_row_power_loss',
    })
    if paired:
        saved_metadata['delta_scale_w'] = float(delta_scale_w)
    save_checkpoint(checkpoint_path, model, detector, training_metadata=saved_metadata)
    with checkpoint_path.with_suffix('.txt').open('x', encoding='utf-8') as log:
        log.write(f'MLP loss: {criterion.config}; checkpoint selection: {monitor_label} all-row power loss.\n')
        if paired:
            log.write(f'Fixed initial-training delta scale: {delta_scale_w:.12g} W.\n')
        log.write('epoch\tweighted_training_loss\tmonitor_loss\tpower_mae\tcenter_power_mae\tcenter_sample_count'
                  + ('\ttraining_nominal_loss\ttraining_delta_loss\tnominal_power_mae\tdelta_mae\tdelta_sign_count\tdelta_sign_accuracy\tperturbation_cost_mae' if paired else '') + '\n')
        for row in history:
            log.write(
                f'{row["epoch"]}\t{row["training_loss"]:.8e}\t{row["validation_loss"]:.8e}\t'
                f'{row["power_mae"]:.8e}\t{row["center_power_mae"]}\t{row["center_sample_count"]}'
                + (f'\t{row["training_nominal_loss"]:.8e}\t{row["training_delta_loss"]:.8e}'
                   f'\t{row["nominal_power_mae"]:.8e}\t{row["delta_mae"]}'
                   f'\t{row["delta_sign_count"]}\t{row["delta_sign_accuracy"]}'
                   f'\t{row["perturbation_cost_mae"]}' if paired else '') + '\n'
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
    group_size = read_design_group_size(dataset_path)
    train_indices, validation_indices = split_pair_indices(
        len(dataset), seed=seed, group_size=group_size,
    )
    references = {}
    if params['loss']['supervision'] == 'nominal_delta':
        train_nominal, train_sizes = design_row_metadata(
            dataset.x, group_size, train_indices, validate_directions=True,
        )
        validation_nominal, validation_sizes = design_row_metadata(
            dataset.x, group_size, validation_indices,
        )
        references = {
            'train_nominal_indices': train_nominal, 'train_group_sizes': train_sizes,
            'validation_nominal_indices': validation_nominal,
            'validation_group_sizes': validation_sizes,
            'delta_scale_w': initial_delta_scale(
                dataset.y[train_indices], train_nominal, params['loss']['delta_scale_floor_w'],
            ),
        }
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_power_predictor(model_config)
    model.set_normalization(dataset.x[train_indices])
    model = model.to(device)
    metadata = {
        'base_data_filename': dataset_path.name,
        'base_data_count': len(dataset),
        'base_group_size': group_size,
        'base_data_sha256': _file_sha256(dataset_path),
        'split_seed': seed,
        'train_indices': train_indices,
        'validation_indices': validation_indices,
        'validation_mode': (
            'training_monitor_only' if len(dataset) == group_size
            else 'held_out_pairs' if group_size == 2 else 'held_out_designs'
        ),
        'current_round': 0,
    }
    print(f'MLP base data: {len(train_indices)} training rows, {len(validation_indices)} monitoring rows.')
    return _fit_model(
        model, dataset.x[train_indices], dataset.y[train_indices],
        torch.ones(len(train_indices)), dataset.x[validation_indices], dataset.y[validation_indices],
        checkpoint_path, detector, params, metadata, seed=seed, **references,
    )
