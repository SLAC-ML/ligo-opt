"""Improve an existing MLP with each new round of Finesse measurements.

Fine-tuning starts from learned weights instead of a new random network. We
replay older training examples to reduce forgetting and give the newest examples
extra weight. The original validation designs stay out of training in every round.
"""

import math
import re
import warnings
from pathlib import Path

import torch
from utils.training_groups import read_design_group_size, validate_group_size

from .checkpoint import load_checkpoint
from .data import PowerDataset
from .groups import design_row_metadata, initial_delta_scale
from .training import (
    DEFAULT_TRAINING_PARAMS, _check_output_paths, _file_sha256,
    _fit_model, _training_params,
)


DEFAULT_FINETUNE_PARAMS = {
    **DEFAULT_TRAINING_PARAMS,
    'n_epochs': 20,
    'learning_rate': 1e-4,
}
ROUND_RE = re.compile(r'round([0-9]+)\.h5')


def discover_rounds(data_dir):
    """Return round number -> path, excluding unrelated files and subfolders."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f'MLP data directory does not exist: {data_dir}')
    rounds = {}
    for path in sorted(data_dir.glob('round*.h5')):
        match = ROUND_RE.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        number = int(match.group(1))
        if number < 1:
            raise ValueError(f'Round numbers must start at 1: {path}')
        if number in rounds:
            raise ValueError(f'Ambiguous files for round {number}: {rounds[number]} and {path}')
        rounds[number] = path
    return rounds


def resolve_finetune_weight(base_graph_count, new_graph_count, finetune_weight=None):
    """Give the latest round total influence comparable to the original dataset.

    For example, 200 base rows and 5 new rows give each new row weight 40.
    Historical rows have weight 1. The reference remains the original base size
    rather than growing every round. A positive explicit weight overrides this.
    """
    if new_graph_count < 1:
        raise ValueError('The newest fine-tuning round must contain at least one sample.')
    if base_graph_count < 1:
        raise ValueError('Fine-tuning requires a non-empty original base dataset.')
    if finetune_weight is None:
        return max(1.0, base_graph_count / new_graph_count)
    if isinstance(finetune_weight, bool):
        raise ValueError('finetune_weight must be finite and positive, or None.')
    weight = float(finetune_weight)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('finetune_weight must be finite and positive, or None.')
    return weight


def _checkpoint_path(template, round_number):
    """Accept a format template with or without the final .pt extension."""
    template = str(template)
    try:
        formatted = template.format(round_number)
        next_formatted = template.format(round_number + 1)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError('checkpoint_template must contain a positional {} round placeholder.') from exc
    if formatted == next_formatted:
        raise ValueError('checkpoint_template must contain a positional {} round placeholder.')
    return Path(formatted if formatted.endswith('.pt') else formatted + '.pt')


def _restore_base_split(metadata, sample_count):
    """Check the persisted split before trusting it for historical replay."""
    try:
        train = metadata['train_indices']
        validation = metadata['validation_indices']
        mode = metadata['validation_mode']
    except KeyError as exc:
        raise ValueError('Checkpoint lacks the original MLP validation split; retrain the base model.') from exc
    for name, indices in (('train_indices', train), ('validation_indices', validation)):
        if not isinstance(indices, list) or not indices:
            raise ValueError(f'Checkpoint {name} must be a non-empty list.')
        if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < sample_count for index in indices):
            raise ValueError(f'Checkpoint {name} contains invalid row indices.')
        if len(set(indices)) != len(indices):
            raise ValueError(f'Checkpoint {name} contains duplicate row indices.')
    train_set, validation_set = set(train), set(validation)
    # Older checkpoints describe pairs and do not persist their group size.
    group_size = validate_group_size(metadata.get('base_group_size', 2))
    if mode == 'training_monitor_only':
        if sample_count != group_size or train_set != set(range(group_size)) or validation_set != train_set:
            raise ValueError('Checkpoint bootstrap split is inconsistent with its base dataset.')
        warnings.warn(
            'Single-design bootstrap: checkpoint selection still uses training samples; '
            'this is not an independent validation estimate.', UserWarning, stacklevel=2,
        )
    elif mode in ('held_out_pairs', 'held_out_designs'):
        if mode == 'held_out_pairs' and group_size != 2:
            raise ValueError('Checkpoint pair validation mode requires a group size of 2.')
        if sample_count % group_size or train_set & validation_set or train_set | validation_set != set(range(sample_count)):
            raise ValueError('Checkpoint must partition all base rows into disjoint training and validation sets.')
        for start in range(0, sample_count, group_size):
            group = set(range(start, start + group_size))
            if not (group <= train_set or group <= validation_set):
                raise ValueError('Checkpoint separates an initial design group across training and validation.')
    else:
        raise ValueError(f'Unsupported checkpoint validation mode: {mode!r}')
    return train, validation


def run_finetune(
    data_dir, base_checkpoint_path, checkpoint_template, current_round=None,
    finetune_weight=None, hyperparams=None,
):
    """Train one feedback round and return its checkpoint prefix, without .pt.

    ``round1.h5`` starts from the base checkpoint. Every subsequent round starts
    from the previous round's checkpoint. All new measurements and all previous
    feedback measurements train the model, together with the original training
    split. Future round files are ignored when a specific round is requested.

    The mean and scale learned during base training remain frozen. Changing
    normalization while keeping learned weights would change what those weights
    mean, so normalization must travel unchanged with the checkpoint.
    """
    data_dir, base_checkpoint_path = Path(data_dir), Path(base_checkpoint_path)
    if not base_checkpoint_path.is_file():
        raise FileNotFoundError(f'MLP base checkpoint does not exist: {base_checkpoint_path}')
    rounds = discover_rounds(data_dir)
    if not rounds:
        raise FileNotFoundError(f'No round*.h5 files found in {data_dir}')
    if current_round is None:
        current_round = max(rounds)
    if isinstance(current_round, bool) or not isinstance(current_round, int) or current_round < 1:
        raise ValueError('current_round must be a positive integer.')
    missing = [number for number in range(1, current_round + 1) if number not in rounds]
    if missing:
        raise FileNotFoundError(f'Missing round data in {data_dir}: {missing}; rounds must be contiguous.')

    checkpoint_path = _check_output_paths(_checkpoint_path(checkpoint_template, current_round))
    pretrained_path = (
        base_checkpoint_path if current_round == 1
        else _checkpoint_path(checkpoint_template, current_round - 1)
    )
    if not pretrained_path.is_file():
        raise FileNotFoundError(f'MLP previous-round checkpoint does not exist: {pretrained_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, payload = load_checkpoint(pretrained_path, device=device)
    metadata = dict(payload.get('training_metadata', {}))
    # Continue the checkpoint's loss unless explicitly overridden. Old models
    # without loss metadata used ordinary log-space Smooth-L1; preserve that
    # behavior for standalone fine-tuning of those checkpoints.
    inherited_loss = metadata.get('loss_config', metadata.get('hyperparams', {}).get(
        'loss', {'mode': 'log_smooth_l1'},
    ))
    params = _training_params(
        hyperparams, defaults={**DEFAULT_FINETUNE_PARAMS, 'loss': inherited_loss},
    )
    if metadata.get('current_round') != current_round - 1:
        raise ValueError(
            f'Checkpoint {pretrained_path} belongs to round {metadata.get("current_round")!r}; '
            f'round {current_round} must resume from round {current_round - 1}.'
        )
    base_checkpoint_hash = _file_sha256(base_checkpoint_path)
    if current_round > 1 and metadata.get('base_checkpoint_sha256') != base_checkpoint_hash:
        raise ValueError('Previous-round checkpoint belongs to a different or modified base checkpoint.')

    # The filename and hash stored during base training make the source explicit.
    # Multiple base*.h5 files could otherwise combine unrelated experiments.
    base_files = sorted(path for path in data_dir.glob('base*.h5') if path.is_file())
    expected_filename = metadata.get('base_data_filename')
    if not isinstance(expected_filename, str) or Path(expected_filename).name != expected_filename:
        raise ValueError('Checkpoint lacks a valid base_data_filename; retrain the base model.')
    expected_base = data_dir / expected_filename
    if len(base_files) > 1 or (base_files and base_files[0] != expected_base):
        raise ValueError(f'Ambiguous base data in {data_dir}; expected only {expected_filename}.')
    if not expected_base.is_file():
        raise FileNotFoundError(f'Original MLP base data is missing: {expected_base}')
    if _file_sha256(expected_base) != metadata.get('base_data_sha256'):
        raise ValueError(f'Original base dataset changed after training: {expected_base}')
    detector = payload['detector']
    base_dataset = PowerDataset([expected_base], detector=detector)
    if len(base_dataset) != metadata.get('base_data_count'):
        raise ValueError('Original base dataset size disagrees with checkpoint metadata.')
    if read_design_group_size(expected_base) != metadata.get('base_group_size', 2):
        raise ValueError('Original base dataset group size disagrees with checkpoint metadata.')
    train_indices, validation_indices = _restore_base_split(metadata, len(base_dataset))

    new_dataset = PowerDataset([rounds[current_round]], detector=detector)
    newest_weight = resolve_finetune_weight(len(base_dataset), len(new_dataset), finetune_weight)
    train_inputs = [base_dataset.x[train_indices]]
    train_targets = [base_dataset.y[train_indices]]
    train_weights = [torch.ones(len(train_indices))]
    references = {}
    paired = params['loss']['supervision'] == 'nominal_delta'
    if paired:
        train_nominal, train_sizes = design_row_metadata(
            base_dataset.x, metadata.get('base_group_size', 2), train_indices,
            validate_directions=True,
        )
        validation_nominal, validation_sizes = design_row_metadata(
            base_dataset.x, metadata.get('base_group_size', 2), validation_indices,
        )
        scale = metadata.get('delta_scale_w')
        if scale is None:
            if inherited_loss.get('supervision') == 'nominal_delta':
                raise ValueError('Nominal/delta checkpoint is missing its fixed delta_scale_w.')
            # Explicitly opting an older checkpoint into the new objective still
            # fits the scale using its original training split, never feedback.
            scale = initial_delta_scale(
                base_dataset.y[train_indices], train_nominal,
                params['loss']['delta_scale_floor_w'],
            )
        nominal_parts, size_parts = [train_nominal], [train_sizes]
        references = {
            'validation_nominal_indices': validation_nominal,
            'validation_group_sizes': validation_sizes, 'delta_scale_w': scale,
        }
    round_counts = {}
    # Replay only training data. A fixed holdout prevents a validation example
    # from accidentally becoming a training example in a later round.
    for number in range(1, current_round):
        historical = PowerDataset([rounds[number]], detector=detector)
        if len(historical) == 0:
            raise ValueError(f'Historical round {number} contains no samples.')
        train_inputs.append(historical.x)
        train_targets.append(historical.y)
        train_weights.append(torch.ones(len(historical)))
        if paired:
            nominal, sizes = design_row_metadata(
                historical.x, 1 if len(historical) == 1 else 5,
                validate_directions=True,
            )
            nominal_parts.append(nominal + sum(len(part) for part in train_inputs[:-1]))
            size_parts.append(sizes)
        round_counts[str(number)] = len(historical)
    train_inputs.append(new_dataset.x)
    train_targets.append(new_dataset.y)
    train_weights.append(torch.full((len(new_dataset),), newest_weight))
    if paired:
        nominal, sizes = design_row_metadata(
            new_dataset.x, 1 if len(new_dataset) == 1 else 5, validate_directions=True,
        )
        nominal_parts.append(nominal + sum(len(part) for part in train_inputs[:-1]))
        size_parts.append(sizes)
        references.update({
            'train_nominal_indices': torch.cat(nominal_parts),
            'train_group_sizes': torch.cat(size_parts),
        })
    round_counts[str(current_round)] = len(new_dataset)

    metadata.update({
        'current_round': current_round,
        'base_checkpoint_sha256': base_checkpoint_hash,
        'previous_checkpoint_filename': pretrained_path.name,
        'replay_rounds': list(range(1, current_round)),
        'round_data_counts': round_counts,
        'newest_round_count': len(new_dataset),
        'newest_round_weight': newest_weight,
    })
    seed = metadata.get('split_seed')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Checkpoint lacks a valid split_seed; retrain the base model.')
    torch.manual_seed(seed + current_round)
    print(
        f'MLP round {current_round}: resume {pretrained_path.name}; '
        f'{len(new_dataset)} new rows with weight {newest_weight:g}.'
    )
    _fit_model(
        model, torch.cat(train_inputs), torch.cat(train_targets), torch.cat(train_weights),
        base_dataset.x[validation_indices], base_dataset.y[validation_indices],
        checkpoint_path, detector, params, metadata, seed=seed + current_round, **references,
    )
    return str(checkpoint_path.with_suffix(''))
