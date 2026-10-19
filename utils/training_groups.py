"""Keep each initial design and its perturbations in the same data split."""

from numbers import Integral
import warnings

import h5py
import torch


INITIAL_SAMPLE_TYPES = ('design', 'ITM_plus', 'ITM_minus', 'ETM_plus', 'ETM_minus')
INITIAL_GROUP_SIZE = len(INITIAL_SAMPLE_TYPES)


def validate_group_size(group_size):
    if isinstance(group_size, bool) or not isinstance(group_size, Integral) or group_size < 2:
        raise ValueError('Design group size must be an integer of at least 2.')
    return int(group_size)


def read_design_group_size(path):
    """Older initial datasets without group metadata contain nominal/perturbed pairs."""
    with h5py.File(path, 'r') as handle:
        return validate_group_size(handle.attrs.get('design_group_size', 2))


def split_design_indices(sample_count, train_fraction=0.8, seed=9302, *, group_size=2):
    """Split complete consecutive design groups reproducibly, including legacy pairs."""
    group_size = validate_group_size(group_size)
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be strictly between 0 and 1.')
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError('sample_count must be an integer.')
    if sample_count < group_size or sample_count % group_size:
        raise ValueError(f'Base training needs complete design groups of {group_size} samples.')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer.')
    group_count = sample_count // group_size
    if group_count == 1:
        warnings.warn(
            f'Single-design bootstrap: all {group_size} samples are used for training '
            'and checkpoint monitoring. The reported loss is not independent validation accuracy.',
            UserWarning, stacklevel=2,
        )
        return list(range(group_size)), list(range(group_size))
    training_groups = min(group_count - 1, max(1, int(train_fraction * group_count)))
    group_order = torch.randperm(
        group_count, generator=torch.Generator().manual_seed(seed),
    ).tolist()

    def expand(groups):
        return [index for group in groups
                for index in range(group_size * group, group_size * (group + 1))]

    return expand(group_order[:training_groups]), expand(group_order[training_groups:])
