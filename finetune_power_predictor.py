'''
This file contains the code for fine-tuning a pretrained power predictor on
additional data (e.g. new geometries, extra sampling near a region of interest).

All data lives together in one directory (e.g. GNN/data/run1/):
  - base*.h5        the original full training set (e.g. base.h5)
  - round{N}.h5      one file per round of new/extra data (round1.h5, round2.h5, ...)

Running this script auto-detects the highest round{N}.h5 present and treats it as
"new" data; every base*.h5 file plus every earlier round gets folded into "old" data.
So each time you want another round of fine-tuning, you just drop in the next
round{N}.h5 and rerun -- it automatically resumes from round N-1's checkpoint (or the
original base_checkpoint_path for round 1), and you never need to hand-edit paths.
'''

import glob
import os
import re
import math

import h5py

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from torch_geometric.loader import DataLoader

from GNN.power_predictor import LinGNN as PowerGNN
from train_power_predictor import PowerDataset

# The very first fine-tune round (round 1, when no earlier round checkpoint exists yet)
# resumes from this checkpoint. Exposed here so other scripts (e.g. run.py, which needs
# to know what GNN checkpoint to start its own loop from) can import the same value
# instead of hardcoding a second copy of this path that can silently drift out of sync.



def add_weight(weight):
    '''Returns a PowerDataset `transform` that stamps a per-node loss weight onto every
    graph as it's loaded. PowerDataset applies `transform` in __getitem__ after its own
    `get()`, so this runs on top of the existing x/y setup without modifying PowerDataset.
    '''
    def _transform(data):
        data.weight = torch.full((data.num_nodes,), weight, dtype=torch.float32)
        return data
    return _transform



def resolve_finetune_weight(base_graph_count, new_graph_count, finetune_weight=None):
    """Balance the newest batch against the initial dataset, or use an override.

    With equal-size graphs, N_new * weight equals N_base unless the minimum
    weight of 1 applies. Earlier fine-tuning rounds do not inflate this reference.
    This changes relative loss contributions; weighted_crit keeps its existing
    unnormalized mean, so absolute gradient magnitudes also change.
    """
    if new_graph_count < 1:
        raise ValueError('The newest fine-tuning round must contain at least one graph.')
    if finetune_weight is None:
        if base_graph_count < 1:
            raise ValueError(
                'Automatic fine-tuning weight requires non-empty base*.h5 data; '
                'provide initial data or an explicit finetune_weight.'
            )
        return max(1.0, base_graph_count / new_graph_count)
    weight = float(finetune_weight)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('finetune_weight must be finite and positive, or None for automatic weighting.')
    return weight


def weighted_crit(pred, target, weight):
    # reduction='none' keeps one loss value per node; multiplying by `weight` before the
    # final .mean() is what makes weight=100 count as 100x the gradient contribution of
    # weight=1, without physically duplicating any data.
    loss = nn.functional.smooth_l1_loss(pred, target, reduction='none')
    return (loss * weight).mean()


# Matches "round1.h5", "round2.h5", etc. -- the digits become the round number.
ROUND_RE = re.compile(r'round(\d+)\.h5')


def discover_rounds(data_dir):
    '''Maps round number -> path, for every round{N}.h5 file directly inside data_dir.'''
    rounds = {}
    for path in glob.glob(os.path.join(data_dir, 'round*.h5')):
        m = ROUND_RE.fullmatch(os.path.basename(path))
        if m:
            rounds[int(m.group(1))] = path
    return rounds


def discover_base_dataset(data_dir):
    '''Every base*.h5 file directly inside data_dir (the original full training set).'''
    return sorted(glob.glob(os.path.join(data_dir, 'base*.h5')))


def finetune(model, hyperparams, data_train, data_val, device, save_path):
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    n_epochs = hyperparams['n_epochs']
    save_loss_interval = hyperparams['save_loss_interval']
    print_interval = hyperparams['print_interval']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    min_val_loss = np.inf

    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        model.train()
        for data in loader:
            data = data.to(device)

            optimizer.zero_grad()
            out = model(data).to(device)
            loss = weighted_crit(out.reshape(data.y.shape), data.y, data.weight)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Validate every save_loss_interval epochs, and only keep the checkpoint with the
        # best validation loss seen so far.
        if epoch % save_loss_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for dp in data_val:
                    dp = dp.to(device)
                    out = model(dp).to(device)
                    val_loss += weighted_crit(out.reshape(dp.y.shape), dp.y, dp.weight)
                val_loss /= len(data_val)
                train_loss = epoch_loss / len(data_train) * batch_size

                if val_loss < min_val_loss:
                    torch.save(model.state_dict(), f'{save_path}.pt')
                    min_val_loss = val_loss
                if epoch % print_interval == 0:
                    print("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}.".format(epoch, train_loss, val_loss))
                    with open(f'{save_path}.txt', 'a+') as f:
                        f.write("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}.\n".format(epoch, train_loss, val_loss))
        scheduler.step()
    return None


def run_finetune(
    data_dir=None,
    base_checkpoint_path=None,
    checkpoint_template=None,
    current_round=None,
    finetune_weight=None,
    gat_layers=10,
    kan_layers=5,
    hyperparams=None,
):
    '''Runs one round of fine-tuning: uses current_round when supplied, otherwise
    auto-detects the highest round{N}.h5 file. All newest-round graphs are included
    in training. By default their per-graph weight is max(1, N_base / N_new),
    using actual graph counts from all base*.h5 files and the selected round.
    Pass a positive numeric finetune_weight to override automatic weighting.
    Historical replay and validation remain at weight 1.0; earlier rounds do not
    contribute to N_base. Resumes from round N-1's checkpoint (or
    base_checkpoint_path for round 1) and returns the save path used.
    '''
    hyperparams = hyperparams or {
        'batch_size': 100,
        'save_loss_interval': 5,
        'print_interval': 1,
        'n_epochs': 50,
        # Much lower than the initial 1e-5 used in train_power_predictor.py: we're
        # nudging an already-converged model, not learning from scratch.
        'learning_rate': 1e-6,
    }

    base_dataset_paths = discover_base_dataset(data_dir)

    rounds = discover_rounds(data_dir)
    if not rounds:
        raise FileNotFoundError(f'No round*.h5 files found in {data_dir}')
    if current_round is None:
        current_round = max(rounds)  # convenient fallback for standalone/manual use
    elif current_round not in rounds:
        raise FileNotFoundError(
            f'Expected round {current_round}.h5 in {data_dir}, but it was not found. '
            f'Rounds present: {sorted(rounds)}.'
        )

    # Round numbers must be contiguous (1, 2, 3, ...): the checkpoint chain relies on
    # round N having actually been fine-tuned from round N-1's checkpoint, so a gap
    # (e.g. round1.h5 and round3.h5 with no round2.h5) means there's no valid checkpoint
    # to resume from for this round. Fail loudly instead of silently falling back to
    # base_checkpoint_path and quietly discarding whatever earlier rounds learned.
    if current_round > 1 and (current_round - 1) not in rounds:
        raise FileNotFoundError(
            f'Round {current_round} requires round {current_round - 1}.h5 (to resume from '
            f'its checkpoint), but it was not found in {data_dir}. Rounds present: {sorted(rounds)}.'
        )

    # Everything the model hasn't already been fine-tuned on this run: the new round.
    new_dataset_paths = [rounds[current_round]]
    # Everything the model has already seen (directly or via a previous round's
    # checkpoint): base data + every round below the current one.
    old_dataset_paths = base_dataset_paths + [rounds[r] for r in sorted(rounds) if r < current_round]

    # Resume from the previous round's checkpoint, or the original base-trained model
    # for round 1. (Contiguity was just checked above, so round - 1's checkpoint is
    # expected to exist; if it doesn't, torch.load below will raise clearly.)
    pretrained_path = (
        checkpoint_template.format(current_round - 1) + '.pt'
        if current_round > 1
        else base_checkpoint_path
    )
    save_path = checkpoint_template.format(current_round)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator().manual_seed(9302)

    # Historical data keeps unit weight. Resolve the latest round's weight from
    # actual file contents, independent of run.py's initial-data configuration.
    if not old_dataset_paths:
        raise FileNotFoundError(
            f'No base*.h5 or earlier round data found in {data_dir}; '
            'historical data is required to create a validation split.'
        )

    old_dataset = PowerDataset(data_files=old_dataset_paths, transform=add_weight(1.0))
    new_dataset = PowerDataset(data_files=new_dataset_paths)
    base_graph_count = 0
    for path in base_dataset_paths:
        with h5py.File(path, 'r') as handle:
            base_graph_count += len(handle)
    resolved_weight = resolve_finetune_weight(
        base_graph_count, len(new_dataset), finetune_weight
    )
    new_dataset.transform = add_weight(resolved_weight)

    if len(old_dataset) < 2:
        raise ValueError(
            f'Need at least 2 historical graphs to create train/validation splits; '
            f'found {len(old_dataset)}.'
        )

    # Validation is drawn only from historical data. Every graph from the newest
    # Finesse round is placed in training, guaranteeing that this feedback round
    # actually contributes gradients (especially important when a round is small).
    old_train_size = max(1, int(0.8 * len(old_dataset)))
    old_val_size = len(old_dataset) - old_train_size
    if old_val_size == 0:
        old_train_size -= 1
        old_val_size = 1
    old_train, data_val = random_split(
        old_dataset, [old_train_size, old_val_size], generator=generator
    )
    data_train = ConcatDataset([old_train, new_dataset])
    print(f'Round {current_round}: fine-tuning from {pretrained_path}')
    print(f'Loaded {len(old_dataset)} old + {len(new_dataset)} new ({rounds[current_round]}) graphs.')
    weight_mode = 'automatic' if finetune_weight is None else 'explicit override'
    print(
        f'Latest-round loss weight: {resolved_weight:g} ({weight_mode}; '
        f'{base_graph_count} initial graphs / {len(new_dataset)} latest graphs). '
        'Historical training and validation weights: 1.'
    )

    # Build the model with the same architecture as the checkpoint, then load its
    # weights instead of starting from random init -- this is what makes it fine-tuning
    # rather than training from scratch.
    model = PowerGNN(hidden_size=1000, num_layers=gat_layers, lin_layers=kan_layers, target_size=1).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))

    finetune(model, hyperparams, data_train, data_val, device, save_path=save_path)
    return save_path


# if __name__ == '__main__':
#     # Usage: drop the next round's data at GNN/data/run1/round{N}.h5 (alongside base.h5
#     # already there), then run this file as-is with `python finetune_power_predictor.py`.
#     # It auto-detects the highest round present and resumes from round N-1's checkpoint
#     # (round 1 resumes from the original base_checkpoint_path).
#     #
#     # To tweak settings per call instead of editing this file, import and call the
#     # function directly, e.g. from a notebook or another script:
#     #   from finetune_power_predictor import run_finetune
#     #   run_finetune(finetune_weight=10.0, hyperparams={'batch_size': 100,
#     #                'save_loss_interval': 5, 'print_interval': 1, 'n_epochs': 20,
#     #                'learning_rate': 5e-7})
#     run_finetune()
