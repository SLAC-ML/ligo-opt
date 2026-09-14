'''
This file contains the code for training the power predictor
'''


import pickle
import time
from pathlib import Path

import networkx as nx
import numpy as np
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm


import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from GNN.power_predictor import LinGNN as PowerGNN

class PowerDataset(Dataset):
    def __init__(self, data_files, max_size = None, transform=None, pre_transform=None):
        self.data_files = data_files
        self.data = []
        for fn, file in enumerate(self.data_files):
            with h5py.File(file, 'r') as f:
                keys = sorted(
                    f.keys(),
                    key=lambda key: int(key.removeprefix('sim_')),
                )
                for i, key in enumerate(keys):
                    if max_size and i > max_size[fn]:
                        break

                    # Read and deserialize each graph
                    serialized_graph = f[key][()]
                    graph = pickle.loads(serialized_graph)
                    self.data.append(graph)
        super(PowerDataset, self).__init__(self.data, transform, pre_transform)

    def len(self):
        return len(self.data)

    def get(self, idx):
        # Load the data from the file
        data = self.data[idx]
        data = pyg.utils.from_networkx(data, group_node_attrs = ['Rc', 'R', 'q_re', 'q_im'], group_edge_attrs=['length', 'nr'])
        data.x = torch.nan_to_num(data.x, posinf=0).float()
        y_log = torch.log1p(data.pd)
        data.y = torch.nan_to_num(torch.clamp(y_log, min=-10),posinf=0).float()
        return data


def split_graph_pairs(dataset, train_fraction=0.8, seed=9302):
    """Split consecutive design/perturbation graph pairs without pair leakage."""
    dataset_size = len(dataset)
    if dataset_size < 4 or dataset_size % 2:
        raise ValueError(
            'Pair-aware splitting requires an even dataset with at least 4 graphs; '
            f'found {dataset_size}.'
        )
    if not 0 < train_fraction < 1:
        raise ValueError(f'train_fraction must be between 0 and 1; got {train_fraction}.')

    pair_count = dataset_size // 2
    train_pair_count = min(pair_count - 1, max(1, int(train_fraction * pair_count)))
    pair_order = torch.randperm(
        pair_count,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    train_pairs = pair_order[:train_pair_count]
    val_pairs = pair_order[train_pair_count:]

    def graph_indices(pair_indices):
        return [
            index
            for pair_index in pair_indices
            for index in (2 * pair_index, 2 * pair_index + 1)
        ]

    return (
        Subset(dataset, graph_indices(train_pairs)),
        Subset(dataset, graph_indices(val_pairs)),
    )


def crit(mod, gt, lam, adj_matr):
    return nn.functional.smooth_l1_loss(gt, mod)

def train(model, hyperparams, data_train, data_val, device, save_path):

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

            edges = data.edge_index.cpu()
            adj_matr = torch.tensor(nx.adjacency_matrix(nx.from_edgelist(edges.detach().numpy().T)).toarray(), dtype=torch.float32).to(device)
            optimizer.zero_grad()
            out = model(data).to(device)
            loss = crit(out.reshape(data.y.shape), data.y, 0.1, adj_matr)
            epoch_loss += loss.item() 
            loss.backward()
            optimizer.step()
        
        # Adding several validation steps throughout training.
        if epoch % save_loss_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                tot_time = 0
                for dp in data_val:
                    dp = dp.to(device)
                    s = time.time()
                    out = model(dp).to(device)
                    e = time.time()
                    tot_time += e-s
                    edges = dp.edge_index.cpu()
                    adj_matr = torch.tensor(nx.adjacency_matrix(nx.from_edgelist(edges.detach().numpy().T)).toarray(), dtype=torch.float32).to(device)
                    
                    val_loss += crit(out.reshape(dp.y.shape), dp.y, 0.1, adj_matr)
                val_loss /= len(data_val)
                train_loss = epoch_loss / len(data_train) * batch_size
            
                if val_loss < min_val_loss:
                    torch.save(model.state_dict(), f'{save_path}.pt')
                    min_val_loss = val_loss
                if epoch % print_interval == 0:
                    print("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}.".format(epoch, train_loss, val_loss))
                    with open(f'{save_path}.txt', 'a+') as f:
                        f.write("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}. Mean Inference took {}.\n".format(epoch, train_loss, val_loss, tot_time / len(data_val)))
        scheduler.step()
    return None

def train_base_power_predictor(
    dataset_path,
    checkpoint_path,
    hyperparams=None,
    gat_layers=10,
    kan_layers=5,
):
    """Train the initial power GNN and return the checkpoint path produced."""
    hyperparams = hyperparams or {
        'batch_size' : 100, 
        'save_loss_interval' : 5, 
        'print_interval' : 1,
        'n_epochs' : 200,
        'learning_rate' : 1e-5
    }
    dataset_path = str(dataset_path)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        raise FileExistsError(f'Base checkpoint already exists: {checkpoint_path}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = PowerDataset(data_files=[dataset_path])

    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError(f'Base training requires at least 2 graphs; found {dataset_size}.')
    if dataset_size == 2:
        # A two-graph bootstrap set is one design plus its fixed-q perturbation. An
        # 80/20 split would train on only one graph,
        # so the q-mismatch graph could be validation-only and contribute no gradient.
        # Use both graphs for training. Reusing them for checkpoint monitoring is not
        # an unbiased validation estimate, but no meaningful holdout is possible with
        # two samples; later Finesse rounds provide independent feedback data.
        data_train = dataset
        data_val = dataset
        print(
            "Two-graph bootstrap dataset: using both design and perturbed graphs "
            "for training and checkpoint monitoring."
        )
    else:
        data_train, data_val = split_graph_pairs(dataset, train_fraction=0.8, seed=9302)
        print(
            f'Pair-aware split: {len(data_train)} training graphs and '
            f'{len(data_val)} validation graphs.'
        )
    print("Loaded base dataset. Beginning initial training.")

    model = PowerGNN(
        hidden_size=1000,
        num_layers=gat_layers,
        lin_layers=kan_layers,
        target_size=1,
    ).to(device)
    save_prefix = str(checkpoint_path.with_suffix(''))
    train(
        model,
        hyperparams,
        data_train,
        data_val,
        device,
        save_path=save_prefix,
    )
    if not checkpoint_path.is_file():
        raise RuntimeError(f'Initial training did not produce checkpoint: {checkpoint_path}')
    return checkpoint_path


if __name__ == '__main__':
    train_base_power_predictor(
        dataset_path='GNN/data/run6/base_10.h5',
        checkpoint_path='GNN/models/run6/base_10_gat10_kan5.pt',
    )

