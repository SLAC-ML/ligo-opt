"""Load graph datasets and train the base power-prediction GNN."""

import pickle
import time
from pathlib import Path

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

from GNN.power_predictor import build_power_predictor
from utils.training_groups import read_design_group_size, split_design_indices

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


def split_graph_pairs(dataset, train_fraction=0.8, seed=9302, *, group_size=2):
    """Split complete designs; the legacy function name defaults to graph pairs."""
    train_indices, validation_indices = split_design_indices(
        len(dataset), train_fraction, seed, group_size=group_size,
    )
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def crit(prediction, target):
    """Mean smooth-L1 loss for log-power predictions."""
    return nn.functional.smooth_l1_loss(prediction, target)

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

            optimizer.zero_grad()
            out = model(data).to(device)
            loss = crit(out.reshape(data.y.shape), data.y)
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
                    
                    val_loss += crit(out.reshape(dp.y.shape), dp.y)
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
    graph_layers=None,
    linear_layers=None,
    *,
    gat_layers=None,
    kan_layers=None,
):
    """Train the initial power GNN and return the checkpoint path produced.

    Layer counts default to the shared model factory. gat_layers/kan_layers are
    retained as keyword aliases for graph_layers/linear_layers.
    """
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

    group_size = read_design_group_size(dataset_path)
    data_train, data_val = split_graph_pairs(
        dataset, train_fraction=0.8, seed=9302, group_size=group_size,
    )
    print(
        f'Design-group split ({group_size} graphs per design): {len(data_train)} training '
        f'graphs and {len(data_val)} checkpoint-monitoring graphs.'
    )
    print("Loaded base dataset. Beginning initial training.")

    model = build_power_predictor(
        graph_layers=graph_layers, linear_layers=linear_layers,
        gat_layers=gat_layers, kan_layers=kan_layers,
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
        dataset_path='GNN/data/run6/base_25.h5',
        checkpoint_path='GNN/models/run6/base_25_gat10_kan5.pt',
    )
