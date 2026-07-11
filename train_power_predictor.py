'''
This file contains the code for training the power predictor
'''


import pickle
import time

import networkx as nx
import numpy as np
import h5py

import torch
import torch.nn as nn
from torch.utils.data import random_split
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
                print(len(f.keys()))
                for i, key in enumerate(f.keys()):
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
        data = pyg.utils.from_networkx(data, group_node_attrs = ['Rc', 'R', 'alpha'], group_edge_attrs=['length', 'nr'])
        data.x = torch.nan_to_num(data.x, posinf=0).float()
        y_log = torch.log1p(data.pd)
        data.y = torch.nan_to_num(torch.clamp(y_log, min=-10),posinf=0).float()
        return data

def crit(mod, gt, lam, adj_matr):
    return nn.functional.smooth_l1_loss(gt, mod)

def train(model, hyperparams, save_path):

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

if __name__ == '__main__':
    dataset_path = 'GNN/models/training_dataset_ligoParams.h5'
    save_path = 'GNN/models/power_predictor_ligoParams_loss_fixed'

    hyperparams = {
        'batch_size' : 100, 
        'save_loss_interval' : 5, 
        'print_interval' : 1,
        'n_epochs' : 200,
        'learning_rate' : 1e-5
    }
    gat_layers = [10]
    kan_layers = [5]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator().manual_seed(9302)
    
    for gt in gat_layers:
        for kl in kan_layers:
            generator = torch.Generator().manual_seed(9302)
            # Train a model with only half_aligo data
            dataset = PowerDataset(data_files=[dataset_path], max_size=[25000,5000])

            dataset_size = len(dataset)
            train_size = int(0.8 * dataset_size)  # 80% for training
            test_size = dataset_size - train_size  # 20% for testing

            # Split the dataset
            data_train, data_val = random_split(dataset, [train_size, test_size], generator=generator) 
            print("Loaded dataset. Beginning training.")
            # dp, lab = data_train[0]

            model = (PowerGNN(hidden_size=1000, num_layers=gt, lin_layers=kl, target_size = 1)).to(device) # needs to be double precision
            model_loss_traj = train(model, hyperparams, save_path=f'{save_path}_gat{gt}_kan{kl}')

