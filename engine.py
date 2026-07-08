"""Custom engine for locking-parameter optimization.

Extends the base Engine with:
- R switch: conditional retraining (approach 1)
- Approach-aware behavior via config
"""

import torch


# GNN model and training code:
class PowGNN(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=700, target_size=1, num_edge_features=2, num_layers=20, lin_layers = 6, lin_size=800):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.num_layers = num_layers
        self.lin_size = lin_size
        
        self.bnn = BatchNorm(in_channels=num_features)
        self.bne = BatchNorm(in_channels=num_edge_features)
        self.batch_norms = nn.ModuleList()
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(self.num_features, self.hidden_size, edge_dim=num_edge_features))
        
        for _ in range(self.num_layers - 2):
            self.convs.append(GATv2Conv(self.hidden_size, self.hidden_size, edge_dim=num_edge_features))
            self.batch_norms.append(BatchNorm(in_channels=self.hidden_size))
        self.convs.append(GATv2Conv(self.hidden_size, self.lin_size, edge_dim=num_edge_features))
        # # # Linear layers
        # self.linears = KAN([lin_size]*(lin_layers)+[target_size])
        self.linears = nn.ModuleList()
        for _ in range(lin_layers - 1):
            self.linears.append(nn.Linear(self.lin_size, self.lin_size))
        self.linears.append(nn.Linear(self.lin_size, self.target_size))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.bnn(x)
        edge_attr = self.bne(edge_attr)

        x = self.convs[0](x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x)
        # Apply GATConv layers with residual connections
        for i, conv in enumerate(self.convs[1:-1]):
            residual = x
            x = self.batch_norms[i](x)
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.leaky_relu(x)
            x = x + residual  # Skip connection

        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        # Apply linear layers with residual connections
        for i, linear in enumerate(self.linears[:-1]):
            residual = x
            x = linear(x)
            x = F.leaky_relu(x)
            x = x + residual  # Skip connection

        x = self.linears[-1](x)  # Final output layer
        return x

def crit(mod, gt, lam, adj_matr):
    return nn.functional.l1_loss(gt, mod) + lam*(torch.log(torch.sum(torch.abs(adj_matr.T@torch.exp(mod) - torch.exp(mod)))))

def train(model, hyperparams, save_path):

    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    n_epochs = hyperparams['n_epochs']
    save_loss_interval = hyperparams['save_loss_interval']
    print_interval = hyperparams['print_interval']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    min_val_loss = np.inf
    
    for epoch in range(n_epochs):
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

