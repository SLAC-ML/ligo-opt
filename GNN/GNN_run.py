from GNN.power_predictor import LinGNN as PowerGNN
from GNN.GNN_utils import model_to_nx_port

import torch
import torch_geometric as pyg


def run_GNN(kat, model_path):
    graph = model_to_nx_port(kat)
    model = PowerGNN(hidden_size=1000, num_layers=10, lin_layers=5, target_size = 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()


    data = pyg.utils.from_networkx(
        graph,
        group_node_attrs=['Rc', 'R', 'alpha'],
        group_edge_attrs=['length', 'nr']
    )
    data.x = torch.nan_to_num(data.x, posinf=0).float()
    data.edge_attr = torch.nan_to_num(data.edge_attr, posinf=0).float()
    with torch.no_grad():
        out = model(data)

    name = [n for n, attrs in graph.nodes(data=True)]

    powers = [torch.expm1(out[i]).item() for i in range(len(name))]
    # for i, node in enumerate(name):
    #     print(f"Predicted power at node {node}: {np.exp(out[i].item())}")

    return name, powers