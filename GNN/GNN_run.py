from GNN.power_predictor import LinGNN as PowerGNN
import networkx as nx
import finesse
import torch
import torch_geometric as pyg
import numpy as np

def model_to_nx_port(model):

    finesse_g = model.optical_network
    g = nx.DiGraph()
    
    for node in finesse_g.nodes():
        
        opt = node.split('.')[0]
        
        if isinstance(getattr(model, opt), finesse.components.mirror.Mirror):
            # Create feature vector

            # Make sure that the attributes are evaluated to floats if they are not already (if the value is not specified in the kat script, it might be a symbolic expression that needs to be evaluated)
            _Rc = getattr(model, opt).Rcx.value.eval() if not isinstance(getattr(model, opt).Rcx.value, float) else getattr(model, opt).Rcx.value
            _R = getattr(model, opt).R.value.eval() if not isinstance(getattr(model, opt).R.value, float) else getattr(model, opt).R.value

            # Add the node to the graph with the feature vector as attributes
            g.add_node(node, Rc=_Rc, R=_R, alpha=0)
            

        elif isinstance(getattr(model, opt), finesse.components.laser.Laser):
            
            g.add_node(node, Rc=0, R = 0, alpha=0)
        
        else:
          
            g.add_node(node, Rc=0, R = 0, alpha=0)
    # Access edge attributes
    for i, edge in enumerate(finesse_g.edges().data()):
     
        dat = list(finesse_g.edges().data())[i][2]['owner']()
        if not isinstance(dat, finesse.components.space.Space):
           
            g.add_edge(str(edge[0]), str(edge[1]), length=0, nr=1)
        else:
           
            g.add_edge(str(edge[0]), str(edge[1]), length=dat.L.value if not isinstance(dat.L.value, finesse.symbols.Symbol) else dat.L.value.eval(), nr=dat.nr.value if not hasattr(dat.nr.value, 'eval') else dat.nr.value.eval())
    
    return g

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

    powers = [out[i].item() for i in range(len(name))]
    # for i, node in enumerate(name):
    #     print(f"Predicted power at node {node}: {np.exp(out[i].item())}")

    return name, powers