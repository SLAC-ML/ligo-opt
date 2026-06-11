# %%
import h5py
import finesse
import networkx as nx
import torch

# %%
kat = """
# Add a Laser named L0 with a power of 1 W.
l L0 P=1

s s1 portA=L0.p1 portB=eom1.p1 L=10

modulator eom1 9M 0.1 order=1

s s2 portA=eom1.p2 portB=ITM.p1 L=10

# Input mirror of cavity.
m ITM L=0 T=0.014 Rc=-1934

# Intra-cavity space with length of 4 km
s CAV ITM.p2 ETM.p1 L=3994.47

# End mirror of cavity.
m ETM L=0 T=5u  Rc=2245

cavity cavArm source=ITM.p2.o

# Power detectors on reflection, circulation and transmission.
pd circ ETM.p1.i

pd1 pdhI node=ITM.p1.o f=eom1.f phase=0 # In phase demodulated signal
pd1 pdhQ node=ITM.p1.o f=eom1.f phase=90 # Quadrature phase demodulated signal

# dof ETMz ETM.dofs.z
# readout_rf pdh_readout ITM.p1.o f=eom1.f output_detectors=true phase=0

# Add a lock
lock lock_length pdhI ETM.phi -1.0673950644453318 1e-12
"""

# %%
def reset_model(kat):
    fabry_perot = finesse.Model()
    fabry_perot.parse(kat)
    fabry_perot.modes(maxtem=6, modes='even')
    return fabry_perot

# %%
def model_to_nx_port(model):

    finesse_g = model.optical_network
    out = model.run()
    g = nx.DiGraph()
    
    for node in finesse_g.nodes():
        name = node.replace('.', '_')
        opt = node.split('.')[0]
        
        if isinstance(getattr(model, opt), finesse.components.mirror.Mirror):
            # Create feature vector
            g.add_node(node, Rc=getattr(model, opt).Rcx.value, R = getattr(model, opt).R.value, alpha=0, 
                       fd=torch.tensor(out[f'f_{name}'], dtype=torch.complex64), pd=out[f'p_{name}'], q=torch.tensor(out[f'q_{name}'], dtype=torch.complex64))
        elif isinstance(getattr(model, opt), finesse.components.laser.Laser):
            g.add_node(node, Rc=out['roc_l0'], R = 0, alpha=0, 
                       fd=torch.tensor(out[f'f_{name}'], dtype=torch.complex64), pd=out[f'p_{name}'], q=torch.tensor(out[f'q_{name}'], dtype=torch.complex64))
        elif isinstance(getattr(model, opt), finesse.components.beamsplitter.Beamsplitter):
            g.add_node(node, Rc=getattr(model, opt).Rcx.value, R = getattr(model, opt).R.value, alpha = getattr(model, opt).alpha.value, 
                       fd=torch.tensor(out[f'f_{name}'], dtype=torch.complex64), pd=out[f'p_{name}'], q=torch.tensor(out[f'q_{name}'], dtype=torch.complex64))
        elif isinstance(getattr(model, opt), finesse.components.lens.Lens):
            g.add_node(node, Rc=2*getattr(model, opt).f.value, R = 0, alpha = 0, 
                       fd=torch.tensor(out[f'f_{name}'], dtype=torch.complex64), pd=out[f'p_{name}'], q=torch.tensor(out[f'q_{name}'], dtype=torch.complex64))
    # Access edge attributes
    for i, edge in enumerate(finesse_g.edges().data()):
        dat = list(finesse_g.edges().data())[i][2]['owner']()
        if not isinstance(dat, finesse.components.space.Space):
            g.add_edge(str(edge[0]), str(edge[1]), length=0, nr=1)
        else:
            g.add_edge(str(edge[0]), str(edge[1]), length=dat.L.value if not isinstance(dat.L.value, finesse.symbols.Symbol) else dat.L.value.eval(), nr=dat.nr.value if not hasattr(dat.nr.value, 'eval') else dat.nr.value.eval())
    
    return g


# %%
finesse_model = reset_model(kat)
# finesse.tb()
result = model_to_nx_port(finesse_model)

# %%
import torch
from torch_geometric.data import Data
from model.power_predictor import LinGNN

# Node features: [num_nodes, num_features]
x = torch.tensor([
    [1.0, 0.1, 0.5],   # node 0
    [0.8, 0.2, 0.3],   # node 1
    [0.5, 0.4, 0.9],   # node 2
], dtype=torch.float)

# Edge index: [2, num_edges]
# Here we use 2 directed edges: 0->1 and 1->2
edge_index = torch.tensor([
    [0, 1],
    [1, 2],
], dtype=torch.long)

# Edge features: [num_edges, num_edge_features]
edge_attr = torch.tensor([
    [0.3, 1.0],   # edge 0->1
    [0.4, 0.9],   # edge 1->2
], dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
