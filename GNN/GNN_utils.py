import numpy as np
import finesse
import networkx as nx
import torch

import time

# def perturb_model(model, pct_pert):
#     len_dofs = ['ls3', 'ls2', 'ls1', 'LX']
#     for comp in model.elements:
#         elem = getattr(model, comp)

#         if isinstance(elem, finesse.components.mirror.Mirror):
#             _elem_R = min(1, elem.R.eval()*(1+(2*np.random.rand()-1)*pct_pert))
#             _elem_T = 1-_elem_R
#             elem.Rc = elem.Rc*(1+(2*np.random.rand()-1)*pct_pert)
#             elem.set_RTL(R=_elem_R, T=_elem_T, L=0)
#         elif isinstance(elem, finesse.components.space.Space) and elem.name in len_dofs:
#             elem.L =  elem.L.value*(1+(2*np.random.rand()-1)*pct_pert)
#     return model


# def model_to_nx_port(model):

#     finesse_g = model.optical_network
#     g = nx.DiGraph()
    
#     for node in finesse_g.nodes():
        
#         opt = node.split('.')[0]
        
#         if isinstance(getattr(model, opt), finesse.components.mirror.Mirror):
#             # Create feature vector
#             print("add a mirror node")

#             # Make sure that the attributes are evaluated to floats if they are not already (if the value is not specified in the kat script, it might be a symbolic expression that needs to be evaluated)
#             _Rc = getattr(model, opt).Rcx.value.eval() if not isinstance(getattr(model, opt).Rcx.value, float) else getattr(model, opt).Rcx.value
#             _R = getattr(model, opt).R.value.eval() if not isinstance(getattr(model, opt).R.value, float) else getattr(model, opt).R.value

#             # Add the node to the graph with the feature vector as attributes
#             g.add_node(node, Rc=_Rc, R=_R, alpha=0)
#             print("Mirror node added with Rc:", _Rc, "R:", _R)

#         elif isinstance(getattr(model, opt), finesse.components.laser.Laser):
#             print("add a laser node")
#             g.add_node(node, Rc=0, R = 0, alpha=0)
        
#         else:
#             print("add an unknown node")
#             g.add_node(node, Rc=0, R = 0, alpha=0)
#     # Access edge attributes
#     for i, edge in enumerate(finesse_g.edges().data()):
#         print(f"Processing edge: {edge}")
#         dat = list(finesse_g.edges().data())[i][2]['owner']()
#         if not isinstance(dat, finesse.components.space.Space):
#             print("add a space edge")
#             g.add_edge(str(edge[0]), str(edge[1]), length=0, nr=1)
#         else:
#             print("add an unknown edge")
#             g.add_edge(str(edge[0]), str(edge[1]), length=dat.L.value if not isinstance(dat.L.value, finesse.symbols.Symbol) else dat.L.value.eval(), nr=dat.nr.value if not hasattr(dat.nr.value, 'eval') else dat.nr.value.eval())
    
#     return g

def _power_ouput(model_result, node_name):
    if model_result == -1:
        return 0

    return model_result[f'p_{node_name}']

def model_to_nx_port(model):

    if not model.cavArm.is_stable:
            out = -1
    else:
        try:
            out = model.run("""Series(
                                run_locks(max_iterations=100000,),
                                noxaxis(),
                                )""")["noxaxis"]
            
        except finesse.exceptions.LostLock:
            out = -1

    
    finesse_g = model.optical_network
    g = nx.DiGraph()
    
    for node in finesse_g.nodes():
        name = node.replace('.', '_')
        opt = node.split('.')[0]
        
        if isinstance(getattr(model, opt), finesse.components.mirror.Mirror):
             # Make sure that the attributes are evaluated to floats if they are not already (if the value is not specified in the kat script, it might be a symbolic expression that needs to be evaluated)
            _Rc = getattr(model, opt).Rcx.value.eval() if not isinstance(getattr(model, opt).Rcx.value, float) else getattr(model, opt).Rcx.value
            _R = getattr(model, opt).R.value.eval() if not isinstance(getattr(model, opt).R.value, float) else getattr(model, opt).R.value

            # Create feature vector
            g.add_node(node, Rc=_Rc, R = _R, alpha=0, 
                       fd=0, pd=_power_ouput(out, name), q=0)
            
        else:
            g.add_node(node, Rc=0, R = 0, alpha = 0, 
                       fd=0, pd=_power_ouput(out, name), q=0)

    # Access edge attributes
    for i, edge in enumerate(finesse_g.edges().data()):
        dat = list(finesse_g.edges().data())[i][2]['owner']()
        if not isinstance(dat, finesse.components.space.Space):
            g.add_edge(str(edge[0]), str(edge[1]), length=0, nr=1)
        else:
            g.add_edge(str(edge[0]), str(edge[1]), length=dat.L.value if not isinstance(dat.L.value, finesse.symbols.Symbol) else dat.L.value.eval(), nr=dat.nr.value if not hasattr(dat.nr.value, 'eval') else dat.nr.value.eval())
    
    return g