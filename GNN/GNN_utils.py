import numpy as np
import finesse
import networkx as nx
import torch
import h5py
import pickle

from finesse.utilities.maps import circular_aperture
from finesse.knm import Map

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

def model_to_nx_port(model, pd_names=None, powers=None, q_names=None, q_values=None):
    '''Builds the nx.DiGraph of a finesse model's optical network -- the node/edge
    feature vectors the GNN trains and infers on. This does NOT run the finesse
    simulation itself: pass pd_names/powers and q_names/q_values
    (finesse_sim(model)'s return value, see utils/sim.py) when you need real pd/q
    labels, e.g. to build training data. Leave them unset when you only need the
    graph's static geometry (Rc/R/alpha/edges), e.g. for GNN inference -- pd and q
    default to 0 and the expensive simulation is skipped entirely.
    '''
    power_by_name = dict(zip(pd_names, powers)) if pd_names is not None else {}
    q_by_name = dict(zip(q_names, q_values)) if q_names is not None else {}

    finesse_g = model.optical_network
    g = nx.DiGraph()

    for node in finesse_g.nodes():
        name = node.replace('.', '_')
        opt = node.split('.')[0]
        pd = power_by_name.get(f'p_{name}', 0)
        q_re = np.real(q_by_name.get(f'q_{name}', 0))
        q_im = np.imag(q_by_name.get(f'q_{name}', 0))

        if isinstance(getattr(model, opt), finesse.components.mirror.Mirror):
             # Make sure that the attributes are evaluated to floats if they are not already (if the value is not specified in the kat script, it might be a symbolic expression that needs to be evaluated)
            _Rc = getattr(model, opt).Rcx.value.eval() if not isinstance(getattr(model, opt).Rcx.value, float) else getattr(model, opt).Rcx.value
            _R = getattr(model, opt).R.value.eval() if not isinstance(getattr(model, opt).R.value, float) else getattr(model, opt).R.value
           
            # Create feature vector
            g.add_node(node, Rc=_Rc, R = _R, 
                       pd=pd, q_re=q_re, q_im=q_im)

        else:
            g.add_node(node, Rc=0, R = 0,
                       pd=pd, q_re=q_re, q_im=q_im)

    # Access edge attributes
    for i, edge in enumerate(finesse_g.edges().data()):
        dat = list(finesse_g.edges().data())[i][2]['owner']()
        if not isinstance(dat, finesse.components.space.Space):
            g.add_edge(str(edge[0]), str(edge[1]), length=0, nr=1)
        else:
            g.add_edge(str(edge[0]), str(edge[1]), length=dat.L.value if not isinstance(dat.L.value, finesse.symbols.Symbol) else dat.L.value.eval(), nr=dat.nr.value if not hasattr(dat.nr.value, 'eval') else dat.nr.value.eval())

    return g


def append_graph_to_h5(graph, path):
    '''Appends a single nx.DiGraph to an h5 file, in the same sim_{i}/pickled format
    fabry_perot.py writes its training data in -- so the file can be used directly as
    fine-tune round data. Safe to call repeatedly on the same path: each call adds one
    more entry, indexed off however many are already in the file (creates the file if
    it doesn't exist yet).
    '''
    with h5py.File(path, 'a') as f:
        i = len(f.keys())
        f.create_dataset(f'sim_{i}', data=np.void(pickle.dumps(graph)))


def kat_manipulation(
    ITM_Roc,
    ETM_Roc,
    base_kat,
    nominal_q_value=None,
    include_aperture_maps=True,
):
    """
    Build a modified copy of a KAT (Finesse) model with specific ITM/ETM
    radii of curvature and a fixed input beam parameter, leaving the
    original base_kat untouched.

    Used internally by cost_fn (see main_wrapper_PSO) to generate the KAT
    models needed for each PSO cost evaluation: the perturbed ITM/ETM ROC
    sweeps and the baseline (unperturbed) run.

    Parameters
    ----------
    ITM_Roc : float
        Radius of curvature to set on the ITM (input test mass) mirror.
    ETM_Roc : float
        Radius of curvature to set on the ETM (end test mass) mirror.
    base_kat : Kat
        Template Finesse KAT model to copy and modify. Never mutated
        directly — a deep copy is modified and returned instead, so the
        same base_kat can be reused safely across many calls (e.g. every
        PSO particle/iteration).
    nominal_q_value : complex or str
        Fixed complex beam parameter (q) to impose at the ITM input,
        parsed directly into the KAT file as a `gauss` command.
    include_aperture_maps : bool, default True
        Add the finite-aperture surface maps required by real Finesse
        simulations. GNN-only inference can disable these maps because they are
        not part of the surrogate graph features and do not affect beam tracing.

    Returns
    -------
    Kat
        A new KAT model instance with the requested aperture maps, fixed
        q-parameter, and ITM/ETM ROC values applied.
    """
    # Deep copy so we don't mutate the caller's base_kat — each PSO
    # particle/perturbation needs its own independent model instance.
    run_kat = base_kat.deepcopy()

    if include_aperture_maps:
        # Define a circular aperture grid (17 cm radius, 1001x1001 sample points)
        # for real Finesse simulations. The GNN does not consume surface maps, so
        # its hot PSO path skips this allocation via include_aperture_maps=False.
        x = y = np.linspace(-0.17, 0.17, 1001)
        run_kat.ITM.surface_map = Map(
            x, y, amplitude=circular_aperture(x, y, 0.17)
        )
        run_kat.ETM.surface_map = Map(
            x, y, amplitude=circular_aperture(x, y, 0.17)
        )

    # Force the input beam's complex beam parameter (q) at the ITM to a
    # fixed nominal value, rather than letting Finesse solve for the
    # cavity's self-consistent eigenmode. This isolates the effect of
    # ROC changes on power/coupling without the beam mode also shifting.
    #
    # Guard against q == 0: an unstable (or otherwise unsolvable) cavity
    # comes back from finesse_sim with q = 0 (see utils/sim.py). Passing
    # that on a `gauss` command raises "Waist size must be a positive
    # number". Treating a zero q the same as None (no forced beam) keeps
    # every caller safe regardless of where the 0 came from.
    if nominal_q_value is not None and nominal_q_value != 0:
        run_kat.parse(f"gauss fixed_q_value ITM.p1.i q={nominal_q_value}")

    # Apply the ROC values being tested — these are the two PSO design
    # variables (params[0], params[1] in cost_fn).
    run_kat.ITM.Rc = ITM_Roc
    run_kat.ETM.Rc = ETM_Roc

    return run_kat