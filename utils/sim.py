"""Mockup simulation for locking-parameter refinement.

In reality, this would be an expensive physics simulation (e.g., particle
tracking in an accelerator lattice) that finds accurate locking parameters
given a design configuration.

The mockup demonstrates the key property: with a good initial guess from
the surrogate model, the simulation converges faster and more accurately.
"""

import numpy as np
import finesse
import matplotlib.pyplot as plt
# from finesse.knm import Map
# from finesse.utilities.maps import circular_aperture

def finesse_sim(Design_param, base_kat):

    ITM_ROC, ETM_ROC = Design_param
    
    kat = base_kat.deepcopy()
    kat.ITM.Rc = ITM_ROC
    kat.ETM.Rc = ETM_ROC
    # x = y = np.linspace(-0.17, 0.17, 101)
    # kat.ETM.surface_map = Map(x, y, amplitude=circular_aperture(x,y,0.17))
    # kat.parse("""fd E_itm ITM.p2.i f=0
    #           fd E_etm ETM.p1.i f=0""")
    out = kat.run("""Series(
                          run_locks(display_progress=false),
                          noxaxis(),
                          )""")
    
    pd_names = []
    kat_g = kat.optical_network
    for node in kat_g.nodes():
            name = node.replace('.', '_')
            pd_names.append(f"p_{name}")
    powers = []
    for name in pd_names:
        powers.append(out["noxaxis"][name])
    # L = kat.ETM.phi.value

    return pd_names, powers
