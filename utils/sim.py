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
from utils.finesse_base import base_kat
from finesse.knm import Map
from finesse.utilities.maps import circular_aperture

def finesse_sim(D, L_init=None):

    ITM_ROC, ETM_ROC = D
    
    kat = base_kat.deepcopy()
    kat.ITM.Rc = ITM_ROC
    kat.ETM.Rc = ETM_ROC
    kat.ETM.phi = L_init
    x = y = np.linspace(-0.17, 0.17, 101)
    kat.ETM.surface_map = Map(x, y, amplitude=circular_aperture(x,y,0.17))
    kat.parse("""fd E_itm ITM.p2.i f=0
              fd E_etm ETM.p1.i f=0""")
    out = kat.run("run_locks(display_progress=true,)")

    L = kat.ETM.phi.value

    return out
