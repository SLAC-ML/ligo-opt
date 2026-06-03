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


def simulate(D, L_init=None):
    """Finesse simulation that produce .

    Simulates the behavior of an expensive physics simulation:
    - With good initial guess (from surrogate): small residual noise
    - Without initial guess: larger noise (convergence difficulties)

    Args:
        D: Design parameters, shape (n, d)
        L_init: Initial guess for locking params from surrogate, shape (n, 2).
                None if no surrogate prediction available.
    Returns:
        L_star: Refined locking parameters, shape (n, 2)
    """
    L_true = true_locking_params(D)

    L_star = L_true
    return L_star

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


def simulate_finesse(D, L_init=None, noise_scale=0.00):

    D = tuple(x * noise_scale for x in D)

    if L_init is not None:
        # Good initial guess: simulation converges accurately
        L_star = finesse_sim(D, L_init)
    else:
        # No initial guess: larger residual (simulation struggles)
        L_star = finesse_sim(D)

    return L_star
