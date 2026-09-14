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

def _power_ouput(model_result, node_name):
    if model_result == -1:
        return 0

    return model_result["noxaxis"][node_name]

def _q_output(model_result, node_name):
    """Extract a beam-parameter (q) detector value from a simulation result.

    The base kat model (see utils/finesse_base.py) already defines a `bp`
    detector named `q_{node}` for every node in the optical network, so the
    q value for a given node is simply `out["q_{node}"]`. If the simulation
    failed (out == -1) we return 0 as a placeholder, matching the power
    convention used by `_power_ouput`.
    """
    if model_result == -1:
        return 0

    return model_result["noxaxis"][node_name]

def finesse_sim(run_kat):
    if not run_kat.cavArm.is_stable:
            out = -1
    else:
        try:
            out = run_kat.run("""Series(
                                run_locks(max_iterations=100000,),
                                noxaxis(),
                                )""")["noxaxis"]
            
        except finesse.exceptions.LostLock:
            out = -1
        except Exception:
            # A cavity that passes the stability check but still fails to solve
            # (e.g. sitting right on the stability boundary, or a lock that never
            # converges) is treated as unsolvable the same way an unstable cavity
            # is: everything comes back as 0, so the GNN learns these geometries
            # as zero-power. Keeps a single bad grid point from aborting a long
            # data-generation run.
            out = -1
    
    pd_names = []
    q_names = []
    kat_g = run_kat.optical_network
    for node in kat_g.nodes():
            name = node.replace('.', '_')
            pd_names.append(f"p_{name}")
            q_names.append(f"q_{name}")

    powers = []
    for name in pd_names:
        powers.append(_power_ouput(out, name))

    # Extract the complex beam parameter (q) for every node. For a simple
    # cavity the q value at the input node (e.g. ITM.p1.i) is the laser's
    # beam parameter matched to the cavity eigenmode. The `bp` detectors are
    # already added to the base kat model in utils/finesse_base.py.
    q_values = []
    for name in q_names:
        q_values.append(_q_output(out, name))
    # L = kat.ETM.phi.value

    return pd_names, powers, q_names, q_values
