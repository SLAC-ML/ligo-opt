"""Mockup objective calculation for locking-parameter optimization.

Computes objectives from design parameters D and locking parameters L.
In reality, this would evaluate physics performance metrics based on
the locking configuration achieved for a given design.
"""

import numpy as np
from finesse.cymath.homs import HGModes
from scipy.integrate import dblquad
from scipy.signal import find_peaks

def calc_objective(D, L):
    """Compute scalar objective from design params and locking params.

    Mockup: minimize a combined metric that depends on both D and L.
    Lower is better.

    Args:
        D: Design parameters, shape (n, d)
        L: Locking parameters, shape (n, 2)

    Returns:
        obj: Objective values, shape (n,)
    """
    # Distance from target in design space
    d_cost = np.sum((D - 0.3) ** 2, axis=1) 

    # Distance from target in locking space
    l_cost = np.sum((L - 0.5) ** 2, axis=1)

    return d_cost + 2.0 * l_cost

def addSingleMod(n, m, kat):
    """
    Creates a Finesse model copy with a single laser mode (HG##) 
    and runs a frequency sweep simulation without applying any surface coatings.
    """
    kat_copy = kat.deepcopy()
    c=3.0e8 # assuming meters
    FSR = c/(2*kat_copy.CAV.L) # Free Spectral Range for cavity

    # Add only the single mode for laser
    kat_copy.parse(f"""
    tem(L0, 0, 0, factor=0,)
    tem(L0, {n}, {m}, factor=1,)
    ad HG{n}{m} node=ETM.p1.o f=L0.f n={n} m={m}
    """)

    # Set max tem mode to simulate
    max_modes = (n + m)*2 # why * 2?
    kat_copy.modes(maxtem=max_modes)

    # Run the model
    result = kat_copy.run(f"xaxis(L0.f, lin, {-FSR}, {FSR}, 10000)")
    return kat_copy, result, FSR

def findMaxPeak(frequency, power):
    """
    Finds the index and value of the highest local peak in the amplitude/power array, 
    considering only peaks where the corresponding frequency is non-negative.
    """
    power = np.asarray(power)  # Ensure data is a numpy array
    frequency = np.asarray(frequency)  # Ensure data is a numpy array

    # Check if the input data arrays are of the same length and do not contain NaN values
    if len(frequency) != len(power):
        raise ValueError("Input data arrays must have the same length.")
    if np.isnan(power).any():
        raise ValueError("Input power contains NaN values, this will mess with find_peak.")
    if np.isnan(frequency).any():
        raise ValueError("Input frequency contains NaN values.")
    peaks, _ = find_peaks(power)
    posPeaks = [i for i in peaks if frequency[i] >= 0] 
    if len(posPeaks) == 0:
        return None,None,None
    else: 
        maxPeakIndex = posPeaks[np.argmax(power[posPeaks])]
        maxPeakValue = power[maxPeakIndex]
        return maxPeakIndex, maxPeakValue, posPeaks 


def run_trails(kat):
    modes = np.linspace(1, 10, 1)
    frqs=[]
    for mode in modes:
        n = mode // 2 # arbitrary selection of specific mode
        m = mode - n
        _, result, FSR = addSingleMod(n,m,kat)
        NPeakIndex, _, _ = findMaxPeak(result.x1, np.abs(result[f"HG{n}{m}"]))
        NPeakFreq = result.x1[NPeakIndex]

    return frqs

def calc_HOM_cost():
    pass # HOM cost is currently a work in progress, dependent functions are mostly implemented above

def calc_round_trip_clipping_loss(kat): # TODO: change to just 00 mode, currently this is accounting for the entire intensity profile
    kat_copy = kat.deepcopy()
    kat_copy.parse("""fd E_itm ITM.p2.i f=0
              fd E_etm ETM.p1.i f=0""")
    out = kat_copy.run()
    r_apeture = 0.17

    HGs_ITM = HGModes(kat_copy.ITM.p2.i.q, kat_copy.homs)
    HGs_ETM = HGModes(kat_copy.ETM.p1.i.q, kat_copy.homs)

    amp_itm = out["E_itm"].flatten()[:, None]
    amp_etm = out["E_etm"].flatten()[:, None]
    def intensity_polar_ITM(r, theta):
        x = np.array([r * np.cos(theta)], dtype=np.float64)
        y = np.array([r * np.sin(theta)], dtype=np.float64)
        a = HGs_ITM.compute_points(x,y) * amp_itm
        E = np.sum(a, axis=0)[0]
        I = np.abs(E)**2 * r #jacobian
        return I
    def intensity_polar_ETM(r, theta):
        x = np.array([r * np.cos(theta)], dtype=np.float64)
        y = np.array([r * np.sin(theta)], dtype=np.float64)
        a = HGs_ETM.compute_points(x,y) * amp_etm
        E = np.sum(a, axis=0)[0]
        I = np.abs(E)**2 * r #jacobian
        return I
    loss_ITM, error_ITM = dblquad(intensity_polar_ITM, 0, 2*np.pi, lambda x: r_apeture, lambda x: 10)
    loss_ETM, error_ETM = dblquad(intensity_polar_ETM, 0, 2*np.pi, lambda x: r_apeture, lambda x: 10)
    loss = loss_ETM + loss_ITM
    power_circ = out['circ']
    loss_ppm = (loss / power_circ) * 1e6
    return loss_ppm


def calc_stability_cost(g_value):
    C_stable = 50 * (g_value - 0.5)**2
    return C_stable

def calc_cost(kat):
    kat.parse("""fd E_itm ITM.p2.i f=0
            fd E_itm ETM.p1.i f=0""")
    out = kat.run()
    HGs_ITM = HGModes(kat.ITM.p2.i.q, kat.homs)
    a = HGs_ITM.compute_points(0, 0) * out["E_itmx1"][:, None]
    C_sen = None

    # C_stable
    g_value = kat.cavArm.g[0]
    C_stable = calc_stability_cost(g_value)

    C_loss = 0
    C_HOM = 0
    C_total = C_sen + C_stable + C_loss + C_HOM
    return C_total