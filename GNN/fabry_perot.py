"""
Generate Fabry-Perot cavity simulations using Finesse.

This module builds KAT model using pregenerated parameters,
and runs repeated perturbation simulations to collect graph-based
representations of the optical network.

The saved file is then used to train a GNN model
"""


import numpy as np
import networkx as nx
import finesse
import pickle
import warnings
import h5py

import finesse.analysis.actions as fac
from finesse.analysis.actions import Xaxis, Series
from finesse.components.readout import ReadoutDetectorOutput

from sympy import symbols, solveset, limit

from utils import model_to_nx_port
from tqdm import tqdm
from finesse_base import base_kat
from finesse.utilities.maps import circular_aperture
from finesse.knm import Map

finesse.init_plotting(fmts=["png"])

if __name__ == '__main__':
    ########
    # Parameters
    ########
    
    
    ITM_ROC_nominal = -1934
    ETM_ROC_nominal = 2245

    ITM_ROC_List = np.linspace(ITM_ROC_nominal - 500, ITM_ROC_nominal + 500, 174)
    ETM_ROC_List = np.linspace(ETM_ROC_nominal - 500, ETM_ROC_nominal + 500, 174)

    fabry_perot = base_kat
    x = y = np.linspace(-0.17, 0.17, 100)
    base_kat.ETM.surface_map = Map(x, y, amplitude=circular_aperture(x,y,0.17))
    
    data = []


    ########
    # Model 
    ########


    with tqdm(total=len(ITM_ROC_List)*len(ETM_ROC_List), position=0, desc="sims") as pbar:
        for ITM_ROC in ITM_ROC_List:
            for ETM_ROC in ETM_ROC_List:
                fabry_perot.ITM.Rc = ITM_ROC
                fabry_perot.ETM.Rc = ETM_ROC
                g = model_to_nx_port(fabry_perot)
                data.append(g)
                pbar.update(1)
            
    ########
    # Saving the result in hdf5 format
    ########
    
    with h5py.File('training_dataset_ligoParams.h5', 'w') as f:
        for i, graph in enumerate(data):
            # Serialize the graph object using pickle
            serialized_graph = pickle.dumps(graph)

            # Store the serialized graph in the HDF5 file
            f.create_dataset(f'sim_{i}', data=np.void(serialized_graph))