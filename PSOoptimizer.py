
from utils.calc import calc_cost
from utils.sim import finesse_sim
from GNN.GNN_run import run_GNN
from perturbation import calc_perturbed_params
import numpy as np

def main_wrapper_PSO_sim(params, kat):
    ITM_Roc, ETM_Roc = params
    # Call your main function with the unpacked parameters
    ITM_ROC_perturbed, ETM_ROC_perturbed, Delta = calc_perturbed_params(ITM_Roc, ETM_Roc)

    ITM_power_list = []
    ITM_name_list = []
    ETM_power_list = []
    ETM_name_list = []

    for ITM_ROC_p in ITM_ROC_perturbed:
        ITM_names, ITM_powers= finesse_sim((ITM_ROC_p, ETM_Roc), kat)
        ITM_power_list.append(ITM_powers)
        ITM_name_list.append(ITM_names)
    
    for ETM_ROC_p in ETM_ROC_perturbed:
        ETM_names, ETM_powers= finesse_sim((ITM_Roc, ETM_ROC_p), kat)
        ETM_power_list.append(ETM_powers)
        ETM_name_list.append(ETM_names)

    
    cost = calc_cost(kat, (name_list, power_list), Delta)
    return name_list, power_list

def main_wrapper_PSO_NN(params, kat):
    ITM_Roc, ETM_Roc = params
    # Call your main function with the unpacked parameters
    ITM_ROC_perturbed, ETM_ROC_perturbed = calc_perturbed_params(ITM_Roc, ETM_Roc)
    results = []
    for ITM_ROC_p, ETM_ROC_p, in zip(ITM_ROC_perturbed, ETM_ROC_perturbed):
        result = run_GNN((ITM_ROC_p, ETM_ROC_p), kat)
        results.append(result)
        print(result)

    # cost = calc_cost(kat, results)
    return None

def PSO_step(self, func):

    for iter_num in range(max_iter):
        self.update_V()
        self.recorder()
        self.update_X()
        self.func = func
        self.cal_y()
        self.update_pbest()
        self.update_gbest()
        self.gbest_y_hist.append(self.gbest_y)
        
        return self.gbest_x.copy(), self.gbest_y

# pso = PSO(func=main_wrapper_PSO, n_dim=3, pop=300, max_iter=1000, lb=[start_ETM_Roc, start_ITM_Roc], ub=[end_ETM_Roc, end_ITM_Roc], w=0.9, c1=1.2, c2=1.8, verbose=True)

# pso.run()
# print("Best parameters found: ", pso.gbest_x)
# print("Best cost found: ", pso.gbest_y)