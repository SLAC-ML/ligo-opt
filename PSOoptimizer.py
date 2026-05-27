from sko.PSO import PSO
from utils.calc import calc_cost
from utils.sim import finesse_sim
from perturbation import calc_perturbed_params


def main_wrapper_PSO_sim(params):
    ETM_Roc, ITM_Roc, Cav_L = params
    # Call your main function with the unpacked parameters
    ETM_ROC_perturbed, ITM_ROC_perturbed, Cav_L_perturbed = calc_perturbed_params(ETM_Roc, ITM_Roc, Cav_L)
    results = []
    for ETM_ROC_p, ITM_ROC_p, Cav_L_p in zip(ETM_ROC_perturbed, ITM_ROC_perturbed, Cav_L_perturbed):
        result = finesse_sim(ETM_ROC_p, ITM_ROC_p, Cav_L_p)
        results.append(result)

    cost = calc_cost(results)
    return cost

def main_wrapper_PSO_NN(params):
    ETM_Roc, ITM_Roc, Cav_L = params
    # Call your main function with the unpacked parameters
    ETM_ROC_perturbed, ITM_ROC_perturbed, Cav_L_perturbed = calc_perturbed_params(ETM_Roc, ITM_Roc, Cav_L)
    results = []
    for ETM_ROC_p, ITM_ROC_p, Cav_L_p in zip(ETM_ROC_perturbed, ITM_ROC_perturbed, Cav_L_perturbed):
        result = NN_sim(ETM_ROC_p, ITM_ROC_p, Cav_L_p)
        results.append(result)

    cost = calc_cost(results)
    return cost


pso = PSO(func=main_wrapper_PSO, n_dim=3, pop=300, max_iter=1000, lb=[start_ETM_Roc, start_ITM_Roc, start_Cav_L], ub=[end_ETM_Roc, end_ITM_Roc, end_Cav_L], w=0.9, c1=1.2, c2=1.8, verbose=True)

pso.run()
print("Best parameters found: ", pso.gbest_x)
print("Best cost found: ", pso.gbest_y)