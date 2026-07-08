import numpy as np

roc_perturbation_percent = 0.0033  # Perturbation for ROC in percentage (0.33%)
cav_length_perturbation_std = 0.001  # Perturbation for cavity length in meters

def calc_perturbed_params(ITM_ROC, ETM_ROC):
    """Calculate perturbed parameters for PSO optimization.

    Args:
        ETM_ROC: End Test Mass Radius of Curvature (meters)
        ITM_ROC: Input Test Mass Radius of Curvature (meters)
        Cav_L: Cavity Length (meters)

    Returns:
        Tuple of perturbed parameters:
        (ETM_ROC_perturbed_pos, ETM_ROC, ETM_ROC_perturbed_neg),
        (ITM_ROC_perturbed_pos, ITM_ROC, ITM_ROC_perturbed_neg),
        (Cav_L_perturbed_pos, Cav_L, Cav_L_perturbed_neg)
    """
    ETM_ROC_Perturb = roc_perturbation_percent * ETM_ROC  
    ITM_ROC_Perturb = roc_perturbation_percent * ITM_ROC
    # Cav_L_Perturb = cav_length_perturbation_std

    # ETM_ROC_noise = np.abs(np.random.normal(0, ETM_ROC_Perturb_std))
    # ITM_ROC_noise = np.abs(np.random.normal(0, ITM_ROC_Perturb_std))
    # Cav_L_noise = np.abs(np.random.normal(0, Cav_L_Perturb_std))

    ETM_ROC_perturbed_pos = ETM_ROC + ETM_ROC_Perturb 
    ETM_ROC_perturbed_neg = ETM_ROC - ETM_ROC_Perturb

    ITM_ROC_perturbed_pos = ITM_ROC + ITM_ROC_Perturb 
    ITM_ROC_perturbed_neg = ITM_ROC - ITM_ROC_Perturb

    # Cav_L_perturbed_pos = Cav_L + Cav_L_Perturb
    # Cav_L_perturbed_neg = Cav_L - Cav_L_Perturb

    return (ITM_ROC_perturbed_pos, ITM_ROC, ITM_ROC_perturbed_neg), (ETM_ROC_perturbed_pos, ETM_ROC, ETM_ROC_perturbed_neg), (ITM_ROC_Perturb, ETM_ROC_Perturb)