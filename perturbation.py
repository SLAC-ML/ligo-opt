roc_perturbation_percent = 0.0033  # Perturbation for ROC in percentage (0.33%)

def calc_perturbed_params(ITM_ROC, ETM_ROC):
    """Return symmetric positive/negative ROC perturbations and step magnitudes.

    Args:
        ITM_ROC: Input Test Mass Radius of Curvature (meters)
        ETM_ROC: End Test Mass Radius of Curvature (meters)

    Returns:
        ``((ITM + delta_ITM, ITM - delta_ITM),
        (ETM + delta_ETM, ETM - delta_ETM),
        (delta_ITM, delta_ETM))``, where both deltas are positive magnitudes.

        Using magnitudes matters for the negative LIGO ITM ROC: the entry named
        ``positive`` should be numerically greater than the design value rather than
        becoming more negative. The perturbation cost is symmetric, so this merely
        makes the labels and denominator convention unambiguous.
    """
    ETM_ROC_Perturb = abs(roc_perturbation_percent * ETM_ROC)
    ITM_ROC_Perturb = abs(roc_perturbation_percent * ITM_ROC)

    ETM_ROC_perturbed_pos = ETM_ROC + ETM_ROC_Perturb
    ETM_ROC_perturbed_neg = ETM_ROC - ETM_ROC_Perturb

    ITM_ROC_perturbed_pos = ITM_ROC + ITM_ROC_Perturb
    ITM_ROC_perturbed_neg = ITM_ROC - ITM_ROC_Perturb

    return (ITM_ROC_perturbed_pos, ITM_ROC_perturbed_neg), (ETM_ROC_perturbed_pos, ETM_ROC_perturbed_neg), (ITM_ROC_Perturb, ETM_ROC_Perturb)