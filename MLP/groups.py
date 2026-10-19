"""Nominal references for loss computation; saved samples remain unchanged."""

import math

import torch


def design_row_metadata(inputs, group_size, row_indices=None, *, validate_directions=False):
    """Map selected rows to their nominal row in the selected tensor's order.

    Initial splits may reorder whole designs. Construct references from original
    row IDs, rather than assuming that a shuffled batch contains adjacent pairs.
    A size-one feedback group supplies nominal supervision without a delta term.
    """
    count = len(inputs)
    if (isinstance(group_size, bool) or not isinstance(group_size, int)
            or group_size < 1 or not count or count % group_size):
        raise ValueError('Loss references require complete consecutive design groups.')
    if validate_directions and group_size == 5:
        grouped = inputs.reshape(-1, 5, inputs.shape[1])
        differences = grouped[:, 1:, :2] - grouped[:, :1, :2]
        for row, axis, sign in ((0, 0, 1), (1, 0, -1), (2, 1, 1), (3, 1, -1)):
            if ((differences[:, row, axis] * sign <= 0).any()
                    or (differences[:, row, 1 - axis].abs() > 1e-3).any()):
                raise ValueError('Unexpected perturbation order; expected nominal, ITM+, ITM-, ETM+, ETM-.')
    indices = (torch.arange(count) if row_indices is None
               else torch.as_tensor(row_indices, dtype=torch.long))
    if (indices.ndim != 1 or not len(indices) or (indices < 0).any()
            or (indices >= count).any() or len(indices.unique()) != len(indices)):
        raise ValueError('Loss references require unique, valid row indices.')
    selected_counts = torch.bincount(indices // group_size, minlength=count // group_size)
    if ((selected_counts != 0) & (selected_counts != group_size)).any():
        raise ValueError('Loss references cannot separate a nominal design from its partners.')
    local = torch.full((count,), -1, dtype=torch.long)
    local[indices] = torch.arange(len(indices))
    nominal = local[(indices // group_size) * group_size]
    return nominal, torch.full((len(indices),), group_size, dtype=torch.long)


def initial_delta_scale(target_log_power, nominal_indices, floor_w):
    """Resolve one fixed RMS scale from initial training labels only."""
    if (isinstance(floor_w, bool) or not isinstance(floor_w, (int, float))
            or not math.isfinite(floor_w) or floor_w <= 0):
        raise ValueError('The delta scale floor must be finite and positive.')
    partners = torch.arange(len(target_log_power)) != nominal_indices
    if not partners.any():
        raise ValueError('Delta supervision requires initial training perturbations.')
    powers = target_log_power.detach().double().expm1().flatten()
    differences = powers[partners] - powers[nominal_indices[partners]]
    rms = differences.square().mean().sqrt().item()
    if not math.isfinite(rms):
        raise FloatingPointError('Initial training delta scale became non-finite.')
    return max(float(floor_w), rms)


@torch.no_grad()
def response_metrics(log_predictions, log_targets, nominal_indices, group_sizes, inputs):
    """Report signed responses without changing the checkpoint-selection score."""
    predicted = log_predictions.double().expm1().flatten()
    target = log_targets.double().expm1().flatten()
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Response metrics require finite powers.')
    rows = torch.arange(len(predicted))
    nominal_mask = rows == nominal_indices
    partners = ~nominal_mask
    true_delta = target - target[nominal_indices]
    predicted_delta = predicted - predicted[nominal_indices]
    error = (predicted_delta - true_delta).abs()
    significant = partners & (true_delta.abs() > 0.1)
    sign_count = int(significant.sum())
    metrics = {
        'nominal_power_mae': (predicted[nominal_mask] - target[nominal_mask]).abs().mean().item(),
        'delta_mae': error[partners].mean().item() if partners.any() else None,
        'delta_max_abs_error': error[partners].max().item() if partners.any() else None,
        'delta_sign_threshold_w': 0.1,
        'delta_sign_count': sign_count,
        'delta_sign_accuracy': (predicted_delta[significant].sign() == true_delta[significant].sign()).double().mean().item() if sign_count else None,
    }
    steps = inputs[:, :2].double() - inputs[nominal_indices, :2].double()
    by_direction = {}
    for name, axis, sign in (('ITM_plus', 0, 1), ('ITM_minus', 0, -1),
                             ('ETM_plus', 1, 1), ('ETM_minus', 1, -1)):
        mask = partners & (group_sizes == 5) & (steps[:, axis] * sign > 0)
        n_sign = mask & significant
        n = int(n_sign.sum())
        by_direction[name] = {
            'count': int(mask.sum()),
            'delta_mae': error[mask].mean().item() if mask.any() else None,
            'sign_count': n,
            'sign_accuracy': (predicted_delta[n_sign].sign() == true_delta[n_sign].sign()).double().mean().item() if n else None,
        }
    metrics['delta_by_direction'] = by_direction
    # Four-direction cost uses positive ROC-step magnitudes, as in the objective.
    cost_errors = []
    for nominal in rows[nominal_mask & (group_sizes == 5)].tolist():
        mask = partners & (nominal_indices == nominal)
        roc_steps = steps[mask].abs().sum(dim=1)
        if (roc_steps > 0).all():
            true_cost = (true_delta[mask].abs() / roc_steps).sum()
            predicted_cost = (predicted_delta[mask].abs() / roc_steps).sum()
            cost_errors.append((predicted_cost - true_cost).abs().item())
    metrics['perturbation_cost_mae'] = sum(cost_errors) / len(cost_errors) if cost_errors else None
    return metrics
