"""Teach the MLP both the overall power curve and the nearly flat central plateau.

The map's bright region is around 282.09, but differences of only a few
thousandths are visible there. Ordinary squared errors in log(1 + power)
make these differences extremely small. We add a more sensitive Smooth-L1
term for examples whose *Finesse label* is close to that plateau.

This is a supervised training loss. It does not change PSO's cavity objective,
replace simulation labels by 282.09, or infer an optimum from plot colors.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_LOSS_CONFIG = {
    'mode': 'center_focused',  # 'log_smooth_l1' restores the original loss.
    'supervision': 'row_power',  # 'nominal_delta' also teaches signed responses.
    'delta_weight': 1.0,
    'delta_scale_floor_w': 1.0,
    'center_power': 282.09,   # Approximate power of the region we want to resolve.
    'center_width': 0.01,     # Width of a soft band around that reference power.
    'center_error_scale': 0.001,  # Desired sensitivity, in the same units as power.
    'center_weight': 10.0,    # Relative influence of the extra plateau loss.
}


def resolve_loss_config(overrides=None):
    """Return a complete independent configuration, catching misspelled settings."""
    config = dict(DEFAULT_LOSS_CONFIG)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError('MLP loss settings must be a dictionary.')
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f'Unknown MLP loss settings: {sorted(unknown)}')
        config.update(overrides)
    if config['mode'] not in ('log_smooth_l1', 'center_focused'):
        raise ValueError('MLP loss mode must be log_smooth_l1 or center_focused.')
    if config['supervision'] not in ('row_power', 'nominal_delta'):
        raise ValueError('MLP supervision must be row_power or nominal_delta.')
    for key in ('center_power', 'center_width', 'center_error_scale', 'center_weight'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite real number.')
        if value < 0 or (key != 'center_weight' and value == 0):
            raise ValueError(f'{key} must be positive (center_weight may also be zero).')
    for key in ('delta_weight', 'delta_scale_floor_w'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite real number.')
        if value < 0 or (key == 'delta_scale_floor_w' and value == 0):
            raise ValueError(f'{key} must be positive (delta_weight may also be zero).')
    return config


def signed_delta_losses(
    predicted_log_power, target_log_power,
    nominal_predicted_log_power, nominal_target_log_power, delta_scale_w,
):
    """Return one scaled signed-response loss per [N, 1] partner/nominal pair.

    Differences are in watts, after converting log-powers in float64. The
    nominal predictions remain attached so both evaluations receive gradients.
    The caller supplies a fixed training-only scale and applies any weights.
    """
    values = (
        predicted_log_power, target_log_power,
        nominal_predicted_log_power, nominal_target_log_power,
    )
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError('Signed delta loss expects tensor predictions and targets.')
    if any(value.ndim != 2 or value.shape[1] != 1 for value in values):
        raise ValueError('Signed delta loss expects [N, 1] predictions and targets.')
    if any(value.shape != predicted_log_power.shape for value in values):
        raise ValueError('Signed delta predictions and targets must have the same shape.')
    if any(not value.is_floating_point() for value in values):
        raise ValueError('Signed delta loss expects real floating-point log-powers.')
    if any(value.device != predicted_log_power.device for value in values):
        raise ValueError('Signed delta predictions and targets must use the same device.')
    if (isinstance(delta_scale_w, bool) or not isinstance(delta_scale_w, (int, float))
            or not math.isfinite(delta_scale_w) or delta_scale_w <= 0):
        raise ValueError('delta_scale_w must be finite and positive.')
    if any(not torch.isfinite(value).all() for value in values):
        raise FloatingPointError('Signed delta log-powers must be finite.')
    powers = tuple(value.double().expm1() for value in values)
    if any(not torch.isfinite(power).all() for power in powers):
        raise FloatingPointError('Signed delta power conversion became non-finite.')
    predicted, target, nominal_predicted, nominal_target = powers
    error = ((predicted - nominal_predicted) - (target - nominal_target)) / delta_scale_w
    if not torch.isfinite(error).all():
        raise FloatingPointError('Scaled signed delta error became non-finite.')
    return F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none').mean(dim=1)


class PowerPredictionLoss(nn.Module):
    """Return one loss per example; the trainer applies feedback weights afterward.

    Let e = predicted_log_power - true_log_power. Our loss is:

        SmoothL1(e, beta=1)
        + center_weight * band(true_power) * SmoothL1(e, beta=small_beta)

    band(P) = exp(-0.5 * ((P - center_power) / center_width)**2).

    The band depends only on labels, so the network cannot avoid the extra loss
    by predicting a power outside the band. It selects a *power band*, including
    any other geometry with that power, rather than a hard-coded ROC rectangle.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = resolve_loss_config(config)
        # Near reference power P, a power error dP corresponds to a log error
        # log(1 + dP / (1 + P)). log1p avoids subtracting two almost equal logs.
        self.center_beta = math.log1p(
            self.config['center_error_scale'] / (1.0 + self.config['center_power'])
        )
        if not math.isfinite(self.center_beta) or self.center_beta <= 0:
            raise ValueError('center_error_scale is too small/large for the reference power.')

    def center_weights(self, target_log_power):
        """Smoothly emphasize measured powers near the plateau; zeros get no boost."""
        power = torch.expm1(target_log_power.detach().double())
        distance = (power - self.config['center_power']) / self.config['center_width']
        return torch.exp(-0.5 * distance.square())

    def forward(self, predicted_log_power, target_log_power):
        if predicted_log_power.shape != target_log_power.shape:
            raise ValueError('Predicted and target log-powers must have the same shape.')
        if predicted_log_power.ndim != 2 or predicted_log_power.shape[1] != 1:
            raise ValueError('MLP power loss expects [N, 1] predictions and targets.')
        # Keep label differences in double precision. Gradients still flow back
        # through this cast into the existing float32 network and its weights.
        predicted = predicted_log_power.double()
        target = target_log_power.double()
        overall = F.smooth_l1_loss(predicted, target, reduction='none')
        if self.config['mode'] == 'log_smooth_l1' or self.config['center_weight'] == 0:
            return overall.mean(dim=1)
        # A smaller beta makes errors around the requested power scale matter.
        # Unlike squared normalized errors, Smooth-L1's slope stays bounded at
        # large errors, which helps while the model is still learning the scale.
        center = F.smooth_l1_loss(predicted, target, beta=self.center_beta, reduction='none')
        return (overall + self.config['center_weight'] * self.center_weights(target) * center).mean(dim=1)

    @torch.no_grad()
    def power_metrics(self, predicted_log_power, target_log_power):
        """Report interpretable errors in power units, including a center-only MAE.

        MAE means mean absolute error. For reporting only, the center is the
        hard band |true_power - center_power| <= center_width. Training uses the
        smooth band above, so it has no abrupt boundary. No center examples
        means an unavailable metric (None), not a misleading error of zero.
        """
        target = torch.expm1(target_log_power.double())
        prediction = torch.expm1(predicted_log_power.double())
        error = (prediction - target).abs()
        if not torch.isfinite(error).all():
            raise FloatingPointError('MLP power error became non-finite during validation.')
        center = (target - self.config['center_power']).abs() <= self.config['center_width']
        count = int(center.sum().item())
        return {
            'power_mae': error.mean().item(),
            'center_power_mae': error[center].mean().item() if count else None,
            'center_sample_count': count,
        }
