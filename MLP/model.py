"""The MLP itself: four physical inputs become one predicted log-power.

An MLP (multilayer perceptron) is a sequence of fully connected ``Linear``
layers. Each neuron can use every output of the previous layer. Nonlinear
activation functions between layers let the network learn a curved response
instead of just a straight-line fit.
"""

import torch
from torch import nn


DEFAULT_MODEL_CONFIG = {
    "input_size": 4,
    "hidden_sizes": [64, 64],
    "output_size": 1,
}


class PowerMLP(nn.Module):
    """Predict ``log1p(power)`` for one detector of the fixed cavity.

    Inputs are rows of [ITM Rc, ETM Rc, Re(input q), Im(input q)]. We keep
    normalization inside the model so training and later PSO predictions use
    exactly the same units and statistics, including after loading a file.
    """

    def __init__(self, input_size=4, hidden_sizes=None, output_size=1):
        super().__init__()
        if input_size != 4 or output_size != 1:
            raise ValueError("This cavity MLP requires input_size=4 and output_size=1.")
        hidden_sizes = list([64, 64] if hidden_sizes is None else hidden_sizes)
        if not hidden_sizes or any(
            not isinstance(width, int) or isinstance(width, bool) or width <= 0
            for width in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain one or more positive integers.")
        self.config = {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "output_size": output_size,
        }

        # Buffers move with model.to(device) and are saved in state_dict, but
        # unlike weights they are not adjusted by the optimizer.
        self.register_buffer("input_mean", torch.zeros(input_size))
        self.register_buffer("input_scale", torch.ones(input_size))

        layers = []
        previous_width = input_size
        for width in hidden_sizes:
            layers.append(nn.Linear(previous_width, width))
            # SiLU is smooth, which is helpful when the objective compares
            # powers at nearby mirror curvatures.
            layers.append(nn.SiLU())
            previous_width = width
        layers.append(nn.Linear(previous_width, output_size))
        # log(1 + power) cannot be negative when physical power is nonnegative.
        # Softplus enforces that condition without a sharp clipping boundary.
        layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)

    @torch.no_grad()
    def set_normalization(self, training_inputs):
        """Fit scaling using training rows only, before optimization starts.

        Standardization subtracts each column's mean and divides by its
        standard deviation. A constant column gets scale 1 to avoid division
        by zero. Never include validation rows when fitting these statistics.
        """
        values = torch.as_tensor(
            training_inputs, dtype=self.input_mean.dtype, device=self.input_mean.device,
        )
        if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
            raise ValueError("Normalization requires a nonempty [N, 4] input tensor.")
        if not torch.isfinite(values).all():
            raise ValueError("Normalization inputs must all be finite.")
        mean = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False)
        self.input_mean.copy_(mean)
        self.input_scale.copy_(torch.where(scale > 0, scale, torch.ones_like(scale)))

    def forward(self, inputs):
        """Accept raw physical values and return [N, 1] predicted log-powers."""
        if inputs.ndim != 2 or inputs.shape[1] != self.config["input_size"]:
            raise ValueError("PowerMLP expects a [N, 4] input tensor.")
        normalized = (inputs - self.input_mean) / self.input_scale
        return self.network(normalized)


def build_power_predictor(config=None):
    """Build the default network or reconstruct the configuration in a checkpoint."""
    settings = {**DEFAULT_MODEL_CONFIG, **({} if config is None else config)}
    unexpected = set(settings) - set(DEFAULT_MODEL_CONFIG)
    if unexpected:
        raise ValueError(f"Unknown MLP configuration fields: {sorted(unexpected)}")
    return PowerMLP(**settings)
