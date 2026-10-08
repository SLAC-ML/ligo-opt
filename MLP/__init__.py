"""A small, fully connected surrogate for this repository's fixed optical cavity.

Start with ``README.md`` for the workflow, ``data.py`` for the four input
numbers, and ``model.py`` for the neural network that turns them into power.
The package does not import training or Finesse automatically: importing the
model alone only needs PyTorch.
"""

from .model import DEFAULT_MODEL_CONFIG, PowerMLP, build_power_predictor

__all__ = ["DEFAULT_MODEL_CONFIG", "PowerMLP", "build_power_predictor"]
