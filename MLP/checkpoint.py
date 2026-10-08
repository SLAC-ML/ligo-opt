"""Save everything needed to reproduce a trained MLP's predictions.

Weights alone are insufficient: a loaded model must use the same feature
order, detector, layer sizes, and normalization statistics. Keeping these in
one versioned checkpoint prevents accidentally combining incompatible pieces.
"""

from pathlib import Path

import torch

from .data import FEATURE_NAMES
from .model import build_power_predictor


SCHEMA_VERSION = 1


def save_checkpoint(path, model, detector, training_metadata=None):
    """Write weights, normalization buffers, and plain training information."""
    if not isinstance(detector, str) or not detector.startswith("p_"):
        raise ValueError("Checkpoint detector must be a per-node power detector name.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_config": dict(model.config),
        "feature_names": list(FEATURE_NAMES),
        "detector": detector,
        # CPU tensors make this file portable between GPU and CPU machines.
        "state_dict": {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()},
        "training_metadata": {} if training_metadata is None else dict(training_metadata),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device=None):
    """Return ``(model, payload)`` and reject incompatible checkpoint formats.

    ``weights_only=True`` restricts loading to tensors and safe plain Python
    values. Architecture is built by our own code, never unpickled as a model.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported MLP checkpoint schema; expected version {SCHEMA_VERSION}.")
    required = {"model_config", "feature_names", "detector", "state_dict", "training_metadata"}
    if not required.issubset(payload):
        raise ValueError(f"MLP checkpoint is missing fields: {sorted(required - set(payload))}")
    if payload["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("MLP checkpoint feature order does not match this implementation.")
    if not isinstance(payload["detector"], str) or not payload["detector"].startswith("p_"):
        raise ValueError("MLP checkpoint has an invalid detector name.")
    if not isinstance(payload["training_metadata"], dict):
        raise ValueError("MLP checkpoint training_metadata must be a dictionary.")
    model = build_power_predictor(payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    if not torch.isfinite(model.input_mean).all() or not (
        torch.isfinite(model.input_scale).all() and (model.input_scale > 0).all()
    ):
        raise ValueError("MLP checkpoint contains invalid input normalization statistics.")
    model.eval()
    return model, payload
