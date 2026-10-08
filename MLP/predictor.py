"""Use a trained MLP in the existing PSO power-prediction interface.

Finesse still supplies a cheap beam trace to describe the beam entering the
cavity. The neural network replaces the expensive optical power solve. No
NetworkX or PyTorch Geometric graph is constructed during MLP inference.
"""

import math

import torch

from .checkpoint import load_checkpoint
from .data import kat_to_features


class MLPPowerPredictor:
    """Predict one checkpoint-selected detector for each prepared KAT model."""

    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, payload = load_checkpoint(model_path, self.device)
        self.detector = payload["detector"]

    @staticmethod
    def _beam_q_values(kat):
        """Return node q values in the same format as the GNN/PSO interface.

        A perturbation's ``gauss`` setting is already on kat, so beam_trace
        respects its fixed nominal input q. Stability must be checked first:
        a forced beam can otherwise trace through an unstable cavity even
        though the reference Finesse workflow gives that cavity zero power.
        """
        nodes = list(kat.optical_network.nodes())
        names = [f"q_{node.replace('.', '_')}" for node in nodes]
        zeros = [0j] * len(nodes)
        try:
            if not kat.cavArm.is_stable:
                return names, zeros
            trace = kat.beam_trace()
            values = []
            for node in nodes:
                entry = trace.get(node)
                value = 0j if entry is None else complex(entry.qx)
                if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                    return names, zeros
                values.append(value)
            return names, values
        except Exception:
            # Match the simulator's convention for untraceable geometries.
            return names, zeros

    def _prepare_data(self, kat):
        """Snapshot one mutable KAT state into an independent feature tensor."""
        names, values = self._beam_q_values(kat)
        try:
            q_at_itm = values[names.index("q_ITM_p1_i")]
        except ValueError as exc:
            raise ValueError("MLP requires the fixed cavity's ITM.p1.i input node.") from exc
        features = kat_to_features(kat, q_at_itm)
        return features, q_at_itm

    def run_batch(self, kats):
        """Return ``(detector_names, powers, input_q)`` in input order.

        PSO can yield the same KAT object repeatedly while changing its mirror
        curvatures. Extract each row before advancing that iterator; collecting
        KAT references first would accidentally predict the final state N times.
        Only valid rows visit the neural network. An unstable or failed beam
        trace returns exactly zero, matching the simulation's convention.
        """
        prepared = [self._prepare_data(kat) for kat in kats]
        if not prepared:
            return []
        powers = [0.0] * len(prepared)
        valid_indices = [index for index, (_, q) in enumerate(prepared) if q != 0]
        if valid_indices:
            inputs = torch.stack([prepared[index][0] for index in valid_indices]).to(self.device)
            with torch.inference_mode():
                log_powers = self.model(inputs)
                # Double precision here gives expm1 extra numerical headroom;
                # the small neural network itself still uses float32.
                predictions = torch.expm1(log_powers.double()).flatten().cpu()
            if not torch.isfinite(predictions).all() or (predictions < 0).any():
                raise RuntimeError("MLP produced invalid power predictions; inspect the trained checkpoint.")
            for index, power in zip(valid_indices, predictions.tolist()):
                powers[index] = power
        return [([self.detector], [power], q)
                for power, (_, q) in zip(powers, prepared)]

    def run(self, kat):
        """Compatibility wrapper for callers evaluating one geometry at a time."""
        names, powers, _ = self.run_batch([kat])[0]
        return names, powers
