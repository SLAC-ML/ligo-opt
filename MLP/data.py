"""Turn the existing simulation graphs into ordinary rows for the MLP.

The simulator already saves graphs to HDF5. We reuse those files, but the MLP
does not do graph message passing: each graph contributes just four inputs and
the measured power at one detector. This also keeps the GNN/MLP comparison on
the same expensive Finesse simulations.
"""

import math
from pathlib import Path
import pickle
import re

import h5py
import torch
from torch.utils.data import Dataset


from utils.detectors import POWER_DETECTOR

FEATURE_NAMES = ("ITM_Rc", "ETM_Rc", "q_input_real", "q_input_imag")

DEFAULT_DETECTOR = POWER_DETECTOR


def _finite_number(value, description):
    """Convert scalar simulation values to a real number with a useful error."""
    try:
        scalar = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{description} must be a real scalar.") from exc
    # Some detector arrays use complex dtype even when their value is real.
    # Accept those without silently discarding a nonzero imaginary component.
    if scalar.imag != 0:
        raise ValueError(f"{description} must be real, got {scalar}.")
    result = scalar.real
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite, got {result}.")
    return result


def _node_attribute(graph, node, attribute):
    try:
        value = graph.nodes[node][attribute]
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError(f"Graph is missing {node!r} attribute {attribute!r}.") from exc
    return _finite_number(value, f"{node}.{attribute}")


def graph_to_features(graph):
    """Read four inputs in FEATURE_NAMES order, returning a float32 tensor.

    The input q must come from ITM.p1.i. Perturbed simulations hold this beam
    fixed at their nominal design's q, so curvature alone is insufficient to
    describe those examples. q=(0, 0) is allowed: failed/unstable simulations
    deliberately store that value alongside zero-power labels.
    """
    values = [
        _node_attribute(graph, "ITM.p1.i", "Rc"),
        _node_attribute(graph, "ETM.p1.i", "Rc"),
        _node_attribute(graph, "ITM.p1.i", "q_re"),
        _node_attribute(graph, "ITM.p1.i", "q_im"),
    ]
    features = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("Graph features exceed the finite float32 range.")
    return features


def kat_to_features(kat, q_input):
    """Extract the same row directly from a prepared KAT, without a graph.

    The predictor supplies q_input from a beam trace after any fixed ``gauss``
    setting has been applied. Rcx matches the x-axis curvature used when the
    simulator wrote the training graphs (the current cavity is spherical).
    """
    values = []
    for mirror_name in ("ITM", "ETM"):
        value = getattr(kat, mirror_name).Rcx.value
        # Finesse parameters can be numbers or symbolic expressions.
        if hasattr(value, "eval"):
            value = value.eval()
        values.append(_finite_number(value, f"{mirror_name}.Rc"))
    values.extend([
        _finite_number(complex(q_input).real, "Re(input q)"),
        _finite_number(complex(q_input).imag, "Im(input q)"),
    ])
    features = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("KAT features exceed the finite float32 range.")
    return features


class PowerDataset(Dataset):
    """Load one or more existing ``sim_<number>`` HDF5 graph files.

    ``x`` contains raw inputs, and ``y`` contains log1p(power) targets. The
    network fits the compressed target because optical powers can span a wide
    range; inference converts it back with expm1. These HDF5 files contain
    pickles and must be trusted local simulation outputs.
    """

    def __init__(self, data_files, detector=DEFAULT_DETECTOR):
        if isinstance(data_files, (str, Path)):
            data_files = [data_files]
        self.data_files = [Path(path) for path in data_files]
        if not self.data_files:
            raise ValueError("At least one MLP training data file is required.")
        if not isinstance(detector, str) or not detector.startswith("p_"):
            raise ValueError("detector must be a per-node power detector name such as p_ITM_p2_i.")
        self.detector = detector
        rows, targets = [], []
        for path in self.data_files:
            with h5py.File(path, "r") as handle:
                keys = list(handle.keys())
                if any(re.fullmatch(r"sim_\d+", key) is None for key in keys):
                    raise ValueError(f"{path}: expected only sim_<number> graph entries.")
                # Lexical sorting puts sim_10 before sim_2. Numeric sorting
                # preserves consecutive design groups for data splitting.
                keys.sort(key=lambda key: int(key.removeprefix("sim_")))
                for key in keys:
                    try:
                        graph = pickle.loads(bytes(handle[key][()]))
                        row = graph_to_features(graph)
                        nodes = [node for node in graph.nodes
                                 if f"p_{node.replace('.', '_')}" == detector]
                        if len(nodes) != 1:
                            raise ValueError(f"Graph does not have a unique detector {detector!r}.")
                        power = _node_attribute(graph, nodes[0], "pd")
                        if power < 0:
                            raise ValueError(f"Detector {detector!r} has negative power {power}.")
                    except (ValueError, TypeError, KeyError, pickle.UnpicklingError) as exc:
                        raise ValueError(f"Invalid MLP sample {path}:{key}: {exc}") from exc
                    rows.append(row)
                    targets.append([math.log1p(power)])
        if not rows:
            raise ValueError("MLP training data contains no simulation graphs.")
        self.x = torch.stack(rows)
        # Preserve tiny plateau differences from the Finesse labels. Converting
        # log(1 + 282.09) to float32 would quantize powers by about 0.000135.
        # Inputs/network remain float32; the loss compares outputs in float64.
        self.y = torch.tensor(targets, dtype=torch.float64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
