"""Read saved pairs/five-row groups without modifying the source artifacts."""

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import h5py
import torch

from MLP.data import DEFAULT_DETECTOR, FEATURE_NAMES, PowerDataset
from utils.training_groups import read_design_group_size

REPO_ROOT = Path(__file__).resolve().parents[2]
DIRECTIONS = ("nominal", "ITM_plus", "ITM_minus", "ETM_plus", "ETM_minus")


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    # Newer runs can record diagnostics absent from older training histories.
    # Keep every field, in first-seen order, and leave unavailable values blank.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fingerprints(paths):
    return {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in sorted(set(map(str, paths)))}


@dataclass
class Group:
    group_id: str
    source: str
    keys: list
    directions: tuple
    x: torch.Tensor
    y: torch.Tensor

    @property
    def valid_fixed_q(self):
        # Failed solves can have q=0; never call those genuine mismatch examples.
        return bool((self.x[:, 3] > 0).all() and torch.allclose(
            self.x[:, 2:], self.x[0, 2:].expand(len(self.x), 2), rtol=1e-6, atol=1e-4,
        ))

    def records(self):
        powers = self.y.expm1().flatten().tolist()
        for i, (key, direction, features, power) in enumerate(zip(
            self.keys, self.directions, self.x.tolist(), powers,
        )):
            yield {
                "group_id": self.group_id, "source": self.source, "h5_key": key,
                "direction": direction, **dict(zip(FEATURE_NAMES, features)),
                "power": power, "nominal_power": powers[0],
                "delta_power": power - powers[0],
                "delta_itm_roc": features[0] - self.x[0, 0].item(),
                "delta_etm_roc": features[1] - self.x[0, 1].item(),
                "zero_output": power == 0, "valid_fixed_q_group": self.valid_fixed_q,
            }


def _load_file(path, detector):
    # PowerDataset sorts numerically; require contiguous keys so recorded row IDs
    # cannot silently refer to a different sample in a damaged file.
    dataset = PowerDataset(path, detector=detector)
    with h5py.File(path, "r") as handle:
        if set(handle) != {f"sim_{i}" for i in range(len(dataset))}:
            raise ValueError(f"Noncontiguous simulation keys in {path}")
    return dataset


def _validate_direction_order(x, path, group_id):
    # Initial five-row designs and feedback share PSOoptimizer's order.
    differences = x[1:, :2] - x[0, :2]
    for row, axis, sign in ((0, 0, 1), (1, 0, -1), (2, 1, 1), (3, 1, -1)):
        if differences[row, axis] * sign <= 0 or abs(differences[row, 1-axis]) > 1e-3:
            raise ValueError(f"Unexpected perturbation order in {path}, group {group_id}")


def load_groups(config, selection):
    root = repo_path(config["data_dir"])
    detector = config.get("detector", DEFAULT_DETECTOR)
    groups = []
    base_ids = selection.get("base_ids", [])
    if "base_id_range" in selection:
        if base_ids:
            raise ValueError("Use base_ids or base_id_range, not both.")
        base_ids = list(range(*selection["base_id_range"]))
    if base_ids == "all" or base_ids:
        paths = list(root.glob("base*.h5"))
        if len(paths) != 1:
            raise ValueError(f"Expected one base*.h5 file in {root}")
        path = paths[0]
        data = _load_file(path, detector)
        group_size = read_design_group_size(path)
        if group_size not in (2, len(DIRECTIONS)):
            raise ValueError(f"Unsupported base design group size {group_size} in {path}; expected 2 or 5.")
        if len(data) % group_size:
            if group_size == 2:
                raise ValueError("Base dataset must contain consecutive nominal/perturbed pairs.")
            raise ValueError(f"{path}: expected complete nominal + four-perturbation groups.")
        directions = ("nominal", "joint") if group_size == 2 else DIRECTIONS
        ids = range(len(data) // group_size) if base_ids == "all" else base_ids
        for group_id in ids:
            if not isinstance(group_id, int) or not 0 <= group_id < len(data) // group_size:
                kind = "pair" if group_size == 2 else "group"
                raise ValueError(f"Invalid base {kind} ID: {group_id}")
            start = group_size * group_id
            x = data.x[start:start + group_size]
            if group_size == len(DIRECTIONS):
                _validate_direction_order(x, path, group_id)
            groups.append(Group(f"base:{group_id}", str(path),
                                [f"sim_{i}" for i in range(start, start + group_size)],
                                directions, x, data.y[start:start + group_size]))
    for number in selection.get("rounds", []):
        if not isinstance(number, int) or number < 1:
            raise ValueError(f"Invalid feedback round: {number}")
        path = root / f"round{number}.h5"
        data = _load_file(path, detector)
        if len(data) % 5:
            raise ValueError(f"{path}: expected complete nominal + four-perturbation groups.")
        for start in range(0, len(data), 5):
            x = data.x[start:start + 5]
            _validate_direction_order(x, path, start // 5)
            groups.append(Group(f"round:{number}:{start // 5}", str(path),
                                [f"sim_{i}" for i in range(start, start + 5)],
                                DIRECTIONS, x, data.y[start:start + 5]))
    identities = [(g.source, tuple(g.keys)) for g in groups]
    if len(set(identities)) != len(identities):
        raise ValueError("Selection repeats a group.")
    if not groups:
        raise ValueError("Selection contains no groups.")
    return groups


def load_splits(config):
    splits, excluded = {}, {}
    seen_groups, seen_geometries = {}, {}
    for split, selection in config["splits"].items():
        groups = load_groups(config, selection)
        excluded[split] = [g.group_id for g in groups if not g.valid_fixed_q]
        if config.get("valid_fixed_q_only", True):
            groups = [g for g in groups if g.valid_fixed_q]
        if not groups:
            raise ValueError(f"No usable groups in {split}.")
        for group in groups:
            identity = (group.source, tuple(group.keys))
            geometry = tuple(group.x[0, :2].tolist())
            for key, seen in ((identity, seen_groups), (geometry, seen_geometries)):
                if key in seen and seen[key] != split:
                    raise ValueError(f"Design/group leakage between {seen[key]} and {split}: {group.group_id}")
                seen[key] = split
        splits[split] = groups
    return splits, excluded


def pack(groups):
    """Concatenate whole groups and return nominal/partner indices for pair loss."""
    nominal, perturbed, offset = [], [], 0
    for group in groups:
        nominal.extend([offset] * (len(group.x) - 1))
        perturbed.extend(range(offset + 1, offset + len(group.x)))
        offset += len(group.x)
    return (torch.cat([g.x for g in groups]), torch.cat([g.y for g in groups]),
            torch.tensor(nominal), torch.tensor(perturbed))
