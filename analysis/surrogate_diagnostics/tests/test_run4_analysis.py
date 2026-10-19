"""Regression checks for comparisons spanning older and nominal/delta runs."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from analysis.surrogate_diagnostics.dataset import DIRECTIONS, Group, write_csv
from analysis.surrogate_diagnostics.training_audit import _initial_delta_scale_summary


class MixedHistoryExportTests(unittest.TestCase):
    def test_new_run_metrics_survive_mixed_training_history_export(self):
        history = [
            {"run": 3, "epoch": 5, "power_mae": 7.1},
            {"run": 4, "epoch": 5, "power_mae": 6.0, "delta_mae": 0.3,
             "training_delta_loss": 0.1},
            {"run": 4, "epoch": 10, "power_mae": 5.0, "delta_mae": 0.2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            write_csv(path, iter(history))
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames,
                                 ["run", "epoch", "power_mae", "delta_mae",
                                  "training_delta_loss"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["delta_mae"], "")
        self.assertEqual(rows[1]["delta_mae"], "0.3")
        self.assertEqual(rows[1]["training_delta_loss"], "0.1")
        self.assertEqual(rows[2]["training_delta_loss"], "")

    def test_delta_scale_audit_uses_training_groups_and_exposes_invalid_share(self):
        def group(name, powers, valid=True):
            inputs = torch.ones((5, 4))
            if not valid:
                inputs[:, 3] = 0
            return Group(name, "base.h5", list(range(5)), DIRECTIONS,
                         inputs, torch.tensor(powers, dtype=torch.float64).log1p()[:, None])
        run = {
            "groups": [group("valid", [10, 13, 6, 10, 10]),
                       group("invalid", [10, 10, 10, 10, 22], False),
                       group("heldout", [0, 1000, 1000, 1000, 1000])],
            "train_ids": [0, 1],
            "payload": {"training_metadata": {
                "delta_scale_w": np.sqrt(169 / 8),
                "loss_config": {"delta_scale_floor_w": 1.0}}},
        }
        summary = _initial_delta_scale_summary(run)
        self.assertEqual(summary["training_partner_rows"], 8)
        self.assertAlmostEqual(summary["training_delta_rms_w"], np.sqrt(169 / 8))
        self.assertAlmostEqual(summary["by_fixed_q_validity"]["invalid"]["squared_delta_share"],
                               144 / 169)
        self.assertAlmostEqual(summary["by_fixed_q_validity"]["valid"]["delta_rms_w"], 2.5)


if __name__ == "__main__":
    unittest.main()
