"""Check saved-data conclusions against production costs and complete design splits."""

import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from utils import calc
from analysis.surrogate_diagnostics.compare_runs import (
    _base_split_groups, _cost_components, _evaluate,
)
from analysis.surrogate_diagnostics.dataset import DIRECTIONS, Group, fingerprints
from analysis.surrogate_diagnostics.training_audit import _selection_ids, _symmetric_summary
from analysis.surrogate_diagnostics.tests.test_diagnostics import write_base


class FixedPowerModel(torch.nn.Module):
    def __init__(self, powers):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("targets", torch.tensor(powers, dtype=torch.float64).log1p().reshape(-1, 1))

    def forward(self, x):
        return self.targets


def response_group(powers):
    x = torch.tensor([
        [-1900., 2200., -1000., 400.],
        [-1894., 2200., -1000., 400.],
        [-1906., 2200., -1000., 400.],
        [-1900., 2207., -1000., 400.],
        [-1900., 2193., -1000., 400.],
    ])
    y = torch.tensor(powers, dtype=torch.float64).log1p().reshape(-1, 1)
    return Group("round:1:0", "fixture.h5", [f"sim_{i}" for i in range(5)],
                 DIRECTIONS, x, y)


class Run3CostTests(unittest.TestCase):
    def test_saved_cost_reconstruction_matches_production_objective(self):
        powers = dict(zip(DIRECTIONS, [20., 22., 17., 24., 13.]))
        cases = [
            (powers, (2., 4.)),
            (dict.fromkeys(DIRECTIONS, 0.), (2., 4.)),
            (powers, (0., 4.)),
            ({**powers, "ETM_plus": -1.}, (2., 4.)),
            ({**powers, "ITM_minus": np.nan}, (2., 4.)),
        ]
        for case, delta in cases:
            for mode in ("design_power", "perturbation", "full"):
                with self.subTest(mode=mode, delta=delta, powers=case):
                    # Include an unrelated detector so this also follows production's
                    # detector selection rather than assuming a scalar power array.
                    names = [["p_unrelated", "p_ITM_p2_i"]] * 3
                    itm = [[999., case[key]] for key in ("ITM_plus", "nominal", "ITM_minus")]
                    etm = [[999., case[key]] for key in ("ETM_plus", "nominal", "ETM_minus")]
                    with mock.patch.multiple(calc, COST_MODE=mode, FULL_POWER_WEIGHT=0.2,
                                             FULL_PERTURBATION_WEIGHT=0.8, RECORD_FULL_COST=False), \
                            mock.patch.object(calc, "_record_full_cost") as record:
                        expected = calc.calc_cost(None, itm, etm, names, names, delta, "p_ITM_p2_i")
                        record.assert_not_called()
                    actual = _cost_components(case, delta, mode, 0.2, 0.8)["total_cost"]
                    if np.isfinite(expected):
                        self.assertAlmostEqual(actual, expected, places=12)
                    else:
                        self.assertIsNone(actual)

    def test_exact_absolute_cost_can_hide_all_wrong_response_signs(self):
        group = response_group([20., 22., 17., 24., 13.])
        # Reflect every perturbation about the same nominal power. Absolute slopes
        # remain exact while the physical response directions are all reversed.
        model = FixedPowerModel([20., 18., 23., 16., 27.])
        rows, metrics, costs = _evaluate(
            model, [group], "feedback", 0.1,
            {"cost_mode": "full", "power_weight": 0.2, "perturbation_weight": 0.8},
            recorded_deltas={group.group_id: (2., 4.)},
            training_exposure="held out from weight updates", evaluation_role="same saved geometries")
        self.assertAlmostEqual(costs[0]["true_perturbation_cost"], 5.25)
        self.assertAlmostEqual(metrics["perturbation_cost_mae"], 0., places=12)
        self.assertAlmostEqual(metrics["full_cost_mae"], 0., places=12)
        self.assertAlmostEqual(metrics["delta_mae"], 8.)
        self.assertEqual(metrics["sign_count"], 4)
        self.assertEqual(metrics["sign_correct"], 0)
        self.assertEqual(metrics["sign_accuracy"], 0.)
        self.assertTrue(all(row["training_exposure"] == "held out from weight updates" for row in rows))
        self.assertTrue(all(row["evaluation_role"] == "same saved geometries" for row in rows))

    def test_zero_nominal_preserves_response_metrics_and_null_costs(self):
        group = response_group([0., 2., 0., 4., 0.])
        rows, metrics, costs = _evaluate(
            FixedPowerModel([1., 3., 1., 5., 1.]), [group], "feedback", 0.1,
            {"cost_mode": "full", "power_weight": 0.5, "perturbation_weight": 0.5})
        self.assertTrue(group.valid_fixed_q)
        self.assertEqual(metrics["groups"], 1)
        self.assertEqual(metrics["rows"], 5)
        self.assertEqual(metrics["sign_count"], 2)
        self.assertEqual(metrics["sign_correct"], 2)
        self.assertEqual(metrics["valid_cost_groups"], 0)
        self.assertEqual(metrics["invalid_cost_groups"], 1)
        self.assertIsNone(costs[0]["true_perturbation_cost"])
        self.assertIsNone(costs[0]["true_total_cost"])
        self.assertIsNotNone(costs[0]["predicted_total_cost"])
        self.assertIsNone(metrics["perturbation_cost_mae"])
        self.assertIsNone(metrics["full_cost_mae"])
        json.dumps({"rows": rows, "metrics": metrics, "costs": costs}, allow_nan=False)

    def test_symmetric_summary_distinguishes_curvature_from_a_linear_tilt(self):
        group = response_group([20., 18., 18., 16., 16.])
        summary = _symmetric_summary(FixedPowerModel([20., 22., 18., 24., 16.]), [group])
        for axis, curvature, tilt in (("ITM", -2., 2.), ("ETM", -4., 4.)):
            with self.subTest(axis=axis):
                self.assertEqual(summary[axis]["true_both_decrease_count"], 1)
                self.assertEqual(summary[axis]["predicted_both_decrease_count"], 0)
                self.assertEqual(summary[axis]["predicted_opposite_sign_count"], 1)
                self.assertAlmostEqual(summary[axis]["true_even_mean"], curvature)
                self.assertAlmostEqual(summary[axis]["predicted_even_mean"], 0., places=12)
                self.assertAlmostEqual(summary[axis]["true_mean_abs_odd"], 0., places=12)
                self.assertAlmostEqual(summary[axis]["predicted_mean_abs_odd"], tilt)


class Run3SplitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "base_10.h5"
        write_base(self.path)
        self.run = {"root": self.root, "metadata": {"files": {"base_data": str(self.path)}}}
        self.payload = {"detector": "p_ITM_p2_i", "training_metadata": {
            "base_group_size": 5, "base_data_count": 10, "train_indices": list(range(5)),
            "validation_indices": list(range(5, 10)),
            "base_data_sha256": fingerprints([self.path])[str(self.path)],
        }}

    def tearDown(self):
        self.directory.cleanup()

    def test_production_split_keeps_invalid_q_rows_and_overlapping_valid_subset(self):
        with h5py.File(self.path, "a") as handle:
            graph = pickle.loads(bytes(handle["sim_6"][()]))
            graph.nodes["ITM.p1.i"].update(q_re=0., q_im=0.)
            del handle["sim_6"]
            handle.create_dataset("sim_6", data=np.void(pickle.dumps(graph)))
        self.payload["training_metadata"]["base_data_sha256"] = fingerprints([self.path])[str(self.path)]
        _, splits = _base_split_groups(self.run, self.payload)
        self.assertEqual([group.group_id for group in splits["training"]], ["base:0"])
        self.assertEqual([group.group_id for group in splits["validation"]], ["base:1"])
        self.assertEqual(sum(len(group.x) for group in splits["validation"]), 5)
        self.assertFalse(splits["validation"][0].valid_fixed_q)
        self.assertEqual(splits["validation_valid_fixed_q"], [])
        self.assertIs(splits["training_valid_fixed_q"][0], splits["training"][0])

    def test_partial_design_split_rejected_by_both_audits(self):
        partial = copy.deepcopy(self.payload)
        partial["training_metadata"]["train_indices"] = [0, 1, 2, 3]
        partial["training_metadata"]["validation_indices"] = list(range(4, 10))
        with self.assertRaisesRegex(ValueError, "complete distinct initial groups"):
            _base_split_groups(self.run, partial)
        with self.assertRaisesRegex(ValueError, "separates a design group"):
            _selection_ids(partial["training_metadata"], 5, "train_indices")

    def test_duplicate_rows_cannot_misrepresent_training_exposure(self):
        duplicate = copy.deepcopy(self.payload)
        duplicate["training_metadata"]["train_indices"].append(0)
        with self.assertRaisesRegex(ValueError, "complete distinct initial groups"):
            _base_split_groups(self.run, duplicate)

    def test_overlapping_or_missing_designs_cannot_be_called_held_out(self):
        overlap = copy.deepcopy(self.payload)
        overlap["training_metadata"]["validation_indices"] = list(range(10))
        with self.assertRaisesRegex(ValueError, "splits overlap"):
            _base_split_groups(self.run, overlap)
        missing = copy.deepcopy(self.payload)
        missing["training_metadata"]["validation_indices"] = []
        with self.assertRaisesRegex(ValueError, "do not cover all base rows"):
            _base_split_groups(self.run, missing)

    def test_equal_row_count_does_not_hide_changed_training_data(self):
        with h5py.File(self.path, "a") as handle:
            graph = pickle.loads(bytes(handle["sim_4"][()]))
            graph.nodes["ITM.p2.i"]["pd"] = 123.
            del handle["sim_4"]
            handle.create_dataset("sim_4", data=np.void(pickle.dumps(graph)))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            _base_split_groups(self.run, self.payload)

    def test_checkpoint_group_layout_must_match_saved_data(self):
        legacy_layout = copy.deepcopy(self.payload)
        legacy_layout["training_metadata"]["base_group_size"] = 2
        with self.assertRaisesRegex(ValueError, "disagree on group size"):
            _base_split_groups(self.run, legacy_layout)


if __name__ == "__main__":
    unittest.main()
