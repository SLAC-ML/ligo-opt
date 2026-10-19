"""Check objective reconstruction and interpretation of signed-response metrics."""

import unittest

import numpy as np

from analysis.surrogate_diagnostics.compare_runs import (
    _assert_saved_cost, _cost_components, _online, _response_metrics,
)


class RunComparisonTests(unittest.TestCase):
    def setUp(self):
        self.powers = {"nominal": 20., "ITM_plus": 22., "ITM_minus": 17.,
                       "ETM_plus": 24., "ETM_minus": 13.}

    def test_distinct_roc_steps_and_saved_weights(self):
        costs = _cost_components(self.powers, (2., 4.), power_weight=0.2,
                                 perturbation_weight=0.8)
        self.assertAlmostEqual(costs["power_cost"], 0.05)
        self.assertAlmostEqual(costs["perturbation_cost"], 5.25)
        self.assertAlmostEqual(costs["total_cost"], 4.21)
        self.assertEqual(_cost_components(self.powers, (2., 4.), "design_power")["total_cost"], 0.05)
        self.assertEqual(_cost_components(self.powers, (2., 4.), "perturbation")["total_cost"], 5.25)

    def test_zero_nominal_is_invalid_even_when_all_differences_are_zero(self):
        zero = dict.fromkeys(self.powers, 0.)
        costs = _cost_components(zero, (2., 4.))
        self.assertIsNone(costs["total_cost"])
        self.assertIsNone(costs["perturbation_cost"])
        _assert_saved_cost({"cost": np.inf}, costs)

    def test_zero_step_and_negative_perturbation_invalidate_response_cost(self):
        self.assertIsNone(_cost_components(self.powers, (0., 4.))["total_cost"])
        invalid = {**self.powers, "ETM_plus": -1.}
        self.assertIsNone(_cost_components(invalid, (2., 4.))["total_cost"])
        self.assertEqual(_cost_components(invalid, (2., 4.), "design_power")["total_cost"], 0.05)

    def test_misrecorded_objective_is_rejected(self):
        costs = _cost_components(self.powers, (2., 4.))
        _assert_saved_cost({"cost": costs["total_cost"] + 1e-8}, costs)
        with self.assertRaisesRegex(ValueError, "differs from saved cost"):
            _assert_saved_cost({"cost": costs["total_cost"] + 0.1}, costs)

    def test_sign_threshold_uses_true_response_and_zero_prediction_is_wrong(self):
        rows = [
            {"delta_power": 0.1, "predicted_delta_power": -0.1, "delta_error": -0.2},
            {"delta_power": -0.2, "predicted_delta_power": 0., "delta_error": 0.2},
            {"delta_power": 0.3, "predicted_delta_power": 0.02, "delta_error": -0.28},
        ]
        metrics = _response_metrics(rows, 0.1)
        self.assertEqual(metrics["sign_count"], 2)
        self.assertEqual(metrics["sign_correct"], 1)
        self.assertEqual(metrics["sign_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["delta_mae"], 0.68 / 3)

    def test_online_powers_must_match_selected_geometry(self):
        run = {"number": 3, "metadata": {"settings": {"training": {"base": {}}}},
               "result": {"data": [[-1434., 2500.]],
               "evaluations_surrogate": [{"params": np.array([-1435., 2500.])}],
               "evaluations_finesse": [{"params": np.array([-1434., 2500.])}]}}
        with self.assertRaisesRegex(ValueError, "misassociated powers"):
            _online(run, 0.1)


if __name__ == "__main__":
    unittest.main()
