"""Check design references and response metrics against independent examples."""

import unittest

import torch

from MLP.groups import design_row_metadata, initial_delta_scale, response_metrics


def log_power(values):
    return torch.tensor(values, dtype=torch.float64).reshape(-1, 1).log1p()


def directional_inputs(count=1):
    groups = []
    for number in range(count):
        itm, etm = -100. - 10. * number, 200. + 20. * number
        groups.extend(((itm, etm, -50., 20.), (itm + 2., etm, -50., 20.),
                       (itm - 2., etm, -50., 20.), (itm, etm + 4., -50., 20.),
                       (itm, etm - 4., -50., 20.)))
    return torch.tensor(groups, dtype=torch.float32)


class DesignRowMetadataTests(unittest.TestCase):
    def test_reordered_rows_reference_local_nominals(self):
        inputs = directional_inputs(2)
        selected = [9, 5, 8, 6, 7, 4, 0, 3, 1, 2]
        nominal, sizes = design_row_metadata(inputs, 5, selected, validate_directions=True)
        torch.testing.assert_close(nominal, torch.tensor([1] * 5 + [6] * 5))
        torch.testing.assert_close(sizes, torch.full((10,), 5))
        # References address the selected table, rather than the source table.
        selected_inputs = inputs[selected]
        torch.testing.assert_close(selected_inputs[nominal[:5]], inputs[5].expand(5, 4))
        torch.testing.assert_close(selected_inputs[nominal[5:]], inputs[0].expand(5, 4))

    def test_nominal_only_and_legacy_pairs_are_supported(self):
        inputs = directional_inputs()
        nominal, sizes = design_row_metadata(inputs[:3], 1, [2, 0])
        torch.testing.assert_close(nominal, torch.tensor([0, 1]))
        torch.testing.assert_close(sizes, torch.ones(2, dtype=torch.long))
        nominal, sizes = design_row_metadata(inputs[:4], 2, [2, 3, 1, 0])
        torch.testing.assert_close(nominal, torch.tensor([0, 0, 3, 3]))
        torch.testing.assert_close(sizes, torch.full((4,), 2))

    def test_incomplete_duplicate_and_out_of_range_groups_are_rejected(self):
        inputs = directional_inputs(2)
        for selected in ([0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 1, 2, 3],
                         [-1, 0, 1, 2, 3], [10], []):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                design_row_metadata(inputs, 5, selected)
        for size in (0, -1, True, 2.5, 3):
            with self.subTest(size=size), self.assertRaises(ValueError):
                design_row_metadata(inputs, size)

    def test_direction_validation_checks_both_changed_and_fixed_axes(self):
        inputs = directional_inputs()
        bad_order = inputs[[0, 2, 1, 3, 4]]
        bad_axis = inputs.clone()
        bad_axis[1, 1] += 1.
        for invalid in (bad_order, bad_axis):
            with self.subTest(inputs=invalid.tolist()), self.assertRaisesRegex(ValueError, 'perturbation order'):
                design_row_metadata(invalid, 5, validate_directions=True)
        # Zero-q samples are retained; the reference map does not filter labels.
        inputs[:, 2:] = 0.
        nominal, _ = design_row_metadata(inputs, 5, validate_directions=True)
        torch.testing.assert_close(nominal, torch.zeros(5, dtype=torch.long))


class InitialDeltaScaleTests(unittest.TestCase):
    def test_scale_uses_only_selected_training_rows_and_their_nominal_map(self):
        inputs = directional_inputs(2)
        targets = log_power([10., 110., 10., 10., 10., 20., 23., 16., 20., 20.])
        selected = [9, 7, 5, 8, 6]
        nominal, _ = design_row_metadata(inputs, 5, selected)
        # Training deltas are [0, -4, 0, 3]: sqrt((16 + 9) / 4) = 2.5 W.
        self.assertAlmostEqual(initial_delta_scale(targets[selected], nominal, 1.), 2.5)
        self.assertEqual(initial_delta_scale(targets[selected], nominal, 3.), 3.)
        full_nominal, _ = design_row_metadata(inputs, 5)
        self.assertGreater(initial_delta_scale(targets, full_nominal, 1.), 30.)

    def test_zero_labels_use_floor_and_initial_nominal_only_data_is_rejected(self):
        nominal = torch.zeros(5, dtype=torch.long)
        self.assertEqual(initial_delta_scale(log_power([0.] * 5), nominal, .25), .25)
        with self.assertRaisesRegex(ValueError, 'initial training perturbations'):
            initial_delta_scale(log_power([10.]), torch.tensor([0]), 1.)
        for invalid in (0., -1., True, float('nan'), float('inf')):
            with self.subTest(floor=invalid), self.assertRaises(ValueError):
                initial_delta_scale(log_power([0.] * 5), nominal, invalid)


class ResponseMetricsTests(unittest.TestCase):
    def test_signed_errors_signs_directions_and_cost_match_hand_calculations(self):
        inputs = directional_inputs()
        nominal, sizes = design_row_metadata(inputs, 5)
        metrics = response_metrics(
            log_power([11., 14., 12., 10., 8.]),
            log_power([10., 12., 8., 14., 6.]), nominal, sizes, inputs,
        )
        # True deltas [2, -2, 4, -4], predictions [3, 1, -1, -3].
        self.assertAlmostEqual(metrics['nominal_power_mae'], 1.)
        self.assertAlmostEqual(metrics['delta_mae'], 2.5)
        self.assertAlmostEqual(metrics['delta_max_abs_error'], 5.)
        self.assertEqual(metrics['delta_sign_count'], 4)
        self.assertEqual(metrics['delta_sign_threshold_w'], .1)
        self.assertEqual(metrics['delta_sign_accuracy'], .5)
        # ROC steps [2, 2, 4, 4] give true cost 4 and predicted cost 3 W/m.
        self.assertAlmostEqual(metrics['perturbation_cost_mae'], 1.)
        for direction, error, sign in (
            ('ITM_plus', 1., 1.), ('ITM_minus', 3., 0.),
            ('ETM_plus', 5., 0.), ('ETM_minus', 1., 1.),
        ):
            actual = metrics['delta_by_direction'][direction]
            self.assertEqual(actual['count'], 1)
            self.assertAlmostEqual(actual['delta_mae'], error)
            self.assertEqual(actual['sign_count'], 1)
            self.assertEqual(actual['sign_accuracy'], sign)

    def test_exact_labels_have_zero_error_even_after_row_reordering(self):
        inputs = directional_inputs()
        selected = [4, 2, 0, 3, 1]
        nominal, sizes = design_row_metadata(inputs, 5, selected)
        targets = log_power([10., 12., 8., 14., 6.])[selected]
        metrics = response_metrics(targets, targets, nominal, sizes, inputs[selected])
        self.assertEqual(metrics['nominal_power_mae'], 0.)
        self.assertEqual(metrics['delta_mae'], 0.)
        self.assertEqual(metrics['delta_max_abs_error'], 0.)
        self.assertEqual(metrics['perturbation_cost_mae'], 0.)
        self.assertEqual(metrics['delta_sign_accuracy'], 1.)

    def test_sign_threshold_legacy_pairs_and_nominal_only_metrics(self):
        inputs = directional_inputs()[:4]
        nominal, sizes = design_row_metadata(inputs, 2)
        metrics = response_metrics(
            log_power([10., 10.04, 20., 20.]), log_power([10., 10.05, 20., 19.8]),
            nominal, sizes, inputs,
        )
        self.assertAlmostEqual(metrics['delta_mae'], .105)
        self.assertEqual(metrics['delta_sign_count'], 1)
        self.assertEqual(metrics['delta_sign_accuracy'], 0.)
        self.assertIsNone(metrics['perturbation_cost_mae'])
        self.assertTrue(all(value['count'] == 0 for value in metrics['delta_by_direction'].values()))
        nominal, sizes = design_row_metadata(inputs[:1], 1)
        metrics = response_metrics(log_power([11.]), log_power([10.]), nominal, sizes, inputs[:1])
        self.assertAlmostEqual(metrics['nominal_power_mae'], 1.)
        for name in ('delta_mae', 'delta_max_abs_error', 'delta_sign_accuracy', 'perturbation_cost_mae'):
            self.assertIsNone(metrics[name])
        self.assertEqual(metrics['delta_sign_count'], 0)


if __name__ == '__main__':
    unittest.main()
