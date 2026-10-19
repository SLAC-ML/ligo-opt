"""Check that plateau training learns measured detail without a constant-power target."""

from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from MLP.checkpoint import load_checkpoint
from MLP.data import PowerDataset
from MLP.finetune import run_finetune
from MLP.losses import PowerPredictionLoss, resolve_loss_config, signed_delta_losses
from MLP.tests.test_workflow import DETECTOR, SMALL_MODEL, write_graphs
from MLP.training import train_base_power_predictor


def log_power(values):
    return torch.log1p(torch.tensor(values, dtype=torch.float64).reshape(-1, 1))


class CenterLossTests(unittest.TestCase):
    def test_small_plateau_errors_matter_while_other_powers_keep_global_loss(self):
        target = log_power([282.0955, 100., 0.])
        predicted = log_power([282.0945, 99.999, 0.001])
        focused = PowerPredictionLoss()(predicted, target)
        original = PowerPredictionLoss({'mode': 'log_smooth_l1'})(predicted, target)
        self.assertGreater(focused[0].item(), 100000 * original[0].item())
        torch.testing.assert_close(focused[1:], original[1:], rtol=0, atol=0)

    def test_minimum_is_the_true_label_not_the_reference_power(self):
        target = log_power([282.091, 282.0955])
        actual = target.clone().requires_grad_()
        loss = PowerPredictionLoss()(actual, target)
        torch.testing.assert_close(loss, torch.zeros(2, dtype=torch.float64))
        loss.sum().backward()
        torch.testing.assert_close(actual.grad, torch.zeros_like(actual))
        # Forcing every answer to the approximate plateau reference is penalized.
        reference = log_power([282.09, 282.09])
        self.assertTrue((PowerPredictionLoss()(reference, target) > 0).all())

    def test_large_errors_have_finite_bounded_gradients_and_cannot_escape_band(self):
        target = log_power([282.0955, 282.0955])
        predicted = torch.tensor([[1000.], [1.]], requires_grad=True)
        criterion = PowerPredictionLoss()
        loss = criterion(predicted, target)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertTrue((predicted.grad.abs() <= 11.).all())
        self.assertGreater(predicted.grad[0].item(), 0)
        self.assertLess(predicted.grad[1].item(), 0)
        # Weighting comes from the true power even when predictions are far away.
        self.assertTrue((criterion.center_weights(target) > 0.8).all())

    def test_old_loss_and_zero_center_weight_restore_original_formula(self):
        target = log_power([0., 8., 282.094])
        prediction = log_power([1., 9., 282.093])
        expected = F.smooth_l1_loss(prediction, target, reduction='none').mean(dim=1)
        for config in ({'mode': 'log_smooth_l1'}, {'center_weight': 0.}):
            torch.testing.assert_close(PowerPredictionLoss(config)(prediction, target), expected)

    def test_metrics_report_actual_power_error_and_absent_center_as_unavailable(self):
        criterion = PowerPredictionLoss()
        metrics = criterion.power_metrics(log_power([282.0945, 99.]), log_power([282.0955, 100.]))
        self.assertEqual(metrics['center_sample_count'], 1)
        self.assertAlmostEqual(metrics['center_power_mae'], 0.001, places=10)
        self.assertAlmostEqual(metrics['power_mae'], 0.5005, places=10)
        missing = criterion.power_metrics(log_power([10.]), log_power([11.]))
        self.assertIsNone(missing['center_power_mae'])
        self.assertEqual(missing['center_sample_count'], 0)

    def test_invalid_loss_settings_fail_early(self):
        for config in ({'center_width': 0}, {'center_error_scale': -1},
                       {'center_weight': float('nan')}, {'center_power': True},
                       {'mode': 'unknown'}, {'center_wdth': .01}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_loss_config(config)

    def test_loader_preserves_sub_float32_log_power_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'base_2.h5'
            write_graphs(path, 2)
            powers = [282.0955, 282.095501]
            with h5py.File(path, 'a') as handle:
                for i, power in enumerate(powers):
                    key = f'sim_{i}'
                    graph = pickle.loads(bytes(handle[key][()]))
                    graph.nodes['ITM.p2.i']['pd'] = power
                    del handle[key]
                    handle.create_dataset(key, data=np.void(pickle.dumps(graph)))
            dataset = PowerDataset(path, DETECTOR)
            self.assertEqual(dataset.y.dtype, torch.float64)
            np.testing.assert_allclose(torch.expm1(dataset.y).numpy().ravel(), powers, rtol=0, atol=1e-10)
            self.assertNotEqual(dataset.y[0].item(), dataset.y[1].item())
            self.assertEqual(dataset.y.float()[0].item(), dataset.y.float()[1].item())


class SignedDeltaLossTests(unittest.TestCase):
    def test_unreduced_losses_follow_signed_watt_errors_and_fixed_scale(self):
        loss = signed_delta_losses(
            log_power([13., 6., 11.]), log_power([12., 8., 10.]),
            log_power([9., 11., 10.]), log_power([10., 10., 10.]), 2.,
        )
        # Signed response errors are [2, -3, 1] W; normalized by 2 W.
        torch.testing.assert_close(loss, torch.tensor([0.5, 1., 0.125], dtype=torch.float64))
        self.assertEqual(loss.shape, (3,))
        self.assertEqual(loss.dtype, torch.float64)

    def test_reversed_response_signs_are_penalized(self):
        target, nominal = log_power([12., 8.]), log_power([10., 10.])
        exact = signed_delta_losses(target, target, nominal, nominal, 1.)
        wrong = signed_delta_losses(log_power([8., 12.]), target, nominal, nominal, 1.)
        torch.testing.assert_close(exact, torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(wrong, torch.full((2,), 3.5, dtype=torch.float64))

    def test_common_watt_offset_preserves_differences(self):
        loss = signed_delta_losses(
            log_power([17., 13., 15.]), log_power([12., 8., 10.]),
            log_power([15., 15., 15.]), log_power([10., 10., 10.]), 1.,
        )
        torch.testing.assert_close(loss, torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-25)

    def test_gradients_reach_partner_and_nominal_predictions(self):
        predicted = log_power([12.]).float().requires_grad_()
        nominal = log_power([10.]).float().requires_grad_()
        loss = signed_delta_losses(predicted, log_power([13.]), nominal, log_power([10.]), 2.)
        loss.sum().backward()
        self.assertEqual(predicted.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertTrue(torch.isfinite(nominal.grad).all())
        self.assertLess(predicted.grad.item(), 0.)
        self.assertGreater(nominal.grad.item(), 0.)

    def test_zero_and_tiny_true_differences_remain_resolvable(self):
        nominal = log_power([0., 282.0955])
        target = log_power([0., 282.095501])
        exact = signed_delta_losses(target, target, nominal, nominal, 1.)
        flat = signed_delta_losses(nominal, target, nominal, nominal, 1.)
        torch.testing.assert_close(exact, torch.zeros(2, dtype=torch.float64))
        self.assertEqual(flat[0].item(), 0.)
        self.assertGreater(flat[1].item(), 0.)

    def test_invalid_shapes_and_dtypes_fail_before_broadcasting(self):
        reference = log_power([10., 11.])
        for invalid in (reference.flatten(), reference[:1], reference.expand(2, 2),
                        reference.long(), reference.cdouble(), [10., 11.]):
            for position in range(4):
                arguments = [reference] * 4
                arguments[position] = invalid
                with self.subTest(position=position, invalid=str(invalid)), self.assertRaises(ValueError):
                    signed_delta_losses(*arguments, 1.)

    def test_invalid_scale_is_rejected(self):
        reference = log_power([10.])
        for invalid in (0., -1., True, float('nan'), float('inf'), '1'):
            with self.subTest(scale=invalid), self.assertRaises(ValueError):
                signed_delta_losses(reference, reference, reference, reference, invalid)

    def test_nonfinite_inputs_conversion_and_scaled_errors_fail_clearly(self):
        reference = log_power([10.])
        for invalid in (float('nan'), float('inf'), -float('inf'), 1000.):
            for position in range(4):
                arguments = [reference] * 4
                arguments[position] = torch.tensor([[invalid]], dtype=torch.float64)
                with self.subTest(position=position, value=invalid), self.assertRaises(FloatingPointError):
                    signed_delta_losses(*arguments, 1.)
        with self.assertRaises(FloatingPointError):
            signed_delta_losses(log_power([12.]), reference, reference, reference, 1e-320)


class DeltaConfigurationTests(unittest.TestCase):
    def test_old_config_defaults_to_row_supervision(self):
        config = resolve_loss_config({'mode': 'log_smooth_l1'})
        self.assertEqual(config['supervision'], 'row_power')
        self.assertEqual(config['delta_weight'], 1.)
        self.assertEqual(config['delta_scale_floor_w'], 1.)
        config['delta_weight'] = 7.
        self.assertEqual(resolve_loss_config()['delta_weight'], 1.)

    def test_delta_config_leaves_base_row_criterion_unchanged(self):
        prediction, target = log_power([282.0945, 10.]), log_power([282.0955, 11.])
        expected = PowerPredictionLoss()(prediction, target)
        actual = PowerPredictionLoss({
            'supervision': 'nominal_delta', 'delta_weight': 0., 'delta_scale_floor_w': .01,
        })(prediction, target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_delta_settings_fail_early(self):
        for invalid in ({'supervision': 'paired'}, {'supervision': True},
                        {'delta_weight': -1.}, {'delta_weight': True},
                        {'delta_weight': float('nan')}, {'delta_weight': float('inf')},
                        {'delta_scale_floor_w': 0.}, {'delta_scale_floor_w': -1.},
                        {'delta_scale_floor_w': True}, {'delta_scale_floor_w': '1'},
                        {'delta_scale_floor_w': float('nan')}, {'delta_scale_floor_w': float('inf')}):
            with self.subTest(config=invalid), self.assertRaises(ValueError):
                resolve_loss_config(invalid)


class LossPersistenceTests(unittest.TestCase):
    def test_finetuning_inherits_the_loss_and_saves_center_validation_metrics(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, 'is_available', return_value=False):
            root = Path(directory)
            write_graphs(root / 'base_12.h5', 12)
            write_graphs(root / 'round1.h5', 5)
            write_graphs(root / 'round2.h5', 5)
            custom_loss = {'center_power': 20., 'center_width': 20., 'center_weight': 3.}
            train_base_power_predictor(
                root / 'base_12.h5', root / 'base.pt',
                hyperparams={'n_epochs': 1, 'loss': custom_loss}, model_config=SMALL_MODEL,
            )
            _, base = load_checkpoint(root / 'base.pt', device='cpu')
            prefix = run_finetune(
                root, root / 'base.pt', str(root / 'round{}'), current_round=1,
                hyperparams={'n_epochs': 1},
            )
            _, first = load_checkpoint(prefix + '.pt', device='cpu')
            base_config = base['training_metadata']['loss_config']
            first_meta = first['training_metadata']
            self.assertEqual(first_meta['loss_config'], base_config)
            self.assertEqual(first_meta['hyperparams']['loss'], base_config)
            self.assertGreater(first_meta['best_power_metrics']['center_sample_count'], 0)
            self.assertIsNotNone(first_meta['best_power_metrics']['center_power_mae'])
            self.assertIn('center_power_mae', first_meta['history'][0])
            prefix = run_finetune(
                root, root / 'base.pt', str(root / 'round{}'), current_round=2,
                hyperparams={'n_epochs': 1, 'loss': {'mode': 'log_smooth_l1'}},
            )
            _, second = load_checkpoint(prefix + '.pt', device='cpu')
            self.assertEqual(second['training_metadata']['loss_config']['mode'], 'log_smooth_l1')
            self.assertEqual(second['training_metadata']['loss_config']['center_power'], 20.)


if __name__ == '__main__':
    unittest.main()
