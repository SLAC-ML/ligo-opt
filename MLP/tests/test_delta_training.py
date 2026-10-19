"""Check signed-difference supervision without changing the row training schedule."""

import contextlib
import io
import math
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import h5py
import networkx as nx
import numpy as np
import torch

from MLP.checkpoint import load_checkpoint, save_checkpoint
from MLP.data import PowerDataset
from MLP.finetune import run_finetune
from MLP.losses import PowerPredictionLoss
from MLP.training import train_base_power_predictor
from utils.training_groups import split_design_indices


DETECTOR = 'p_ITM_p2_i'
MODEL_CONFIG = {'hidden_sizes': [8, 8]}
SEED = 9302
DELTA_KEYS = ('supervision', 'delta_weight', 'delta_scale_floor_w')


def write_designs(path, amplitudes, *, nominal_power=200., offset=0., zero_groups=(), group_size=5):
    """Save fixed-q five-direction groups, legacy pairs, or nominal-only feedback."""
    if group_size not in (1, 2, 5):
        raise ValueError('The fixture supports nominal rows, legacy pairs, or five-row designs.')
    with h5py.File(path, 'w') as handle:
        if group_size != 2:  # Historical pairs have no group-size attribute.
            handle.attrs['design_group_size'] = group_size
        for group in reversed(range(len(amplitudes))):
            itm, etm = -1900. - offset - 30. * group, 2200. + offset + 40. * group
            geometry = ((itm, etm), (itm + 6., etm), (itm - 6., etm),
                        (itm, etm + 7.), (itm, etm - 7.))
            amplitude = amplitudes[group]
            center = nominal_power + 10. * group
            powers = (center, center + amplitude, center - .5 * amplitude,
                      center + 2. * amplitude, center - amplitude)
            if group_size == 2:
                geometry = ((itm, etm), (itm - 6., etm + 7.))
                powers = (center, center + amplitude)
            elif group_size == 1:
                geometry, powers = ((itm, etm),), (center,)
            q_real, q_imag = -1000. + group, 400. + group
            if group in zero_groups:
                powers = (0.,) * group_size
                q_real, q_imag = 0., 0.
            for direction in reversed(range(group_size)):
                graph = nx.DiGraph()
                graph.add_node('ITM.p1.i', Rc=geometry[direction][0],
                               q_re=q_real, q_im=q_imag, pd=1.)
                graph.add_node('ETM.p1.i', Rc=geometry[direction][1], pd=1.)
                graph.add_node('ITM.p2.i', pd=powers[direction])
                index = group_size * group + direction
                handle.create_dataset(f'sim_{index}', data=np.void(pickle.dumps(graph)))


def delta_rms(dataset, indices):
    """Compute the expected scale directly from saved powers in selected groups."""
    selected = set(indices)
    deltas = []
    powers = dataset.y.expm1().flatten()
    for start in range(0, len(dataset), 5):
        if start in selected:
            deltas.append(powers[start + 1:start + 5] - powers[start])
    return torch.cat(deltas).square().mean().sqrt().item()


class DeltaTrainingTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        self.cpu = patch.object(torch.cuda, 'is_available', return_value=False)
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = {
            'n_epochs': 2, 'batch_size': 4, 'save_loss_interval': 1,
            'print_interval': 100,
            'loss': {'supervision': 'nominal_delta', 'delta_weight': 1.25,
                     'delta_scale_floor_w': 1.},
        }

    def train(self, data_path, checkpoint_path, settings=None):
        with contextlib.redirect_stdout(io.StringIO()):
            train_base_power_predictor(
                data_path, checkpoint_path, hyperparams=settings or self.settings,
                detector=DETECTOR, model_config=MODEL_CONFIG, seed=SEED,
            )
        return load_checkpoint(checkpoint_path, device='cpu')

    def assert_power_only_checkpoint_monitor(self, model, dataset, metadata):
        criterion = PowerPredictionLoss(metadata['loss_config'])
        indices = metadata['validation_indices']
        batch_size = metadata['hyperparams']['batch_size']
        total = 0.
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                total += criterion(model(dataset.x[batch]), dataset.y[batch]).sum().item()
        self.assertAlmostEqual(metadata['best_validation_loss'], total / len(indices), places=12)

    def test_delta_scale_uses_only_initial_training_groups_and_applies_floor(self):
        train_indices, validation_indices = split_design_indices(30, seed=SEED, group_size=5)
        validation_groups = {index // 5 for index in validation_indices}
        for amplitude, expected_floor in ((2., False), (.01, True)):
            with self.subTest(amplitude=amplitude):
                amplitudes = [80. if group in validation_groups else amplitude for group in range(6)]
                data_path = self.root / f'base_{amplitude}.h5'
                write_designs(data_path, amplitudes)
                _, payload = self.train(data_path, self.root / f'model_{amplitude}.pt')
                metadata = payload['training_metadata']
                dataset = PowerDataset(data_path, DETECTOR)
                expected = max(1., delta_rms(dataset, train_indices))
                self.assertAlmostEqual(metadata['delta_scale_w'], expected, places=10)
                self.assertEqual(metadata['train_indices'], train_indices)
                self.assertEqual(metadata['validation_indices'], validation_indices)
                self.assertLess(metadata['delta_scale_w'], delta_rms(dataset, range(30)))
                if expected_floor:
                    self.assertEqual(metadata['delta_scale_w'], 1.)

    def test_two_rounds_freeze_scale_loss_split_and_normalization_without_filtering_rows(self):
        train_indices, _ = split_design_indices(30, seed=SEED, group_size=5)
        zero_group = train_indices[0] // 5
        data_path = self.root / 'base_30.h5'
        write_designs(data_path, [1., 2., 3., 4., 5., 6.], zero_groups=(zero_group,))
        write_designs(self.root / 'round1.h5', [300.], nominal_power=1000., offset=100.)
        write_designs(self.root / 'round2.h5', [600.], nominal_power=2000., offset=200.)
        # A future file must not affect either replay round or its scale.
        write_designs(self.root / 'round3.h5', [900.], nominal_power=3000., offset=300.)
        base_model, base_payload = self.train(data_path, self.root / 'base.pt')
        base = base_payload['training_metadata']
        dataset = PowerDataset(data_path, DETECTOR)
        self.assert_power_only_checkpoint_monitor(base_model, dataset, base)
        self.assertEqual(base['training_row_count'], len(train_indices))
        self.assertEqual(base['loss_config']['supervision'], 'nominal_delta')
        for number in (1, 2):
            with contextlib.redirect_stdout(io.StringIO()):
                prefix = run_finetune(
                    self.root, self.root / 'base.pt', str(self.root / 'checkpoint_round{}'),
                    current_round=number,
                    hyperparams={'n_epochs': 2, 'batch_size': 4, 'save_loss_interval': 1},
                )
            model, payload = load_checkpoint(prefix + '.pt', device='cpu')
            metadata = payload['training_metadata']
            self.assert_power_only_checkpoint_monitor(model, dataset, metadata)
            self.assertEqual(metadata['delta_scale_w'], base['delta_scale_w'])
            self.assertEqual(metadata['loss_config'], base['loss_config'])
            self.assertEqual(metadata['hyperparams']['loss'], base['loss_config'])
            self.assertEqual(metadata['train_indices'], base['train_indices'])
            self.assertEqual(metadata['validation_indices'], base['validation_indices'])
            self.assertEqual(metadata['training_row_count'], len(train_indices) + 5 * number)
            self.assertEqual(metadata['validation_row_count'], len(base['validation_indices']))
            self.assertEqual(metadata['newest_round_weight'], 6.)
            self.assertEqual(metadata['round_data_counts'], {str(i): 5 for i in range(1, number + 1)})
            torch.testing.assert_close(model.input_mean, base_model.input_mean, rtol=0, atol=0)
            torch.testing.assert_close(model.input_scale, base_model.input_scale, rtol=0, atol=0)
            self.assertTrue(math.isfinite(metadata['best_validation_loss']))

    def test_legacy_checkpoints_keep_row_power_supervision_without_delta_scale(self):
        data_path = self.root / 'base_20.h5'
        write_designs(data_path, [1., 2., 3., 4.])
        write_designs(self.root / 'round1.h5', [20.], offset=100.)
        settings = {**self.settings, 'loss': {'mode': 'center_focused'}}
        model, payload = self.train(data_path, self.root / 'current.pt', settings)
        for missing_loss_metadata in (False, True):
            with self.subTest(missing_loss_metadata=missing_loss_metadata):
                metadata = dict(payload['training_metadata'])
                metadata['hyperparams'] = dict(metadata['hyperparams'])
                old_config = {key: value for key, value in metadata['loss_config'].items()
                              if key not in DELTA_KEYS}
                metadata['loss_config'] = old_config
                metadata['hyperparams']['loss'] = dict(old_config)
                metadata.pop('delta_scale_w', None)
                if missing_loss_metadata:
                    metadata.pop('loss_config')
                    metadata['hyperparams'].pop('loss')
                legacy = self.root / f'legacy_{missing_loss_metadata}.pt'
                save_checkpoint(legacy, model, DETECTOR, metadata)
                with contextlib.redirect_stdout(io.StringIO()):
                    prefix = run_finetune(
                        self.root, legacy, str(self.root / f'legacy_{missing_loss_metadata}_round{{}}'),
                        current_round=1, hyperparams={'n_epochs': 1},
                    )
                _, restored = load_checkpoint(prefix + '.pt', device='cpu')
                saved = restored['training_metadata']
                self.assertEqual(saved['loss_config']['supervision'], 'row_power')
                self.assertEqual(saved['loss_config']['mode'],
                                 'log_smooth_l1' if missing_loss_metadata else 'center_focused')
                self.assertNotIn('delta_scale_w', saved)

    def test_legacy_pairs_replay_directional_and_nominal_only_feedback(self):
        data_path = self.root / 'base_8.h5'
        write_designs(data_path, [1., 2., 3., 4.], group_size=2)
        write_designs(self.root / 'round1.h5', [20.], offset=100.)
        write_designs(self.root / 'round2.h5', [0.], offset=200., group_size=1)
        base_model, payload = self.train(data_path, self.root / 'base.pt')
        base = payload['training_metadata']
        self.assertEqual(base['base_group_size'], 2)
        self.assertEqual(base['validation_mode'], 'held_out_pairs')
        for number, added_rows, newest_count in ((1, 5, 5), (2, 6, 1)):
            with contextlib.redirect_stdout(io.StringIO()):
                prefix = run_finetune(
                    self.root, self.root / 'base.pt', str(self.root / 'mixed_round{}'),
                    current_round=number, hyperparams={'n_epochs': 1, 'batch_size': 3},
                )
            model, payload = load_checkpoint(prefix + '.pt', device='cpu')
            metadata = payload['training_metadata']
            self.assertEqual(metadata['delta_scale_w'], base['delta_scale_w'])
            self.assertEqual(metadata['train_indices'], base['train_indices'])
            self.assertEqual(metadata['validation_indices'], base['validation_indices'])
            self.assertEqual(metadata['training_row_count'], len(base['train_indices']) + added_rows)
            self.assertEqual(metadata['newest_round_count'], newest_count)
            self.assertEqual(metadata['newest_round_weight'], 8. / newest_count)
            self.assertEqual(metadata['loss_config'], base['loss_config'])
            torch.testing.assert_close(model.input_mean, base_model.input_mean, rtol=0, atol=0)
            torch.testing.assert_close(model.input_scale, base_model.input_scale, rtol=0, atol=0)

    def test_recorded_training_loss_matches_nominal_and_weighted_signed_delta_objective(self):
        import MLP.training as training

        data_path = self.root / 'base_20.h5'
        write_designs(data_path, [1., 2., 3., 4.])
        write_designs(self.root / 'round1.h5', [10.], offset=100.)
        original_builder = training.build_power_predictor

        def constant_builder(config):
            model = original_builder(config)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            return model

        # Freeze optimizer updates so the epoch loss has one independently
        # calculable value, including the unchanged newest-feedback weights.
        settings = {**self.settings, 'n_epochs': 1, 'batch_size': 3}
        with patch.object(training, 'build_power_predictor', side_effect=constant_builder), \
                patch.object(torch.optim.AdamW, 'step', new=lambda *args, **kwargs: None):
            _, base_payload = self.train(data_path, self.root / 'base.pt', settings)
            with contextlib.redirect_stdout(io.StringIO()):
                prefix = run_finetune(
                    self.root, self.root / 'base.pt', str(self.root / 'objective_round{}'),
                    current_round=1, hyperparams={'n_epochs': 1, 'batch_size': 3},
                )
        model, payload = load_checkpoint(prefix + '.pt', device='cpu')
        metadata = payload['training_metadata']
        base_dataset = PowerDataset(data_path, DETECTOR)
        feedback = PowerDataset(self.root / 'round1.h5', DETECTOR)
        selected = base_payload['training_metadata']['train_indices']
        x = torch.cat((base_dataset.x[selected], feedback.x))
        target = torch.cat((base_dataset.y[selected], feedback.y))
        nominal = torch.arange(len(x)) // 5 * 5
        partners = torch.arange(len(x)) != nominal
        weights = torch.cat((torch.ones(len(selected), dtype=torch.float64),
                             torch.full((5,), metadata['newest_round_weight'], dtype=torch.float64)))
        with torch.no_grad():
            prediction = model(x).double()
            power_loss = PowerPredictionLoss(metadata['loss_config'])(prediction[nominal], target[nominal])
            true_power, predicted_power = target.expm1().flatten(), prediction.expm1().flatten()
            signed_error = ((predicted_power - predicted_power[nominal]) -
                            (true_power - true_power[nominal])) / metadata['delta_scale_w']
            absolute = signed_error.abs()
            huber = torch.where(absolute < 1., .5 * signed_error.square(), absolute - .5)
            expected_rows = power_loss + (5. / 4.) * metadata['loss_config']['delta_weight'] * partners * huber
            expected = (expected_rows * weights).mean().item()
        self.assertAlmostEqual(metadata['history'][0]['training_loss'], expected, places=12)

    def test_delta_supervision_preserves_shuffled_row_batches_and_optimizer_steps(self):
        import MLP.training as training

        data_path = self.root / 'base_30.h5'
        write_designs(data_path, [1., 2., 3., 4., 5., 6.])
        original_loader = training.DataLoader
        original_step = torch.optim.AdamW.step
        schedules, step_counts = {}, {}
        for supervision in ('row_power', 'nominal_delta'):
            batches, steps = [], []

            def make_loader(dataset, *args, **kwargs):
                loader = original_loader(dataset, *args, **kwargs)
                if not kwargs.get('shuffle', False):
                    return loader

                class RecordingLoader:
                    def __iter__(self):
                        for batch in loader:
                            features = next(value for value in batch
                                            if value.ndim == 2 and value.shape[1] == 4)
                            batches.append(features.detach().cpu().clone())
                            yield batch

                    def __len__(self):
                        return len(loader)

                return RecordingLoader()

            def record_step(optimizer, *args, **kwargs):
                steps.append(1)
                return original_step(optimizer, *args, **kwargs)

            settings = {**self.settings, 'batch_size': 3,
                        'loss': {**self.settings['loss'], 'supervision': supervision}}
            with patch.object(training, 'DataLoader', side_effect=make_loader), \
                    patch.object(torch.optim.AdamW, 'step', new=record_step):
                _, payload = self.train(data_path, self.root / f'{supervision}.pt', settings)
            schedules[supervision] = batches
            step_counts[supervision] = len(steps)
            expected = settings['n_epochs'] * math.ceil(
                len(payload['training_metadata']['train_indices']) / settings['batch_size'])
            self.assertEqual(len(steps), expected)
            self.assertEqual(len(batches), expected)
            self.assertTrue(all(len(batch) <= settings['batch_size'] for batch in batches))
        self.assertEqual(step_counts['nominal_delta'], step_counts['row_power'])
        self.assertEqual(len(schedules['nominal_delta']), len(schedules['row_power']))
        for actual, reference in zip(schedules['nominal_delta'], schedules['row_power']):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
