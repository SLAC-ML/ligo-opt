"""Check initial simulations and complete-design splits for both surrogates."""

import csv
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

import fabry_perot
import perturbation
import train_power_predictor as gnn_training
from MLP.training import split_pair_indices
from MLP.tests.test_workflow import write_graphs
from utils.training_groups import read_design_group_size


class InitialDataGenerationTests(unittest.TestCase):
    def check_generated_groups(self, samples, percentage, *, nominal_q=4000 + 2000j):
        built_models = []

        def build(itm, etm, _base, nominal_q_value=None):
            model = SimpleNamespace(itm=float(itm), etm=float(etm), fixed_q=nominal_q_value)
            built_models.append(model)
            return model

        def simulate(model):
            q_value = nominal_q if model.fixed_q is None else model.fixed_q
            return ['p_test'], [1.0], ['q_ITM_p1_i'], [q_value]

        def graph(model, *_args):
            return vars(model).copy()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'base.h5'
            with (
                patch.object(fabry_perot, 'optimized_lhs_roc_samples', return_value=samples),
                patch.object(fabry_perot, 'kat_manipulation', side_effect=build),
                patch.object(fabry_perot, 'finesse_sim', side_effect=simulate),
                patch.object(fabry_perot, 'model_to_nx_port', side_effect=graph),
                patch.object(perturbation, 'roc_perturbation_percent', percentage),
            ):
                self.assertEqual(fabry_perot.generate_initial_data(
                    output, num_design_points=len(samples),
                    itm_bounds=(-4134, -1434), etm_bounds=(1745, 4245),
                ), output)
                expected_geometries = []
                for itm, etm in samples:
                    (ip, im), (ep, em), _ = perturbation.calc_perturbed_params(itm, etm)
                    expected_geometries.extend([(itm, etm), (ip, etm), (im, etm),
                                                (itm, ep), (itm, em)])
            with output.with_name('initial_samples.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            types = ('design', 'ITM_plus', 'ITM_minus', 'ETM_plus', 'ETM_minus')
            self.assertEqual(len(rows), 5 * len(samples))
            self.assertEqual(read_design_group_size(output), 5)
            with h5py.File(output, 'r') as handle:
                self.assertEqual(set(handle), {f'sim_{i}' for i in range(len(rows))})
                for index, row in enumerate(rows):
                    self.assertEqual(row['h5_key'], f'sim_{index}')
                    self.assertEqual(int(row['design_id']), index // 5)
                    self.assertEqual(row['sample_type'], types[index % 5])
                    saved = pickle.loads(handle[row['h5_key']][()])
                    self.assertEqual((saved['itm'], saved['etm']), expected_geometries[index])
                    self.assertEqual(float(row['itm_roc_m']), saved['itm'])
                    self.assertEqual(float(row['etm_roc_m']), saved['etm'])
            self.assertEqual(len(built_models), len(expected_geometries))
            for index, model in enumerate(built_models):
                self.assertEqual((model.itm, model.etm), expected_geometries[index])
                expected_q = None if index % 5 == 0 or nominal_q == 0 else nominal_q
                self.assertEqual(model.fixed_q, expected_q)
                if index % 5:
                    itm, etm = samples[index // 5]
                    self.assertEqual(int(model.itm != itm) + int(model.etm != etm), 1)
            # Existing artifacts must remain intact when overwrite was not requested.
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                fabry_perot.generate_initial_data(output, num_design_points=2)
            self.assertEqual(output.read_bytes(), original)

    def test_each_design_writes_four_single_mirror_fixed_q_perturbations(self):
        samples = np.array([[-4000., 2000.], [-2000., 4000.]])
        for percentage in (0.0033, 0.01):
            with self.subTest(percentage=percentage):
                self.check_generated_groups(samples, percentage)

    def test_boundary_designs_keep_both_cost_steps_without_clipping(self):
        self.check_generated_groups(np.array([[-4134., 1745.], [-1434., 4245.]]), 0.0033)

    def test_zero_q_designs_retain_all_five_samples(self):
        self.check_generated_groups(np.array([[-4000., 2000.]]), 0.0033, nominal_q=0j)


class DesignGroupSplitTests(unittest.TestCase):
    def test_both_trainers_keep_five_sample_designs_together(self):
        # 500 can be divided into pairs too; the explicit group size avoids that ambiguity.
        gnn_train, gnn_val = gnn_training.split_graph_pairs(list(range(500)), group_size=5)
        mlp_train, mlp_val = split_pair_indices(500, group_size=5)
        self.assertEqual(gnn_train.indices, mlp_train)
        self.assertEqual(gnn_val.indices, mlp_val)
        self.assertEqual(split_pair_indices(500, group_size=5), (mlp_train, mlp_val))
        train, validation = set(mlp_train), set(mlp_val)
        self.assertEqual((len(train), len(validation)), (400, 100))
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(range(500)))
        for start in range(0, 500, 5):
            group = set(range(start, start + 5))
            self.assertTrue(group <= train or group <= validation)

    def test_legacy_datasets_and_splits_still_use_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.h5'
            write_graphs(path, 10)
            self.assertEqual(read_design_group_size(path), 2)
        train, validation = split_pair_indices(10)
        self.assertEqual((len(train), len(validation)), (8, 2))
        for start in range(0, 10, 2):
            self.assertTrue({start, start + 1} <= set(train) or
                            {start, start + 1} <= set(validation))

    def test_incomplete_design_groups_are_rejected(self):
        for count in (0, 4, 6, 12):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'complete design groups'):
                split_pair_indices(count, group_size=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.h5'
            for group_size in (True, 1, 2.5, '5'):
                write_graphs(path, 10, group_size=group_size)
                with self.subTest(group_size=group_size), self.assertRaises(ValueError):
                    read_design_group_size(path)

    def test_gnn_base_training_reads_five_sample_group_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'base_25.h5'
            checkpoint = Path(directory) / 'base.pt'
            write_graphs(path, 25, group_size=5)
            with (
                patch.object(gnn_training, 'build_power_predictor'),
                patch.object(gnn_training.torch.cuda, 'is_available', return_value=False),
                patch.object(gnn_training, 'train',
                             side_effect=lambda *_args, **_kwargs: checkpoint.write_bytes(b'test')) as train,
            ):
                self.assertEqual(gnn_training.train_base_power_predictor(path, checkpoint), checkpoint)
            training, validation = train.call_args.args[2:4]
            self.assertEqual((len(training), len(validation)), (20, 5))
            for start in range(0, 25, 5):
                group = set(range(start, start + 5))
                self.assertTrue(group <= set(training.indices) or group <= set(validation.indices))


if __name__ == '__main__':
    unittest.main()
