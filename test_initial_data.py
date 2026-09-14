import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from scipy.stats import qmc

import fabry_perot
from train_power_predictor import split_graph_pairs


class OptimizedLhsTests(unittest.TestCase):
    def test_samples_are_reproducible_stratified_and_optimized(self):
        count = 10
        itm_bounds = (-4134, -1434)
        etm_bounds = (1745, 4245)
        first = fabry_perot.optimized_lhs_roc_samples(
            count, itm_bounds, etm_bounds, seed=9302
        )
        second = fabry_perot.optimized_lhs_roc_samples(
            count, itm_bounds, etm_bounds, seed=9302
        )
        np.testing.assert_array_equal(first, second)

        unit_samples = qmc.scale(
            first,
            [itm_bounds[0], etm_bounds[0]],
            [itm_bounds[1], etm_bounds[1]],
            reverse=True,
        )
        for dimension in range(2):
            strata = np.floor(unit_samples[:, dimension] * count).astype(int)
            np.testing.assert_array_equal(np.sort(strata), np.arange(count))

        unoptimized = qmc.LatinHypercube(d=2, seed=9302).random(n=count)
        self.assertLessEqual(
            qmc.discrepancy(unit_samples, method='CD'),
            qmc.discrepancy(unoptimized, method='CD'),
        )

    def test_perturbation_stays_inside_bounds(self):
        rng = np.random.default_rng(9302)
        itm_bounds = (-4134, -1434)
        etm_bounds = (1745, 4245)
        for itm, etm in [(-4134, 1745), (-1434, 4245)]:
            perturbed_itm, perturbed_etm = fabry_perot._perturb_pair(
                itm,
                etm,
                rng=rng,
                itm_bounds=itm_bounds,
                etm_bounds=etm_bounds,
            )
            self.assertTrue(itm_bounds[0] <= perturbed_itm <= itm_bounds[1])
            self.assertTrue(etm_bounds[0] <= perturbed_etm <= etm_bounds[1])
            self.assertNotEqual(perturbed_itm, itm)
            self.assertNotEqual(perturbed_etm, etm)

    def test_invalid_design_count_is_rejected(self):
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                fabry_perot.optimized_lhs_roc_samples(
                    count, (-4134, -1434), (1745, 4245)
                )


class InitialDataGenerationTests(unittest.TestCase):
    def test_each_design_point_writes_a_design_and_fixed_q_graph(self):
        samples = np.array([[-4000.0, 2000.0], [-2000.0, 4000.0]])
        built_models = []

        def fake_kat_manipulation(itm, etm, _base_kat, nominal_q_value=None):
            model = SimpleNamespace(
                itm=float(itm),
                etm=float(etm),
                fixed_q=nominal_q_value,
            )
            built_models.append(model)
            return model

        def fake_finesse_sim(model):
            q_value = (
                complex(abs(model.itm), model.etm)
                if model.fixed_q is None
                else model.fixed_q
            )
            return ['p_test'], [1.0], ['q_ITM_p1_i'], [q_value]

        def fake_model_to_graph(model, *_args):
            return {
                'itm': model.itm,
                'etm': model.etm,
                'fixed_q': model.fixed_q,
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / 'base_4.h5'
            with (
                patch.object(
                    fabry_perot,
                    'optimized_lhs_roc_samples',
                    return_value=samples,
                ),
                patch.object(
                    fabry_perot,
                    'kat_manipulation',
                    side_effect=fake_kat_manipulation,
                ),
                patch.object(fabry_perot, 'finesse_sim', side_effect=fake_finesse_sim),
                patch.object(
                    fabry_perot,
                    'model_to_nx_port',
                    side_effect=fake_model_to_graph,
                ),
            ):
                returned_path = fabry_perot.generate_initial_data(
                    output_path,
                    num_design_points=2,
                    itm_bounds=(-4134, -1434),
                    etm_bounds=(1745, 4245),
                    seed=9302,
                )

            self.assertEqual(returned_path, output_path)
            with h5py.File(output_path, 'r') as h5_file:
                self.assertEqual(set(h5_file.keys()), {'sim_0', 'sim_1', 'sim_2', 'sim_3'})

        self.assertEqual(len(built_models), 4)
        self.assertIsNone(built_models[0].fixed_q)
        self.assertEqual(built_models[1].fixed_q, complex(4000, 2000))
        self.assertIsNone(built_models[2].fixed_q)
        self.assertEqual(built_models[3].fixed_q, complex(2000, 4000))


class PairAwareSplitTests(unittest.TestCase):
    def test_pairs_remain_together(self):
        dataset = list(range(10))
        train, validation = split_graph_pairs(dataset, train_fraction=0.8, seed=9302)
        train_indices = set(train.indices)
        validation_indices = set(validation.indices)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(validation), 2)
        self.assertFalse(train_indices & validation_indices)
        for pair_start in range(0, len(dataset), 2):
            pair = {pair_start, pair_start + 1}
            self.assertTrue(pair <= train_indices or pair <= validation_indices)


if __name__ == '__main__':
    unittest.main()