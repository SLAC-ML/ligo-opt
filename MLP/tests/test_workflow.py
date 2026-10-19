"""Exercise the MLP's data, saved models, training, and PSO integration.

Run from the repository root with:
    /home/xuesi.ma/.conda/envs/ligoopt/bin/python -m unittest discover -s MLP/tests -v

The training fixtures are tiny synthetic graphs. The inference tests use real
Finesse beam tracing, but never run its expensive optical power simulation.
"""

from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import h5py
import networkx as nx
import numpy as np
import torch

from GNN.GNN_utils import kat_manipulation
from MLP.checkpoint import load_checkpoint, save_checkpoint
from MLP.data import PowerDataset
from MLP.model import build_power_predictor
from MLP.predictor import MLPPowerPredictor


DETECTOR = "p_ITM_p2_i"
SMALL_MODEL = {"hidden_sizes": [8, 8]}


def write_graphs(path, count, *, offset=0, group_size=None):
    """Write the same graph format as the real simulator, with known answers.

    Insertion order is deliberately reversed. HDF5's default iteration is also
    alphabetical (sim_10 before sim_2), so correct group splitting requires the
    loader to explicitly recover the numeric simulation order.
    """
    with h5py.File(path, "w") as handle:
        if group_size is not None:
            handle.attrs['design_group_size'] = group_size
        for index in reversed(range(count)):
            value = float(offset + index)
            graph = nx.DiGraph()
            graph.add_node(
                "ITM.p1.i", Rc=-1934.0 - value, q_re=-1800.0 + value,
                q_im=300.0 + value, pd=999.0,
            )
            graph.add_node("ETM.p1.i", Rc=2245.0 + value, pd=200.0 + value)
            # The target is neither the first nor the second graph node.
            graph.add_node("ITM.p2.i", Rc=-1934.0 - value, pd=10.0 + value)
            handle.create_dataset(f"sim_{index}", data=np.void(pickle.dumps(graph)))


class CPUTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny matrix operations are quicker with one CPU thread; restore the
        # previous setting so these tests do not affect another test suite.
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)


class DataAndCheckpointTests(CPUTestCase):
    def test_numeric_graph_order_raw_features_and_selected_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base_12.h5"
            write_graphs(path, 12)
            dataset = PowerDataset(path, DETECTOR)
            self.assertEqual(tuple(dataset.x.shape), (12, 4))
            self.assertEqual(tuple(dataset.y.shape), (12, 1))
            index = np.arange(12)
            expected = np.column_stack((
                -1934 - index, 2245 + index, -1800 + index, 300 + index,
            ))
            np.testing.assert_array_equal(dataset.x.numpy(), expected)
            np.testing.assert_allclose(
                dataset.y.numpy().ravel(), np.log1p(10 + index), rtol=1e-6,
            )
            other = PowerDataset(path, "p_ETM_p1_i")
            np.testing.assert_allclose(
                other.y.numpy().ravel(), np.log1p(200 + index), rtol=1e-6,
            )

    def test_missing_detector_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base_2.h5"
            write_graphs(path, 2)
            with self.assertRaises((KeyError, ValueError)):
                PowerDataset(path, "p_nonexistent")

    def test_normalization_and_checkpoint_roundtrip(self):
        model = build_power_predictor(SMALL_MODEL)
        # A constant feature should receive scale 1, avoiding division by zero.
        inputs = torch.tensor([[1., 5., -4., 2.], [3., 5., 4., 6.]])
        model.set_normalization(inputs)
        torch.testing.assert_close(model.input_mean, inputs.mean(dim=0))
        torch.testing.assert_close(model.input_scale, torch.tensor([1., 1., 4., 2.]))
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, DETECTOR, training_metadata={"example": 7})
            restored, payload = load_checkpoint(path, device="cpu")
            restored.eval()
            self.assertEqual(payload["detector"], DETECTOR)
            self.assertEqual(payload["training_metadata"]["example"], 7)
            torch.testing.assert_close(restored.input_mean, model.input_mean)
            torch.testing.assert_close(restored.input_scale, model.input_scale)
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs), model(inputs))


class TrainingTests(CPUTestCase):
    def test_base_and_two_rounds_keep_holdout_and_normalization(self):
        self.check_base_and_two_rounds(group_size=None, design_count=6)

    def test_five_sample_groups_keep_holdout_and_normalization_across_rounds(self):
        self.check_base_and_two_rounds(group_size=5, design_count=5)

    def check_base_and_two_rounds(self, *, group_size, design_count):
        from MLP.finetune import run_finetune
        from MLP.training import train_base_power_predictor

        settings = {
            "n_epochs": 2, "batch_size": 4, "learning_rate": 0.001,
            "save_loss_interval": 1, "print_interval": 100,
        }
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch.cuda, "is_available", return_value=False):
            root = Path(directory)
            effective_group_size = group_size or 2
            sample_count = effective_group_size * design_count
            base_data = root / f"base_{sample_count}.h5"
            base_checkpoint = root / "base.pt"
            write_graphs(base_data, sample_count, group_size=group_size)
            # Five rows represent a nominal design and its four perturbations.
            write_graphs(root / "round1.h5", 5, offset=100)
            write_graphs(root / "round2.h5", 5, offset=200)
            write_graphs(root / "round3.h5", 7, offset=300)
            train_base_power_predictor(
                base_data, base_checkpoint, hyperparams=settings,
                detector=DETECTOR, model_config=SMALL_MODEL, seed=9302,
            )
            base_model, base_payload = load_checkpoint(base_checkpoint, device="cpu")
            metadata = base_payload["training_metadata"]
            train_indices = metadata["train_indices"]
            validation_indices = metadata["validation_indices"]
            training = set(train_indices)
            validation = set(validation_indices)
            self.assertTrue(training)
            self.assertTrue(validation)
            self.assertFalse(training & validation)
            self.assertEqual(training | validation, set(range(sample_count)))
            self.assertEqual(metadata['base_group_size'], effective_group_size)
            self.assertEqual(metadata['validation_mode'],
                             'held_out_pairs' if effective_group_size == 2 else 'held_out_designs')
            for start in range(0, sample_count, effective_group_size):
                group = set(range(start, start + effective_group_size))
                self.assertTrue(group <= training or group <= validation)

            dataset = PowerDataset(base_data, DETECTOR)
            training_inputs = dataset.x[train_indices]
            torch.testing.assert_close(base_model.input_mean, training_inputs.mean(dim=0))
            torch.testing.assert_close(
                base_model.input_scale, training_inputs.std(dim=0, unbiased=False),
            )
            self.assertTrue(np.isfinite(metadata["best_validation_loss"]))

            # Fine-tuning replays prior rounds, while the newest round keeps all
            # five related examples together in training. Future files are ignored.
            for round_number in (1, 2):
                prefix = run_finetune(
                    data_dir=root, base_checkpoint_path=base_checkpoint,
                    checkpoint_template=str(root / "checkpoint_round{}"),
                    current_round=round_number, hyperparams=settings,
                )
                self.assertEqual(prefix, str(root / f"checkpoint_round{round_number}"))
                model, payload = load_checkpoint(f"{prefix}.pt", device="cpu")
                latest = payload["training_metadata"]
                self.assertEqual(latest["train_indices"], train_indices)
                self.assertEqual(latest["validation_indices"], validation_indices)
                self.assertEqual(latest["current_round"], round_number)
                self.assertEqual(latest["newest_round_count"], 5)
                self.assertEqual(latest["newest_round_weight"], sample_count / 5)
                self.assertEqual(latest['base_group_size'], effective_group_size)
                self.assertEqual(latest["training_row_count"], len(training) + 5 * round_number)
                self.assertEqual(
                    latest["round_data_counts"],
                    {str(index): 5 for index in range(1, round_number + 1)},
                )
                self.assertTrue(np.isfinite(latest["best_validation_loss"]))
                torch.testing.assert_close(model.input_mean, base_model.input_mean, rtol=0, atol=0)
                torch.testing.assert_close(model.input_scale, base_model.input_scale, rtol=0, atol=0)

    def test_single_five_sample_design_trains_and_finetunes_without_holdout(self):
        from MLP.finetune import run_finetune
        from MLP.training import train_base_power_predictor

        settings = {'n_epochs': 1, 'batch_size': 5}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch.cuda, 'is_available', return_value=False):
            root = Path(directory)
            write_graphs(root / 'base_5.h5', 5, group_size=5)
            write_graphs(root / 'round1.h5', 5, offset=100)
            with self.assertWarnsRegex(UserWarning, 'Single-design bootstrap'):
                train_base_power_predictor(
                    root / 'base_5.h5', root / 'base.pt', hyperparams=settings,
                    detector=DETECTOR, model_config=SMALL_MODEL,
                )
            with self.assertWarnsRegex(UserWarning, 'Single-design bootstrap'):
                prefix = run_finetune(
                    data_dir=root, base_checkpoint_path=root / 'base.pt',
                    checkpoint_template=str(root / 'round{}'), current_round=1,
                    hyperparams=settings,
                )
            _, payload = load_checkpoint(f'{prefix}.pt', device='cpu')
            metadata = payload['training_metadata']
            self.assertEqual(metadata['validation_mode'], 'training_monitor_only')
            self.assertEqual(metadata['train_indices'], list(range(5)))
            self.assertEqual(metadata['validation_indices'], list(range(5)))
            self.assertEqual(metadata['training_row_count'], 10)

    def test_restored_splits_reject_design_leakage_and_accept_legacy_pairs(self):
        from MLP.finetune import _restore_base_split

        metadata = dict(base_group_size=5, validation_mode='held_out_designs',
                        train_indices=list(range(5)), validation_indices=list(range(5, 10)))
        self.assertEqual(_restore_base_split(metadata, 10),
                         (metadata['train_indices'], metadata['validation_indices']))
        # Still a disjoint partition of all rows, but the first design is split.
        leaked = {**metadata, 'train_indices': [0, 1, 2, 3, 5],
                  'validation_indices': [4, 6, 7, 8, 9]}
        with self.assertRaisesRegex(ValueError, 'design group'):
            _restore_base_split(leaked, 10)
        legacy = dict(validation_mode='held_out_pairs', train_indices=[0, 1],
                      validation_indices=[2, 3])
        self.assertEqual(_restore_base_split(legacy, 4), ([0, 1], [2, 3]))


class PredictorIntegrationTests(CPUTestCase):
    def setUp(self):
        from utils.finesse_base import gnn_base_kat

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "model.pt"
        model = build_power_predictor(SMALL_MODEL)
        model.set_normalization(torch.tensor([
            [-1934., 2245., -1900., 300.], [-3500., 3500., -1700., 500.],
        ]))
        # A positive output makes the unstable-cavity zero test meaningful.
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.fill_(0.01)
                    module.bias.fill_(0.5)
        save_checkpoint(self.checkpoint, model, DETECTOR)
        self.predictor = MLPPowerPredictor(self.checkpoint, device="cpu")
        self.base = gnn_base_kat.deepcopy()

    def fresh(self, itm, etm, q_value=None):
        return kat_manipulation(
            itm, etm, self.base, nominal_q_value=q_value,
            include_aperture_maps=False,
        )

    def assert_outputs_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for (names, powers, q_value), (ref_names, ref_powers, ref_q) in zip(actual, expected):
            self.assertEqual(names, ref_names)
            np.testing.assert_allclose(powers, ref_powers, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(q_value, ref_q, rtol=1e-11, atol=1e-9)

    def test_scalar_batch_and_reused_fixed_q_iterator_agree(self):
        import PSOoptimizer as optimizer

        specs = [
            (-1934., 2245., None), (-3500., 3500., None),
            (-1900., 2250., -1900 + 300j), (-1950., 2300., -1700 + 500j),
            (-1434., 1745., None), (-1934., 2245., None),
        ]
        expected = self.predictor.run_batch([self.fresh(*spec) for spec in specs])
        scalar = [self.predictor.run(self.fresh(*spec)) for spec in specs]
        for (names, powers), (expected_names, expected_powers, _) in zip(scalar, expected):
            self.assertEqual(names, expected_names)
            np.testing.assert_allclose(powers, expected_powers, rtol=1e-6)
        self.assertGreater(expected[0][1][0], 0)
        np.testing.assert_allclose(expected[2][2], specs[2][2])
        np.testing.assert_allclose(expected[3][2], specs[3][2])
        reuse = optimizer._ReusableGNNKats(self.base)
        for _ in range(2):
            self.assert_outputs_close(
                self.predictor.run_batch(reuse.update(*spec) for spec in specs), expected,
            )
        with patch.object(self.predictor.model, "forward", wraps=self.predictor.model.forward) as forward:
            self.assertEqual(self.predictor.run_batch(iter(())), [])
            forward.assert_not_called()

    def test_unstable_cavity_has_exact_zero_even_with_fixed_q(self):
        for q_value in (None, -1900 + 300j):
            kat = self.fresh(-1434., 1745., q_value)
            self.assertFalse(kat.cavArm.is_stable)
            names, powers, q_at_itm = self.predictor.run_batch([kat])[0]
            self.assertEqual(names, [DETECTOR])
            self.assertEqual(powers, [0.0])
            self.assertEqual(q_at_itm, 0j)

    def test_pso_rejects_checkpoint_for_another_detector(self):
        import PSOoptimizer as optimizer

        with self.assertRaisesRegex(ValueError, "detector|predicts"):
            optimizer.main_wrapper_PSO(
                self.base, "p_ETM_p1_i", model_path=self.checkpoint,
                surrogate_type="MLP",
            )


    def test_pso_nominal_and_perturbed_batches_match_individual_evaluations(self):
        import PSOoptimizer as optimizer
        from perturbation import calc_perturbed_params
        from utils import calc
        from utils.power_results import PowerEvaluationRecorder, reconstruct_cost

        points = np.array([[-1934., 2245.], [-3500., 3500.], [-1434., 1745.]])
        for mode, evaluation_mode in (("design_power", "nominal"), ("full", "perturbed")):
            with (
                self.subTest(mode=mode),
                patch.object(calc, "COST_MODE", mode),
                patch.object(calc, "RECORD_FULL_COST", False),
                patch.object(optimizer, "COST_MODE", mode),
                patch.object(optimizer, "EVALUATION_MODE", evaluation_mode),
            ):
                expected_costs = []
                for itm, etm in points:
                    nominal = self.predictor.run_batch([self.fresh(itm, etm)])[0]
                    names, powers, q_input = nominal
                    if evaluation_mode == "nominal":
                        expected_costs.append(calc.calc_cost(
                            None, [powers] * 3, [powers] * 3,
                            [names] * 3, [names] * 3, (0., 0.), DETECTOR,
                        ))
                    else:
                        (ip, im), (ep, em), delta = calc_perturbed_params(itm, etm)
                        # Independent copies provide a reference for PSO's reused
                        # mutable models and preserve each nominal beam's q.
                        plus_i, minus_i, plus_e, minus_e = self.predictor.run_batch([
                            self.fresh(ip, etm, q_input), self.fresh(im, etm, q_input),
                            self.fresh(itm, ep, q_input), self.fresh(itm, em, q_input),
                        ])
                        expected_costs.append(calc.calc_cost(
                            None, [plus_i[1], powers, minus_i[1]],
                            [plus_e[1], powers, minus_e[1]],
                            [plus_i[0], names, minus_i[0]],
                            [plus_e[0], names, minus_e[0]], delta, DETECTOR,
                        ))
                recorder = PowerEvaluationRecorder()
                with (
                    patch("MLP.predictor.MLPPowerPredictor", return_value=self.predictor),
                    patch.object(self.predictor, "run_batch", wraps=self.predictor.run_batch) as batches,
                ):
                    objective = optimizer.main_wrapper_PSO(
                        self.base, DETECTOR, model_path=self.checkpoint,
                        surrogate_type="MLP", evaluation_recorder=recorder,
                    )
                    actual = objective(points)
                    self.assertEqual(batches.call_count, 1 if evaluation_mode == "nominal" else 2)
                    self.assertEqual(objective(np.empty((0, 2))).shape, (0,))
                np.testing.assert_allclose(actual, expected_costs, rtol=1e-6, atol=1e-7)
                self.assertTrue(np.isinf(actual[-1]))
                self.assertAlmostEqual(reconstruct_cost(recorder.best), recorder.best["cost"])
                self.assertEqual(recorder.best["model_path"], str(self.checkpoint))
                if evaluation_mode == "nominal":
                    self.assertIsNone(recorder.best["powers"]["ITM_plus"])


if __name__ == "__main__":
    unittest.main()
