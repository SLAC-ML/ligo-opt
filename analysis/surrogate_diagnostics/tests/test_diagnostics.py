"""Check pairing, leakage prevention, metrics, and the diagnostic training loop."""

import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import h5py
import networkx as nx
import numpy as np
import torch

from MLP.checkpoint import load_checkpoint
from analysis.surrogate_diagnostics.dataset import load_groups, load_splits, pack, REPO_ROOT
from analysis.surrogate_diagnostics.evaluate import predictions
from analysis.surrogate_diagnostics.tiny_fit import model_fingerprint, paired_loss, run_experiment


def write_round(path, offset=0, powers=(20., 21., 19., 24., 17.)):
    geometry = [(-1900-offset, 2200+offset), (-1894-offset, 2200+offset),
                (-1906-offset, 2200+offset), (-1900-offset, 2207+offset),
                (-1900-offset, 2193+offset)]
    with h5py.File(path, "w") as handle:
        for i in reversed(range(5)):
            graph = nx.DiGraph()
            graph.add_node("ITM.p1.i", Rc=geometry[i][0], q_re=-1000., q_im=400., pd=1.)
            graph.add_node("ETM.p1.i", Rc=geometry[i][1], pd=1.)
            graph.add_node("ITM.p2.i", pd=powers[i])
            handle.create_dataset(f"sim_{i}", data=np.void(pickle.dumps(graph)))


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        write_round(self.root / "round1.h5")
        write_round(self.root / "round2.h5", offset=50)
        write_round(self.root / "round3.h5", offset=100)
        self.config = json.loads((REPO_ROOT / "analysis/surrogate_diagnostics/configs/tiny_fit.json").read_text())
        self.config.update(data_dir=str(self.root), seeds=[7], epochs=5, record_every=1,
                           model_config={"hidden_sizes": [8]}, splits={"train": {"rounds": [1]}})
        self.old_threads = torch.get_num_threads()

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        self.directory.cleanup()

    def test_group_order_and_nominal_indices(self):
        groups = load_groups(self.config, {"rounds": [1, 2]})
        x, y, nominal, partners = pack(groups)
        self.assertEqual(x.shape, (10, 4))
        self.assertEqual(nominal.tolist(), [0, 0, 0, 0, 5, 5, 5, 5])
        self.assertEqual(partners.tolist(), [1, 2, 3, 4, 6, 7, 8, 9])
        torch.testing.assert_close(y[:5].expm1().flatten(), torch.tensor([20., 21., 19., 24., 17.], dtype=torch.float64))
        self.assertTrue(all(g.valid_fixed_q for g in groups))
        self.assertEqual(list(groups[0].records())[-1]["direction"], "ETM_minus")

    def test_duplicate_geometry_and_group_leakage_rejected(self):
        self.config["splits"] = {"train": {"rounds": [1]}, "test": {"rounds": [1]}}
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_splits(self.config)
        write_round(self.root / "round2.h5", offset=0)
        self.config["splits"]["test"] = {"rounds": [2]}
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_splits(self.config)

    def test_wrong_direction_and_missing_keys_rejected(self):
        with h5py.File(self.root / "round1.h5", "a") as handle:
            handle.move("sim_4", "sim_8")
        with self.assertRaisesRegex(ValueError, "Noncontiguous"):
            load_groups(self.config, {"rounds": [1]})
        write_round(self.root / "round1.h5")
        with h5py.File(self.root / "round1.h5", "a") as handle:
            graph = pickle.loads(bytes(handle["sim_1"][()]))
            graph.nodes["ITM.p1.i"]["Rc"] = -1906.
            del handle["sim_1"]
            handle.create_dataset("sim_1", data=np.void(pickle.dumps(graph)))
        with self.assertRaisesRegex(ValueError, "perturbation order"):
            load_groups(self.config, {"rounds": [1]})

    def test_zero_q_group_is_not_a_valid_mismatch_example(self):
        group = load_groups(self.config, {"rounds": [1]})[0]
        group.x[1, 2:] = 0
        self.assertFalse(group.valid_fixed_q)

    def test_pair_loss_uses_power_differences_and_both_gradients(self):
        target = torch.tensor([[20.], [24.]], dtype=torch.float64).log1p()
        offset_prediction = torch.tensor([[120.], [124.]], dtype=torch.float64).log1p()
        nominal, partner = torch.tensor([0]), torch.tensor([1])
        self.assertAlmostEqual(paired_loss(offset_prediction, target, nominal, partner, 1.).item(), 0., places=20)
        bad = torch.tensor([[20.], [20.]], dtype=torch.float64).log1p().requires_grad_()
        loss = paired_loss(bad, target, nominal, partner, 1.)
        loss.backward()
        self.assertGreater(bad.grad[0].item(), 0.)
        self.assertLess(bad.grad[1].item(), 0.)

    def test_exact_predictions_give_correct_metrics_and_cost(self):
        group = load_groups(self.config, {"rounds": [1]})[0]

        class Exact(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.placeholder = torch.nn.Parameter(torch.zeros(()))

            def forward(self, x):
                return group.y

        _, metrics, costs = predictions(Exact(), [group], "train")
        self.assertEqual(metrics["delta_mae"], 0.)
        self.assertEqual(metrics["sign_accuracy"], 1.)
        self.assertAlmostEqual(costs[0]["true_perturbation_cost"], 2/6 + 7/7)
        self.assertEqual(metrics["perturbation_cost_mae"], 0.)

    def test_training_identical_initialization_and_checkpoint_replay(self):
        output = self.root / "fit"
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_experiment(self.config, output)
        self.assertEqual(len(summary), 3)
        hashes = set()
        for variant in self.config["losses"]:
            directory = output / f"{variant}_seed7"
            config = json.loads((directory / "config.json").read_text())
            hashes.add(config["initial_state_sha256"])
            model, payload = load_checkpoint(directory / "checkpoint.pt", device="cpu")
            self.assertTrue(payload["training_metadata"]["diagnostic_only"])
            _, metrics, _ = predictions(model, load_groups(self.config, {"rounds": [1]}), "train")
            saved = json.loads((directory / "metrics.json").read_text())["splits"]["train"]
            self.assertEqual(metrics, saved)
        self.assertEqual(len(hashes), 1)
        with self.assertRaises(FileExistsError):
            run_experiment(self.config, output)

    def test_test_labels_never_affect_training_or_selection(self):
        self.config.update(mode="generalization", losses=["paired"], splits={
            "train": {"rounds": [1]}, "validation": {"rounds": [2]}, "test": {"rounds": [3]},
        })
        with contextlib.redirect_stdout(io.StringIO()):
            run_experiment(self.config, self.root / "before")
            write_round(self.root / "round3.h5", offset=100, powers=(100., 80., 90., 70., 110.))
            run_experiment(self.config, self.root / "after")
        first, p1 = load_checkpoint(self.root / "before/paired_seed7/checkpoint.pt", device="cpu")
        second, p2 = load_checkpoint(self.root / "after/paired_seed7/checkpoint.pt", device="cpu")
        self.assertEqual(model_fingerprint(first), model_fingerprint(second))
        self.assertEqual(p1["training_metadata"]["best_epoch"], p2["training_metadata"]["best_epoch"])


if __name__ == "__main__":
    unittest.main()
