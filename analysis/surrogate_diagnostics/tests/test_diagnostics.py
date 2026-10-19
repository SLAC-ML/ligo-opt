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
from analysis.surrogate_diagnostics.dataset import DIRECTIONS, load_groups, load_splits, pack, REPO_ROOT
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


def write_base(path, group_count=2, group_size=5, include_metadata=True):
    with h5py.File(path, "w") as handle:
        if include_metadata:
            handle.attrs["design_group_size"] = group_size
        for group_id in range(group_count):
            itm, etm = -1900. - 50 * group_id, 2200. + 50 * group_id
            geometry = ([(itm, etm), (itm - 6, etm + 7)] if group_size == 2 else
                        [(itm, etm), (itm + 6, etm), (itm - 6, etm),
                         (itm, etm + 7), (itm, etm - 7)])
            for offset, (itm_roc, etm_roc) in enumerate(geometry):
                graph = nx.DiGraph()
                graph.add_node("ITM.p1.i", Rc=itm_roc, q_re=-1000., q_im=400., pd=1.)
                graph.add_node("ETM.p1.i", Rc=etm_roc, pd=1.)
                graph.add_node("ITM.p2.i", pd=20. + offset)
                handle.create_dataset(f"sim_{group_size * group_id + offset}",
                                      data=np.void(pickle.dumps(graph)))


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

    def test_five_row_initial_data_counts_and_selection(self):
        write_base(self.root / "base_500.h5", group_count=100)
        groups = load_groups(self.config, {"base_ids": "all"})
        self.assertEqual(len(groups), 100)
        self.assertEqual(sum(len(group.x) for group in groups), 500)
        self.assertEqual(groups[-1].group_id, "base:99")
        self.assertTrue(all(group.directions == DIRECTIONS for group in groups))
        selected = load_groups(self.config, {"base_id_range": [98, 100]})
        self.assertEqual([group.group_id for group in selected], ["base:98", "base:99"])
        self.assertEqual(selected[-1].keys, [f"sim_{i}" for i in range(495, 500)])
        x, y, nominal, partners = pack(selected)
        self.assertEqual(x.shape, (10, 4))
        self.assertEqual(nominal.tolist(), [0, 0, 0, 0, 5, 5, 5, 5])
        self.assertEqual(partners.tolist(), [1, 2, 3, 4, 6, 7, 8, 9])
        torch.testing.assert_close(y[:5].expm1().flatten(),
                                   torch.arange(20., 25., dtype=torch.float64))
        records = list(selected[-1].records())
        self.assertEqual([row["direction"] for row in records], list(DIRECTIONS))
        self.assertEqual([(row["delta_itm_roc"], row["delta_etm_roc"]) for row in records],
                         [(0., 0.), (6., 0.), (-6., 0.), (0., 7.), (0., -7.)])
        with self.assertRaisesRegex(ValueError, "Invalid base group ID"):
            load_groups(self.config, {"base_ids": [100]})

    def test_legacy_initial_pairs_with_and_without_metadata(self):
        path = self.root / "base_4.h5"
        for include_metadata in (False, True):
            with self.subTest(include_metadata=include_metadata):
                write_base(path, group_size=2, include_metadata=include_metadata)
                groups = load_groups(self.config, {"base_ids": "all"})
                self.assertEqual([group.group_id for group in groups], ["base:0", "base:1"])
                self.assertTrue(all(group.directions == ("nominal", "joint") for group in groups))
                selected = load_groups(self.config, {"base_ids": [1]})[0]
                self.assertEqual(selected.keys, ["sim_2", "sim_3"])
                records = list(selected.records())
                self.assertEqual(records[1]["direction"], "joint")
                self.assertEqual(records[1]["delta_itm_roc"], -6.)
                self.assertEqual(records[1]["delta_etm_roc"], 7.)
                self.assertAlmostEqual(records[1]["delta_power"], 1.)
                _, _, nominal, partners = pack(groups)
                self.assertEqual(nominal.tolist(), [0, 2])
                self.assertEqual(partners.tolist(), [1, 3])

    def test_five_row_initial_direction_corruption_rejected(self):
        path = self.root / "base_10.h5"
        for corruption in ("wrong_sign", "other_mirror_changed"):
            with self.subTest(corruption=corruption):
                write_base(path)
                with h5py.File(path, "a") as handle:
                    graph = pickle.loads(bytes(handle["sim_1"][()]))
                    if corruption == "wrong_sign":
                        graph.nodes["ITM.p1.i"]["Rc"] = -1906.
                    else:
                        graph.nodes["ETM.p1.i"]["Rc"] = 2207.
                    del handle["sim_1"]
                    handle.create_dataset("sim_1", data=np.void(pickle.dumps(graph)))
                with self.assertRaisesRegex(ValueError, "perturbation order"):
                    load_groups(self.config, {"base_ids": "all"})

    def test_initial_incomplete_and_unsupported_group_layouts_rejected(self):
        path = self.root / "base_10.h5"
        write_base(path)
        with h5py.File(path, "a") as handle:
            del handle["sim_9"]
        with self.assertRaisesRegex(ValueError, "complete nominal"):
            load_groups(self.config, {"base_ids": "all"})
        write_base(path)
        with h5py.File(path, "a") as handle:
            handle.attrs["design_group_size"] = 3
        with self.assertRaisesRegex(ValueError, "Unsupported base design group size"):
            load_groups(self.config, {"base_ids": "all"})

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
