"""Protect audit exposure claims and failed-feedback reporting."""

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from MLP.model import build_power_predictor
from analysis.surrogate_diagnostics import training_audit as audit
from analysis.surrogate_diagnostics.tests.test_diagnostics import write_base, write_round


class TrainingAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.model = build_power_predictor({"hidden_sizes": [8]})

    def tearDown(self):
        self.directory.cleanup()

    def base_fixture(self, group_size=5, legacy=False):
        path = self.root / "base.h5"
        write_base(path, group_count=2, group_size=group_size,
                   include_metadata=not legacy)
        metadata = {"base_data_count": 2 * group_size,
                    "base_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "train_indices": list(range(group_size)),
                    "validation_indices": list(range(group_size, 2 * group_size))}
        if not legacy:
            metadata["base_group_size"] = group_size
        (self.root / "metadata.json").write_text(json.dumps({
            "files": {"base_checkpoint": str(self.root / "base.pt")}}))
        return {"detector": "p_ITM_p2_i", "training_metadata": metadata}

    def load_fixture(self, payload):
        def resolve(value):
            return self.root if str(value).startswith("MLP/data/") else Path(value)
        with patch.object(audit, "repo_path", side_effect=resolve), \
                patch.object(audit, "load_checkpoint", return_value=(self.model, payload)):
            return audit._load_run(7)

    def test_duplicate_indices_rejected_even_for_complete_group(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit._selection_ids({"train_indices": [0, 1, 2, 3, 4, 0]}, 5, "train_indices")

    def test_checkpoint_group_size_must_match_saved_data(self):
        payload = self.base_fixture()
        payload["training_metadata"]["base_group_size"] = 2
        with self.assertRaisesRegex(ValueError, "group size"):
            self.load_fixture(payload)

    def test_modified_base_data_cannot_be_claimed_as_training_exposure(self):
        payload = self.base_fixture()
        # Add a legitimate HDF5 attribute after the checkpoint's data hash.
        with h5py.File(self.root / "base.h5", "a") as handle:
            handle.attrs["edited_after_training"] = True
        with self.assertRaisesRegex(ValueError, "hash"):
            self.load_fixture(payload)

    def test_legacy_pair_metadata_still_loads_with_verified_hash(self):
        payload = self.base_fixture(group_size=2, legacy=True)
        run = self.load_fixture(payload)
        self.assertEqual(run["train_ids"], [0])
        self.assertEqual(run["validation_ids"], [1])

    def feedback_fixture(self, powers, invalid_q=False):
        path = self.root / "round1.h5"
        write_round(path, powers=powers)
        if invalid_q:
            with h5py.File(path, "a") as handle:
                for key in list(handle):
                    graph = pickle.loads(bytes(handle[key][()]))
                    graph.nodes["ITM.p1.i"]["q_re"] = 0.
                    graph.nodes["ITM.p1.i"]["q_im"] = 0.
                    del handle[key]
                    handle.create_dataset(key, data=np.void(pickle.dumps(graph)))
        run = {"run": 7, "root": self.root, "model": self.model,
               "payload": {"detector": "p_ITM_p2_i"}}
        payload = {"training_metadata": {"current_round": 1, "best_epoch": 1,
                   "newest_round_weight": 100., "best_validation_loss": 0.1,
                   "training_row_count": 405}}
        with patch.object(audit, "load_checkpoint", return_value=(self.model, payload)):
            return audit._feedback_summary(run, 0.1, [])

    def test_valid_q_zero_nominal_excludes_unavailable_cost_only(self):
        result, rows, _ = self.feedback_fixture((0., 1., 0., 2., 0.))
        self.assertEqual(result["valid_fixed_q_groups"], 1)
        for stage in ("pre", "post"):
            metrics = result["pre_post"][stage]
            self.assertIsNotNone(metrics["delta_mae"])
            self.assertIsNone(metrics["perturbation_cost_mae"])
            self.assertEqual(metrics["valid_cost_groups"], 0)
            self.assertEqual(metrics["invalid_cost_groups"], 1)
            self.assertEqual(metrics["cost_excluded_group_ids"], ["round:1:0"])
        self.assertTrue(all(not row["valid_perturbation_cost"] for row in rows))
        json.dumps(result, allow_nan=False)

    def test_all_invalid_feedback_returns_explicit_empty_metrics(self):
        result, _, _ = self.feedback_fixture((0., 0., 0., 0., 0.), invalid_q=True)
        self.assertFalse(result["has_valid_fixed_q_groups"])
        self.assertEqual(result["excluded_fixed_q_group_ids"], ["round:1:0"])
        self.assertIsNone(result["base_on_all_feedback"])
        self.assertIsNone(result["final_on_all_feedback"])
        for stage in ("pre", "post"):
            metrics = result["pre_post"][stage]
            self.assertEqual(metrics["groups"], 0)
            self.assertIsNone(metrics["power_mae"])
            self.assertIsNone(metrics["delta_mae"])
            self.assertIsNone(metrics["perturbation_cost_mae"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
