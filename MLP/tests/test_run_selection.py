"""Check that run.py's single switch selects a complete, isolated MLP run.

The driver is executed in a private module with an edited AST constant and a
fake repository root. This exercises its real import-time path construction
without changing run.py or creating any files in real experiment directories.
Only expensive simulation, training, and searching are mocked.
"""

import ast
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch

from MLP.checkpoint import load_checkpoint, save_checkpoint
from MLP.model import DEFAULT_MODEL_CONFIG, build_power_predictor
from MLP.tests.test_workflow import DETECTOR, SMALL_MODEL, write_graphs


RUN_SOURCE = Path(__file__).resolve().parents[2] / "run.py"


def load_driver(root, surrogate):
    """Evaluate the actual driver with the requested switch and temporary paths."""
    tree = ast.parse(RUN_SOURCE.read_text(), filename=str(RUN_SOURCE))
    values = {
        "SURROGATE_TYPE": surrogate, "RUN_NUMBER": 7, "TOTAL_ROUNDS": 2,
        "INITIAL_DESIGN_POINTS": 3, "INITIAL_DATA_SEED": 321,
        "DETECTOR": "p_ETM_p1_i",
    }
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in values:
                    statement.value = ast.Constant(values[target.id])
    module = types.ModuleType("mlp_selection_test_driver")
    module.__file__ = str(root / "run.py")
    exec(compile(ast.fix_missing_locations(tree), str(RUN_SOURCE), "exec"), module.__dict__)
    return module


class RunSelectionTests(unittest.TestCase):
    def test_switch_selects_model_paths_settings_and_base_trainer(self):
        for surrogate in ("GNN", "MLP"):
            with self.subTest(surrogate=surrogate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                driver = load_driver(root, surrogate)
                expected_data = root / surrogate / "data" / "run7"
                expected_models = root / surrogate / "models" / "run7"
                self.assertEqual(driver.DATA_DIR, expected_data)
                self.assertEqual(driver.MODEL_DIR, expected_models)
                self.assertEqual(driver.BASE_DATA_PATH, expected_data / "base_6.h5")
                filename = "base_6_mlp.pt" if surrogate == "MLP" else "base_6_gat10_kan5.pt"
                self.assertEqual(driver.BASE_CHECKPOINT_PATH, expected_models / filename)
                self.assertEqual(driver.FINETUNE_DATA_DIR, expected_data)
                self.assertEqual(
                    driver.CHECKPOINT_TEMPLATE,
                    str(expected_models / "power_predictor_ligoParams_finetuned_round{}"),
                )
                with (
                    patch.object(driver, "generate_initial_data") as generate,
                    patch.object(driver, "write_run_metadata") as metadata,
                    patch.object(driver, "train_base_power_predictor") as train_gnn,
                    patch("MLP.training.train_base_power_predictor") as train_mlp,
                ):
                    driver._prepare_new_run()
                self.assertTrue(expected_data.is_dir())
                self.assertTrue(expected_models.is_dir())
                generate.assert_called_once_with(
                    driver.BASE_DATA_PATH, num_design_points=3,
                    itm_bounds=driver.ITM_ROC_BOUNDS, etm_bounds=driver.ETM_ROC_BOUNDS,
                    seed=321,
                )
                selected_base, selected_finetune, selected_config = driver._active_training_settings()
                if surrogate == "MLP":
                    train_gnn.assert_not_called()
                    train_mlp.assert_called_once_with(
                        dataset_path=driver.BASE_DATA_PATH,
                        checkpoint_path=driver.BASE_CHECKPOINT_PATH,
                        hyperparams=driver.MLP_BASE_TRAINING_PARAMS,
                        detector="p_ETM_p1_i", model_config=DEFAULT_MODEL_CONFIG, seed=321,
                    )
                    self.assertIs(selected_base, driver.MLP_BASE_TRAINING_PARAMS)
                    self.assertIs(selected_finetune, driver.MLP_FINETUNE_PARAMS)
                else:
                    train_mlp.assert_not_called()
                    train_gnn.assert_called_once_with(
                        dataset_path=driver.BASE_DATA_PATH,
                        checkpoint_path=driver.BASE_CHECKPOINT_PATH,
                        hyperparams=driver.GNN_BASE_TRAINING_PARAMS,
                    )
                    self.assertIs(selected_base, driver.GNN_BASE_TRAINING_PARAMS)
                    self.assertIs(selected_finetune, driver.GNN_FINETUNE_PARAMS)
                metadata.assert_called_once()
                self.assertEqual(metadata.call_args.args[0], expected_data / "metadata.json")
                settings = metadata.call_args.kwargs["settings"]
                self.assertEqual(settings["surrogate_type"], surrogate)
                self.assertEqual(settings[surrogate.lower()], selected_config)
                self.assertEqual(settings["training"]["base"], selected_base)
                self.assertEqual(settings["training"]["finetune"], selected_finetune)
                self.assertEqual(settings["objective"]["detector"], "p_ETM_p1_i")
                self.assertEqual(settings["sampling"]["seed"], 321)
                self.assertEqual(
                    metadata.call_args.kwargs["paths"]["base_checkpoint"],
                    driver.BASE_CHECKPOINT_PATH,
                )

    def test_mlp_main_chains_round_checkpoints_and_labels_results(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = load_driver(Path(directory), "MLP")
            driver.MODEL_DIR.mkdir(parents=True)

            def fake_search(round_number, *args, model_path, **kwargs):
                powers = {"nominal": float(round_number)}
                predicted = {"powers": powers, "model_path": str(model_path)}
                measured = {"powers": {"nominal": round_number + 0.5}, "model_path": None}
                return [round_number, round_number], round_number / 10, round_number / 20, predicted, measured

            def fake_finetune(**kwargs):
                prefix = driver.CHECKPOINT_TEMPLATE.format(kwargs["current_round"])
                Path(prefix + ".pt").touch()
                return prefix

            with (
                patch.object(driver, "_prepare_new_run") as prepare,
                patch.object(driver, "_full_gnn_loop", side_effect=fake_search) as search,
                patch.object(driver, "run_finetune") as finetune_gnn,
                patch("MLP.finetune.run_finetune", side_effect=fake_finetune) as finetune_mlp,
                patch.object(driver, "tqdm", side_effect=lambda values: values),
            ):
                result = driver.main()
            prepare.assert_called_once_with()
            finetune_gnn.assert_not_called()
            self.assertEqual(search.call_count, 2)
            self.assertEqual(finetune_mlp.call_count, 2)
            previous_checkpoints = [
                driver.BASE_CHECKPOINT_PATH,
                Path(driver.CHECKPOINT_TEMPLATE.format(1) + ".pt"),
            ]
            for number, (search_call, training_call) in enumerate(
                zip(search.call_args_list, finetune_mlp.call_args_list), start=1,
            ):
                self.assertEqual(search_call.args[:2], (number, "p_ETM_p1_i"))
                self.assertEqual(search_call.kwargs["model_path"], previous_checkpoints[number - 1])
                self.assertEqual(
                    search_call.kwargs["finetune_data_path"], driver.DATA_DIR / f"round{number}.h5",
                )
                self.assertEqual(training_call.kwargs, {
                    "data_dir": driver.DATA_DIR,
                    "base_checkpoint_path": driver.BASE_CHECKPOINT_PATH,
                    "checkpoint_template": driver.CHECKPOINT_TEMPLATE,
                    "current_round": number, "finetune_weight": driver.FINETUNE_WEIGHT,
                    "hyperparams": driver.MLP_FINETUNE_PARAMS,
                })
            for field in ("costs", "powers", "evaluations"):
                self.assertIs(result[f"{field}_mlp"], result[f"{field}_surrogate"])
                self.assertNotIn(f"{field}_gnn", result)
                self.assertEqual(len(result[f"{field}_finesse"]), 2)
            self.assertEqual(result["costs_surrogate"], [0.1, 0.2])
            self.assertEqual(result["powers_mlp"], [{"nominal": 1.}, {"nominal": 2.}])
            self.assertEqual(
                [record["model_path"] for record in result["evaluations_mlp"]],
                [str(path) for path in previous_checkpoints],
            )

    def test_mlp_search_forwards_switch_to_pso(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = load_driver(Path(directory), "MLP")
            finesse_cost = Mock(return_value=3.0)
            surrogate_cost = Mock()
            pso = Mock(gbest_x=[-1934., 2245.], gbest_y=2.0)
            finesse_record = {"powers": {"nominal": 5.0}}
            surrogate_record = {"powers": {"nominal": 4.0}}
            finesse_recorder = Mock()
            finesse_recorder.verified_record.return_value = finesse_record
            surrogate_recorder = Mock()
            surrogate_recorder.verified_record.return_value = surrogate_record
            with (
                patch.object(driver, "main_wrapper_PSO", side_effect=[finesse_cost, surrogate_cost]) as wrappers,
                patch.object(driver, "SteppablePSO", return_value=pso) as swarm,
                patch.object(driver, "PowerEvaluationRecorder", side_effect=[finesse_recorder, surrogate_recorder]),
                patch.object(driver, "tqdm", return_value=[]),
            ):
                output = driver._full_gnn_loop(
                    1, driver.DETECTOR, -4000., -1000., 1700., 4300.,
                    model_path=driver.BASE_CHECKPOINT_PATH,
                    finetune_data_path=driver.DATA_DIR / "round1.h5",
                )
            self.assertIsNone(wrappers.call_args_list[0].kwargs["model_path"])
            surrogate_call = wrappers.call_args_list[1]
            self.assertEqual(surrogate_call.kwargs["surrogate_type"], "MLP")
            self.assertEqual(surrogate_call.kwargs["model_path"], driver.BASE_CHECKPOINT_PATH)
            self.assertIs(swarm.call_args.kwargs["func"], surrogate_cost)
            finesse_cost.assert_called_once_with(pso.gbest_x)
            self.assertIs(output[3], surrogate_record)
            self.assertIs(output[4], finesse_record)

    def test_invalid_switch_fails_before_files_or_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            driver = load_driver(root, "unknown")
            with (
                patch.object(driver, "generate_initial_data") as generate,
                patch.object(driver, "write_run_metadata") as metadata,
                patch.object(driver, "train_base_power_predictor") as train_gnn,
                patch("MLP.training.train_base_power_predictor") as train_mlp,
                self.assertRaisesRegex(ValueError, "SURROGATE_TYPE"),
            ):
                driver.main()
            generate.assert_not_called()
            metadata.assert_not_called()
            train_gnn.assert_not_called()
            train_mlp.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])


class ArtifactValidationTests(unittest.TestCase):
    def test_checkpoint_rejects_unknown_schema_and_reordered_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.pt"
            save_checkpoint(path, build_power_predictor(SMALL_MODEL), DETECTOR)
            original = torch.load(path, weights_only=True, map_location="cpu")
            for key, value, message in (
                ("schema_version", -1, "schema"),
                ("feature_names", list(reversed(original["feature_names"])), "feature order"),
            ):
                with self.subTest(key=key):
                    torch.save({**original, key: value}, path)
                    with self.assertRaisesRegex(ValueError, message):
                        load_checkpoint(path, device="cpu")

    def test_base_training_preserves_existing_checkpoint(self):
        from MLP.training import train_base_power_predictor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "base.pt"
            checkpoint.write_bytes(b"existing experiment")
            with self.assertRaises(FileExistsError):
                train_base_power_predictor(root / "unused.h5", checkpoint)
            self.assertEqual(checkpoint.read_bytes(), b"existing experiment")

    def test_finetuning_rejects_modified_base_and_missing_round(self):
        from MLP.finetune import run_finetune

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_data = root / "base_4.h5"
            checkpoint = root / "base.pt"
            write_graphs(base_data, 4)
            metadata = {
                "current_round": 0, "base_data_filename": base_data.name,
                "base_data_count": 4,
                "base_data_sha256": hashlib.sha256(base_data.read_bytes()).hexdigest(),
                "split_seed": 9302, "train_indices": [0, 1],
                "validation_indices": [2, 3], "validation_mode": "held_out_pairs",
            }
            save_checkpoint(checkpoint, build_power_predictor(SMALL_MODEL), DETECTOR, metadata)
            write_graphs(root / "round1.h5", 5)
            write_graphs(base_data, 4, offset=100)
            arguments = dict(
                data_dir=root, base_checkpoint_path=checkpoint,
                checkpoint_template=str(root / "finetuned_round{}"),
            )
            with patch("MLP.finetune._fit_model") as fit:
                with self.assertRaisesRegex(ValueError, "dataset changed"):
                    run_finetune(**arguments, current_round=1)
                (root / "round1.h5").unlink()
                write_graphs(root / "round2.h5", 5)
                with self.assertRaisesRegex(FileNotFoundError, "Missing round"):
                    run_finetune(**arguments, current_round=2)
                fit.assert_not_called()
            self.assertFalse((root / "finetuned_round1.pt").exists())
            self.assertFalse((root / "finetuned_round2.pt").exists())


if __name__ == "__main__":
    unittest.main()
