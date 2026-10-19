"""Verify MLP cost-map CSV isolation and weighted cost recording."""
import csv
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from analysis import perturbation_cost_map_mlp as mlp_map
from utils import calc


class MlpCostRecordingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.shared_csv = self.root / 'shared.csv'
        self.model = self.root / 'model.pt'
        self.model.write_bytes(b'test model checkpoint')
        settings = patch.multiple(
            calc, COST_MODE='full', RECORD_FULL_COST=True,
            FULL_POWER_WEIGHT=0.5, FULL_PERTURBATION_WEIGHT=0.5,
            FULL_COST_CSV=self.shared_csv,
        )
        settings.start()
        self.addCleanup(settings.stop)
        grid = patch.multiple(
            mlp_map, GRID_SIZE=2,
            ITM_ROC_LIST=np.array([-2.0, -1.0]),
            ETM_ROC_LIST=np.array([2.0, 3.0]),
        )
        grid.start()
        self.addCleanup(grid.stop)

    def _cost(self, pd_name='p_test', full_cost_csv=None):
        names = [[pd_name]] * 3
        return calc.calc_cost(
            None, [[4.0], [2.0], [1.0]], [[3.0], [2.0], [1.0]],
            names, names, (1.0, 2.0), pd_name,
            full_cost_csv=full_cost_csv,
        )

    def _rows(self, path):
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    def _run_map(self, output, *, overwrite=False):
        predictor_module = types.ModuleType('MLP.predictor')
        predictor_module.MLPPowerPredictor = Mock(
            return_value=types.SimpleNamespace(detector=mlp_map.DEFAULT_PD_NAME)
        )

        def evaluate(itm_roc, etm_roc, pd_name, predictor, *, full_cost_csv=None):
            return self._cost(pd_name=pd_name, full_cost_csv=full_cost_csv)

        argv = ['--model-path', str(self.model), '--output', str(output)]
        if overwrite:
            argv.append('--overwrite')
        with patch.dict(sys.modules, {'MLP.predictor': predictor_module}), \
                patch.object(mlp_map, 'full_mlp_cost', side_effect=evaluate) as objective, \
                patch.object(mlp_map, 'tqdm'), patch('sys.stdout', new=io.StringIO()):
            result = mlp_map.main(argv)
        self.assertEqual(result, output.resolve())
        return objective

    def test_override_records_weighted_components_only_in_dedicated_csv(self):
        dedicated = self.root / 'nested/map_records.csv'
        self.assertEqual(self._cost(full_cost_csv=str(dedicated)), 2.25)
        self.assertFalse(self.shared_csv.exists())
        rows = self._rows(dedicated)
        self.assertEqual(len(rows), 1)
        expected = {
            'nominal_power': 2.0, 'power_cost': 0.5, 'perturbation_cost': 4.0,
            'power_weight': 0.5, 'perturbation_weight': 0.5,
            'weighted_power_cost': 0.25, 'weighted_perturbation_cost': 2.0,
            'total_cost': 2.25,
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                self.assertEqual(float(rows[0][field]), value)

    def test_default_csv_remains_available_to_existing_callers(self):
        names = [['p_test']] * 3
        result = calc.calc_cost(
            None, [[4.0], [2.0], [1.0]], [[3.0], [2.0], [1.0]],
            names, names, (1.0, 2.0), 'p_test',
        )
        self.assertEqual(result, 2.25)
        self.assertEqual(len(self._rows(self.shared_csv)), 1)

    def test_full_mlp_cost_forwards_the_dedicated_csv_to_the_calculator(self):
        manipulation = types.ModuleType('GNN.GNN_utils')
        manipulation.kat_manipulation = Mock(return_value=object())
        perturbation = types.ModuleType('perturbation')
        perturbation.calc_perturbed_params = Mock(
            return_value=((-1.0, -3.0), (4.0, 0.0), (1.0, 2.0))
        )
        finesse = types.ModuleType('utils.finesse_base')
        finesse.base_kat = object()
        predictor = Mock()
        predictor.run_batch.side_effect = [
            [(['p_test'], [2.0], 'nominal q')],
            [(['p_test'], [power], None) for power in (4.0, 1.0, 3.0, 1.0)],
        ]
        dedicated = self.root / 'map_records.csv'
        with patch.dict(sys.modules, {
            'GNN.GNN_utils': manipulation,
            'perturbation': perturbation,
            'utils.finesse_base': finesse,
        }):
            result = mlp_map.full_mlp_cost(
                -2.0, 2.0, 'p_test', predictor, full_cost_csv=dedicated
            )
        self.assertEqual(result, 2.25)
        self.assertEqual(len(self._rows(dedicated)), 1)
        self.assertFalse(self.shared_csv.exists())

    def test_custom_output_routes_every_evaluation_to_its_sidecar(self):
        output = self.root / 'chosen/map.npz'
        sidecar = output.with_name('map_records.csv')
        objective = self._run_map(output)
        self.assertEqual(objective.call_count, 4)
        for call in objective.call_args_list:
            self.assertEqual(call.kwargs['full_cost_csv'], sidecar.resolve())
        self.assertEqual(len(self._rows(sidecar)), 4)
        self.assertFalse(self.shared_csv.exists())

    def test_resume_preserves_rows_and_evaluates_only_incomplete_points(self):
        output = self.root / 'map.npz'
        sidecar = self.root / 'map_records.csv'
        self._cost(pd_name=mlp_map.DEFAULT_PD_NAME, full_cost_csv=sidecar)
        recorded = sidecar.read_bytes()
        mlp_map._save_checkpoint(
            output, np.array([[2.25, np.nan], [np.nan, np.nan]]),
            np.array([[True, False], [False, False]]),
            pd_name=mlp_map.DEFAULT_PD_NAME,
            model_metadata={
                'mlp_checkpoint_path': str(self.model),
                'mlp_checkpoint_sha256': hashlib.sha256(self.model.read_bytes()).hexdigest(),
            },
        )
        objective = self._run_map(output)
        self.assertEqual(objective.call_count, 3)
        self.assertEqual(
            [call.args[:2] for call in objective.call_args_list],
            [(-2.0, 3.0), (-1.0, 2.0), (-1.0, 3.0)],
        )
        self.assertTrue(sidecar.read_bytes().startswith(recorded))
        self.assertEqual(len(self._rows(sidecar)), 4)
        self.assertEqual(self._run_map(output).call_count, 0)
        self.assertEqual(len(self._rows(sidecar)), 4)

    def test_overwrite_resets_only_the_dedicated_csv(self):
        output = self.root / 'map.npz'
        sidecar = self.root / 'map_records.csv'
        self._run_map(output)
        self.shared_csv.write_bytes(b'shared records must remain')
        objective = self._run_map(output, overwrite=True)
        self.assertEqual(objective.call_count, 4)
        self.assertEqual(len(self._rows(sidecar)), 4)
        self.assertEqual(self.shared_csv.read_bytes(), b'shared records must remain')

    def test_overwrite_rejects_a_sidecar_that_is_the_shared_csv(self):
        output = self.root / 'map.npz'
        sidecar = self.root / 'map_records.csv'
        sidecar.write_bytes(b'shared records must remain')
        with patch.object(calc, 'FULL_COST_CSV', sidecar):
            with self.assertRaises(ValueError):
                self._run_map(output, overwrite=True)
        self.assertEqual(sidecar.read_bytes(), b'shared records must remain')


if __name__ == '__main__':
    unittest.main()
