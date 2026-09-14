import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import pickle

import h5py
import networkx as nx
import numpy as np
import torch

import finetune_power_predictor as ft


class FinetuneWeightTests(unittest.TestCase):
    def test_scales_with_initial_and_latest_graph_counts(self):
        self.assertEqual(ft.resolve_finetune_weight(30, 5), 6.0)
        self.assertEqual(ft.resolve_finetune_weight(300, 5), 60.0)
        self.assertEqual(ft.resolve_finetune_weight(300, 10), 30.0)
        self.assertEqual(ft.resolve_finetune_weight(3, 5), 1.0)

    def test_explicit_override_and_invalid_inputs(self):
        self.assertEqual(ft.resolve_finetune_weight(300, 5, 4.0), 4.0)
        self.assertEqual(ft.resolve_finetune_weight(0, 5, 2.0), 2.0)
        for base, latest, weight in [
            (0, 5, None), (30, 0, None), (30, 0, 2.0),
            (30, 5, 0), (30, 5, -1), (30, 5, float('nan')),
            (30, 5, float('inf')),
        ]:
            with self.subTest(base=base, latest=latest, weight=weight):
                with self.assertRaises(ValueError):
                    ft.resolve_finetune_weight(base, latest, weight)

    @staticmethod
    def write_graphs(path, count):
        graph = nx.DiGraph()
        graph.add_node('a', Rc=1.0, R=0.9, q_re=1.0, q_im=2.0, pd=1.0)
        graph.add_node('b', Rc=2.0, R=0.9, q_re=1.0, q_im=2.0, pd=2.0)
        graph.add_edge('a', 'b', length=1.0, nr=1.0)
        with h5py.File(path, 'w') as handle:
            for index in range(count):
                handle.create_dataset(f'sim_{index}', data=np.void(pickle.dumps(graph)))

    def test_actual_files_drive_weights_and_latest_graphs_all_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_graphs(root / 'base_a.h5', 4)
            self.write_graphs(root / 'base_b.h5', 6)
            self.write_graphs(root / 'round1.h5', 3)
            self.write_graphs(root / 'round2.h5', 2)
            # A future round must neither be replayed nor affect the weight.
            self.write_graphs(root / 'round3.h5', 20)
            model = Mock()
            model.to.return_value = model
            for override, expected_weight in [(None, 5.0), (4.0, 4.0)]:
                with self.subTest(override=override), \
                        patch.object(ft, 'PowerGNN', return_value=model), \
                        patch.object(ft.torch, 'load', return_value={}), \
                        patch.object(ft, 'finetune') as train:
                    result = ft.run_finetune(
                        data_dir=root, base_checkpoint_path=root / 'base.pt',
                        checkpoint_template=str(root / 'checkpoint_round{}'),
                        current_round=2, finetune_weight=override,
                    )
                self.assertEqual(result, str(root / 'checkpoint_round2'))
                training = train.call_args.args[2]
                validation = train.call_args.args[3]
                replay, latest = training.datasets
                self.assertEqual(len(replay), 10)
                self.assertEqual(len(validation), 3)
                self.assertEqual(len(latest), 2)
                for index in range(len(latest)):
                    self.assertTrue(torch.all(latest[index].weight == expected_weight))
                for subset in [replay, validation]:
                    for index in range(len(subset)):
                        self.assertTrue(torch.all(subset[index].weight == 1.0))

    def test_new_round_weight_changes_its_gradient_contribution(self):
        prediction = torch.tensor([1.0, 1.0], requires_grad=True)
        weight = ft.resolve_finetune_weight(30, 5)
        ft.weighted_crit(prediction, torch.zeros(2), torch.tensor([1.0, weight])).backward()
        self.assertAlmostEqual(float(prediction.grad[1] / prediction.grad[0]), weight)


if __name__ == '__main__':
    unittest.main()
