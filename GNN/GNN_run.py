from GNN.power_predictor import LinGNN as PowerGNN
from GNN.GNN_utils import model_to_nx_port

import torch
import torch_geometric as pyg

class GNNPowerPredictor:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

    def _load_model(self):
        model = PowerGNN(hidden_size=1000, num_layers=10, lin_layers=5, target_size=1)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _beam_q_values(kat):
        """Per-node complex beam parameters, populated exactly the way the training
        data was generated (see fabry_perot.py / oracles.py):

        - Stable cavity: run a cheap ABCD beam trace (no matrix/knm solve) and read
          the q at every node. This reproduces the q_re/q_im features the model was
          trained on.
        - Unstable / untraceable cavity: leave q = 0 for every node, matching how
          finesse_sim labels unstable cavities (all zeros). This is what lets the
          model learn "unstable cavity -> everything 0".
        """
        q_names = [f"q_{node.replace('.', '_')}" for node in kat.optical_network.nodes()]
        q_values = [0] * len(q_names)
        try:
            # An unstable cavity must stay all-zero even when a `gauss` command is
            # present: beam_trace would otherwise happily return finite q there,
            # which is NOT what the training data looks like.
            if not kat.cavArm.is_stable:
                return q_names, q_values

            trace = kat.beam_trace()
            for i, node in enumerate(kat.optical_network.nodes()):
                entry = trace.get(node)
                if entry is not None:
                    q_values[i] = complex(entry.qx)
        except Exception:
            # Any cavity that cannot be traced is treated as unsolvable -> all zeros.
            pass
        return q_names, q_values

    def _prepare_data(self, kat):
        """Create one inference graph and retain its design-point input q.

        KAT construction, beam tracing, and NetworkX conversion remain CPU-side.
        ``run_batch`` batches the resulting PyG graphs, which is the GPU-bound part
        of a PSO objective evaluation.
        """
        q_names, q_values = self._beam_q_values(kat)
        graph = model_to_nx_port(kat, q_names=q_names, q_values=q_values)
        names = [f"p_{node.replace('.', '_')}" for node, _ in graph.nodes(data=True)]
        data = pyg.utils.from_networkx(
            graph,
            group_node_attrs=['Rc', 'R', 'q_re', 'q_im'],
            group_edge_attrs=['length', 'nr'],
        )
        data.x = torch.nan_to_num(data.x, posinf=0).float()
        data.edge_attr = torch.nan_to_num(data.edge_attr, posinf=0).float()
        q_at_itm = q_values[q_names.index('q_ITM_p1_i')]
        return data, names, q_at_itm

    def run_batch(self, kats):
        """Prepare KAT states sequentially, then predict in one PyG GPU batch.

        ``kats`` may be an iterator yielding successive states of a reused model.
        Finish _prepare_data for each state before advancing the iterator: only
        the independent graph tensors may be collected, never the KAT references.

        Results are ``(names, powers, q_at_ITM)`` tuples in input order. The graphs
        remain disconnected in the PyG batch, so message passing cannot cross PSO
        particle boundaries. The returned q keeps each nominal particle associated
        with only its own four q-mismatched perturbations.
        """
        prepared = [self._prepare_data(kat) for kat in kats]
        if not prepared:
            return []

        batch = pyg.data.Batch.from_data_list([item[0] for item in prepared])
        batch = batch.to(self.device, non_blocking=True)
        with torch.no_grad():
            output = self.model(batch)
        output = torch.expm1(output).squeeze(-1).cpu()

        node_offsets = batch.ptr.cpu().tolist()
        results = []
        for index, (_, names, q_at_itm) in enumerate(prepared):
            powers = output[node_offsets[index]:node_offsets[index + 1]].tolist()
            if len(names) != len(powers):
                raise RuntimeError(
                    f'Batched GNN output size mismatch for graph {index}: '
                    f'{len(names)} names but {len(powers)} powers.'
                )
            results.append((names, powers, q_at_itm))
        return results

    def run(self, kat):
        """Scalar compatibility wrapper around :meth:`run_batch`."""
        names, powers, _ = self.run_batch([kat])[0]
        return names, powers
