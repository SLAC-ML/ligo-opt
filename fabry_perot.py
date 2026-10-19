"""Generate initial training graphs from sampled Fabry-Perot cavities.

The generators module supplies optimized-LHS design geometries. This module
simulates each design and four single-mirror perturbations with the nominal beam
parameter held fixed, then saves five graphs per design to an HDF5 dataset.
"""


import csv
from pathlib import Path

import h5py
import finesse
import perturbation
from generators import optimized_lhs_roc_samples

from GNN.GNN_utils import model_to_nx_port, append_graph_to_h5, kat_manipulation
from tqdm import tqdm
from utils.finesse_base import base_kat
from oracles import finesse_sim
from utils.training_groups import INITIAL_GROUP_SIZE, INITIAL_SAMPLE_TYPES


finesse.init_plotting(fmts=["png"])


def generate_initial_data(
    output_path,
    num_design_points=5,
    itm_bounds=(-4134, -1434),
    etm_bounds=(1745, 4245),
    overwrite=False,
    seed=9302,
):
    """Generate optimized-LHS design graphs and their fixed-q perturbations.

    Every LHS design point contributes five consecutive HDF5 graphs: its nominal
    cavity followed by ITM +/-, then ETM +/- with the nominal q held fixed. The
    perturbations use the cost's helper and may extend beyond the nominal sampling
    bounds. The output contains exactly ``5 * num_design_points`` graphs, with
    group size recorded in its attributes. A sibling ``initial_samples.csv``
    records sample types, zero-based design IDs, HDF5 keys, and ROCs in metres.
    """
    roc_samples = optimized_lhs_roc_samples(
        num_design_points,
        itm_bounds,
        etm_bounds,
        seed=seed,
    )
    total_graphs = INITIAL_GROUP_SIZE * int(num_design_points)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path = output_path.with_name('initial_samples.csv')
    for path in (output_path, samples_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f'{path} already exists. Pass overwrite=True to regenerate it.'
            )
    if output_path.exists():
        output_path.unlink()
    with h5py.File(output_path, 'w') as handle:
        handle.attrs['design_group_size'] = INITIAL_GROUP_SIZE

    print(
        f'Generating {num_design_points} optimized-LHS design points across the '
        f'full ROC space. Each point adds one design graph and four fixed-q '
        f'perturbation graphs ({total_graphs} total graphs, seed={seed}).'
    )

    valid_q_designs = 0
    with (
        samples_path.open('w' if overwrite else 'x', newline='', encoding='utf-8') as stream,
        tqdm(total=total_graphs, position=0, desc="initial sims") as pbar,
    ):
        writer = csv.writer(stream)
        writer.writerow(['design_id', 'sample_type', 'h5_key', 'itm_roc_m', 'etm_roc_m'])
        for design_id, (ITM_ROC, ETM_ROC) in enumerate(roc_samples):
            design_kat = kat_manipulation(
                ITM_ROC,
                ETM_ROC,
                base_kat,
                nominal_q_value=None,
            )
            pd_names, powers, q_names, q_values = finesse_sim(design_kat)
            graph = model_to_nx_port(
                design_kat, pd_names, powers, q_names, q_values
            )
            append_graph_to_h5(graph, output_path)
            group_start = INITIAL_GROUP_SIZE * design_id
            writer.writerow([design_id, 'design', f'sim_{group_start}', ITM_ROC, ETM_ROC])
            stream.flush()
            pbar.update(1)

            design_q_value = q_values[q_names.index('q_ITM_p1_i')]
            fixed_q = None if design_q_value == 0 else design_q_value
            valid_q_designs += fixed_q is not None
            (itm_plus, itm_minus), (etm_plus, etm_minus), _ = perturbation.calc_perturbed_params(
                ITM_ROC, ETM_ROC,
            )
            geometries = (
                (itm_plus, ETM_ROC), (itm_minus, ETM_ROC),
                (ITM_ROC, etm_plus), (ITM_ROC, etm_minus),
            )
            for offset, (sample_type, (perturbed_ITM_ROC, perturbed_ETM_ROC)) in enumerate(
                zip(INITIAL_SAMPLE_TYPES[1:], geometries), start=1,
            ):
                perturbed_kat = kat_manipulation(
                    perturbed_ITM_ROC, perturbed_ETM_ROC, base_kat,
                    nominal_q_value=fixed_q,
                )
                pd_names, powers, q_names, q_values = finesse_sim(perturbed_kat)
                graph = model_to_nx_port(
                    perturbed_kat, pd_names, powers, q_names, q_values
                )
                append_graph_to_h5(graph, output_path)
                writer.writerow([
                    design_id, sample_type, f'sim_{group_start + offset}',
                    perturbed_ITM_ROC, perturbed_ETM_ROC,
                ])
                stream.flush()
                pbar.update(1)

    print(
        f'Generated {total_graphs} graphs. {valid_q_designs}/{num_design_points} '
        'designs supplied a valid fixed q to their four perturbations; groups whose '
        'design cavity was unstable/unsolvable were retained as zero-power data.'
    )

    return output_path


if __name__ == '__main__':
    generate_initial_data('GNN/data/run6/base_25.h5', num_design_points=5)
