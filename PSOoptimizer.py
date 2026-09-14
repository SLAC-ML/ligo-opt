import os

from utils.calc import COST_MODE, calc_cost
from utils.sim import finesse_sim
from GNN.GNN_run import GNNPowerPredictor
from GNN.GNN_utils import model_to_nx_port, append_graph_to_h5, kat_manipulation
from perturbation import calc_perturbed_params
import numpy as np
from sko.PSO import PSO
from sko.tools import func_transformer


# Controls how many cavity evaluations are performed for each objective call.
#
# "nominal":   run only the design point D (1 GNN/Finesse evaluation). This mode
#              is valid only with utils.calc.COST_MODE == "design_power".
# "perturbed": run D plus ITM +/- Delta and ETM +/- Delta (5 evaluations total),
#              retaining the data needed by perturbation/full cost modes.
EVALUATION_MODE = "perturbed"

VALID_EVALUATION_MODES = {"nominal", "perturbed"}


def _power_ouput(ITM_Roc, ETM_Roc, q_value, kat, model_path=None, finetune_data_path=None, predictor_GNN=None):
    """Run the cavity (ITM_Roc, ETM_Roc) and return (names, powers, q_at_ITM).

    Builds a deep-copied KAT with the given ITM/ETM ROC and, when q_value is
    not None, forces the input beam's complex beam parameter to that fixed
    value via a `gauss fixed_q_value` command.

    q_at_ITM is the complex beam parameter q at the ITM input node ("ITM.p1.i")
    reported by finesse_sim:
      - For the nominal run (q_value=None) this is the cavity's own
        self-consistent eigenmode q — cost_fn grabs it here and re-applies it
        as the fixed q for the perturbed runs, avoiding a separate eigenmode
        solve.
      - For perturbed runs (fixed q_value) it is the forced q.
      - In the GNN surrogate path the predictor itself does not return q, so we
        pull the complex beam parameter at "ITM.p1.i" from the same beam-trace
        helper GNNPowerPredictor uses to build its node features
        (predictor_GNN._beam_q_values). That q is the design point's eigenmode q
        and is re-applied as the fixed q for the perturbed runs — reproducing the
        same q-mismatch the Finesse path and the training data (fabry_perot.py)
        encode.
    """
    run_kat = kat_manipulation(
        ITM_Roc, ETM_Roc, kat, nominal_q_value=q_value,
        include_aperture_maps=model_path is None,
    )

    if model_path is None:
        pd_names, pd_powers, q_names, q_values = finesse_sim(run_kat)
        if finetune_data_path is not None:
            append_graph_to_h5(model_to_nx_port(run_kat, pd_names, pd_powers, q_names, q_values), finetune_data_path)
        # finesse_sim labels the q detector on node "ITM.p1.i" as "q_ITM_p1_i".
        idx = q_names.index('q_ITM_p1_i')
        return pd_names, pd_powers, q_values[idx]
    else:
        names, powers = predictor_GNN.run(run_kat)
        # Mirror the finesse path: return the beam-trace q at the ITM input node so
        # the perturbed runs get a fixed q equal to the design point's eigenmode q.
        # Without this the perturbed GNN runs would use their own (mode-matched)
        # eigenmode q and the surrogate's gain cost would miss the q-mismatch
        # sensitivity that the Finesse path and the training data encode.
        # (For an unstable cavity _beam_q_values returns q = 0 everywhere, and
        # kat_manipulation treats a zero q the same as None — identical to the
        # finesse path's behaviour there.)
        qtrace_names, qtrace_values = predictor_GNN._beam_q_values(run_kat)
        qtrace_idx = qtrace_names.index('q_ITM_p1_i')
        return names, powers, qtrace_values[qtrace_idx]


def main_wrapper_PSO(base_kat, pd_name, model_path=None, finetune_data_path=None, overwrite=False):
    """
    Closure factory: bakes in the fixed simulation parameters (base_kat,
    pd_name, model_path) so the returned cost_fn only
    needs the PSO particle vector `params`. This is what gets passed to
    PSO(func=...) / pso.set_func(...).

    finetune_data_path : str, optional
        Only takes effect when model_path is None (i.e. cost_fn actually runs
        Finesse instead of the GNN surrogate). If given, every finesse_sim() result
        is also converted to a graph and appended to this h5 file, in the same
        format fabry_perot.py writes -- so it can be used directly as fine-tune
        round data (see finetune_power_predictor.py). In EVALUATION_MODE="nominal",
        each cost_fn call appends one design-point graph. In "perturbed" mode it
        appends 5 graphs: D, ITM +/- Delta, and ETM +/- Delta. Appending happens
        across every cost_fn call made with the
        cost_fn returned from this one main_wrapper_PSO call (e.g. every particle,
        every PSO iteration) -- that's intentional, it's how one PSO run's worth of
        data accumulates into a single round file.
    overwrite : bool, default False
        Only checked once, here, when the wrapper (and thus finetune_data_path) is
        set up -- not on every cost_fn call. If finetune_data_path already exists,
        raises FileExistsError instead of silently appending to it, since that file
        could be leftover from an earlier, unrelated run. Pass True to delete it and
        start fresh.
    """
    if EVALUATION_MODE not in VALID_EVALUATION_MODES:
        raise ValueError(
            f'Unknown EVALUATION_MODE {EVALUATION_MODE!r}; expected one of '
            f'{sorted(VALID_EVALUATION_MODES)}.'
        )
    if EVALUATION_MODE == "nominal" and COST_MODE != "design_power":
        raise ValueError(
            f'EVALUATION_MODE="nominal" cannot supply the perturbed powers required '
            f'by utils.calc.COST_MODE={COST_MODE!r}. Set EVALUATION_MODE="perturbed" '
            f'or COST_MODE="design_power".'
        )

    if finetune_data_path is not None and os.path.exists(finetune_data_path):
        if overwrite:
            os.remove(finetune_data_path)
        else:
            raise FileExistsError(
                f'{finetune_data_path} already exists. Pass overwrite=True to main_wrapper_PSO '
                f'to delete it and start fresh, otherwise cost_fn would silently append to '
                f'whatever is already in it.'
            )

    if model_path is not None:
        predictor_GNN = GNNPowerPredictor(model_path)
    else:
        predictor_GNN = None

    def cost_fn(params):
        # PSO calls this with a single particle's position vector.
        # n_dim=2 -> unpack into the two design variables being optimized.
        ITM_Roc, ETM_Roc = params

        # Nominal D run at the design point (ITM_Roc, ETM_Roc), solved as the
        # cavity's own eigenmode (no fixed q). finesse_sim returns the q at
        # ITM.p1.i for free, so grab it here and re-apply it as the fixed q
        # for the perturbed runs — no separate q-finding simulation needed.
        D_names, D_powers, q_value = _power_ouput(ITM_Roc, ETM_Roc, None, base_kat, model_path=model_path, finetune_data_path=finetune_data_path, predictor_GNN=predictor_GNN)

        if EVALUATION_MODE == "nominal":
            # calc_cost's existing interface expects three-entry ITM/ETM lists.
            # Repeating D is only an adapter: design_power reads the middle entry,
            # and the compatibility guard above prevents perturbation/full cost from
            # consuming these placeholders. No additional simulation is performed.
            nominal_power_list = [D_powers, D_powers, D_powers]
            nominal_name_list = [D_names, D_names, D_names]
            return calc_cost(
                None,
                nominal_power_list, nominal_power_list,
                nominal_name_list, nominal_name_list,
                (0.0, 0.0), pd_name,
            )

        # Generate perturbed ROC values around the current (ITM_Roc, ETM_Roc)
        # point, used for finite-difference-style sensitivity/gradient info.
        ITM_ROC_perturbed, ETM_ROC_perturbed, Delta = calc_perturbed_params(ITM_Roc, ETM_Roc)
        ITM_ROC_perturbed_pos, ITM_ROC_perturbed_neg = ITM_ROC_perturbed
        ETM_ROC_perturbed_pos, ETM_ROC_perturbed_neg = ETM_ROC_perturbed

        # ITM perturbed powers: only the ITM ROC moves (ETM stays at ETM_Roc),
        # with the fixed q_value from the nominal D run.
        ITM_names_p_pos, ITM_powers_p_pos, _ = _power_ouput(ITM_ROC_perturbed_pos, ETM_Roc, q_value, base_kat, model_path=model_path, finetune_data_path=finetune_data_path, predictor_GNN=predictor_GNN)
        ITM_names_p_neg, ITM_powers_p_neg, _ = _power_ouput(ITM_ROC_perturbed_neg, ETM_Roc, q_value, base_kat, model_path=model_path, finetune_data_path=finetune_data_path, predictor_GNN=predictor_GNN)

        # ETM perturbed powers: only the ETM ROC moves (ITM stays at ITM_Roc),
        # with the fixed q_value from the nominal D run.
        ETM_names_p_pos, ETM_powers_p_pos, _ = _power_ouput(ITM_Roc, ETM_ROC_perturbed_pos, q_value, base_kat, model_path=model_path, finetune_data_path=finetune_data_path, predictor_GNN=predictor_GNN)
        ETM_names_p_neg, ETM_powers_p_neg, _ = _power_ouput(ITM_Roc, ETM_ROC_perturbed_neg, q_value, base_kat, model_path=model_path, finetune_data_path=finetune_data_path, predictor_GNN=predictor_GNN)

        ITM_power_list = [ITM_powers_p_pos, D_powers, ITM_powers_p_neg]
        ITM_name_list = [ITM_names_p_pos, D_names, ITM_names_p_neg]
        ETM_power_list = [ETM_powers_p_pos, D_powers, ETM_powers_p_neg]
        ETM_name_list = [ETM_names_p_pos, D_names, ETM_names_p_neg]

        # Reuse a kat built at the design point D for the stability check
        # inside calc_cost — its cavity g-factor depends only on the mirror
        # geometry (Rc), not on the beam parameter, so the fixed-q model is
        # fine for that.
        find_q_kat = kat_manipulation(
            ITM_Roc, ETM_Roc, base_kat, nominal_q_value=q_value,
            include_aperture_maps=model_path is None,
        )

        cost = calc_cost(
            find_q_kat,
            ITM_power_list, ETM_power_list,
            ITM_name_list, ETM_name_list,
            Delta, pd_name
        )

        return cost

    # Only the surrogate path is vectorized. The Finesse feedback call remains a
    # scalar five-simulation evaluation so it continues to append exactly one round
    # of real graphs at the selected PSO optimum.
    if model_path is None:
        return cost_fn

    def batched_cost_fn(params_matrix):
        """Evaluate one PSO swarm in two GPU GNN batches.

        ``params_matrix`` has shape ``(n_particles, 2)``. First the nominal KAT for
        every particle is graph-built and inferred as one batch. Its individual q is
        then used to build that particle's four perturbed KATs, and all perturbations
        are inferred in a second batch. CPU KAT/beam-trace construction remains
        sequential; only independent PyG GNN forwards are batched.
        """
        params_matrix = np.asarray(params_matrix, dtype=float)
        if params_matrix.ndim != 2 or params_matrix.shape[1] != 2:
            raise ValueError(
                'Batched PSO objective requires params with shape (n_particles, 2); '
                f'got {params_matrix.shape}.'
            )
        if len(params_matrix) == 0:
            return np.empty(0, dtype=float)

        nominal_kats = [
            kat_manipulation(
                itm_roc, etm_roc, base_kat, nominal_q_value=None,
                include_aperture_maps=False,
            )
            for itm_roc, etm_roc in params_matrix
        ]
        nominal_results = predictor_GNN.run_batch(nominal_kats)

        if EVALUATION_MODE == "nominal":
            costs = []
            for names, powers, _ in nominal_results:
                power_list = [powers, powers, powers]
                name_list = [names, names, names]
                costs.append(
                    calc_cost(None, power_list, power_list, name_list, name_list,
                              (0.0, 0.0), pd_name)
                )
            return np.asarray(costs, dtype=float)

        # Keep each particle's four perturbations contiguous. This makes the output
        # reconstruction explicit and prevents q/particle associations from drifting.
        perturbation_kats = []
        perturbation_metadata = []
        for (itm_roc, etm_roc), (_, _, q_value) in zip(params_matrix, nominal_results):
            (itm_pos, itm_neg), (etm_pos, etm_neg), delta = calc_perturbed_params(
                itm_roc, etm_roc
            )
            perturbation_kats.extend([
                kat_manipulation(
                    itm_pos, etm_roc, base_kat, nominal_q_value=q_value,
                    include_aperture_maps=False,
                ),
                kat_manipulation(
                    itm_neg, etm_roc, base_kat, nominal_q_value=q_value,
                    include_aperture_maps=False,
                ),
                kat_manipulation(
                    itm_roc, etm_pos, base_kat, nominal_q_value=q_value,
                    include_aperture_maps=False,
                ),
                kat_manipulation(
                    itm_roc, etm_neg, base_kat, nominal_q_value=q_value,
                    include_aperture_maps=False,
                ),
            ])
            perturbation_metadata.append(delta)

        perturbation_results = predictor_GNN.run_batch(perturbation_kats)
        costs = []
        for index, ((d_names, d_powers, _), delta) in enumerate(
            zip(nominal_results, perturbation_metadata)
        ):
            itm_pos, itm_neg, etm_pos, etm_neg = perturbation_results[4 * index:4 * index + 4]
            itm_power_list = [itm_pos[1], d_powers, itm_neg[1]]
            itm_name_list = [itm_pos[0], d_names, itm_neg[0]]
            etm_power_list = [etm_pos[1], d_powers, etm_neg[1]]
            etm_name_list = [etm_pos[0], d_names, etm_neg[0]]
            costs.append(
                calc_cost(
                    None,
                    itm_power_list,
                    etm_power_list,
                    itm_name_list,
                    etm_name_list,
                    delta,
                    pd_name,
                )
            )
        return np.asarray(costs, dtype=float)

    # scikit-opt's func_transformer passes the full swarm matrix through unchanged
    # when this marker is present; PSO.cal_y then reshapes the returned (N,) costs.
    batched_cost_fn.mode = "vectorization"
    return batched_cost_fn

class SteppablePSO(PSO):
    def update_gbest(self):
        """Keep the best historical cost paired with its historical position.

        The installed scikit-opt implementation selects the index from ``pbest_y``
        but copies ``X[idx]``. Once that particle has moved, ``gbest_x`` therefore
        no longer corresponds to ``gbest_y``. The real-Finesse feedback point must
        be the same geometry whose surrogate cost won, so copy ``pbest_x`` here.
        """
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.pbest_x[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min].copy()

    def step(self):
        """Run exactly one PSO iteration (equivalent to one loop of run())."""
        self.update_V()
        self.recorder()
        self.update_X()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()
        self.gbest_y_hist.append(self.gbest_y)
        return self.gbest_x, self.gbest_y

    def set_func(self, func):
        """Swap the objective function between steps."""
        self.func = func_transformer(func)

# pso = SteppablePSO(func=main_wrapper_PSO, n_dim=2, pop=300, max_iter=1000, lb=[start_ETM_Roc, start_ITM_Roc], ub=[end_ETM_Roc, end_ITM_Roc], w=0.9, c1=2.0, c2=0.4, verbose=True)

# for i in range(150):
#     if i == 75:
#         pso.set_func(cost_fn_b)   # change cost function mid-run
#     best_x, best_y = pso.step()

# print("Best parameters found: ", pso.gbest_x)
# print("Best cost found: ", pso.gbest_y)