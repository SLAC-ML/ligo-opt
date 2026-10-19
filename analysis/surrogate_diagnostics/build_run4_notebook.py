"""Adapt the run3 diagnostic notebook and optionally execute it using IPython.

Run with /home/xuesi.ma/.conda/envs/ligoopt/bin/python and --execute to refresh
the saved run4 notebook, including rich tables and PNG figures. No training or
simulation runs are launched. IPython supplies notebook output capture without
requiring nbclient, nbformat, or a separate Jupyter kernel installation.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
TARGET = HERE / "results_overview_run4.ipynb"


def source(cell):
    return "".join(cell["source"])


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def make_cell(kind, text, cell_id):
    cell = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


PRODUCTION_INTRO = """## Production base-training trajectories

These curves read the original production checkpoint histories. Run 4 uses
`nominal_delta`: nominal log-power loss plus a signed perturbation-difference
term. The small diagnostic `paired` fit above retains the older row log-power
plus difference objective, so its loss components and selected epoch have a
different interpretation.

Production history and the diagnostic `validation` split cover all **20 held-out
initial groups**; `validation_valid_fixed_q` retains **15 valid fixed-q groups**. The
initial geometric designs and split membership are matched between runs 3 and
4. Saved run 3 histories omit nominal/delta diagnostics: those values remain
missing here, rather than being inferred or filled with zero. Vertical lines
mark the lowest saved validation-loss epoch for each production history.
"""


PRODUCTION_CODE = """production_history_dir = OUTPUTS_DIR / 'run4_training_audit'
production_history_path = production_history_dir / 'base_training_history.csv'
if not production_history_path.exists():
    production_history_path = production_history_dir / 'training_history.csv'
production_history = read_csv(production_history_path)
table(production_history, title='Original production base-training history', collapsed=True)
production_best_rows = []
if production_history:
    production_run_numbers = sorted({r['run'] for r in production_history})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for index, run_number in enumerate(production_run_numbers):
        history = sorted([r for r in production_history if r['run'] == run_number],
                         key=lambda row: row['epoch'])
        color, label = f'C{index % 10}', f'run {run_number}'
        epoch = values(history, 'epoch')
        monitored = [r for r in history if r.get('validation_loss') is not None]
        best = min(monitored, key=lambda r: r['validation_loss']) if monitored else None
        if best:
            production_best_rows.append(best)
        panels = [
            (axes[0, 0], [('training_loss', 'train', '-'),
                           ('validation_loss', 'validation', '--')], 'Production total loss'),
            (axes[0, 1], [('power_mae', 'all-row power', '-'),
                           ('center_power_mae', 'center-band power', '--')], 'Validation power MAE (W)'),
            (axes[0, 2], [('nominal_power_mae', 'nominal power', '-')], 'Validation nominal MAE (W)'),
            (axes[1, 0], [('delta_mae', 'mean', '-'),
                           ('delta_max_abs_error', 'maximum', '--')], 'Validation signed ΔPower error (W)'),
            (axes[1, 1], [('training_nominal_loss', 'nominal', '-'),
                           ('training_delta_loss', 'signed delta', '--')], 'Production training loss components'),
            (axes[1, 2], [('perturbation_cost_mae', 'four-direction cost', '-')], 'Validation cost MAE (W/m)'),
        ]
        for ax, fields, ylabel in panels:
            for field, component, style in fields:
                y = values(history, field)
                if np.isfinite(y).any():
                    ax.plot(epoch, y, style, color=color, label=f'{label} · {component}')
            if best and ax.lines:
                ax.axvline(best['epoch'], color=color, ls=':', alpha=0.55)
            ax.set(xlabel='Epoch', ylabel=ylabel)
            ax.set_yscale('symlog', linthresh=1e-5)
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No saved diagnostic values', ha='center', va='center',
                    transform=ax.transAxes)
    fig.suptitle('Original production training · 20 held-out initial groups')
    show_figure(fig, 'production_base_training_run3_run4')
    table(production_best_rows, title='Lowest saved production validation-loss epoch per run')
else:
    display(Markdown('No saved production base-training history is available.'))
"""


def build_notebook():
    notebook = json.loads((HERE / "results_overview_run3.ipynb").read_text())
    for cell in notebook["cells"]:
        text = source(cell)
        for old, new in [("run3", "run4"), ("run 3", "run 4"),
                         ("Run 3", "Run 4"), ("RUN3", "RUN4"),
                         ("run2", "run3"), ("run 2", "run 3"),
                         ("Run 2", "Run 3"), ("RUN2", "RUN3")]:
            text = text.replace(old, new)
        set_source(cell, text)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    cells = notebook["cells"]
    set_source(cells[0], """# Surrogate diagnostics: run 4 results overview

Run all cells using the Python kernel from **ligoopt** to refresh this report
from saved files in `analysis/surrogate_diagnostics/outputs/`. Open the notebook
from the repository root or this directory. Refreshing reads CSV/JSON artifacts
and does not train models or launch Finesse simulations.

This report covers data and inference-feature audits, the matched initial
designs and held-out validation, original production training, online PSO
behavior, checkpoint replay, five-group reconstruction and transfer, signed
perturbation responses, and source provenance.

Run 4 production uses `nominal_delta`, combining nominal log-power and signed
power-difference terms. The diagnostic `paired` tiny-fit experiment retains
the older row log-power plus difference objective. Tiny-fit scores measure
training reconstruction; feedback checkpoint replay includes prior online
training exposure and does not establish held-out generalization.

Only run 4 artifact folders are included. Run 3 checkpoints remain reference
baselines within matched run 4 comparisons. Open the [run 3 notebook](results_overview_run3.ipynb)
for the earlier analysis, or the [run 4 findings overview](RUN4_FINDINGS.html)
for the concise report. Figures can optionally be exported as PNG files.
""")
    set_source(cells[1], source(cells[1]).replace("RUN_NUMBER = 3", "RUN_NUMBER = 4"))
    set_source(cells[3], source(cells[3]).replace(
        "# Initial datasets differ between runs; give each original split its own\n    # experiment instead of combining nominal/joint and five-point labels.",
        "# Preserve each production checkpoint's saved initial split metadata\n    # in a separate experiment, even when initial designs are matched.",
    ))
    set_source(cells[5], """## Data coverage and label audit

The audits include invalid fixed-q groups and zero-output rows. A zero label
alone cannot distinguish an unstable cavity from a solver or lock failure.
Runs 3 and 4 share initial geometric designs, each recorded as five-row
nominal/ITM+/ITM-/ETM+/ETM- groups. Their online feedback visits different
designs. The center band comes from each saved config.

All ROC sample-coverage panels use the full `run4_audit` ITM/ETM axis ranges.
Nominal q coverage panels likewise share its nominal Re(q)/Im(q) axis ranges,
so selected feedback designs can be compared with the complete coverage.
""")
    set_source(cells[9], """## Run 4: matched designs, held-out validation, and online optimization

The [written findings](RUN4_FINDINGS.md) and [HTML overview](RUN4_FINDINGS.html)
separate the matched initial-design comparison, held-out validation, original
online PSO decisions, and later checkpoint replay. Runs 3 and 4 reuse the same
initial geometric designs and production split membership. They change the
production training objective and subsequently visit different online designs;
this is a matched-start comparison rather than an isolated causal experiment.

`base_on_initial` preserves each model's original split metadata. Both the full
`validation` cohort of 20 groups and its `validation_valid_fixed_q` subset of
15 groups appear in the diagnostic tables. Replay on run 4 feedback asks how
saved checkpoints reproduce those labels, including feedback seen during
training; original online logs record the predictions available at the time.
""")
    set_source(cells[11], """## Experiment conditions and comparable results

Compare checkpoints on the same saved geometry, detector, and split. Fresh
five-group fits use local scaling; warm fits reuse the run 4 round-40 checkpoint
and its scaling. Both initialization and normalization change, so their
difference cannot be assigned to either factor alone. These diagnostic loss
variants retain the older `paired` row log-power plus delta implementation;
they do not reproduce production `nominal_delta` training.

Transfer uses the remaining feedback groups outside the five fitted groups.
Those labels are unseen by the tiny-fit stage; a warm checkpoint can already
carry exposure through its earlier production training. The saved conditions,
checkpoint metadata, and source hashes below make that scope explicit.
""")
    set_source(cells[10], source(cells[10]).replace(
        "    table(read_csv(comparison_path.parent / 'online_rounds.csv'),",
        "    table(read_csv(comparison_path.parent / 'response_baselines.csv'),\n"
        "          title=f'{name}: zero-response baselines on the same saved labels')\n"
        "    table(read_csv(comparison_path.parent / 'online_rounds.csv'),",
    ))
    set_source(cells[16], """## Diagnostic fitting trajectories and selected checkpoints

Solid curves show training; dashed curves show the monitored split. Vertical
dotted lines mark the selected epoch. These are the five-group diagnostic fits;
the production checkpoint history is plotted separately below. The diagnostic
`paired` loss combines per-row log-power loss with signed-difference loss.
""")
    set_source(cells[20], """## Four-direction perturbation cost

The `costs.json` tables contain **sum |DeltaPower / DeltaROC|**, in W/m,
for complete five-point groups, including both runs' initial designs. The
plots here show the perturbation term. The run-comparison section separately
reconstructs weighted nominal and total production costs from saved settings.
Unscorable entries remain visible as missing values.
""")
    response_source = source(cells[23]).replace(
        "label = tags['variant'] + (f' / {tags[\"seed\"]}' if tags['seed'] is not None else '')",
        "label = Path(run_id).name if tags['variant'] == 'checkpoint' else (tags['variant'] + (f' / {tags[\"seed\"]}' if tags['seed'] is not None else ''))",
    ).replace("columns=['variant', 'seed', 'direction'", "columns=['run_id', 'variant', 'seed', 'direction'")
    set_source(cells[23], response_source)
    # The explicit checkpoint folder also distinguishes labels in response tables.
    cells[18:18] = [make_cell("markdown", PRODUCTION_INTRO, "run4-production-intro"),
                    make_cell("code", PRODUCTION_CODE, "run4-production-history")]
    notebook["metadata"]["language_info"]["version"] = sys.version.split()[0]
    notebook["metadata"]["run4_refresh"] = {
        "source_notebook": "results_overview_run3.ipynb",
        "python": sys.executable,
        "command": "/home/xuesi.ma/.conda/envs/ligoopt/bin/python analysis/surrogate_diagnostics/build_run4_notebook.py --execute",
    }
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            ast.parse(source(cell), filename=f"run4 cell {index}")
    return notebook


def execute_notebook(notebook):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.display import Image, display
    from IPython.utils.capture import capture_output

    shell = InteractiveShell.instance()

    def show_figures(*args, **kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight")
            display(Image(data=buffer.getvalue()))

    plt.show = show_figures
    execution_count = 0
    started = time.monotonic()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        print(f"Executing cell {index} ({execution_count})", flush=True)
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(source(cell), store_history=True)
        cell["execution_count"] = execution_count
        outputs = []
        for name, content in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
            if content:
                outputs.append({"output_type": "stream", "name": name, "text": content})
        for output in captured.outputs:
            data = {mime: base64.b64encode(value).decode("ascii") if isinstance(value, bytes) else value
                    for mime, value in output.data.items()}
            outputs.append({"output_type": "display_data", "data": data,
                            "metadata": output.metadata or {}})
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            outputs.append({"output_type": "error", "ename": type(error).__name__,
                            "evalue": str(error), "traceback": [str(error)]})
        cell["outputs"] = outputs
        if error is not None:
            TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
            raise RuntimeError(f"Notebook cell {index} failed: {error}") from error
    print(f"Executed {execution_count} code cells in {time.monotonic() - started:.1f}s", flush=True)
    notebook["metadata"]["run4_refresh"]["executed_code_cells"] = execution_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute every code cell and save rich outputs.")
    args = parser.parse_args()
    notebook = build_notebook()
    if args.execute:
        execute_notebook(notebook)
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
