"""Create standalone diagnostic figures from saved experiment predictions."""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import repo_path


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_directory(directory):
    rows = read_rows(directory / "predictions.csv")
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    for split in sorted({r["split"] for r in rows}):
        selected = [r for r in rows if r["split"] == split]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for direction in sorted({r["direction"] for r in selected}):
            subset = [r for r in selected if r["direction"] == direction]
            axes[0].scatter([float(r["power"]) for r in subset],
                            [float(r["predicted_power"]) for r in subset], s=18, label=direction)
            if direction != "nominal":
                axes[1].scatter([float(r["delta_power"]) for r in subset],
                                [float(r["predicted_delta_power"]) for r in subset], s=18, label=direction)
        for ax, label in zip(axes, ("Power (W)", "Signed ΔPower (W)")):
            limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(limits, limits, "k--", lw=1)
            ax.set(xlabel=f"Finesse {label}", ylabel=f"MLP {label}")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{directory.name}: {split}")
        fig.savefig(output / f"{split}_parity.png", dpi=160)
        plt.close(fig)
        # Real sampled response, without presenting interpolation as simulator truth.
        full_groups = [gid for gid in dict.fromkeys(r["group_id"] for r in selected)
                       if any(r["group_id"] == gid and r["direction"] == "ETM_minus" for r in selected)]
        for gid in full_groups[:8]:
            subset = [r for r in selected if r["group_id"] == gid]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
            for ax, mirror in zip(axes, ("ITM", "ETM")):
                points = sorted([r for r in subset if r["direction"] == "nominal" or r["direction"].startswith(mirror)],
                                key=lambda r: float(r[f"delta_{mirror.lower()}_roc"]))
                dx = [float(r[f"delta_{mirror.lower()}_roc"]) for r in points]
                ax.plot(dx, [float(r["delta_power"]) for r in points], "o--", label="Finesse samples")
                ax.plot(dx, [float(r["predicted_delta_power"]) for r in points], "o--", label="MLP samples")
                ax.set(xlabel=f"Δ{mirror} ROC (m)", ylabel="ΔPower (W)")
                ax.grid(alpha=.25)
                ax.legend()
            fig.suptitle(f"{gid}, fixed input q; lines connect three sampled points")
            fig.savefig(output / f"{split}_{gid.replace(':', '_')}_response.png", dpi=160)
            plt.close(fig)
    history = directory / "training_history.csv"
    if history.exists():
        values = read_rows(history)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        for ax, metric in zip(axes, ("power_mae", "delta_mae")):
            for prefix in ("train", "monitor"):
                ax.plot([int(r["epoch"]) for r in values],
                        [float(r[f"{prefix}_{metric}"]) for r in values], label=prefix)
            ax.set(xlabel="Epoch", ylabel=f"{metric} (W)", yscale="symlog")
            ax.legend()
            ax.grid(alpha=.25)
        fig.savefig(output / "training_history.png", dpi=160)
        plt.close(fig)
    print(f"Figures: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Experiment root or individual variant directory.")
    args = parser.parse_args()
    root = repo_path(args.experiment)
    directories = [root] if (root / "predictions.csv").exists() else sorted(p.parent for p in root.glob("*/predictions.csv"))
    if not directories:
        raise FileNotFoundError(f"No predictions.csv under {root}")
    for directory in directories:
        plot_directory(directory)
    summary_path = root / "summary.csv"
    if summary_path.exists():
        rows = read_rows(summary_path)
        splits = sorted({r["split"] for r in rows})
        fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4), squeeze=False, constrained_layout=True)
        for ax, split in zip(axes[0], splits):
            variants = sorted({r["variant"] for r in rows})
            for index, variant in enumerate(variants):
                values = [float(r["delta_mae"]) for r in rows if r["split"] == split and r["variant"] == variant]
                ax.scatter(np.full(len(values), index), values)
            ax.set(xticks=range(len(variants)), xticklabels=variants, ylabel="ΔPower MAE (W)", title=split)
            ax.grid(axis="y", alpha=.25)
        fig.savefig(root / "loss_comparison.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
