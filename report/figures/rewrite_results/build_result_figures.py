"""Generate manuscript diagnostics directly from the saved development results."""

from pathlib import Path
import hashlib
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sc4000_methods_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output = Path(__file__).resolve().parent
    comparison_path = (
        root / "artifacts/temporal/temporal-performance-v2/gbdt_comparison_by_lab_action.csv"
    )
    importance_path = (
        root / "artifacts/phase8/gbdt-performance-v2/permutation_importance.parquet"
    )
    comparison = pd.read_csv(comparison_path)
    importance = (
        pl.read_parquet(importance_path)
        .group_by("group")
        .agg(
            pl.col("average_precision_drop").mean().alias("mean_ap_drop"),
            pl.len().alias("route_count"),
        )
    )
    values = {row["group"]: row for row in importance.to_dicts()}
    categories = [
        ("temporal", "Temporal"),
        ("metric", "Metric"),
        ("metadata", "Metadata"),
        ("quality", "Tracking quality"),
        ("egocentric", "Egocentric"),
        ("group", "Group context"),
        ("invariant", "Invariant"),
        ("boundary", "Boundary"),
    ]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(7, 2.8))
    left = fig.add_axes([0.075, 0.20, 0.325, 0.67])
    right = fig.add_axes([0.650, 0.20, 0.325, 0.67])

    left.scatter(
        comparison["gbdt_fbeta"], comparison["neural_fbeta"],
        s=17, marker="o", facecolors="0.35", edgecolors="0.15",
        linewidths=0.55, alpha=0.75, zorder=3,
    )
    left.plot([0, 1], [0, 1], "--", color="0.55", linewidth=0.8, zorder=1)
    left.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025),
             xlabel="Decoded tree F1", ylabel="Temporal model F1")
    left.set_xticks(np.arange(0, 1.01, 0.25))
    left.set_yticks(np.arange(0, 1.01, 0.25))
    left.set_title("(a) Laboratory–action results", fontsize=9, pad=9)

    drops = [values[key]["mean_ap_drop"] for key, _ in categories]
    labels = [f"{label} (n={values[key]['route_count']})" for key, label in categories]
    y = np.arange(len(categories))
    right.barh(y, drops, color="0.70", edgecolor="0.2", linewidth=0.55, height=0.65)
    right.set_yticks(y, labels, fontsize=7.7)
    right.invert_yaxis()
    right.set_xlim(0, 0.215)
    right.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    right.set_xlabel("Mean AP decrease after permutation")
    right.set_title("(b) Feature diagnostics", fontsize=9, pad=9)
    for i, drop in enumerate(drops):
        right.text(drop + 0.003, i, f"{drop:.4f}", va="center", fontsize=7.4)
    for boundary in [3.5, 5.5]:
        right.axhline(boundary, color="0.75", linestyle=":", linewidth=0.7)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5, length=3)
        ax.grid(axis="x", color="0.92", linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)

    fig.savefig(output / "diagnostics.png", dpi=360)
    fig.savefig(output / "diagnostics.svg")
    plt.close(fig)
    provenance = {
        "sources": {
            str(path.relative_to(root)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (comparison_path, importance_path)
        },
        "lab_action_comparisons": len(comparison),
        "win_counts": {str(k): int(v) for k, v in comparison["stronger_model"].value_counts().items()},
        "permutation_means": [values[key] for key, _ in categories],
        "notes": [
            "Saved lab-action comparison uses an outer merge and fills missing scores with zero.",
            "Permutation columns are grouped by route coverage, not globally ranked by importance.",
            "No uncertainty intervals are inferred from these summaries.",
        ],
    }
    (output / "figure_data.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(output / "diagnostics.png")
    print(json.dumps(provenance["win_counts"]))


if __name__ == "__main__":
    main()
