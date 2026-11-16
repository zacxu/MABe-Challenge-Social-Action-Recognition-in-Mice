"""Generate manuscript figures directly from the frozen performance-v2 reports.

Run from the project root with ``.venv/Scripts/python.exe
report/generate_phase6_13_figures.py``. All plots are exported as 350-dpi PNGs
and vector PDFs with embedded TrueType fonts. No training or OOF rescoring is
performed by this script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "figures" / "phase6_13"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "_mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 350,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SOURCE_PATHS = {
    "phase7": ROOT / "artifacts/xgb/xgb-performance-v2/cv_report.json",
    "phase8": ROOT / "artifacts/phase8/gbdt-performance-v2/phase8_report.json",
    "phase9": ROOT / "artifacts/temporal/temporal-performance-v2/phase9_report.json",
    "phase11": ROOT / "artifacts/phase11/ensemble-performance-v2/phase11_report.json",
}


def read_reports() -> dict:
    return {key: json.loads(path.read_text(encoding="utf-8")) for key, path in SOURCE_PATHS.items()}


def export(fig: plt.Figure, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.07, facecolor="white")
        paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    plt.close(fig)
    return paths


def box(ax, x, y, w, h, text, *, shade="0.97", fontsize=9.3, dashed=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.9,
        edgecolor="0.25",
        facecolor=shade,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.12)
    return patch


def arrow(ax, start, end, *, style="arc3,rad=0", dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            linewidth=0.9, color="0.25", connectionstyle=style,
            linestyle="--" if dashed else "-",
        )
    )


def architecture_figure(reports: dict) -> list[str]:
    architecture = reports["phase9"]["architecture_config"]
    assert architecture["family"] == "conv_attention"
    assert architecture["hidden_channels"] == 128
    assert architecture["stem_kernel_sizes"] == [3, 5, 7, 9]
    assert architecture["tcn_dilations"] == [1, 2, 4, 8, 16, 32]
    assert architecture["attention_layers"] == 4 and architecture["attention_heads"] == 8
    assert architecture["use_bigru"] is False
    fig = plt.figure(figsize=(6.8, 7.5))
    top = fig.add_axes([0.03, 0.51, 0.94, 0.45])
    top.set(xlim=(0, 1), ylim=(0, 1))
    top.axis("off")
    top.text(0.0, 1.02, "(A) Feature, model, ensemble, and deployment pipeline", fontsize=11, fontweight="bold")
    box(top, 0.12, 0.90, 0.76, 0.07, "Pose tracks, laboratory metadata, and annotation masks", shade="0.90")
    box(top, 0.02, 0.70, 0.44, 0.15, "Phase 6: engineered feature cache\nsolo/pair geometry and kinetics,\nrolling context, validity, and metadata", fontsize=9.1)
    box(top, 0.54, 0.70, 0.44, 0.15, "Phase 9: normalized sequences\n30 Hz; 512-frame windows\nsolo: 23 channels; pair: 54 channels", fontsize=9.1)
    arrow(top, (0.30, 0.895), (0.24, 0.86))
    arrow(top, (0.70, 0.895), (0.76, 0.86))
    box(top, 0.02, 0.49, 0.44, 0.15, "Phases 7–8: XGBoost and calibration\nvideo-wise CV; lab/action routes\ncross-fitted interval parameters", fontsize=9.1)
    box(top, 0.54, 0.49, 0.44, 0.15, "Phase 9: convolution–attention\nindependent solo and pair encoders\nframe probabilities from five folds", fontsize=9.1)
    arrow(top, (0.24, 0.693), (0.24, 0.65))
    arrow(top, (0.76, 0.693), (0.76, 0.65))
    top.text(0.76, 0.445, "Phase 10: v1 screens; no model retained", ha="center", fontsize=8.7, color="0.3")
    box(top, 0.08, 0.265, 0.84, 0.13, "Phase 11: cross-fitted probability ensemble\nnonnegative action-wise weights with shrinkage\nnested thresholds, gap filling, and interval conflict resolution", shade="0.88", fontsize=9.1)
    arrow(top, (0.24, 0.482), (0.24, 0.405))
    arrow(top, (0.76, 0.433), (0.76, 0.405))
    box(top, 0.02, 0.025, 0.44, 0.15, "Phase 12: offline inference\nfold averaging; streaming and recovery\nsubmission validation and frozen assets", fontsize=9.1)
    box(top, 0.54, 0.025, 0.44, 0.15, "Phase 13: release-audit framework\nhashes, interval parity, and rescoring\nperformance-v2 release audit pending", dashed=True, fontsize=9.1)
    arrow(top, (0.24, 0.257), (0.24, 0.183))
    arrow(top, (0.47, 0.10), (0.53, 0.10))

    bottom = fig.add_axes([0.03, 0.025, 0.94, 0.445])
    bottom.set(xlim=(0, 1), ylim=(0, 1))
    bottom.axis("off")
    bottom.text(0.0, 1.02, "(B) Temporal architecture (same design, separate weights)", fontsize=11, fontweight="bold")
    box(bottom, 0.10, 0.87, 0.80, 0.10, "Solo input: B × 512 × 23    |    Directed-pair input: B × 512 × 54", shade="0.90", fontsize=9.5)
    arrow(bottom, (0.50, 0.862), (0.50, 0.83))
    box(bottom, 0.06, 0.675, 0.56, 0.15, "Multi-scale Conv1D: kernels 3, 5, 7, 9\n4 × 32 → concatenate → 1 × 1 → 128\nGroupNorm + GELU", fontsize=9.1)
    box(bottom, 0.70, 0.675, 0.28, 0.15, "Lab + metadata MLP\n16 + 3 → 64 → 128\nplus lab adapter 16 → 128", fontsize=8.8)
    arrow(bottom, (0.696, 0.752), (0.635, 0.752))
    box(bottom, 0.06, 0.455, 0.56, 0.15, "Six residual depthwise TCN blocks\nkernel 3; dilations 1, 2, 4, 8, 16, 32\npointwise + GroupNorm + GELU; dropout 0.25", fontsize=9.0)
    arrow(bottom, (0.34, 0.667), (0.34, 0.615))
    box(bottom, 0.70, 0.455, 0.28, 0.15, "Convolution support\n135 frames ≈ 4.5 s\nGroupNorm spans window", fontsize=8.8)
    box(bottom, 0.06, 0.235, 0.56, 0.15, "Four pre-LayerNorm attention blocks\n8 heads (16-dim); attention/FFN residuals\nFFN: 128 → 512 → 128", fontsize=9.2)
    arrow(bottom, (0.34, 0.446), (0.34, 0.395))
    box(bottom, 0.70, 0.235, 0.28, 0.15, "Full-window attention\n512 frames = 17.07 s\nBiGRU disabled", fontsize=8.9)
    box(bottom, 0.06, 0.005, 0.92, 0.185, "Linear heads: 128 → 11 solo actions and 128 → 26 pair actions\nvalidity-masked logits → sigmoid probabilities\nweighted overlap aggregation; stride 128, edge taper 32\n1,915,749 trainable parameters across both branches", shade="0.88", fontsize=9.2)
    arrow(bottom, (0.34, 0.225), (0.34, 0.20))
    return export(fig, "figure1_pipeline_architecture")


def model_comparison_figure(reports: dict) -> tuple[list[str], list[dict]]:
    scores = reports["phase11"]["candidate_scores"]
    items = [
        ("Phase 7: XGBoost", reports["phase7"]["overall_fbeta"], "base"),
        ("Phase 8: calibrated GBDT", reports["phase8"]["final_overall_fbeta"], "base"),
        ("Phase 9: temporal model", reports["phase9"]["neural_overall_fbeta"], "base"),
        ("Phase 11: equal blend", scores["phase11-equal-gbdt-phase9"], "candidate"),
        ("Phase 11: action blend (selected)", scores["phase11-crossfit-action"], "selected"),
        ("Phase 11: global blend", scores["phase11-crossfit-global"], "candidate"),
        ("Phase 11: lab blend", scores["phase11-crossfit-lab"], "candidate"),
        ("Phase 11: lab/action blend", scores["phase11-crossfit-lab-action"], "candidate"),
        ("Phase 11: per-lab selector", scores["phase11-crossfit-per-lab-selector"], "candidate"),
        ("Phase 11: probability stacker", scores["phase11-crossfit-stacker"], "candidate"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    fig.subplots_adjust(left=0.425, right=0.98, top=0.94, bottom=0.28)
    y = np.arange(len(items))
    for index, (_, value, status) in enumerate(items):
        facecolor = "0.18" if status == "selected" else "0.57" if status == "base" else "0.88"
        ax.barh(index, value, height=0.66, color=facecolor, edgecolor="0.25", linewidth=0.7, hatch="///" if status == "candidate" else None)
        ax.text(value + 0.004, index, f"{value:.5f}", va="center", ha="left", fontsize=9)
    ax.set_yticks(y, [item[0] for item in items])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.53)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_xlabel("Exact competition-metric OOF F1")
    ax.xaxis.grid(True, color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    fig.text(0.425, 0.99, "Model and ensemble comparison", fontsize=11, fontweight="bold", va="top")
    legend = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.18", edgecolor="0.25", label="Selected deployment"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.88", edgecolor="0.25", hatch="///", label="Unselected diagnostic"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.51, 0.075), frameon=False, ncol=2, fontsize=9)
    fig.text(0.425, 0.024, "Five video folds; 106,642,451 OOF rows; no uncertainty bars.", fontsize=8.7)
    records = [{"label": label, "exact_oof_f1": value, "status": status} for label, value, status in items]
    return export(fig, "figure2_exact_oof_comparison"), records


def ablation_figure(reports: dict) -> list[str]:
    phase8 = reports["phase8"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.5))
    fig.subplots_adjust(left=0.21, right=0.99, top=0.83, bottom=0.27, wspace=0.80)
    subsets = [
        (axes[0], phase8["feature_ablation_summary"], ["Baseline", "No metadata", "Raw geometry"], "(A) Feature ablations"),
        (axes[1], phase8["hard_negative_summary"], ["Baseline", "Mined negatives"], "(B) Hard negatives"),
    ]
    for ax, rows, labels, title in subsets:
        for index, row in enumerate(rows):
            ap, f1 = row["macro_average_precision"], row["macro_best_frame_f1"]
            ax.plot([f1, ap], [index, index], color="0.6", linewidth=1.0, zorder=1)
            ax.plot(ap, index, marker="o", markersize=7, markerfacecolor="white", markeredgecolor="0.15", linewidth=0, zorder=2)
            ax.plot(f1, index, marker="s", markersize=6, markerfacecolor="0.15", markeredgecolor="0.15", linewidth=0, zorder=2)
            ax.text(ap, index - 0.19, f"{ap:.4f}", ha="center", va="bottom", fontsize=8.8)
            ax.text(f1, index + 0.19, f"{f1:.4f}", ha="center", va="top", fontsize=8.8)
        ax.set_yticks(range(len(rows)), labels)
        ax.set_ylim(len(rows) - 0.55, -0.55)
        ax.set_xlim(0.70, 0.76)
        ax.set_xticks([0.70, 0.72, 0.74, 0.76])
        ax.set_xlabel("Sampled-screen metric")
        ax.set_title(title, fontsize=10, loc="left", pad=14, fontweight="bold")
        ax.xaxis.grid(True, color="0.88", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=7)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="0.15", markersize=7, label="Macro average precision"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="0.15", markeredgecolor="0.15", markersize=6, label="Macro best frame F1"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.53, 0.07), ncol=2, frameon=False, fontsize=9)
    fig.text(0.50, 0.015, "Bounded Phase 8 screens; these are not exact competition scores.", ha="center", fontsize=9)
    return export(fig, "figure3_feature_hardnegative_ablations")


def main() -> None:
    reports = read_reports()
    files = architecture_figure(reports)
    comparison_files, scores = model_comparison_figure(reports)
    files.extend(comparison_files)
    files.extend(ablation_figure(reports))
    manifest = {
        "source_reports": {key: str(path.relative_to(ROOT)).replace("\\", "/") for key, path in SOURCE_PATHS.items()},
        "files": files,
        "png_dpi": 350,
        "pdf_fonttype": 42,
        "model_comparison": scores,
        "notes": [
            "Phase 10 screens belong to the earlier v1 lineage and are not scored beside performance-v2.",
            "Phase 13 is an implemented audit framework; performance-v2 audit completion is not asserted.",
            "The feature/hard-negative ablation figure uses sampled screen AP and best frame F1, not the exact competition metric.",
            "Model comparison uncertainty bars are omitted because no valid confidence intervals are present in the reports.",
        ],
    }
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Generated " + str(len(files)) + " figure artifacts:")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
