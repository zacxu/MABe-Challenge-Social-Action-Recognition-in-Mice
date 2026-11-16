"""Render the implementation-grounded architecture figure for the rewritten paper."""

from pathlib import Path
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sc4000_methods_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(7, 3.15))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.99)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, w, h, text, *, shade=False, size=7.7):
        patch = FancyBboxPatch(
            (x, y - h / 2), w, h,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            linewidth=0.8, edgecolor="black",
            facecolor="0.94" if shade else "white",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=size,
                linespacing=1.25)

    def arrow(points, *, head=True):
        for index, (start, stop) in enumerate(zip(points, points[1:])):
            last = index == len(points) - 2
            ax.add_patch(FancyArrowPatch(
                start, stop, arrowstyle="-|>" if head and last else "-",
                mutation_scale=8, linewidth=0.85, color="black",
                shrinkA=0, shrinkB=1.2 if head and last else 0,
            ))

    ax.text(0.012, 0.98, "(a) Prediction pipeline", fontsize=9, weight="bold",
            va="top")
    upper, lower, middle = 0.845, 0.635, 0.740
    box(0.012, middle, 0.173, 0.155, "Cleaned pose\n+ metadata", shade=True)
    box(0.235, upper, 0.210, 0.145, "Engineered features\n240 individual / 321 pair")
    box(0.235, lower, 0.210, 0.145, "Sequence features\n23 individual / 54 pair")
    box(0.490, upper, 0.165, 0.145, "Laboratory/action\nXGBoost models")
    box(0.490, lower, 0.165, 0.145, "Independent temporal\nencoders + heads", shade=True)
    box(0.715, middle, 0.110, 0.155, "Action-wise\nprobability\nblend", size=7.8)
    box(0.875, middle, 0.110, 0.155, "Interval\ndecoder", size=7.8)
    arrow([(0.185, middle), (0.210, middle), (0.210, upper), (0.235, upper)])
    arrow([(0.210, middle), (0.210, lower), (0.235, lower)])
    arrow([(0.445, upper), (0.490, upper)])
    arrow([(0.445, lower), (0.490, lower)])
    arrow([(0.655, upper), (0.685, upper), (0.685, middle), (0.715, middle)])
    arrow([(0.655, lower), (0.685, lower), (0.685, middle)], head=False)
    arrow([(0.825, middle), (0.875, middle)])

    ax.plot([0.012, 0.985], [0.515, 0.515], color="0.70", linewidth=0.55)
    ax.text(0.012, 0.475, "(b) Temporal encoder", fontsize=9, weight="bold",
            va="top")
    ax.text(0.283, 0.475, "Same structure; independent individual and pair parameters",
            fontsize=8, va="top")

    y = 0.285
    box(0.012, y, 0.180, 0.205,
        "Parallel Conv1D\nkernels 3, 5, 7, 9\nConcatenate + 1×1\nGN + GELU → 128", shade=True)
    box(0.235, y, 0.150, 0.205,
        "Add laboratory/\nmetadata condition\n+ laboratory adapter")
    box(0.430, y, 0.170, 0.205,
        "6 residual TCN blocks\nDepthwise + 1×1\nDilations 1, 2, 4,\n8, 16, 32")
    box(0.645, y, 0.170, 0.205,
        "4 attention blocks\n8-head self-attention\nPre-LayerNorm\n4× feed-forward")
    box(0.860, y, 0.125, 0.205,
        "Linear head\n11 / 26 logits\n+ sigmoid", shade=True)
    for start, stop in [(0.192, 0.235), (0.385, 0.430),
                        (0.600, 0.645), (0.815, 0.860)]:
        arrow([(start, y), (stop, y)])
    ax.text(0.5, 0.09,
            "512 frames at 30 Hz; 128 hidden channels in each encoder",
            ha="center", va="center", fontsize=8)

    output = Path(__file__).resolve().parent
    fig.savefig(output / "model_architecture.png", dpi=360)
    fig.savefig(output / "model_architecture.svg")
    plt.close(fig)
    print(output / "model_architecture.png")
    print(output / "model_architecture.svg")


if __name__ == "__main__":
    main()
