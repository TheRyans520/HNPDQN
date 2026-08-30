"""Draw the revised 48-dimensional HNP-DDQN architecture as vector artwork."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "light": "#F6F8FA",
    "ink": "#1F2933",
    "gray": "#667085",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    detail: str,
    color: str,
    *,
    title_size: float = 7.7,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=color,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h * 0.67),
            w,
            h * 0.33,
            boxstyle="round,pad=0.016,rounding_size=0.025",
            linewidth=0,
            facecolor=color,
            alpha=0.13,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.825,
        title,
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=title_size,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h * 0.35,
        detail,
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=6.6,
        linespacing=1.25,
        zorder=3,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["gray"],
    style: str = "-|>",
    connectionstyle: str = "arc3",
    linewidth: float = 1.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
            zorder=5,
        )
    )


def create_figure() -> plt.Figure:
    _style()
    fig = plt.figure(figsize=(7.05, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0])
    ax_top = fig.add_subplot(grid[0])
    ax_bottom = fig.add_subplot(grid[1])
    for ax in (ax_top, ax_bottom):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax_top.text(
        0.0, 0.99, "A", transform=ax_top.transAxes, fontweight="bold", fontsize=10, va="top"
    )
    ax_top.text(
        0.035,
        0.99,
        "HNP Q-network (32,841 trainable parameters)",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
    )
    xs = [0.01, 0.165, 0.325, 0.485, 0.645, 0.805]
    ws = [0.125, 0.13, 0.13, 0.13, 0.13, 0.175]
    y, h = 0.31, 0.49
    _box(
        ax_top,
        xs[0], y, ws[0], h,
        "Observation",
        "48 values\n40 RF + 8 previous-action\none-hot",
        COLORS["blue"],
    )
    _box(
        ax_top,
        xs[1], y, ws[1], h,
        "Feature block 1",
        "Linear 48→128\nBatchNorm + ReLU",
        COLORS["sky"],
    )
    _box(
        ax_top,
        xs[2], y, ws[2], h,
        "Feature block 2",
        "Linear 128→64\nBatchNorm + ReLU",
        COLORS["green"],
    )
    _box(
        ax_top,
        xs[3], y, ws[3], h,
        "Expansion",
        "Concatenate\n$[h,h^2]\\in\\mathbb{R}^{128}$\nLayerNorm",
        COLORS["orange"],
    )
    _box(
        ax_top,
        xs[4], y, ws[4], h,
        "Projection",
        "Linear 128→128\nReLU",
        COLORS["vermillion"],
    )
    _box(
        ax_top,
        xs[5], y, ws[5], h,
        "Dueling output",
        "Value: 128→1\nAdvantage: 128→8\nQ=V+A−mean(A)",
        COLORS["purple"],
    )
    for i in range(len(xs) - 1):
        _arrow(
            ax_top,
            (xs[i] + ws[i] + 0.003, y + h / 2),
            (xs[i + 1] - 0.004, y + h / 2),
        )
    ax_top.text(
        0.01,
        0.16,
        "Train-fitted standardization is applied only to the 40 RF entries; the previous-action one-hot remains unchanged.",
        fontsize=6.8,
        color=COLORS["gray"],
    )

    ax_bottom.text(
        0.0, 0.99, "B", transform=ax_bottom.transAxes, fontweight="bold", fontsize=10, va="top"
    )
    ax_bottom.text(
        0.035,
        0.99,
        "Double-DQN target and Polyak update",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
    )
    _box(
        ax_bottom, 0.02, 0.24, 0.18, 0.47,
        "Replay batch",
        "$(s_t,a_t,r_t,s_{t+1})$\nseeded buffer",
        COLORS["blue"],
    )
    _box(
        ax_bottom, 0.27, 0.24, 0.18, 0.47,
        "Online network",
        "selects\n$a^*=\\arg\\max_a Q_{online}(s_{t+1},a)$",
        COLORS["green"],
    )
    _box(
        ax_bottom, 0.52, 0.24, 0.18, 0.47,
        "Target network",
        "evaluates $Q_{target}(s_{t+1},a^*)$\n$y=r+\\gamma Q_{target}$",
        COLORS["orange"],
    )
    _box(
        ax_bottom, 0.78, 0.24, 0.19, 0.47,
        "Update",
        "MSE TD objective\n$\\theta^-\\leftarrow(1-\\tau)\\theta^-+\\tau\\theta$\n$\\tau=0.005$",
        COLORS["vermillion"],
        title_size=7.5,
    )
    _arrow(ax_bottom, (0.202, 0.475), (0.265, 0.475))
    _arrow(ax_bottom, (0.452, 0.475), (0.515, 0.475))
    _arrow(ax_bottom, (0.702, 0.475), (0.775, 0.475))
    _arrow(
        ax_bottom,
        (0.88, 0.22),
        (0.62, 0.20),
        color=COLORS["vermillion"],
        connectionstyle="arc3,rad=-0.22",
        linewidth=0.9,
    )
    ax_bottom.text(
        0.72,
        0.045,
        "soft target update",
        color=COLORS["vermillion"],
        fontsize=6.5,
        ha="center",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = create_figure()
    stem = output_dir / "hnp_architecture_48d"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
