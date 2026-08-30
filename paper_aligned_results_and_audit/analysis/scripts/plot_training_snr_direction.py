"""Create the development-training-only audit figure for the raw ``snr`` field.

This script does not load the pilot or either cross-configuration evaluation
condition.  Each plotted pair is one fixed-jammer source CSV from training
scans 0--6.  The figure is descriptive and is used only to document the score
direction of the maximum-quality/minimum-interference baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE_ITO = {
    "orange": "#E69F00",
    "blue": "#0072B2",
    "gray": "#8A8A8A",
}


def _load_training_rows(project_root: Path) -> tuple[pd.DataFrame, float]:
    experiment_root = project_root / "experiment_v2"
    sys.path.insert(0, str(experiment_root))
    from src.data import STATE_FEATURES, build_dataset  # noqa: PLC0415

    bundle = build_dataset(
        experiment_root / "data" / "raw",
        split_scan_ids={"train": tuple(range(7))},
        distance_cm=20,
        power_dbm=10,
        window_size=32,
        stride=1,
    )
    split = bundle.train
    records: list[dict[str, object]] = []
    correct = 0
    total = 0
    for key, trajectory, _, source_file in split.iter_trajectories():
        matrix = trajectory.reshape(
            len(trajectory), len(split.channels), len(STATE_FEATURES)
        )
        raw_snr_mean = matrix[:, :, 0]
        jammer_index = split.channels.index(key.jammer_channel)
        jammed = raw_snr_mean[:, jammer_index]
        nonjammed = np.delete(raw_snr_mean, jammer_index, axis=1).mean(axis=1)
        records.append(
            {
                "scan_id": key.scan_id,
                "jammer_channel_mhz": key.jammer_channel,
                "source_file": str(source_file),
                "jammed_raw_snr_mean": float(np.mean(jammed)),
                "nonjammed_raw_snr_mean": float(np.mean(nonjammed)),
                "n_windows": int(len(trajectory)),
            }
        )
        correct += int(np.sum(np.argmax(raw_snr_mean, axis=1) == jammer_index))
        total += int(len(trajectory))
    return pd.DataFrame.from_records(records), correct / total


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def create_figure(rows: pd.DataFrame, identification_rate: float) -> plt.Figure:
    _configure_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    x_left, x_right = 0.0, 1.0
    for row in rows.itertuples(index=False):
        ax.plot(
            [x_left, x_right],
            [row.nonjammed_raw_snr_mean, row.jammed_raw_snr_mean],
            color=OKABE_ITO["gray"],
            alpha=0.22,
            linewidth=0.55,
            zorder=1,
        )
    ax.scatter(
        np.full(len(rows), x_left),
        rows["nonjammed_raw_snr_mean"],
        s=13,
        facecolor=OKABE_ITO["blue"],
        edgecolor="white",
        linewidth=0.3,
        alpha=0.85,
        label="Non-jammed channels",
        zorder=2,
    )
    ax.scatter(
        np.full(len(rows), x_right),
        rows["jammed_raw_snr_mean"],
        s=13,
        facecolor=OKABE_ITO["orange"],
        edgecolor="black",
        linewidth=0.25,
        alpha=0.85,
        label="Jammed channel",
        zorder=3,
    )
    medians = [
        float(rows["nonjammed_raw_snr_mean"].median()),
        float(rows["jammed_raw_snr_mean"].median()),
    ]
    for x, median in zip((x_left, x_right), medians):
        ax.plot([x - 0.13, x + 0.13], [median, median], color="black", linewidth=1.4)

    ax.set_xlim(-0.38, 1.38)
    ax.set_xticks([0, 1], ["Non-jammed\nchannel mean", "Jammed\nchannel"])
    ax.set_ylabel("Dataset-provided raw snr mean (a.u.)")
    ax.text(
        0.03,
        0.97,
        f"Train only: 56 source files\nargmax identifies jammer: {100*identification_rate:.3f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, identification_rate = _load_training_rows(project_root)
    rows.to_csv(output_dir / "training_snr_direction_source_data.csv", index=False)
    fig = create_figure(rows, identification_rate)
    stem = output_dir / "training_snr_direction"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print(
        f"wrote {len(rows)} source-file pairs; "
        f"argmax jammer identification={identification_rate:.8f}"
    )


if __name__ == "__main__":
    main()
