from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import generate_panels_bc as bc
import plot_main_lobe_gps_galileo as core


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "panel_c_individual"
SIGNALS = ["GPS L1", "GPS L5", "Galileo E1", "Galileo E5a"]
FILE_NAMES = {
    "GPS L1": "gps_l1_ca",
    "GPS L5": "gps_l5q",
    "Galileo E1": "galileo_e1c",
    "Galileo E5a": "galileo_e5a_q",
}
Y_SIDES = {
    "GPS L1": "left",
    "GPS L5": "right",
    "Galileo E1": "left",
    "Galileo E5a": "right",
}
TRANSITION_COLOR = "#6B7280"


def style_axis(ax: plt.Axes, y_side: str) -> None:
    ax.set_xlim(0, 75)
    ax.set_ylim(-45, 4.4)
    ax.set_xticks(np.arange(0, 76, 10))
    ax.set_yticks([-40, -30, -20, -10, 0])
    ax.grid(True, color=core.COLORS["grid"], linewidth=0.45, alpha=0.82)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color(core.COLORS["ink"])

    ax.set_xlabel(r"Off-boresight angle, $\theta$ (deg)")
    ax.set_ylabel("Relative EIRP (dB)")
    if y_side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", labelleft=False, labelright=True)
    else:
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        ax.tick_params(axis="y", labelleft=True, labelright=False)


def draw_signal(
    ax: plt.Axes,
    signal: str,
    summary: core.PatternSummary,
    observations: pd.DataFrame,
    valley: dict[str, float],
    y_side: str,
) -> list[object]:
    color = core.COLORS[signal]
    ax.fill_between(
        summary.theta,
        summary.p05,
        summary.p95,
        color=color,
        alpha=0.22,
        linewidth=0,
        zorder=2,
    )
    ax.plot(summary.theta, summary.median, color=color, linewidth=1.65, zorder=5)

    angles = observations["tx_off_boresight_deg"].to_numpy(dtype=float)
    angle_min = float(np.nanmin(angles))
    angle_max = float(np.nanmax(angles))
    ax.axvspan(
        angle_min,
        angle_max,
        facecolor=core.COLORS["observation"],
        alpha=0.12,
        linewidth=0,
        zorder=0,
    )
    ax.text(
        0.5 * (angle_min + angle_max),
        -12.7,
        "LuGRE",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=core.COLORS["observation"],
        zorder=8,
    )


    low = float(valley["p05"])
    theta = np.asarray(summary.theta, dtype=float)
    median_pattern = np.asarray(summary.median, dtype=float)
    search = np.flatnonzero((theta >= 15.0) & (theta <= 60.0))
    local_minima = [
        idx
        for idx in search[1:-1]
        if median_pattern[idx] <= median_pattern[idx - 1]
        and median_pattern[idx] < median_pattern[idx + 1]
    ]
    boundary = float(
        theta[local_minima[0]] if local_minima else valley["median"]
    )
    high = float(valley["p95"])
    ax.axvline(
        boundary,
        color=TRANSITION_COLOR,
        linewidth=1.0,
        linestyle=(0, (3.0, 2.2)),
        alpha=0.95,
        zorder=4,
    )

    arrow_y = -6.0
    arrow_gap = 0.8
    for start, end, label in (
        (1.0, boundary - arrow_gap, "Main lobe"),
        (boundary + arrow_gap, 74.0, "Side lobes"),
    ):
        ax.annotate(
            "",
            xy=(end, arrow_y),
            xytext=(start, arrow_y),
            arrowprops={
                "arrowstyle": "<->",
                "color": TRANSITION_COLOR,
                "linewidth": 0.95,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=9,
        )
        ax.text(
            0.5 * (start + end),
            arrow_y + 1.25,
            label,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=TRANSITION_COLOR,
            fontweight="bold",
            zorder=9,
        )

    label_low = int(round(low))
    label_high = int(round(high))
    ax.text(
        boundary,
        -42.0,
        rf"${label_low}^\circ$-${label_high}^\circ$",
        ha="center",
        va="center",
        fontsize=7.0,
        color=TRANSITION_COLOR,
        fontweight="bold",
        zorder=9,
    )

    ax.set_title(
        core.SIGNAL_LABELS[signal],
        loc="left",
        color=color,
        fontweight="bold",
        pad=4,
        fontsize=9.0,
    )
    style_axis(ax, y_side)

    return [
        Line2D([0], [0], color=color, linewidth=1.65, label="Pattern median"),
        patches.Patch(
            facecolor=color,
            alpha=0.22,
            edgecolor="none",
            label="Pattern P5-P95",
        ),
        patches.Patch(
            facecolor=core.COLORS["observation"],
            alpha=0.12,
            edgecolor="none",
            label="LuGRE range",
        ),

        Line2D(
            [0],
            [0],
            color=TRANSITION_COLOR,
            linewidth=1.0,
            linestyle=(0, (3.0, 2.2)),
            label="Median first-valley boundary",
        ),
    ]


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    paths = []
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.06, "facecolor": "white"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def save_individual(
    signal: str,
    summary: core.PatternSummary,
    observations: pd.DataFrame,
    valley: dict[str, float],
) -> list[Path]:
    y_side = Y_SIDES[signal]
    fig = plt.figure(figsize=(5.35, 3.85))
    if y_side == "left":
        fig.subplots_adjust(left=0.13, right=0.975, bottom=0.23, top=0.88)
    else:
        fig.subplots_adjust(left=0.055, right=0.87, bottom=0.23, top=0.88)
    ax = fig.add_subplot(1, 1, 1)
    handles = draw_signal(ax, signal, summary, observations, valley, y_side)
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        columnspacing=1.25,
        handlelength=2.0,
        handletextpad=0.45,
        fontsize=6.1,
    )
    paths = save_figure(fig, OUTPUT_DIR / f"panel_c_{FILE_NAMES[signal]}")
    plt.close(fig)
    return paths


def save_combined(
    points: pd.DataFrame,
    summaries: dict[str, core.PatternSummary],
    valleys: dict[str, dict[str, float]],
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 5.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.12, top=0.94, wspace=0.12, hspace=0.20)
    handles = None
    for ax, signal in zip(axes.flat, SIGNALS):
        subset = points[points["signal_name"] == signal]
        panel_handles = draw_signal(
            ax,
            signal,
            summaries[signal],
            subset,
            valleys[signal],
            Y_SIDES[signal],
        )
        ax.legend(
            handles=panel_handles[:2],
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.82,
            fancybox=False,
            borderpad=0.22,
            handlelength=1.7,
            handletextpad=0.4,
            fontsize=5.6,
        )
        handles = panel_handles
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    if handles is not None:
        fig.legend(
            handles=handles[2:],
            frameon=False,
            loc="center",
            bbox_to_anchor=(0.5, 0.53),
            ncol=2,
            columnspacing=1.0,
            handlelength=1.8,
            handletextpad=0.38,
            fontsize=5.8,
        )
    paths = save_figure(fig, OUTPUT_DIR / "panel_c_combined_with_transition_ranges")
    plt.close(fig)
    return paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points, summaries, gps_valleys, e1, e5a, _ = bc.build_inputs()
    valleys = {
        "GPS L1": gps_valleys["GPS L1"],
        "GPS L5": gps_valleys["GPS L5"],
        "Galileo E1": e1,
        "Galileo E5a": e5a,
    }

    paths = []
    for signal in SIGNALS:
        observations = points[points["signal_name"] == signal]
        paths.extend(
            save_individual(signal, summaries[signal], observations, valleys[signal])
        )
    paths.extend(save_combined(points, summaries, valleys))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()