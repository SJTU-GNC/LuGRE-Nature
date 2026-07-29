from __future__ import annotations

from pathlib import Path
import math
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import plot_main_lobe_nature_figures as base


ROOT = Path(__file__).resolve().parent
POINT_CSV = ROOT / "lugre_gnss_main_lobe_point_check.csv"
GRAP_DIR = ROOT / "GRAP_metadata"
OUTPUT_STEM = ROOT / "lugre_main_lobe_ab_figure_nature_grap"

SIGNAL_ORDER = ["GPS L1", "GPS L5", "Galileo E1", "Galileo E5a"]
SIGNAL_LABELS = {
    "GPS L1": "GPS L1 C/A",
    "GPS L5": "GPS L5Q",
    "Galileo E1": "Galileo E1C",
    "Galileo E5a": "Galileo E5a-Q",
}
COLORS = {
    "GPS L1": "#2F6FA3",
    "GPS L5": "#77A9CF",
    "Galileo E1": "#E2A02B",
    "Galileo E5a": "#D66B2B",
    "observed": "#16866F",
    "central": "#6AAED6",
    "side": "#8D73B8",
    "ink": "#202833",
    "muted": "#66717D",
    "grid": "#E2E8EE",
    "ci": "#76818D",
}


def configure_style() -> None:
    base.configure_style()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7.2,
            "legend.fontsize": 5.8,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
        }
    )


def load_points() -> pd.DataFrame:
    usecols = ["signal_name", "tx_off_boresight_deg"]
    df = pd.read_csv(POINT_CSV, usecols=usecols)
    df = df[df["signal_name"].isin(SIGNAL_ORDER)].copy()
    df["tx_off_boresight_deg"] = pd.to_numeric(df["tx_off_boresight_deg"], errors="coerce")
    df = df.dropna(subset=["tx_off_boresight_deg"])
    df["signal_name"] = pd.Categorical(df["signal_name"], categories=SIGNAL_ORDER, ordered=True)
    return df


def _column_number(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Invalid XLSX cell reference: {cell_ref}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value


def _read_numeric_sheet(path: Path, sheet_number: int) -> np.ndarray:
    grid = np.full((363, 93), np.nan, dtype=float)
    worksheet = f"xl/worksheets/sheet{sheet_number}.xml"
    cell_tag = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"
    value_tag = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v"

    with ZipFile(path) as archive, archive.open(worksheet) as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != cell_tag:
                continue
            ref = element.attrib.get("r")
            if not ref:
                element.clear()
                continue
            row_match = re.search(r"(\d+)$", ref)
            value_node = element.find(value_tag)
            if row_match and value_node is not None and value_node.text is not None:
                row = int(row_match.group(1))
                col = _column_number(ref)
                if 1 <= row <= grid.shape[0] and 1 <= col <= grid.shape[1]:
                    try:
                        grid[row - 1, col - 1] = float(value_node.text)
                    except ValueError:
                        pass
            element.clear()
    return grid


def load_grap_band(path: Path) -> dict[str, np.ndarray]:
    eirp_grid = _read_numeric_sheet(path, 1)
    ub_grid = _read_numeric_sheet(path, 2)
    lb_grid = _read_numeric_sheet(path, 3)

    theta = eirp_grid[1, 2:93]
    azimuth = eirp_grid[2:363, 1]
    eirp = eirp_grid[2:363, 2:93]
    upper = ub_grid[2:363, 2:93]
    lower = lb_grid[2:363, 2:93]

    valid_theta = np.isfinite(theta)
    theta = theta[valid_theta]
    eirp = eirp[:, valid_theta]
    upper = upper[:, valid_theta]
    lower = lower[:, valid_theta]

    # The 360-degree cut duplicates 0 degrees and is omitted from azimuth statistics.
    if np.isclose(azimuth[-1], 360.0):
        azimuth = azimuth[:-1]
        eirp = eirp[:-1]
        upper = upper[:-1]
        lower = lower[:-1]

    return {
        "theta": theta,
        "azimuth": azimuth,
        "eirp": eirp,
        "upper": upper,
        "lower": lower,
        "median": np.nanmedian(eirp, axis=0),
        "p05": np.nanquantile(eirp, 0.05, axis=0),
        "p95": np.nanquantile(eirp, 0.95, axis=0),
        "ci_lower_median": np.nanmedian(lower, axis=0),
        "ci_upper_median": np.nanmedian(upper, axis=0),
    }


def first_valley_summary(grap: dict[str, np.ndarray]) -> dict[str, float]:
    theta = grap["theta"]
    valleys: list[float] = []
    kernel = np.ones(5, dtype=float) / 5.0
    for profile in grap["eirp"]:
        smooth = np.convolve(profile, kernel, mode="same")
        candidates = [
            idx
            for idx in range(7, min(60, len(theta) - 1))
            if smooth[idx] <= smooth[idx - 1] and smooth[idx] < smooth[idx + 1]
        ]
        if candidates:
            valleys.append(float(theta[candidates[0]]))
    values = np.asarray(valleys, dtype=float)
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def smoothed_histogram(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(values, bins=bins, density=True)
    x = 0.5 * (edges[:-1] + edges[1:])
    kernel_x = np.arange(-8, 9, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / 2.1) ** 2)
    kernel /= kernel.sum()
    return x, np.convolve(density, kernel, mode="same")


def style_boxed_axis(ax: plt.Axes) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COLORS["ink"])
        ax.spines[side].set_linewidth(0.72)


def draw_panel_a(
    fig: plt.Figure,
    subplot_spec,
    df: pd.DataFrame,
) -> tuple[plt.Axes, plt.Axes, float, float]:
    gs = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=subplot_spec,
        height_ratios=[1.95, 1.08],
        hspace=0.06,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[1, 0], sharex=ax)

    observed_min = float(df["tx_off_boresight_deg"].min())
    observed_max = float(df["tx_off_boresight_deg"].max())
    x_min = math.floor((observed_min - 0.35) * 10) / 10
    x_max = math.ceil((observed_max + 0.35) * 10) / 10
    bins = np.linspace(x_min, x_max, 72)

    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    max_y = 0.0
    for signal in SIGNAL_ORDER:
        values = df.loc[df["signal_name"] == signal, "tx_off_boresight_deg"].to_numpy()
        x, y = smoothed_histogram(values, bins)
        curves[signal] = (values, x, y)
        max_y = max(max_y, float(np.max(y)))

    for signal in SIGNAL_ORDER:
        values, x, y = curves[signal]
        color = COLORS[signal]
        ax.fill_between(x, y, 0, color=color, alpha=0.11, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.55, label=f"{SIGNAL_LABELS[signal]} (n={len(values):,})")

    ax.axvline(13.8, color="#7D8793", linewidth=0.8, linestyle=(0, (2.5, 2.4)))
    ax.text(13.84, max_y * 1.10, "Earth limb 13.8°", fontsize=5.6, color="#6F7A86", va="top")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_y * 1.18)
    ax.set_ylabel("Probability density")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper right", ncol=2, handlelength=2.0, columnspacing=1.05, borderpad=0.1)
    ax.text(
        0.01,
        0.91,
        "all valid matched points; no Galileo angle threshold",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=COLORS["muted"],
    )
    ax.text(
        0.0,
        1.13,
        "a  LuGRE transmitter off-boresight distributions by signal",
        transform=ax.transAxes,
        fontsize=10.4,
        fontweight="bold",
        va="bottom",
        color=COLORS["ink"],
    )
    style_boxed_axis(ax)

    y_positions = {signal: float(len(SIGNAL_ORDER) - index - 1) for index, signal in enumerate(SIGNAL_ORDER)}
    for signal in SIGNAL_ORDER:
        values = curves[signal][0]
        y = y_positions[signal]
        q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        color = COLORS[signal]
        ax_box.plot([q[0], q[6]], [y, y], color=color, linewidth=0.9, alpha=0.72)
        ax_box.plot([q[1], q[5]], [y, y], color=color, linewidth=4.5, alpha=0.58, solid_capstyle="round")
        ax_box.add_patch(
            patches.Rectangle(
                (q[2], y - 0.13),
                q[4] - q[2],
                0.26,
                facecolor="white",
                edgecolor=color,
                linewidth=0.85,
                alpha=0.78,
            )
        )
        ax_box.plot([q[3], q[3]], [y - 0.17, y + 0.17], color=color, linewidth=1.7)
        ax_box.scatter([q[0], q[6]], [y, y], s=10, color=color, zorder=3)
        ax_box.text(
            x_max - 0.03,
            y + 0.20,
            f"median {q[3]:.2f}°",
            ha="right",
            va="center",
            fontsize=5.5,
            color=COLORS["muted"],
        )

    ax_box.axvline(13.8, color="#7D8793", linewidth=0.8, linestyle=(0, (2.5, 2.4)))
    ax_box.set_ylim(-0.52, len(SIGNAL_ORDER) - 0.48)
    ax_box.set_yticks([y_positions[signal] for signal in SIGNAL_ORDER])
    ax_box.set_yticklabels([SIGNAL_LABELS[signal] for signal in SIGNAL_ORDER])
    ax_box.set_xlabel(r"Transmitter off-boresight angle, $\alpha$ (deg)")
    ax_box.grid(True, axis="x", color=COLORS["grid"], linewidth=0.5)
    ax_box.grid(False, axis="y")
    style_boxed_axis(ax_box)
    return ax, ax_box, observed_min, observed_max


def polar_xy(origin: tuple[float, float], radius: float, angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    return origin[0] + radius * math.cos(angle), origin[1] + radius * math.sin(angle)


def draw_panel_b(
    ax: plt.Axes,
    observed_min: float,
    observed_max: float,
    e1_valley: dict[str, float],
    e5a_valley: dict[str, float],
) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-6.35, 6.35)
    ax.set_ylim(-13.85, 1.58)
    ax.axis("off")

    earth_limb = 13.8
    sat_image = (0.0, 0.04)
    antenna = (0.0, -0.38)
    earth = (0.0, -11.02)
    center_distance = antenna[1] - earth[1]
    earth_radius = center_distance * math.sin(math.radians(earth_limb))
    beam_length = center_distance + earth_radius * 1.25

    def beam_angle(offset: float) -> float:
        return -90.0 + offset

    # A common high-EIRP central region is shown without assigning a hard edge.
    ax.add_patch(
        patches.Wedge(
            antenna,
            beam_length,
            beam_angle(-23.0),
            beam_angle(23.0),
            facecolor=COLORS["central"],
            alpha=0.15,
            edgecolor="none",
            zorder=0,
        )
    )

    # GRAP-derived first-valley ranges. These are diagnostics, not official cut-offs.
    for sign in (-1.0, 1.0):
        e1_a0, e1_a1 = sorted((sign * e1_valley["min"], sign * e1_valley["max"]))
        e5_a0, e5_a1 = sorted((sign * e5a_valley["p05"], sign * e5a_valley["p95"]))
        ext_a0, ext_a1 = sorted((sign * e5a_valley["p95"], sign * e5a_valley["max"]))
        ax.add_patch(
            patches.Wedge(
                antenna,
                beam_length,
                beam_angle(e1_a0),
                beam_angle(e1_a1),
                facecolor=COLORS["Galileo E1"],
                alpha=0.18,
                edgecolor="none",
                zorder=1,
            )
        )
        ax.add_patch(
            patches.Wedge(
                antenna,
                beam_length,
                beam_angle(e5_a0),
                beam_angle(e5_a1),
                facecolor=COLORS["Galileo E5a"],
                alpha=0.13,
                edgecolor="none",
                zorder=1,
            )
        )
        ax.add_patch(
            patches.Wedge(
                antenna,
                beam_length,
                beam_angle(ext_a0),
                beam_angle(ext_a1),
                facecolor=COLORS["Galileo E5a"],
                alpha=0.055,
                edgecolor="none",
                hatch="...",
                zorder=1,
            )
        )

    for start, end, length, alpha in [
        (-67, -39, beam_length * 0.50, 0.13),
        (39, 67, beam_length * 0.50, 0.13),
        (-82, -68, beam_length * 0.36, 0.08),
        (68, 82, beam_length * 0.36, 0.08),
    ]:
        ax.add_patch(
            patches.Wedge(
                antenna,
                length,
                beam_angle(start),
                beam_angle(end),
                facecolor=COLORS["side"],
                alpha=alpha,
                edgecolor="none",
                zorder=0,
            )
        )

    for start, end, alpha in [
        (observed_min, observed_max, 0.30),
        (-observed_max, -observed_min, 0.22),
    ]:
        ax.add_patch(
            patches.Wedge(
                antenna,
                beam_length * 0.98,
                beam_angle(start),
                beam_angle(end),
                facecolor=COLORS["observed"],
                alpha=alpha,
                edgecolor="none",
                zorder=2,
            )
        )

    # Only observational and Earth-limb guide rays are drawn; no Galileo hard boundary.
    for offset in (-observed_max, -observed_min, observed_min, observed_max):
        end = polar_xy(antenna, beam_length * 0.96, beam_angle(offset))
        ax.plot(
            [antenna[0], end[0]],
            [antenna[1], end[1]],
            color=COLORS["observed"],
            linewidth=0.72,
            alpha=0.82,
            zorder=4,
        )

    center_end = (earth[0], earth[1] + earth_radius)
    limb_end = polar_xy(antenna, beam_length * 0.96, beam_angle(earth_limb))
    ax.plot(
        [antenna[0], center_end[0]],
        [antenna[1], center_end[1]],
        color="#7D8793",
        linewidth=0.68,
        linestyle=(0, (3.0, 2.5)),
        zorder=3,
    )
    ax.plot(
        [antenna[0], limb_end[0]],
        [antenna[1], limb_end[1]],
        color="#7D8793",
        linewidth=0.68,
        linestyle=(0, (3.0, 2.5)),
        zorder=3,
    )

    base.GPS_III_CUTOUT = ROOT / "gps_iii_satellite_cutout.png"
    base.EARTH_CUTOUT = ROOT / "earth_user_reference_globe.png"
    base.draw_satellite(ax, sat_image, compact=True)
    base.draw_earth_scaled(ax, earth, earth_radius)

    lugre_point = polar_xy(antenna, beam_length * 0.55, beam_angle(-14.2))
    ax.scatter(
        [lugre_point[0]],
        [lugre_point[1]],
        s=20,
        facecolor=COLORS["observed"],
        edgecolor="white",
        linewidth=0.55,
        zorder=12,
    )
    ax.annotate(
        "LuGRE",
        xy=lugre_point,
        xytext=(lugre_point[0] - 1.90, lugre_point[1] + 0.48),
        ha="right",
        va="center",
        fontsize=6.4,
        color=COLORS["observed"],
        fontweight="bold",
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.7,
            "color": COLORS["observed"],
            "shrinkA": 0.5,
            "shrinkB": 1.0,
        },
        zorder=13,
    )

    ax.text(
        3.10,
        sat_image[1] + 0.16,
        "GNSS satellite",
        fontsize=6.3,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
        zorder=14,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.35},
    )
    ax.text(
        0.0,
        -3.18,
        "high-EIRP central region",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#2F6F96",
        fontweight="bold",
    )
    ax.text(-4.05, -4.35, "side lobes", ha="center", fontsize=5.8, color="#705795")
    ax.text(4.05, -4.35, "side lobes", ha="center", fontsize=5.8, color="#705795")
    ax.text(
        0.50,
        -0.075,
        "Earth",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color="#1E3A5F",
        clip_on=False,
    )

    # One-sided Earth-limb angle, matching the geometry used in the earlier panel.
    arc_y = earth[1] + earth_radius + 0.08
    arc_radius = (arc_y - antenna[1]) / math.sin(math.radians(beam_angle(earth_limb)))
    arc_start = (earth[0], arc_y)
    arc_end = polar_xy(antenna, arc_radius, beam_angle(earth_limb))
    ax.add_patch(
        patches.FancyArrowPatch(
            arc_start,
            arc_end,
            connectionstyle="arc3,rad=-0.28",
            arrowstyle="<->",
            mutation_scale=9.5,
            linewidth=0.9,
            color="#7D8793",
            zorder=18,
        )
    )
    ax.text(
        0.5 * (arc_start[0] + arc_end[0]),
        earth[1] + earth_radius + 0.76,
        "Earth-limb 13.8°",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#3B4651",
        zorder=19,
    )

    legend_x = -6.02
    legend_y = -0.82
    legend_rows = [
        (
            COLORS["observed"],
            f"LuGRE observed {observed_min:.1f}–{observed_max:.1f}°",
            0.30,
            None,
        ),
        (
            COLORS["Galileo E1"],
            f"GRAP E1 first valley {e1_valley['min']:.0f}–{e1_valley['max']:.0f}°",
            0.18,
            None,
        ),
        (
            COLORS["Galileo E5a"],
            f"GRAP E5a {e5a_valley['p05']:.0f}–{e5a_valley['p95']:.0f}° (5–95% azimuth)",
            0.15,
            None,
        ),
        (
            COLORS["Galileo E5a"],
            f"E5a extreme azimuth cuts to {e5a_valley['max']:.0f}°",
            0.07,
            "...",
        ),
    ]
    ax.text(
        legend_x,
        legend_y + 0.42,
        "GRAP angular context (not hard cut-offs)",
        fontsize=5.55,
        color=COLORS["muted"],
        ha="left",
        va="bottom",
    )
    for index, (color, label, alpha, hatch) in enumerate(legend_rows):
        y = legend_y - index * 0.72
        ax.add_patch(
            patches.Rectangle(
                (legend_x, y - 0.10),
                0.58,
                0.20,
                facecolor=color,
                edgecolor=color if hatch else "none",
                linewidth=0.4,
                alpha=alpha,
                hatch=hatch,
                zorder=10,
            )
        )
        ax.text(
            legend_x + 0.72,
            y,
            label,
            fontsize=4.95,
            color=COLORS["ink"],
            ha="left",
            va="center",
            zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.18},
        )


def draw_grap_profile(
    ax: plt.Axes,
    grap: dict[str, np.ndarray],
    color: str,
    band_label: str,
    observed_min: float,
    observed_max: float,
    valley: dict[str, float],
    show_xlabels: bool,
) -> None:
    theta = grap["theta"]
    keep = theta <= 40.0
    theta = theta[keep]

    ax.axvspan(observed_min, observed_max, color=COLORS["observed"], alpha=0.10, linewidth=0, zorder=0)
    ax.axvline(13.8, color="#7D8793", linewidth=0.72, linestyle=(0, (2.3, 2.2)), zorder=1)

    if band_label == "Galileo E1":
        ax.axvspan(valley["min"], valley["max"], color=color, alpha=0.10, linewidth=0, zorder=0)
        valley_text = f"first valley {valley['min']:.0f}–{valley['max']:.0f}°"
        valley_x = 0.5 * (valley["min"] + valley["max"])
    else:
        ax.axvspan(valley["p05"], valley["p95"], color=color, alpha=0.10, linewidth=0, zorder=0)
        ax.axvspan(
            valley["p95"],
            valley["max"],
            facecolor=color,
            alpha=0.035,
            hatch="...",
            edgecolor=color,
            linewidth=0.0,
            zorder=0,
        )
        valley_text = f"first valley {valley['p05']:.0f}–{valley['p95']:.0f}°; max {valley['max']:.0f}°"
        valley_x = 0.5 * (valley["p05"] + valley["p95"])

    p05 = grap["p05"][keep]
    p95 = grap["p95"][keep]
    median = grap["median"][keep]
    ci_lower = grap["ci_lower_median"][keep]
    ci_upper = grap["ci_upper_median"][keep]

    ax.fill_between(theta, p05, p95, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(theta, median, color=color, linewidth=1.35, zorder=4)
    ax.plot(theta, ci_lower, color=COLORS["ci"], linewidth=0.68, linestyle=(0, (3.0, 2.2)), zorder=3)
    ax.plot(theta, ci_upper, color=COLORS["ci"], linewidth=0.68, linestyle=(0, (3.0, 2.2)), zorder=3)

    ax.text(
        0.02,
        0.91,
        band_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color=color,
    )
    ax.text(
        valley_x,
        -11.0,
        valley_text,
        ha="center",
        va="bottom",
        fontsize=5.1,
        color=color,
    )
    ax.set_xlim(0, 40)
    ax.set_ylim(-15, 32.5)
    ax.set_yticks([-10, 0, 10, 20, 30])
    ax.grid(True, color=COLORS["grid"], linewidth=0.45)
    ax.tick_params(labelsize=5.5, pad=1.5)
    if show_xlabels:
        ax.set_xlabel(r"Co-elevation / off-boresight angle, $\theta$ (deg)", labelpad=2)
        ax.set_xticks([0, 10, 13.8, 20, 30, 40])
        ax.set_xticklabels(["0", "10", "13.8", "20", "30", "40"])
    else:
        ax.tick_params(labelbottom=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_panel_c(
    fig: plt.Figure,
    subplot_spec,
    e1: dict[str, np.ndarray],
    e5a: dict[str, np.ndarray],
    observed_min: float,
    observed_max: float,
    e1_valley: dict[str, float],
    e5a_valley: dict[str, float],
) -> tuple[plt.Axes, plt.Axes]:
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, hspace=0.13)
    ax_e1 = fig.add_subplot(gs[0, 0])
    ax_e5a = fig.add_subplot(gs[1, 0], sharex=ax_e1, sharey=ax_e1)

    draw_grap_profile(
        ax_e1,
        e1,
        COLORS["Galileo E1"],
        "Galileo E1",
        observed_min,
        observed_max,
        e1_valley,
        show_xlabels=False,
    )
    draw_grap_profile(
        ax_e5a,
        e5a,
        COLORS["Galileo E5a"],
        "Galileo E5a",
        observed_min,
        observed_max,
        e5a_valley,
        show_xlabels=True,
    )
    ax_e1.text(
        -0.10,
        1.12,
        "c  Galileo GRAP EIRP profiles",
        transform=ax_e1.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="bottom",
        color=COLORS["ink"],
    )
    fig.text(
        ax_e1.get_position().x0 - 0.042,
        0.5 * (ax_e5a.get_position().y0 + ax_e1.get_position().y1),
        "Expected EIRP (dBW)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.1,
        color=COLORS["ink"],
    )

    handles = [
        Line2D([0], [0], color=COLORS["Galileo E1"], lw=1.35, label="azimuth median"),
        patches.Patch(facecolor=COLORS["Galileo E1"], alpha=0.18, label="azimuth 5–95%"),
        Line2D([0], [0], color=COLORS["ci"], lw=0.75, linestyle=(0, (3.0, 2.2)), label="median GRAP 95% bounds"),
        patches.Patch(facecolor=COLORS["observed"], alpha=0.10, label="LuGRE observed angles"),
    ]
    ax_e1.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.01, 1.02),
        frameon=False,
        ncol=2,
        handlelength=1.55,
        columnspacing=0.75,
        labelspacing=0.30,
        fontsize=4.9,
    )
    return ax_e1, ax_e5a


def save_figure(fig: plt.Figure) -> list[Path]:
    outputs = [
        OUTPUT_STEM.with_suffix(".png"),
        OUTPUT_STEM.with_suffix(".pdf"),
        OUTPUT_STEM.with_suffix(".svg"),
    ]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.05)
    return outputs


def write_caption(
    df: pd.DataFrame,
    e1_valley: dict[str, float],
    e5a_valley: dict[str, float],
) -> Path:
    observed_min = float(df["tx_off_boresight_deg"].min())
    observed_max = float(df["tx_off_boresight_deg"].max())
    path = ROOT / "extended_data_fig3_grap_revised_caption.txt"
    text = f"""Extended Data Fig. 3 | LuGRE off-boresight geometry evaluated with the Galileo Reference Antenna Pattern.

a, Distributions of all valid LuGRE transmitter off-boresight samples, shown separately for GPS L1 C/A, GPS L5Q, Galileo E1C and Galileo E5a-Q. No Galileo observation is selected or rejected with a scalar beam-angle threshold. The dotted reference marks the representative one-sided Earth-limb angle of 13.8 deg.

b, Schematic Earth-limb geometry. The LuGRE observations span {observed_min:.2f}-{observed_max:.2f} deg. Coloured edge bands summarize a diagnostic first valley of the public JRC Galileo Reference Antenna Pattern (GRAP): {e1_valley['min']:.0f}-{e1_valley['max']:.0f} deg for E1 and {e5a_valley['p05']:.0f}-{e5a_valley['p95']:.0f} deg for the central 5-95% of E5a azimuth cuts, with extreme E5a cuts extending to {e5a_valley['max']:.0f} deg. These bands are not official hard main-lobe cut-offs.

c, Public GRAP expected EIRP as a function of co-elevation/off-boresight angle. Solid curves are medians over antenna-frame azimuth; coloured envelopes span the 5th-95th azimuth percentiles. Grey dashed curves show the azimuth-median lower and upper GRAP 95% model bounds. Green shading marks the observed LuGRE angle range. Because the Galileo direction pattern depends on both co-elevation and antenna-frame azimuth, the analysis does not assign Galileo lobe identity from one fixed angle.

Source: European Commission Joint Research Centre, Galileo Reference Antenna Pattern metadata (E1 and E5a, 1-deg grids) and Menzione, Sgammini & Paonni (2024), JRC135110, doi:10.2760/765842.
"""
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    configure_style()
    df = load_points()
    e1 = load_grap_band(GRAP_DIR / "GRAP_File_E1_.xlsx")
    e5a = load_grap_band(GRAP_DIR / "GRAP_File_E5a_.xlsx")
    e1_valley = first_valley_summary(e1)
    e5a_valley = first_valley_summary(e5a)

    fig = plt.figure(figsize=(8.15, 6.95))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.03, 1.34], hspace=0.36)
    ax_a, ax_box, observed_min, observed_max = draw_panel_a(fig, outer[0], df)

    lower = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[0.92, 1.18], wspace=0.42)
    ax_b = fig.add_subplot(lower[0, 0])
    draw_panel_b(ax_b, observed_min, observed_max, e1_valley, e5a_valley)
    ax_b.text(
        -0.17,
        1.03,
        "b  Earth-limb and GRAP angular geometry",
        transform=ax_b.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="bottom",
        color=COLORS["ink"],
    )
    ax_e1, ax_e5a = draw_panel_c(
        fig,
        lower[0, 1],
        e1,
        e5a,
        observed_min,
        observed_max,
        e1_valley,
        e5a_valley,
    )

    lower_left = ax_b.get_position().x0
    lower_right = ax_e1.get_position().x1
    for top_axis in (ax_a, ax_box):
        position = top_axis.get_position()
        top_axis.set_position([lower_left, position.y0, lower_right - lower_left, position.height])

    fig.text(
        0.5,
        0.008,
        "GRAP first-valley ranges are derived diagnostics, not official fixed main-lobe boundaries.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    outputs = save_figure(fig)
    plt.close(fig)
    caption = write_caption(df, e1_valley, e5a_valley)

    print(f"observed range: {observed_min:.4f}-{observed_max:.4f} deg")
    print("E1 first valley:", e1_valley)
    print("E5a first valley:", e5a_valley)
    for path in outputs:
        print(path)
    print(caption)


if __name__ == "__main__":
    main()
