from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


INK = "#172033"
AXIS = "#334155"
MUTED = "#64748B"
GRID = "#D9E2EC"
BLUE = "#2F6DB5"
LIGHT_BLUE = "#66A6D9"
GOLD = "#E3A01A"
ORANGE = "#D95F2D"
GREEN = "#1A9B7C"
PURPLE = "#7658D6"
RED = "#CC4138"
PANEL_FACE = "#FBFCFE"
VELOCITY_CMAP = "turbo"

SIGNALS = ["GPS L1", "GPS L5", "Galileo E1", "Galileo E5a"]
SIGNAL_COLORS = {
    "GPS L1": BLUE,
    "GPS L5": LIGHT_BLUE,
    "Galileo E1": GOLD,
    "Galileo E5a": ORANGE,
}
SIGNAL_LABELS = {
    "GPS L1": "GPS L1 C/A",
    "GPS L5": "GPS L5Q",
    "Galileo E1": "Galileo E1C",
    "Galileo E5a": "Galileo E5a-Q",
}
EARTH_LIMB = {"GPS": 13.8, "Galileo": 12.4}
ED4_COLORS = {"Galileo E1": "#3568B3", "Galileo E5a": "#CF4038"}

ED5_EVENTS = {
    "north": "north_track187_OP76_G26",
    "south": "south_track017_OP38_E23",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.facecolor": PANEL_FACE,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.9,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "text.color": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
        }
    )


def _finish(fig: plt.Figure, output: Path, *, dpi: int = 220) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)


def _smoothed_density(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(values, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    kernel_x = np.arange(-8, 9, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / 2.1) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(density, kernel, mode="same")
    support = (centres >= np.min(values)) & (centres <= np.max(values))
    smooth[~support] = np.nan
    if support.sum() > 1:
        area = float(np.trapezoid(smooth[support], centres[support]))
        if area > 0:
            smooth[support] /= area
    return centres, smooth


def _render_ed3_a(root: Path, output: Path) -> dict[str, object]:
    source = root / "data" / "panel_ready" / "ED3" / "ed3_a_off_boresight_points.csv.gz"
    frame = pd.read_csv(source)
    frame = _numeric(frame, ["h_tan_km", "tx_off_boresight_deg"]).dropna()
    frame = frame[frame["signal_name"].isin(SIGNALS)]

    values_all = frame["tx_off_boresight_deg"].to_numpy()
    x_min = math.floor((float(values_all.min()) - 0.3) * 10) / 10
    x_max = math.ceil((float(values_all.max()) + 0.3) * 10) / 10
    bins = np.linspace(x_min, x_max, 72)

    fig = plt.figure(figsize=(7.4, 4.2))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.9, 1.05],
        hspace=0.06,
        left=0.13,
        right=0.98,
        bottom=0.15,
        top=0.83,
    )
    ax = fig.add_subplot(grid[0])
    strip = fig.add_subplot(grid[1], sharex=ax)

    maximum = 0.0
    summaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for signal in SIGNALS:
        values = frame.loc[
            frame["signal_name"].eq(signal), "tx_off_boresight_deg"
        ].to_numpy(dtype=float)
        x, y = _smoothed_density(values, bins)
        quantiles = np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        summaries[signal] = values, x, y, quantiles
        maximum = max(maximum, float(np.nanmax(y)))

    for signal in SIGNALS:
        values, x, y, quantiles = summaries[signal]
        color = SIGNAL_COLORS[signal]
        finite = np.isfinite(y)
        central = finite & (x >= quantiles[1]) & (x <= quantiles[5])
        ax.fill_between(x, 0, y, where=finite, color=color, alpha=0.08, linewidth=0)
        ax.fill_between(x, 0, y, where=central, color=color, alpha=0.20, linewidth=0)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.8,
            label=f"{SIGNAL_LABELS[signal]} (n={len(values):,})",
        )
        ax.scatter(
            [quantiles[3]],
            [np.interp(quantiles[3], x[finite], y[finite])],
            s=26,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )

    for constellation, angle in EARTH_LIMB.items():
        color = GOLD if constellation == "Galileo" else BLUE
        linestyle = ":" if constellation == "Galileo" else "--"
        ax.axvline(angle, color=color, linewidth=1.0, linestyle=linestyle)
        ax.text(
            angle,
            maximum * 1.20,
            f"{constellation} limb {angle:.1f}°",
            ha="center",
            va="top",
            fontsize=7,
            color=color,
        )
        strip.axvline(angle, color=color, linewidth=1.0, linestyle=linestyle)

    for row, signal in enumerate(SIGNALS[::-1]):
        values, _, _, quantiles = summaries[signal]
        color = SIGNAL_COLORS[signal]
        y = row
        strip.plot([quantiles[0], quantiles[-1]], [y, y], color=color, lw=3, alpha=0.5)
        strip.plot([quantiles[1], quantiles[5]], [y, y], color=color, lw=8, alpha=0.34, solid_capstyle="round")
        strip.add_patch(
            patches.Rectangle(
                (quantiles[2], y - 0.18),
                quantiles[4] - quantiles[2],
                0.36,
                facecolor=color,
                edgecolor=color,
                alpha=0.28,
            )
        )
        strip.plot([quantiles[3], quantiles[3]], [y - 0.23, y + 0.23], color=color, lw=1.8)
        strip.scatter([quantiles[0], quantiles[-1]], [y, y], s=20, color=color, zorder=4)
        strip.text(
            x_max - 0.02,
            y + 0.18,
            f"median {quantiles[3]:.2f}°",
            ha="right",
            va="center",
            fontsize=7,
            color=MUTED,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, maximum * 1.25)
    ax.set_ylabel("Probability density")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=7, handlelength=2.2)
    strip.set_yticks(range(4))
    strip.set_yticklabels([SIGNAL_LABELS[s] for s in SIGNALS[::-1]])
    strip.set_xlabel(r"Transmitter off-boresight angle, $\theta$ (deg)")
    strip.grid(axis="x")
    strip.grid(axis="y", visible=False)
    _style_axis(ax)
    _style_axis(strip)
    fig.suptitle(
        "ED3-A | LuGRE transmitter off-boresight distributions",
        x=0.13,
        y=0.965,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.13,
        0.90,
        "Analysis-ready observations with WGS84 tangent heights from 50 to 1,000 km",
        fontsize=8,
        color=MUTED,
    )
    _finish(fig, output)
    return {
        "reproduction_level": "analysis-ready numeric points to panel",
        "source_rows": len(frame),
        "signal_counts": frame["signal_name"].value_counts().to_dict(),
        "scientific_note": "Density smoothing and quantile strips are recalculated at runtime.",
    }


def _render_ed3_b(root: Path, output: Path) -> dict[str, object]:
    source = root / "data" / "panel_ready" / "ED3" / "ed3_b_geometry_parameters.csv"
    parameters = pd.read_csv(source)
    values = dict(zip(parameters["quantity"], parameters["value"]))
    galileo = float(values["Galileo nominal Earth-limb angle"])
    gps = float(values["GPS nominal Earth-limb angle"])

    fig, ax = plt.subplots(figsize=(6.0, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    satellite_y = 0.89
    earth_center = (0.50, 0.16)
    earth_radius = 0.145
    ax.add_patch(
        patches.Circle(
            earth_center,
            earth_radius,
            facecolor="#2D7EBB",
            edgecolor="#1B4F72",
            linewidth=1.8,
            zorder=6,
        )
    )
    ax.add_patch(
        patches.Wedge(
            earth_center,
            earth_radius * 0.98,
            15,
            195,
            facecolor="#70B56D",
            edgecolor="none",
            alpha=0.65,
            zorder=7,
        )
    )
    ax.text(0.50, 0.16, "Earth", color="white", ha="center", va="center", fontweight="bold", zorder=8)

    # Main and side lobes are intentionally schematic, with the two nominal
    # Earth-limb angles read from the packaged numeric parameter table.
    main = np.array([[0.48, satellite_y], [0.13, 0.05], [0.87, 0.05], [0.52, satellite_y]])
    lugre = np.array([[0.492, satellite_y], [0.34, 0.04], [0.66, 0.04], [0.508, satellite_y]])
    left_side = np.array([[0.47, satellite_y], [0.00, 0.43], [0.16, 0.30], [0.49, satellite_y]])
    right_side = np.array([[0.51, satellite_y], [0.84, 0.30], [1.00, 0.43], [0.53, satellite_y]])
    ax.add_patch(patches.Polygon(main, closed=True, facecolor="#C9DDEA", edgecolor="#7CA6BE", linewidth=1.0, alpha=0.75))
    ax.add_patch(patches.Polygon(lugre, closed=True, facecolor="#79C8BE", edgecolor=GREEN, linewidth=1.2, alpha=0.45))
    ax.add_patch(patches.Polygon(left_side, closed=True, facecolor="#D9D0EA", edgecolor="#9780B8", linewidth=0.8, alpha=0.60))
    ax.add_patch(patches.Polygon(right_side, closed=True, facecolor="#D9D0EA", edgecolor="#9780B8", linewidth=0.8, alpha=0.60))

    ax.plot([0.5, 0.5], [satellite_y, 0.30], color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.add_patch(patches.Rectangle((0.455, 0.88), 0.09, 0.045, facecolor="#DCE6F2", edgecolor=INK, linewidth=1.0, zorder=10))
    ax.add_patch(patches.Rectangle((0.33, 0.885), 0.12, 0.035, facecolor="#4373B9", edgecolor=INK, linewidth=0.8, zorder=9))
    ax.add_patch(patches.Rectangle((0.55, 0.885), 0.12, 0.035, facecolor="#4373B9", edgecolor=INK, linewidth=0.8, zorder=9))
    ax.text(0.50, 0.95, "GNSS transmitter", ha="center", va="bottom", fontsize=10, fontweight="bold")

    angle_y = 0.49
    ax.annotate("", xy=(0.36, angle_y), xytext=(0.50, angle_y), arrowprops={"arrowstyle": "<->", "color": BLUE, "lw": 1.4})
    ax.annotate("", xy=(0.64, angle_y - 0.035), xytext=(0.50, angle_y - 0.035), arrowprops={"arrowstyle": "<->", "color": GOLD, "lw": 1.4})
    ax.text(0.405, angle_y + 0.015, f"GPS {gps:.1f}°", color=BLUE, fontsize=9, ha="center", fontweight="bold")
    ax.text(0.595, angle_y - 0.075, f"Galileo {galileo:.1f}°", color=GOLD, fontsize=9, ha="center", fontweight="bold")

    ax.annotate("LuGRE observation cone", xy=(0.65, 0.38), xytext=(0.78, 0.47), color=GREEN, fontsize=9, fontweight="bold", arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.3})
    ax.text(0.72, 0.20, "Main lobe", color=BLUE, fontsize=11, fontweight="bold")
    ax.text(0.79, 0.55, "Side lobe", color="#74599C", fontsize=11, fontweight="bold")
    ax.text(
        0.02,
        0.02,
        "ALTERNATIVE SCHEMATIC - NON-EXACT",
        fontsize=8.5,
        color="#9A3412",
        fontweight="bold",
        zorder=20,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF2E8", "edgecolor": "#F0A36B"},
    )
    fig.suptitle(
        "ED3-B | Earth-limb and antenna-lobe geometry",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.94,
        "Scientific geometry is reproduced; the accepted illustration artwork and placement were not archived.",
        fontsize=8,
        color=MUTED,
    )
    _finish(fig, output)
    return {
        "reproduction_level": "numeric geometry to alternative schematic",
        "pixel_exact": False,
        "alternative_schematic": True,
        "warning": "This panel is intentionally labelled non-exact because the accepted illustration assembler was not archived.",
        "gps_limb_angle_deg": gps,
        "galileo_limb_angle_deg": galileo,
    }


def _render_ed3_c(root: Path, output: Path) -> dict[str, object]:
    data_dir = root / "data" / "panel_ready" / "ED3"
    envelope = pd.read_csv(data_dir / "ed3_c_pattern_envelopes.csv")
    ranges = pd.read_csv(data_dir / "ed3_c_observed_ranges.csv")
    envelope = _numeric(
        envelope,
        ["theta_deg", "median_delta_eirp_db", "p05_delta_eirp_db", "p95_delta_eirp_db"],
    ).dropna(subset=["theta_deg", "median_delta_eirp_db"])

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.5), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    valley_rows: dict[str, float] = {}
    for ax, signal in zip(axes_flat, SIGNALS):
        use = envelope[envelope["signal"].eq(signal)].sort_values("theta_deg")
        use = use[use["theta_deg"].between(0, 75)]
        color = SIGNAL_COLORS[signal]
        x = use["theta_deg"].to_numpy(dtype=float)
        median = use["median_delta_eirp_db"].to_numpy(dtype=float)
        p05 = use["p05_delta_eirp_db"].to_numpy(dtype=float)
        p95 = use["p95_delta_eirp_db"].to_numpy(dtype=float)
        ax.fill_between(x, p05, p95, color=color, alpha=0.18, linewidth=0, label="Pattern P5–P95")
        ax.plot(x, median, color=color, lw=2.0, label="Pattern median")

        observed = ranges[ranges["signal_name"].eq(signal)]
        if not observed.empty:
            lower = float(observed["angle_min_deg"].iloc[0])
            upper = float(observed["angle_max_deg"].iloc[0])
            ax.axvspan(lower, upper, color=GREEN, alpha=0.13, linewidth=0)
            ax.text((lower + upper) / 2, -12, "LuGRE", ha="center", va="center", color=GREEN, fontsize=8, fontweight="bold")

        search = use[use["theta_deg"].between(18, 45)].reset_index(drop=True)
        if not search.empty:
            search_y = search["median_delta_eirp_db"].to_numpy(dtype=float)
            local_minima = np.where(
                (search_y[1:-1] <= search_y[:-2])
                & (search_y[1:-1] < search_y[2:])
                & (search_y[1:-1] < -5.0)
            )[0] + 1
            valley_row = int(local_minima[0]) if local_minima.size else int(np.argmin(search_y))
            valley = float(search.loc[valley_row, "theta_deg"])
            valley_rows[signal] = valley
            ax.axvline(valley, color=MUTED, lw=1.0, ls=(0, (3, 3)))
            ax.text(valley + 1, -41.5, f"{valley:.0f}°", color=MUTED, fontsize=7)

        limb = EARTH_LIMB["GPS" if signal.startswith("GPS") else "Galileo"]
        ax.axvline(limb, color=INK, lw=0.75, ls=":")
        ax.set_title(SIGNAL_LABELS[signal], loc="left", color=color, fontweight="bold")
        ax.set_xlim(0, 75)
        ax.set_ylim(-45, 4.5)
        _style_axis(ax)

    axes[0, 0].set_ylabel("Relative EIRP (dB)")
    axes[1, 0].set_ylabel("Relative EIRP (dB)")
    axes[1, 0].set_xlabel(r"Off-boresight angle, $\theta$ (deg)")
    axes[1, 1].set_xlabel(r"Off-boresight angle, $\theta$ (deg)")
    handles = [
        Line2D([0], [0], color=BLUE, lw=2, label="Pattern median"),
        patches.Patch(facecolor=BLUE, alpha=0.18, label="Pattern P5–P95"),
        patches.Patch(facecolor=GREEN, alpha=0.13, label="LuGRE observed range"),
        Line2D([0], [0], color=MUTED, lw=1, ls=(0, (3, 3)), label="Derived first-valley marker"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.53, 0.01))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.87, wspace=0.14, hspace=0.18)
    fig.suptitle(
        "ED3-C | GNSS antenna transmit-pattern evidence",
        x=0.10,
        y=0.97,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.10,
        0.91,
        "Archived NAVCEN/GRAP-derived median and P5–P95 envelopes; LuGRE ranges from 50–1,000 km tangent points",
        fontsize=8,
        color=MUTED,
    )
    _finish(fig, output)
    return {
        "reproduction_level": "analysis-ready pattern envelopes to panel",
        "source_rows": len(envelope),
        "first_valley_markers_deg": valley_rows,
        "scientific_note": "Transmit-pattern envelopes and LuGRE angle ranges are numeric, not raster traces.",
    }


def _ed4_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = root / "data" / "panel_ready" / "ED4"
    cn0 = pd.read_csv(data_dir / "op38_e23_anomaly_focused_cn0_data.csv")
    height = pd.read_csv(data_dir / "op38_e23_anomaly_focused_tangent_height.csv")
    footprint = pd.read_csv(data_dir / "op38_e23_anomaly_focused_simulated_footprint.csv")
    cn0["time"] = pd.to_datetime(cn0["time_dt"], utc=True, errors="coerce")
    height["time"] = pd.to_datetime(height["time_dt"], utc=True, errors="coerce")
    footprint["time"] = pd.to_datetime(footprint["time_dt"], utc=True, errors="coerce")
    cn0 = _numeric(cn0, ["cn0_dbhz", "cn0_trend_dbhz", "cn0_detrended_db", "h_tan_km"])
    height = _numeric(height, ["h_tan_wgs84_km", "lat_wgs84_deg", "lon_wgs84_deg"])
    footprint = _numeric(footprint, ["tp_lat_wgs84_deg", "tp_lon_wgs84_deg", "h_use_wgs84_km"])
    return cn0, height, footprint


def _plot_time_segments(
    ax: plt.Axes,
    frame: pd.DataFrame,
    y_column: str,
    *,
    color: str,
    linewidth: float = 1.5,
    linestyle: str | tuple = "-",
    gap_seconds: float = 90.0,
    alpha: float = 1.0,
) -> int:
    use = frame.dropna(subset=["time", y_column]).sort_values("time").copy()
    if use.empty:
        return 0
    group = use["time"].diff().dt.total_seconds().fillna(0).gt(gap_seconds).cumsum()
    count = 0
    for _, segment in use.groupby(group):
        ax.plot(
            segment["time"],
            segment[y_column],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
        count += 1
    return count


def _format_ed4_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC on 2025-03-03")
    _style_axis(ax)


def _render_ed4_a(root: Path, output: Path) -> dict[str, object]:
    cn0, _, _ = _ed4_data(root)
    observed = cn0[~cn0["cn0_is_filled"].astype(str).str.lower().eq("true")].copy()
    fig, ax = plt.subplots(figsize=(9.3, 4.2))
    rows = {}
    segments = 0
    for signal, marker in [("Galileo E1", "s"), ("Galileo E5a", "D")]:
        use = observed[observed["signal_name"].eq(signal)]
        color = ED4_COLORS[signal]
        rows[signal] = len(use)
        ax.scatter(use["time"], use["cn0_dbhz"], s=10, marker=marker, color=color, alpha=0.48, linewidths=0, label=SIGNAL_LABELS[signal])
        segments += _plot_time_segments(ax, use, "cn0_trend_dbhz", color=INK, linewidth=1.5, linestyle=(0, (4, 3)), gap_seconds=120)
    ax.set_ylabel(r"Receiver-reported $C/N_0$ (dB-Hz)")
    ax.set_title(r"ED4-A | Raw receiver-reported $C/N_0$ and fourth-order baseline", loc="left", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=INK, lw=1.5, ls=(0, (4, 3))))
    labels.append("fourth-order polynomial fit")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False)
    _format_ed4_time_axis(ax)
    fig.tight_layout()
    _finish(fig, output)
    return {
        "reproduction_level": "analysis-ready numeric table to helper-independent panel",
        "source_rows": len(cn0),
        "observed_rows": len(observed),
        "rows_by_signal": rows,
        "visible_time_segments": segments,
    }


def _render_ed4_b(root: Path, output: Path) -> dict[str, object]:
    cn0, _, _ = _ed4_data(root)
    observed = cn0[~cn0["cn0_is_filled"].astype(str).str.lower().eq("true")].copy()
    fig, ax = plt.subplots(figsize=(9.3, 3.8))
    for signal, marker in [("Galileo E1", "s"), ("Galileo E5a", "D")]:
        use = observed[observed["signal_name"].eq(signal)]
        ax.scatter(
            use["time"],
            use["cn0_detrended_db"],
            s=10,
            marker=marker,
            color=ED4_COLORS[signal],
            alpha=0.50,
            linewidths=0,
            label=SIGNAL_LABELS[signal],
        )
    ax.axhline(0, color=INK, lw=1.0)
    robust = observed["cn0_detrended_db"].abs().quantile(0.995)
    limit = max(2.0, min(4.0, math.ceil(float(robust) * 2) / 2))
    ax.set_ylim(-limit, limit)
    ax.set_ylabel(r"Residual $\delta C/N_0$ (dB)")
    ax.set_title("ED4-B | Residual after whole-arc fourth-order detrending", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    _format_ed4_time_axis(ax)
    fig.tight_layout()
    _finish(fig, output)
    return {
        "reproduction_level": "analysis-ready residual table to helper-independent panel",
        "source_rows": len(observed),
        "residual_std_db": {
            signal: float(group["cn0_detrended_db"].std())
            for signal, group in observed.groupby("signal_name")
        },
    }


def _render_ed4_c(root: Path, output: Path) -> dict[str, object]:
    _, height, _ = _ed4_data(root)
    height = height.dropna(subset=["time", "h_tan_wgs84_km"])
    fig, ax = plt.subplots(figsize=(9.3, 4.0))
    ax.axhspan(50, 1000, color="#F4DFE6", alpha=0.55, zorder=0)
    ax.text(
        0.985,
        0.92,
        "50–1,000 km ionosphere shell",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#8D3C56",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    segment_count = 0
    for signal in ["Galileo E1", "Galileo E5a"]:
        use = height[height["signal_name"].eq(signal)].sort_values("time")
        color = ED4_COLORS[signal]
        inside = use["h_tan_wgs84_km"].between(50, 1000)
        outside = use.copy()
        outside.loc[inside, "h_tan_wgs84_km"] = np.nan
        inner = use.copy()
        inner.loc[~inside, "h_tan_wgs84_km"] = np.nan
        segment_count += _plot_time_segments(ax, outside, "h_tan_wgs84_km", color="#8A929E", linewidth=1.2, linestyle="--", gap_seconds=120)
        segment_count += _plot_time_segments(ax, inner, "h_tan_wgs84_km", color=color, linewidth=2.2, gap_seconds=120)
    ax.set_ylim(-650, 1040)
    ax.set_ylabel("WGS84 tangent height (km)")
    ax.set_title("ED4-C | Tangent height through the 50–1,000 km shell", loc="left", fontweight="bold")
    _format_ed4_time_axis(ax)
    fig.tight_layout()
    _finish(fig, output)
    return {
        "reproduction_level": "analysis-ready WGS84 tangent geometry to helper-independent panel",
        "source_rows": len(height),
        "shell_rows": int(height["h_tan_wgs84_km"].between(50, 1000).sum()),
        "visible_time_segments": segment_count,
    }


def _render_ed4_d(root: Path, output: Path) -> dict[str, object]:
    _, _, footprint = _ed4_data(root)
    footprint = footprint.dropna(subset=["tp_lat_wgs84_deg", "tp_lon_wgs84_deg"])
    fig, ax = plt.subplots(figsize=(5.1, 7.2))
    shell_rows = 0
    for signal in ["Galileo E1", "Galileo E5a"]:
        use = footprint[footprint["signal_name"].eq(signal)].sort_values("time").copy()
        color = ED4_COLORS[signal]
        inside = use["h_use_wgs84_km"].between(50, 1000)
        shell_rows += int(inside.sum())
        ax.plot(
            use.loc[~inside, "tp_lon_wgs84_deg"],
            use.loc[~inside, "tp_lat_wgs84_deg"],
            color="#8A929E",
            lw=1.6,
            ls="--",
            alpha=0.9,
        )
        ax.plot(
            use.loc[inside, "tp_lon_wgs84_deg"],
            use.loc[inside, "tp_lat_wgs84_deg"],
            color=color,
            lw=2.4,
            label=SIGNAL_LABELS[signal],
        )
        if inside.any():
            first = use.loc[inside].iloc[0]
            last = use.loc[inside].iloc[-1]
            ax.scatter([first["tp_lon_wgs84_deg"]], [first["tp_lat_wgs84_deg"]], marker=">", s=45, color=color, edgecolor="white", linewidth=0.6, zorder=5)
            ax.scatter([last["tp_lon_wgs84_deg"]], [last["tp_lat_wgs84_deg"]], marker="o", s=40, facecolor="white", edgecolor=color, linewidth=1.4, zorder=5)
    ax.set_xlim(-105, 45)
    ax.set_ylim(-80, 33)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("ED4-D | WGS84 tangent-point footprint", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    _finish(fig, output)
    return {
        "reproduction_level": "analysis-ready simulated WGS84 footprint to helper-independent panel",
        "source_rows": len(footprint),
        "shell_rows": shell_rows,
    }


def _draw_altitude_layers(ax: plt.Axes) -> None:
    ax.axhspan(50, 90, color="#FFF2B8", alpha=0.75, zorder=0)
    ax.axhspan(90, 150, color="#FAD8CE", alpha=0.70, zorder=0)
    ax.axhspan(150, 600, color="#DFF2E7", alpha=0.65, zorder=0)
    ax.axhspan(600, 1000, color="#E3EEF7", alpha=0.72, zorder=0)


def _set_height_axis(ax: plt.Axes, *, side: str = "left") -> None:
    ax.set_yscale("log")
    ax.set_ylim(50, 1000)
    ticks = [50, 90, 150, 300, 600, 1000]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(value) for value in ticks])
    ax.minorticks_off()
    ax.set_ylabel("Tangent height (km)")
    if side == "right":
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    _style_axis(ax)


def _ed5_profiles(root: Path, event: str) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    data_dir = root / "data" / "panel_ready" / "ED5"
    lugre = pd.read_csv(data_dir / "ed5_lugre_profiles.csv.gz")
    leoro = pd.read_csv(data_dir / "ed5_leoro_profiles.csv.gz")
    high = pd.read_csv(data_dir / "ed5_leoro_highrate_l1.csv.gz")
    frames: list[pd.DataFrame] = []

    lugre = lugre[lugre["event"].eq(event)].copy()
    lugre["time"] = pd.to_datetime(lugre["time_utc"], utc=True, errors="coerce")
    lugre = _numeric(lugre, ["height_km", "dcn0_db"]).dropna(subset=["time", "height_km", "dcn0_db"])
    for (signal, label), group in lugre.groupby(["signal", "label"], sort=True):
        color = {
            "GPS L1": "#2F80ED",
            "GPS L5": "#22A766",
            "Galileo E1": "#2F80ED",
            "Galileo E5a": "#22A766",
        }.get(signal, BLUE)
        marker = "o" if signal in {"GPS L1", "Galileo E1"} else "^"
        display_signal = signal.replace("GPS ", "").replace("Galileo ", "")
        satellite = "G26" if "north" in event else "E23"
        use = group[["time", "height_km", "dcn0_db"]].copy()
        use["label"] = f"LuGRE {satellite} {display_signal}"
        use["color"] = color
        use["marker"] = marker
        frames.append(use.sort_values("time"))

    leoro = leoro[leoro["event"].eq(event)].copy()
    leoro["time"] = pd.to_datetime(leoro["time_utc"], utc=True, errors="coerce")
    leoro = _numeric(leoro, ["height_km", "dcn0_db"]).dropna(subset=["time", "height_km", "dcn0_db"])
    for (mission, signal, event_label), group in leoro.groupby(["mission", "signal", "event_label"], sort=True):
        color = "#E45756" if str(signal).upper() == "L1" else PURPLE
        marker = "o" if str(signal).upper() == "L1" else "D"
        tokens = str(event_label).split("/")
        prn = ""
        if len(tokens) > 1:
            parts = tokens[1].strip().split()
            if len(parts) > 1:
                prn = parts[1]
        use = group[["time", "height_km", "dcn0_db"]].copy()
        use["label"] = f"{mission} {prn} {signal}".strip()
        use["color"] = color
        use["marker"] = marker
        frames.append(use.sort_values("time"))

    high = high[high["event"].eq(event)].copy()
    high["time"] = pd.to_datetime(high["time_utc"], utc=True, errors="coerce")
    high = _numeric(high, ["height_km", "dcn0_db"]).dropna(subset=["time", "height_km", "dcn0_db"])
    return frames, high


def _render_ed5_height(root: Path, output: Path, *, hemisphere: str) -> dict[str, object]:
    event = ED5_EVENTS[hemisphere]
    profiles, _ = _ed5_profiles(root, event)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    _draw_altitude_layers(ax)
    rows = {}
    labelled: set[str] = set()
    for frame in profiles:
        label = str(frame["label"].iloc[0])
        rows[label] = len(frame)
        legend_label = label if label not in labelled else "_nolegend_"
        ax.plot(
            frame["time"],
            frame["height_km"],
            color=frame["color"].iloc[0],
            lw=1.8,
            alpha=0.9,
            label=legend_label,
        )
        labelled.add(label)
    _set_height_axis(ax)
    all_times = pd.concat([frame["time"] for frame in profiles], ignore_index=True)
    start = all_times.min().floor("min")
    end = all_times.max().ceil("min")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC")
    title = "OP76 GPS G26 | north polar night" if hemisphere == "north" else "OP38 Galileo E23 | south polar day"
    ax.set_title(f"ED5-{'B' if hemisphere == 'north' else 'G'} | Height–time", loc="left", fontweight="bold")
    ax.text(0.98, 0.96, title, transform=ax.transAxes, ha="right", va="top", fontsize=7.5, fontweight="bold")
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    fig.tight_layout()
    _finish(fig, output)
    return {
        "reproduction_level": "selected-event panel-ready LuGRE and LEO-RO profiles to panel",
        "event": event,
        "profile_rows": rows,
    }


def _render_ed5_dcn0(root: Path, output: Path, *, hemisphere: str) -> dict[str, object]:
    event = ED5_EVENTS[hemisphere]
    profiles, high = _ed5_profiles(root, event)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    _draw_altitude_layers(ax)
    if not high.empty:
        ax.scatter(high["dcn0_db"], high["height_km"], s=5, color="#7B8490", alpha=0.17, linewidths=0, rasterized=True)
    handles = []
    rows = {}
    seen = set()
    for frame in profiles:
        label = str(frame["label"].iloc[0])
        rows[label] = len(frame)
        ax.scatter(
            frame["dcn0_db"],
            frame["height_km"],
            s=17,
            marker=frame["marker"].iloc[0],
            color=frame["color"].iloc[0],
            alpha=0.67,
            linewidths=0,
            rasterized=True,
        )
        if label not in seen:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=frame["marker"].iloc[0],
                    color="none",
                    markerfacecolor=frame["color"].iloc[0],
                    markeredgecolor="none",
                    label=label,
                    markersize=6,
                )
            )
            seen.add(label)
    ax.axvline(0, color=AXIS, lw=0.9, ls=(0, (3, 3)))
    ax.set_xlim(-6, 6)
    ax.set_xticks([-6, -3, 0, 3, 6])
    _set_height_axis(ax, side="right")
    ax.set_xlabel(r"$\delta C/N_0$ (dB)")
    panel = "C" if hemisphere == "north" else "H"
    ax.set_title(rf"ED5-{panel} | Detrended $C/N_0$ by tangent height", loc="left", fontweight="bold")
    ax.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=6.7, columnspacing=0.7, handletextpad=0.2)
    fig.tight_layout()
    _finish(fig, output)
    return {
        "reproduction_level": "selected-event panel-ready LuGRE and LEO-RO residuals to panel",
        "event": event,
        "profile_rows": rows,
        "highrate_background_rows": len(high),
    }


def _render_ed5_ne(root: Path, output: Path, *, hemisphere: str) -> dict[str, object]:
    event = ED5_EVENTS[hemisphere]
    source = root / "data" / "panel_ready" / "ED5" / "ed5_ne_profiles.csv.gz"
    frame = pd.read_csv(source)
    frame = frame[frame["event"].eq(event)].copy()
    frame = _numeric(frame, ["height_km", "log10_ne_m3"]).dropna()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    _draw_altitude_layers(ax)
    styles = {
        "IRI": {"color": INK, "lw": 2.0, "ls": (0, (4, 3))},
        "YunYao": {"color": "#C53024", "lw": 2.1, "ls": "-"},
        "PFISR": {"color": "#1D6FA3", "lw": 1.7, "ls": "-"},
    }
    counts = {}
    for (profile_type, label), group in frame.groupby(["profile_type", "label"], sort=True):
        use = group.sort_values("height_km")
        style = styles.get(profile_type, {"color": MUTED, "lw": 1.5, "ls": "-"})
        ax.plot(use["log10_ne_m3"], use["height_km"], label=label, **style)
        counts[label] = len(use)
    _set_height_axis(ax)
    ax.set_xlim(8, 12.3)
    ax.set_xticks([8, 10, 12])
    ax.set_xlabel(r"$\log_{10} N_e$ (m$^{-3}$)")
    panel = "D" if hemisphere == "north" else "I"
    ax.set_title(f"ED5-{panel} | Electron-density profile, $N_e(h)$", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    _finish(fig, output)
    return {
        "reproduction_level": "selected-event YunYao/IRI/PFISR numeric profiles to panel",
        "event": event,
        "profile_rows": counts,
    }


def _polar_radius(latitude: pd.Series, hemisphere: str) -> np.ndarray:
    lat = pd.to_numeric(latitude, errors="coerce").to_numpy(dtype=float)
    return 90.0 - lat if hemisphere == "north" else 90.0 + lat


def _render_ed5_map(root: Path, output: Path, *, hemisphere: str) -> dict[str, object]:
    event = ED5_EVENTS[hemisphere]
    data_dir = root / "data" / "panel_ready" / "ED5"
    cells = pd.read_csv(data_dir / "ed5_map_mean_abs_velocity_cells.csv.gz")
    tracks = pd.read_csv(data_dir / "ed5_map_tracks.csv.gz")
    cells = cells[cells["event"].eq(event)].copy()
    tracks = tracks[tracks["event"].eq(event)].copy()
    cells = _numeric(cells, ["latitude_deg", "longitude_deg", "mean_abs_velocity_ms"]).dropna(
        subset=["latitude_deg", "longitude_deg", "mean_abs_velocity_ms"]
    )
    tracks = _numeric(tracks, ["latitude_deg", "longitude_deg"]).dropna(
        subset=["latitude_deg", "longitude_deg"]
    )

    cell_radius = _polar_radius(cells["latitude_deg"], hemisphere)
    cell_theta = np.deg2rad(cells["longitude_deg"].to_numpy(dtype=float))
    keep = (cell_radius >= 0) & (cell_radius <= 30)
    fig = plt.figure(figsize=(6.4, 5.8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    scatter = ax.scatter(
        cell_theta[keep],
        cell_radius[keep],
        c=cells.loc[keep, "mean_abs_velocity_ms"],
        cmap=VELOCITY_CMAP,
        norm=Normalize(0, 2000),
        s=13,
        marker="s",
        linewidths=0,
        alpha=0.78,
        rasterized=True,
        zorder=2,
    )
    track_counts = {}
    for track_type, group in tracks.groupby("track_type", sort=True):
        radius = _polar_radius(group["latitude_deg"], hemisphere)
        theta = np.unwrap(np.deg2rad(group["longitude_deg"].to_numpy(dtype=float)))
        valid = (radius >= 0) & (radius <= 30)
        color = GREEN if track_type == "LuGRE" else "#B65F00"
        ax.plot(theta[valid], radius[valid], color=color, lw=2.2, label=track_type, zorder=5)
        track_counts[track_type] = int(valid.sum())
    ax.set_rlim(0, 30)
    ax.set_rticks([0, 10, 20, 30])
    labels = ["90°N", "80°N", "70°N", "60°N"] if hemisphere == "north" else ["90°S", "80°S", "70°S", "60°S"]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_thetagrids([0, 90, 180, 270], ["0°", "90°E", "180°", "90°W"])
    ax.grid(color="white", linewidth=0.65, alpha=0.8)
    ax.set_facecolor("#DCEAF4")
    panel = "A" if hemisphere == "north" else "F"
    event_label = "OP76 GPS G26 | north polar night" if hemisphere == "north" else "OP38 Galileo E23 | south polar day"
    ax.set_title(f"ED5-{panel} | {event_label}", loc="left", pad=18, fontweight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.08, -0.10), frameon=False, fontsize=8)
    colorbar = fig.colorbar(scatter, ax=ax, orientation="vertical", pad=0.10, shrink=0.72)
    colorbar.set_label(r"SuperDARN mean $|V_{\rm LOS}|$ (m/s)")
    colorbar.set_ticks([0, 500, 1000, 1500, 2000])
    fig.text(
        0.11,
        0.02,
        "Final 2026-07-27 observable; values above 2,000 m/s saturate at the upper color limit.",
        fontsize=7.5,
        color=MUTED,
    )
    _finish(fig, output)
    return {
        "reproduction_level": "selected-event numeric polar tracks and 2026-07-27 mean absolute SuperDARN velocity cells to panel",
        "event": event,
        "mean_abs_velocity_cells_total": len(cells),
        "mean_abs_velocity_cells_inside_60deg_boundary": int(keep.sum()),
        "track_rows": track_counts,
        "velocity_limit_ms": 2000,
    }


def _midpoint_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if centres.size == 1:
        return np.array([centres[0] - 1.0, centres[0] + 1.0])
    interior = 0.5 * (centres[:-1] + centres[1:])
    first = centres[0] - 0.5 * (centres[1] - centres[0])
    last = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return np.concatenate(([first], interior, [last]))


def _render_ed5_fan(root: Path, output: Path, *, hemisphere: str) -> dict[str, object]:
    filename = "ed5_fan_north_mean_abs_velocity.csv.gz" if hemisphere == "north" else "ed5_fan_south_mean_abs_velocity.csv.gz"
    source = root / "data" / "panel_ready" / "ED5" / filename
    fan = pd.read_csv(source)
    fan = _numeric(
        fan,
        ["beam", "range_gate", "bmazm_deg", "range_start_km", "range_center_km", "range_width_km", "mean_abs_velocity_ms"],
    )
    fan = fan.dropna(subset=["beam", "range_gate", "bmazm_deg", "range_start_km", "range_width_km"])
    bearing_by_beam = fan.groupby("beam")["bmazm_deg"].median()
    if hemisphere == "south" and len(bearing_by_beam) >= 4:
        raw_center = float(np.nanmedian(bearing_by_beam.to_numpy(dtype=float)))
        raw_span = float(
            np.nanpercentile(
                np.abs(
                    (
                        bearing_by_beam.to_numpy(dtype=float)
                        - raw_center
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                ),
                98,
            )
        )
        if raw_span < 2.0:
            beam_mid = float(np.nanmedian(bearing_by_beam.index.to_numpy(dtype=float)))
            bearing_by_beam = pd.Series(
                raw_center
                + (bearing_by_beam.index.to_numpy(dtype=float) - beam_mid) * 3.24,
                index=bearing_by_beam.index,
            )
    bearing_by_beam = bearing_by_beam.sort_values()
    beams = bearing_by_beam.index.to_numpy(dtype=float)
    bearings = bearing_by_beam.to_numpy(dtype=float)
    range_meta = (
        fan.groupby("range_gate")[["range_start_km", "range_width_km"]]
        .median()
        .sort_values("range_start_km")
    )
    gates = range_meta.index.to_numpy(dtype=float)
    starts = range_meta["range_start_km"].to_numpy(dtype=float)
    widths = range_meta["range_width_km"].to_numpy(dtype=float)
    values = (
        fan.pivot_table(index="range_gate", columns="beam", values="mean_abs_velocity_ms", aggfunc="mean")
        .reindex(index=gates, columns=beams)
        .to_numpy(dtype=float)
    )
    theta_edges = np.deg2rad(_midpoint_edges(bearings))
    range_edges = np.concatenate(([starts[0]], starts + widths))
    masked = np.ma.masked_invalid(values)

    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(111, projection="polar")
    cmap = plt.get_cmap(VELOCITY_CMAP).copy()
    cmap.set_bad("#E6EAF0")
    mesh = ax.pcolormesh(
        theta_edges,
        range_edges,
        masked,
        cmap=cmap,
        norm=Normalize(0, 2000),
        shading="flat",
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(float(np.min(np.rad2deg(theta_edges))))
    ax.set_thetamax(float(np.max(np.rad2deg(theta_edges))))
    ax.set_ylim(0, float(range_edges[-1]))
    ax.grid(color=GRID, linewidth=0.6)
    panel = "E" if hemisphere == "north" else "J"
    radar = str(fan["radar"].dropna().iloc[0]).upper() if len(fan["radar"].dropna()) else "SuperDARN"
    ax.set_title(f"ED5-{panel} | SuperDARN {radar} fan", loc="left", pad=16, fontweight="bold")
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.14, shrink=0.85)
    colorbar.set_label(r"Mean $|V_{\rm LOS}|$ (m/s)")
    colorbar.set_ticks([0, 500, 1000, 1500, 2000])
    fig.text(
        0.11,
        0.02,
        "Contiguous beam × range-gate cells; missing velocity estimates are light grey.",
        fontsize=7.5,
        color=MUTED,
    )
    _finish(fig, output)
    finite = np.isfinite(values)
    return {
        "reproduction_level": "selected-event 2026-07-27 mean absolute SuperDARN velocity table to filled fan panel",
        "event": ED5_EVENTS[hemisphere],
        "fan_rows": len(fan),
        "finite_velocity_cells": int(finite.sum()),
        "beam_count": len(beams),
        "range_gate_count": len(gates),
        "velocity_limit_ms": 2000,
    }


def render_panel(task_id: str, root: Path, output: Path) -> dict[str, object]:
    _configure_style()
    dispatch: dict[str, Callable[[], dict[str, object]]] = {
        "ED3_A": lambda: _render_ed3_a(root, output),
        "ED3_B": lambda: _render_ed3_b(root, output),
        "ED3_C": lambda: _render_ed3_c(root, output),
        "ED4_A": lambda: _render_ed4_a(root, output),
        "ED4_B": lambda: _render_ed4_b(root, output),
        "ED4_C": lambda: _render_ed4_c(root, output),
        "ED4_D": lambda: _render_ed4_d(root, output),
        "ED5_A": lambda: _render_ed5_map(root, output, hemisphere="north"),
        "ED5_B": lambda: _render_ed5_height(root, output, hemisphere="north"),
        "ED5_C": lambda: _render_ed5_dcn0(root, output, hemisphere="north"),
        "ED5_D": lambda: _render_ed5_ne(root, output, hemisphere="north"),
        "ED5_E": lambda: _render_ed5_fan(root, output, hemisphere="north"),
        "ED5_F": lambda: _render_ed5_map(root, output, hemisphere="south"),
        "ED5_G": lambda: _render_ed5_height(root, output, hemisphere="south"),
        "ED5_H": lambda: _render_ed5_dcn0(root, output, hemisphere="south"),
        "ED5_I": lambda: _render_ed5_ne(root, output, hemisphere="south"),
        "ED5_J": lambda: _render_ed5_fan(root, output, hemisphere="south"),
    }
    if task_id not in dispatch:
        raise ValueError(f"No panel renderer registered for {task_id}")
    return dispatch[task_id]()
