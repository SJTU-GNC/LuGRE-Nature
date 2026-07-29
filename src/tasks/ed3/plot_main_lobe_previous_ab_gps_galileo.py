from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

import plot_main_lobe_gps_galileo as core
import plot_main_lobe_nature_figures as base


ROOT = Path(__file__).resolve().parent
POINT_CSV = ROOT / "lugre_gnss_main_lobe_point_check.csv"
OUTPUT_STEM = ROOT / "lugre_main_lobe_ab_figure_nature_previous_ab_gps_galileo"
CURRENT_STEM = ROOT / "lugre_main_lobe_ab_figure_nature"
TANGENT_HEIGHT_MIN_KM = 50.0
TANGENT_HEIGHT_MAX_KM = 1000.0

OLD_COLORS = {
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
}


def load_previous_points() -> pd.DataFrame:
    df = pd.read_csv(
        POINT_CSV,
        usecols=["signal_name", "h_tan_km", "tx_off_boresight_deg"],
    )
    df = df[df["signal_name"].isin(core.SIGNALS)].copy()
    for column in ("h_tan_km", "tx_off_boresight_deg"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["h_tan_km", "tx_off_boresight_deg"])
    return df[
        df["h_tan_km"].between(
            TANGENT_HEIGHT_MIN_KM,
            TANGENT_HEIGHT_MAX_KM,
            inclusive="both",
        )
    ].copy()


def smoothed_histogram(
    values: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(values, bins=bins, density=True)
    x = 0.5 * (edges[:-1] + edges[1:])
    kernel_x = np.arange(-8, 9, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / 2.1) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(density, kernel, mode="same")

    # Keep the smoothed density on the exact support used by the summary below.
    support = (x >= float(np.min(values))) & (x <= float(np.max(values)))
    smooth[~support] = np.nan
    if np.count_nonzero(support) > 1:
        area = float(np.trapezoid(smooth[support], x[support]))
        if area > 0:
            smooth[support] /= area
    return x, smooth


def style_boxed_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=OLD_COLORS["grid"], linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(OLD_COLORS["ink"])
        spine.set_linewidth(0.72)


def draw_previous_panel_a(
    fig: plt.Figure,
    subplot_spec,
    df: pd.DataFrame,
) -> tuple[plt.Axes, plt.Axes, float, float]:
    grid = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=subplot_spec,
        height_ratios=[1.95, 1.08],
        hspace=0.06,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_box = fig.add_subplot(grid[1, 0], sharex=ax)

    observed_min = float(df["tx_off_boresight_deg"].min())
    observed_max = float(df["tx_off_boresight_deg"].max())
    x_min = math.floor((observed_min - 0.35) * 10) / 10
    x_max = math.ceil((observed_max + 0.35) * 10) / 10
    bins = np.linspace(x_min, x_max, 72)

    curves: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    max_y = 0.0
    for signal in core.SIGNALS:
        values = df.loc[
            df["signal_name"] == signal, "tx_off_boresight_deg"
        ].to_numpy(dtype=float)
        x, y = smoothed_histogram(values, bins)
        quantiles = np.quantile(
            values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
        )
        curves[signal] = values, x, y, quantiles
        max_y = max(max_y, float(np.nanmax(y)))

    for signal in core.SIGNALS:
        values, x, y, quantiles = curves[signal]
        color = OLD_COLORS[signal]
        finite = np.isfinite(y)
        central = finite & (x >= quantiles[1]) & (x <= quantiles[5])
        ax.fill_between(x, y, 0, where=finite, color=color, alpha=0.08, linewidth=0)
        ax.fill_between(
            x,
            y,
            0,
            where=central,
            color=color,
            alpha=0.19,
            linewidth=0,
        )
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.55,
            label=f"{core.SIGNAL_LABELS[signal]} (n={len(values):,})",
        )
        median_density = float(np.interp(quantiles[3], x[finite], y[finite]))
        ax.scatter(
            [quantiles[3]],
            [median_density],
            s=13,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
        )

    for constellation in ("Galileo", "GPS"):
        limb_angle = core.EARTH_LIMB_DEG[constellation]
        ax.axvline(
            limb_angle,
            color=core.EARTH_LIMB_COLOR[constellation],
            linewidth=0.9,
            linestyle=core.EARTH_LIMB_STYLE[constellation],
            zorder=1,
        )
        ax.text(
            limb_angle,
            max_y * 1.205,
            rf"{constellation} limb ${limb_angle:.1f}^\circ$",
            fontsize=5.5,
            color=core.EARTH_LIMB_COLOR[constellation],
            ha="center",
            va="top",
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_y * 1.25)
    ax.set_ylabel("Probability density")
    ax.tick_params(labelbottom=False)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.05,
        borderpad=0.1,
        fontsize=5.8,
    )

    ax.text(
        0.0,
        1.13,
        "a  LuGRE off-boresight distributions at 50-1,000 km tangent heights",
        transform=ax.transAxes,
        fontsize=10.4,
        fontweight="bold",
        va="bottom",
        color=OLD_COLORS["ink"],
    )
    style_boxed_axis(ax)

    positions = {
        signal: float(len(core.SIGNALS) - index - 1)
        for index, signal in enumerate(core.SIGNALS)
    }
    for signal in core.SIGNALS:
        values, _, _, q = curves[signal]
        y = positions[signal]
        color = OLD_COLORS[signal]
        ax_box.plot([q[0], q[6]], [y, y], color=color, linewidth=0.9, alpha=0.72)
        ax_box.plot(
            [q[1], q[5]],
            [y, y],
            color=color,
            linewidth=4.5,
            alpha=0.58,
            solid_capstyle="round",
        )
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
        ax_box.plot(
            [q[3], q[3]], [y - 0.17, y + 0.17], color=color, linewidth=1.7
        )
        ax_box.scatter([q[0], q[6]], [y, y], s=10, color=color, zorder=3)
        ax_box.text(
            x_max - 0.03,
            y + 0.20,
            rf"median ${q[3]:.2f}^\circ$",
            ha="right",
            va="center",
            fontsize=5.4,
            color=OLD_COLORS["muted"],
        )

    for constellation in ("Galileo", "GPS"):
        ax_box.axvline(
            core.EARTH_LIMB_DEG[constellation],
            color=core.EARTH_LIMB_COLOR[constellation],
            linewidth=0.9,
            linestyle=core.EARTH_LIMB_STYLE[constellation],
            zorder=1,
        )
    ax_box.set_ylim(-0.52, len(core.SIGNALS) - 0.48)
    ax_box.set_yticks([positions[signal] for signal in core.SIGNALS])
    ax_box.set_yticklabels([core.SIGNAL_LABELS[signal] for signal in core.SIGNALS])
    ax_box.set_xlabel(r"Transmitter off-boresight angle, $\alpha$ (deg)")
    ax_box.grid(True, axis="x", color=OLD_COLORS["grid"], linewidth=0.5)
    ax_box.grid(False, axis="y")
    style_boxed_axis(ax_box)
    return ax, ax_box, observed_min, observed_max


def first_valley_summary(grap: dict[str, np.ndarray]) -> dict[str, float]:
    theta = grap["theta"]
    valleys: list[float] = []
    kernel = np.ones(5, dtype=float) / 5.0
    for profile in grap["eirp"]:
        smooth = np.convolve(profile, kernel, mode="same")
        candidates = [
            index
            for index in range(7, min(60, len(theta) - 1))
            if smooth[index] <= smooth[index - 1]
            and smooth[index] < smooth[index + 1]
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


def load_gps_pattern_set(
    signal: str,
    raw_windows: pd.DataFrame,
    prn_svn: dict[str, int],
) -> tuple[
    dict[int, list[tuple[np.ndarray, np.ndarray]]],
    list[int],
    list[int],
]:
    observed_prns = set(
        raw_windows.loc[raw_windows["signal_name"] == signal, "sat"].astype(str)
    )
    observed_svns = sorted(
        {prn_svn[prn] for prn in observed_prns if prn in prn_svn}
    )
    target_svns = set(observed_svns)
    patterns: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    patterns.update(
        core.load_iir_patterns(target_svns & set(range(41, 62)), signal)
    )
    patterns.update(
        core.load_iif_patterns(target_svns & set(range(62, 74)), signal)
    )
    patterns.update(
        core.load_gps_iii_patterns(target_svns & set(range(74, 79)), signal)
    )
    missing = sorted(target_svns - set(patterns))
    return patterns, observed_svns, missing


def detect_prominent_first_valley(
    theta: np.ndarray,
    gain: np.ndarray,
) -> float | None:
    valid = np.isfinite(theta) & np.isfinite(gain)
    theta = np.asarray(theta, dtype=float)[valid]
    gain = np.asarray(gain, dtype=float)[valid]
    if theta.size < 8:
        return None
    order = np.argsort(theta)
    theta = theta[order]
    gain = gain[order]
    if float(np.max(theta)) < 30.0:
        return None

    grid = np.arange(0.0, min(75.0, float(np.max(theta))) + 0.01, 1.0)
    profile = np.interp(grid, theta, gain)
    kernel = np.asarray([1, 2, 3, 4, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    smooth = np.convolve(
        np.pad(profile, pad, mode="edge"), kernel, mode="valid"
    )

    for index in range(8, min(55, len(grid) - 8)):
        is_minimum = (
            smooth[index] <= smooth[index - 1]
            and smooth[index] < smooth[index + 1]
        )
        if not is_minimum:
            continue
        main_to_valley = float(np.max(smooth[: index + 1]) - smooth[index])
        following_rise = float(
            np.max(smooth[index + 1 : min(index + 16, len(smooth))])
            - smooth[index]
        )
        if main_to_valley >= 3.0 and following_rise >= 1.0:
            return float(grid[index])
    return None


def summarize_gps_first_valleys(
    patterns: dict[int, list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, object]:
    per_svn: dict[int, np.ndarray] = {}
    for svn, profiles in sorted(patterns.items()):
        values = [
            valley
            for theta, gain in profiles
            if (valley := detect_prominent_first_valley(theta, gain)) is not None
        ]
        if values:
            per_svn[svn] = np.asarray(values, dtype=float)
    if not per_svn:
        raise RuntimeError("No GPS first-valley diagnostics could be derived")

    equal_weight_values = np.concatenate(
        [
            np.quantile(values, np.linspace(0.05, 0.95, 19))
            for values in per_svn.values()
        ]
    )
    quantiles = np.quantile(
        equal_weight_values, [0.0, 0.05, 0.50, 0.95, 1.0]
    )
    return {
        "min": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "max": float(quantiles[4]),
        "n_sv": len(per_svn),
        "n_profiles": int(sum(len(values) for values in per_svn.values())),
        "per_svn": per_svn,
        "method": (
            "7-deg triangular smoothing; first local minimum after 8 deg "
            "with >=3 dB main-to-valley drop and >=1 dB following rise"
        ),
    }


def write_derived_pattern_tables(
    summaries: dict[str, core.PatternSummary],
    gps_valleys: dict[str, dict[str, object]],
    gps_patterns: dict[
        str, dict[int, list[tuple[np.ndarray, np.ndarray]]]
    ],
) -> tuple[Path, Path, Path]:
    envelope_path = ROOT / "lugre_panel_c_derived_pattern_envelopes.csv"
    records: list[dict[str, object]] = []
    for signal, summary in summaries.items():
        for index, theta in enumerate(summary.theta):
            records.append(
                {
                    "signal": signal,
                    "theta_deg": float(theta),
                    "median_delta_eirp_db": float(summary.median[index]),
                    "p05_delta_eirp_db": float(summary.p05[index]),
                    "p95_delta_eirp_db": float(summary.p95[index]),
                    "patterned_sv_count": (
                        int(summary.n_sv[index])
                        if signal.startswith("GPS")
                        else ""
                    ),
                    "model_lower_95_db": (
                        float(summary.model_lower[index])
                        if summary.model_lower is not None
                        else ""
                    ),
                    "model_upper_95_db": (
                        float(summary.model_upper[index])
                        if summary.model_upper is not None
                        else ""
                    ),
                    "provenance": summary.source_note,
                }
            )
    pd.DataFrame.from_records(records).to_csv(envelope_path, index=False)

    valley_path = ROOT / "lugre_panel_b_gps_first_valley_by_svn.csv"
    valley_records: list[dict[str, object]] = []
    for signal, summary in gps_valleys.items():
        per_svn = summary["per_svn"]
        for svn, values in sorted(per_svn.items()):
            valley_records.append(
                {
                    "signal": signal,
                    "svn": f"SVN{svn}",
                    "profile_count": len(values),
                    "first_valley_min_deg": float(np.min(values)),
                    "first_valley_p05_deg": float(np.quantile(values, 0.05)),
                    "first_valley_median_deg": float(np.median(values)),
                    "first_valley_p95_deg": float(np.quantile(values, 0.95)),
                    "first_valley_max_deg": float(np.max(values)),
                    "diagnostic_method": summary["method"],
                }
            )
    pd.DataFrame.from_records(valley_records).to_csv(valley_path, index=False)

    manifest_path = ROOT / "lugre_gps_pattern_source_manifest.csv"
    iif_urls = {
        62: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20199%20%282023%2004-10%29_Doc%20-%20GPS%20IIF%20SVN62%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        63: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20198%20%282023%2004-11%29_Doc%20-%20GPS%20IIF%20SVN63%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        64: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20196%20%282023%2004-12%29_Doc%20-%20GPS%20IIF%20SVN64%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        65: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20195%20%282023%2004-13%29_Doc%20-%20GPS%20IIF%20SVN65%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        66: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20214%20%282023%2004-14%29_Docume%20GPS%20IIF%20SVN66%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        67: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20193%20%282023%2004-15%29%20GPS%20IIF%20SVN67%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        68: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20194%20%282023%2004-16%29_Doc%20-%20%20GPS%20IIF%20SVN68%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        69: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20203%20%282023%2004-17%29_Document%20-%20GPS%20IIF%20SVN69%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        70: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20206%20%282023%2004-18%29_Doc%20GPS%20IIF%20SVN70%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        71: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20207%20%282023%2004-19%29_Document%20-%20GPS%20IIF%20SVN71%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        72: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20208%20%282023%2004-20%29_Document%20-%20GPS%20IIF%20SVN72%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
        73: "https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20209%20%282023%2004-21%29_Doc%20-%20GPS%20IIF%20SVN73%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx",
    }
    gps_iii_urls = {
        74: "https://navcen.uscg.gov/sites/default/files/pdf/gps/GPS_ZIP/GPS_III_SVN74_EC_Antenna_Patterns_Directivity.zip",
        75: "https://navcen.uscg.gov/sites/default/files/pdf/gps/GPS_ZIP/GPS_III_SVN75_EC_L1_2_5_Antenna_Patterns_Directivity.zip",
        76: "https://navcen.uscg.gov/sites/default/files/pdf/gps/GPS_ZIP/GPS_III_SVN76_EC_L_2_5_Antenna_Patterns_Directivity.zip",
        77: "https://navcen.uscg.gov/sites/default/files/pdf/gps/GPS_ZIP/GPS_III_SVN77_EC_L1_2_5_Antenna_Patterns_Directivity.zip",
        78: "https://navcen.uscg.gov/sites/default/files/pdf/gps/GPS_ZIP/GPS_III_SVN78_EC_L1_2_5_Antenna_Patterns_Directivity.zip",
    }
    manifest_rows: list[dict[str, object]] = []
    for signal, patterns in gps_patterns.items():
        valley_svns = set(gps_valleys[signal]["per_svn"])
        iir_svns = sorted(set(patterns) & set(range(41, 62)))
        if iir_svns:
            manifest_rows.append(
                {
                    "signal": signal,
                    "gps_block": "GPS IIR/IIR-M",
                    "svn_or_group": " ".join(f"SVN{svn}" for svn in iir_svns),
                    "local_source": "GPS_IIR_IIRM_SV_specific_patterns_data.pptx",
                    "official_source_url": "https://navcen.uscg.gov/sites/default/files/ppt/gps/AppBAntennaPanelPatterns.pptx",
                    "used_in_panel_c": "yes",
                    "used_in_panel_b_first_valley": " ".join(
                        f"SVN{svn}" for svn in iir_svns if svn in valley_svns
                    ),
                    "note": "Derived statistics; not a NAVCEN-published chart.",
                }
            )
        for svn in sorted(set(patterns) & set(range(62, 74))):
            manifest_rows.append(
                {
                    "signal": signal,
                    "gps_block": "GPS IIF",
                    "svn_or_group": f"SVN{svn}",
                    "local_source": f"GPS_IIF_SVN{svn}_L1_L2_L5.xlsx",
                    "official_source_url": iif_urls[svn],
                    "used_in_panel_c": "yes",
                    "used_in_panel_b_first_valley": "yes" if svn in valley_svns else "no",
                    "note": "Derived statistics; not a NAVCEN-published chart.",
                }
            )
        for svn in sorted(set(patterns) & set(range(74, 79))):
            manifest_rows.append(
                {
                    "signal": signal,
                    "gps_block": "GPS III",
                    "svn_or_group": f"SVN{svn}",
                    "local_source": f"GPS_III_SVN{svn}_Directivity.zip",
                    "official_source_url": gps_iii_urls[svn],
                    "used_in_panel_c": "yes",
                    "used_in_panel_b_first_valley": "yes" if svn in valley_svns else "no",
                    "note": "Derived statistics; not a NAVCEN-published chart.",
                }
            )
    pd.DataFrame.from_records(manifest_rows).to_csv(manifest_path, index=False)
    return envelope_path, valley_path, manifest_path

def polar_xy(
    origin: tuple[float, float], radius: float, angle_deg: float
) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    return origin[0] + radius * math.cos(angle), origin[1] + radius * math.sin(angle)


def draw_previous_panel_b(
    ax: plt.Axes,
    observed_min: float,
    observed_max: float,
    gps_l1_valley: dict[str, object],
    gps_l5_valley: dict[str, object],
    e1_valley: dict[str, float],
    e5a_valley: dict[str, float],
) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-6.35, 6.35)
    ax.set_ylim(-13.85, 1.58)
    ax.axis("off")

    earth_limb = 13.8
    satellite = (0.0, 0.04)
    antenna = (0.0, -0.38)
    earth = (0.0, -11.02)
    center_distance = antenna[1] - earth[1]
    earth_radius = center_distance * math.sin(math.radians(earth_limb))
    beam_length = center_distance + earth_radius * 1.25

    def beam_angle(offset: float) -> float:
        return -90.0 + offset

    ax.add_patch(
        patches.Wedge(
            antenna,
            beam_length,
            beam_angle(-23.0),
            beam_angle(23.0),
            facecolor=OLD_COLORS["central"],
            alpha=0.18,
            edgecolor="none",
            zorder=0,
        )
    )


    # First-valley statistics are shown as open arcs so they cannot be
    # mistaken for physical beam-lobe extents.
    diagnostic_arcs = [
        (OLD_COLORS["GPS L1"], gps_l1_valley["p05"], gps_l1_valley["p95"], 2.90),
        (OLD_COLORS["GPS L5"], gps_l5_valley["p05"], gps_l5_valley["p95"], 3.45),
        (OLD_COLORS["Galileo E1"], e1_valley["p05"], e1_valley["p95"], 4.00),
        (OLD_COLORS["Galileo E5a"], e5a_valley["p05"], e5a_valley["p95"], 4.55),
    ]
    for color, low, high, radius in diagnostic_arcs:
        for sign in (-1.0, 1.0):
            start, end = sorted((sign * float(low), sign * float(high)))
            ax.add_patch(
                patches.Arc(
                    antenna,
                    2.0 * radius,
                    2.0 * radius,
                    theta1=beam_angle(start),
                    theta2=beam_angle(end),
                    color=color,
                    linewidth=1.15,
                    alpha=0.95,
                    zorder=7,
                )
            )
            for boundary in (start, end):
                inner = polar_xy(antenna, radius - 0.10, beam_angle(boundary))
                outer = polar_xy(antenna, radius + 0.10, beam_angle(boundary))
                ax.plot(
                    [inner[0], outer[0]],
                    [inner[1], outer[1]],
                    color=color,
                    linewidth=0.82,
                    alpha=0.95,
                    zorder=7,
                )

    # The central field is deliberately longest; side lobes terminate well
    # before the Earth so the schematic hierarchy is unambiguous.
    for start, end, length, alpha in [
        (-55, -31, beam_length * 0.42, 0.18),
        (31, 55, beam_length * 0.42, 0.18),
        (-72, -58, beam_length * 0.27, 0.12),
        (58, 72, beam_length * 0.27, 0.12),
    ]:
        ax.add_patch(
            patches.Wedge(
                antenna,
                length,
                beam_angle(start),
                beam_angle(end),
                facecolor=OLD_COLORS["side"],
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
                facecolor=OLD_COLORS["observed"],
                alpha=alpha,
                edgecolor="none",
                zorder=2,
            )
        )

    for offset in (-observed_max, -observed_min, observed_min, observed_max):
        endpoint = polar_xy(antenna, beam_length * 0.96, beam_angle(offset))
        ax.plot(
            [antenna[0], endpoint[0]],
            [antenna[1], endpoint[1]],
            color=OLD_COLORS["observed"],
            linewidth=0.72,
            alpha=0.82,
            zorder=4,
        )

    center_endpoint = (earth[0], earth[1] + earth_radius)
    limb_endpoint = polar_xy(antenna, beam_length * 0.96, beam_angle(earth_limb))
    ax.plot(
        [antenna[0], center_endpoint[0]],
        [antenna[1], center_endpoint[1]],
        color="#7D8793",
        linewidth=0.68,
        linestyle=(0, (3.0, 2.5)),
        zorder=3,
    )
    ax.plot(
        [antenna[0], limb_endpoint[0]],
        [antenna[1], limb_endpoint[1]],
        color="#7D8793",
        linewidth=0.68,
        linestyle=(0, (3.0, 2.5)),
        zorder=3,
    )

    base.GPS_III_CUTOUT = core.SATELLITE_IMAGE
    base.EARTH_CUTOUT = core.EARTH_IMAGE
    base.draw_satellite(ax, satellite, compact=True)
    base.draw_earth_scaled(ax, earth, earth_radius)

    lugre_point = polar_xy(antenna, beam_length * 0.55, beam_angle(-14.2))
    ax.scatter(
        [lugre_point[0]],
        [lugre_point[1]],
        s=20,
        facecolor=OLD_COLORS["observed"],
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
        color=OLD_COLORS["observed"],
        fontweight="bold",
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.7,
            "color": OLD_COLORS["observed"],
            "shrinkA": 0.5,
            "shrinkB": 1.0,
        },
        zorder=13,
    )
    ax.text(
        3.10,
        satellite[1] + 0.16,
        "GNSS satellite",
        fontsize=6.3,
        fontweight="bold",
        color=OLD_COLORS["ink"],
        ha="left",
        va="center",
        zorder=14,
    )

    ax.text(-3.80, -3.90, "side lobes", ha="center", fontsize=5.8, color="#705795")
    ax.text(3.80, -3.90, "side lobes", ha="center", fontsize=5.8, color="#705795")
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

    limb_markers = [
        ("GPS", 13.8, 0.08, core.EARTH_LIMB_COLOR["GPS"]),
        ("Galileo", 12.4, 0.54, core.EARTH_LIMB_COLOR["Galileo"]),
    ]
    for constellation, limb_angle, vertical_offset, color in limb_markers:
        arc_y = earth[1] + earth_radius + vertical_offset
        arc_radius = (arc_y - antenna[1]) / math.sin(
            math.radians(beam_angle(limb_angle))
        )
        arc_start = (earth[0], arc_y)
        arc_end = polar_xy(antenna, arc_radius, beam_angle(limb_angle))
        ax.add_patch(
            patches.FancyArrowPatch(
                arc_start,
                arc_end,
                connectionstyle="arc3,rad=-0.25",
                arrowstyle="<->",
                mutation_scale=8.5,
                linewidth=0.82,
                color=color,
                zorder=18,
            )
        )
        ax.text(
            0.5 * (arc_start[0] + arc_end[0]),
            arc_y + 0.27,
            rf"{constellation} ${limb_angle:.1f}^\circ$",
            ha="center",
            va="bottom",
            fontsize=5.55,
            color=color,
            zorder=19,
        )

    legend_x = -6.02
    legend_y = -0.28
    ax.text(
        legend_x,
        legend_y + 0.43,
        "First-valley ranges (thin arcs)",
        fontsize=4.8,
        color=OLD_COLORS["muted"],
        ha="left",
        va="bottom",
        zorder=10,
    )
    rows = [
        (
            OLD_COLORS["GPS L1"],
            rf"GPS L1  ${gps_l1_valley['p05']:.0f}^\circ$-${gps_l1_valley['p95']:.0f}^\circ$",
        ),
        (
            OLD_COLORS["GPS L5"],
            rf"GPS L5  ${gps_l5_valley['p05']:.0f}^\circ$-${gps_l5_valley['p95']:.0f}^\circ$",
        ),
        (
            OLD_COLORS["Galileo E1"],
            rf"Galileo E1  ${e1_valley['p05']:.0f}^\circ$-${e1_valley['p95']:.0f}^\circ$",
        ),
        (
            OLD_COLORS["Galileo E5a"],
            rf"Galileo E5a  ${e5a_valley['p05']:.0f}^\circ$-${e5a_valley['p95']:.0f}^\circ$",
        ),
    ]
    for index, (color, label) in enumerate(rows):
        y = legend_y - index * 0.64
        ax.add_patch(
            patches.Rectangle(
                (legend_x, y - 0.10),
                0.58,
                0.20,
                facecolor=color,
                edgecolor="none",
                alpha=0.68,
                zorder=10,
            )
        )
        ax.text(
            legend_x + 0.72,
            y,
            label,
            fontsize=4.65,
            color=OLD_COLORS["ink"],
            ha="left",
            va="center",
            zorder=10,
        )

def save_figure(fig: plt.Figure) -> list[Path]:
    paths: list[Path] = []
    gps_alias = ROOT / "lugre_main_lobe_ab_figure_nature_gps_galileo"
    for stem in (OUTPUT_STEM, CURRENT_STEM, gps_alias):
        for suffix in (".png", ".pdf", ".svg"):
            path = stem.with_suffix(suffix)
            kwargs: dict[str, object] = {
                "bbox_inches": "tight",
                "pad_inches": 0.05,
                "facecolor": "white",
            }
            if suffix == ".png":
                kwargs["dpi"] = 600
            fig.savefig(path, **kwargs)
            paths.append(path)
    return paths


def save_panel_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    tight: bool = True,
) -> list[Path]:
    paths: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        kwargs: dict[str, object] = {"facecolor": "white"}
        if tight:
            kwargs.update({"bbox_inches": "tight", "pad_inches": 0.06})
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def save_standalone_panels(
    point_df: pd.DataFrame,
    observed_min: float,
    observed_max: float,
    gps_l1_valley: dict[str, object],
    gps_l5_valley: dict[str, object],
    e1_valley: dict[str, float],
    e5a_valley: dict[str, float],
    summaries: dict[str, core.PatternSummary],
) -> list[Path]:
    paths: list[Path] = []

    fig_a = plt.figure(figsize=(6.35, 3.25))
    grid_a = fig_a.add_gridspec(
        1, 1, left=0.18, right=0.985, bottom=0.16, top=0.83
    )
    draw_previous_panel_a(fig_a, grid_a[0, 0], point_df)
    paths.extend(
        save_panel_figure(fig_a, ROOT / "lugre_main_lobe_panel_a", tight=False)
    )
    plt.close(fig_a)

    fig_b = plt.figure(figsize=(4.25, 5.25))
    grid_b = fig_b.add_gridspec(
        1, 1, left=0.03, right=0.97, bottom=0.10, top=0.90
    )
    ax_b = fig_b.add_subplot(grid_b[0, 0])
    draw_previous_panel_b(
        ax_b,
        observed_min,
        observed_max,
        gps_l1_valley,
        gps_l5_valley,
        e1_valley,
        e5a_valley,
    )
    position_b = ax_b.get_position()
    fig_b.text(
        position_b.x0,
        position_b.y1 + 0.022,
        "b  Earth-limb and pattern-derived lobe diagnostics",
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=OLD_COLORS["ink"],
    )
    paths.extend(save_panel_figure(fig_b, ROOT / "lugre_main_lobe_panel_b"))
    plt.close(fig_b)

    fig_c = plt.figure(figsize=(6.35, 4.85))
    grid_c = fig_c.add_gridspec(
        1, 1, left=0.10, right=0.985, bottom=0.18, top=0.86
    )
    axes_c, legend_c = core.draw_panel_c(fig_c, grid_c[0, 0], summaries, point_df)
    fig_c.text(
        axes_c[0].get_position().x0,
        axes_c[0].get_position().y1 + 0.038,
        "c  Transmit-pattern evidence by signal",
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=OLD_COLORS["ink"],
    )
    fig_c.legend(
        handles=legend_c,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.025),
        ncol=3,
        columnspacing=1.15,
        handlelength=1.8,
        handletextpad=0.42,
        labelspacing=0.45,
        fontsize=5.8,
    )
    paths.extend(
        save_panel_figure(fig_c, ROOT / "lugre_main_lobe_panel_c", tight=False)
    )
    plt.close(fig_c)

    return paths

def save_reference_layout_figure(
    point_df: pd.DataFrame,
    observed_min: float,
    observed_max: float,
    gps_l1_valley: dict[str, object],
    gps_l5_valley: dict[str, object],
    e1_valley: dict[str, float],
    e5a_valley: dict[str, float],
    summaries: dict[str, core.PatternSummary],
) -> list[Path]:
    """Save the requested publication layout: panel b left, a/c right."""
    fig = plt.figure(figsize=(14.0, 8.4))
    outer = fig.add_gridspec(
        1,
        2,
        left=0.025,
        right=0.985,
        bottom=0.125,
        top=0.895,
        width_ratios=[0.88, 1.46],
        wspace=0.13,
    )

    ax_b = fig.add_subplot(outer[0, 0])
    draw_previous_panel_b(
        ax_b,
        observed_min,
        observed_max,
        gps_l1_valley,
        gps_l5_valley,
        e1_valley,
        e5a_valley,
    )

    right = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[0, 1],
        height_ratios=[0.86, 1.34],
        hspace=0.36,
    )
    ax_a, _, _, _ = draw_previous_panel_a(fig, right[0, 0], point_df)
    axes_c, legend_c = core.draw_panel_c(fig, right[1, 0], summaries, point_df)

    fig.canvas.draw()
    a_position = ax_a.get_position()
    b_position = ax_b.get_position()
    c_left = axes_c[0].get_position().x0
    c_right = axes_c[1].get_position().x1

    fig.suptitle(
        "LuGRE reception geometry relative to GPS and Galileo "
        "transmit-antenna patterns",
        x=0.025,
        y=0.966,
        ha="left",
        va="top",
        fontsize=15.0,
        fontweight="bold",
        color=OLD_COLORS["ink"],
    )
    fig.text(
        b_position.x0,
        a_position.y1 + 0.13 * a_position.height,
        "b  Earth-limb and pattern-derived lobe diagnostics",
        ha="left",
        va="bottom",
        fontsize=10.4,
        fontweight="bold",
        color=OLD_COLORS["ink"],
    )
    fig.text(
        c_left,
        axes_c[0].get_position().y1 + 0.034,
        "c  Transmit-pattern evidence by signal",
        ha="left",
        va="bottom",
        fontsize=10.4,
        fontweight="bold",
        color=OLD_COLORS["ink"],
    )
    fig.legend(
        handles=legend_c,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=((c_left + c_right) / 2.0, 0.070),
        ncol=3,
        columnspacing=1.25,
        handlelength=1.9,
        handletextpad=0.45,
        labelspacing=0.45,
        fontsize=5.9,
    )

    fig.text(
        0.025,
        0.040,
        "GPS: Marquis & Reigh, NAVIGATION 62, 329-347 (2015), "
        "doi:10.1002/navi.123; pattern data: USCG NAVCEN.",
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=OLD_COLORS["muted"],
    )
    fig.text(
        0.025,
        0.020,
        "Galileo: Menzione, Sgammini & Paonni, JRC135110 / EUR 31812 EN "
        "(2024), doi:10.2760/765842 (GRAP).",
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=OLD_COLORS["muted"],
    )

    paths = save_panel_figure(
        fig,
        ROOT / "lugre_main_lobe_abc_reference_layout",
        tight=False,
    )
    plt.close(fig)
    return paths

def write_notes(
    point_df: pd.DataFrame,
    independent_windows: pd.DataFrame,
    observed_min: float,
    observed_max: float,
    e1_valley: dict[str, float],
    e5a_valley: dict[str, float],
    gps_missing_svns: dict[str, list[int]],
    gps_valleys: dict[str, dict[str, object]],
) -> tuple[Path, Path]:
    caption_path = ROOT / "lugre_main_lobe_ab_figure_nature_gps_galileo_caption.txt"
    methods_path = ROOT / "lugre_main_lobe_ab_figure_nature_gps_galileo_references_and_methods.txt"
    counts = point_df.groupby("signal_name").size().to_dict()
    caption = f"""Extended Data Fig. 3 | LuGRE angular geometry and signal-specific transmit-pattern evidence.

+a, Point-level transmitter off-boresight-angle distributions for GPS L1 C/A (n={counts['GPS L1']:,}), GPS L5Q (n={counts['GPS L5']:,}), Galileo E1C (n={counts['Galileo E1']:,}) and Galileo E5a-Q (n={counts['Galileo E5a']:,}). Only samples with geometric tangent-point heights from 50 to 1,000 km are included. Vertical references mark the nominal GPS (13.8 deg) and Galileo (12.4 deg) Earth-limb angles. Curves show probability density on the exact retained data support; darker fills and circles mark the same 5-95% intervals and medians shown in the lower summaries, which additionally show the full range and interquartile range. No fixed off-boresight-angle threshold is applied.

+b, Earth-limb geometry and the altitude-filtered LuGRE observed angular sector ({observed_min:.2f}-{observed_max:.2f} deg). Two compact schematic markers show the nominal GPS (13.8 deg) and Galileo (12.4 deg) Earth-limb angles; panel c retains the same constellation-specific references. Four thin colored arcs show pattern-derived first-valley 5-95% ranges: GPS L1 {gps_valleys['GPS L1']['p05']:.0f}-{gps_valleys['GPS L1']['p95']:.0f} deg, GPS L5 {gps_valleys['GPS L5']['p05']:.0f}-{gps_valleys['GPS L5']['p95']:.0f} deg, Galileo E1 {e1_valley['p05']:.0f}-{e1_valley['p95']:.0f} deg and Galileo E5a {e5a_valley['p05']:.0f}-{e5a_valley['p95']:.0f} deg. The long blue central field and shorter purple fields are schematic main and side lobes. The colored arcs only describe where a smoothed pattern first reaches a prominent post-boresight valley; they are not beam extents or official fixed lobe limits.

+c, Signal-specific transmit patterns normalized to boresight EIRP. The GPS solid curves and P5-P95 bands are a derived statistical product, not a chart published by NAVCEN: every available measured azimuth cut is normalized to its boresight value, summarized within each observed SVN, and then combined with equal SVN weight. Galileo uses azimuth cuts from the JRC GRAP FOC population model. At each off-boresight angle, P5 and P95 are the 5th and 95th percentiles of normalized-EIRP values across the equally weighted satellite/azimuth pattern realizations; the band contains the central 90% and is not a 95% confidence interval. Pale green bands labelled LuGRE show the full minimum-to-maximum range of retained 50-1,000-km tangent-point off-boresight observations, without percentile trimming. These descriptive ranges are not selection filters or lobe boundaries. Each subplot uses its constellation-specific nominal Earth-limb reference: GPS 13.8 deg and Galileo 12.4 deg. No fixed main-lobe threshold is imposed.
""".replace("\n+", "\n")
    caption_path.write_text(caption, encoding="utf-8")

    methods = f"""METHOD AND REFERENCE NOTES

Panels a and c use the same point-level LuGRE geometry cache, lugre_gnss_main_lobe_point_check.csv, restricted to geometric tangent-point heights from 50 to 1,000 km inclusive. The green band in panel c spans the full retained LuGRE minimum-to-maximum angle for each signal, without P5-P95 trimming. Panel a separately reports distribution percentiles. These intervals are descriptive and never selection filters.

GPS patterns: U.S. Coast Guard Navigation Center GPS Technical References, including IIR/IIR-M SV-specific patterns, IIF SVN62-SVN73 L1/L2/L5 workbooks and GPS III SVN74-SVN78 directivity files. NAVCEN supplies the source pattern files; it does not publish the median and 5-95% envelope shown here. That envelope and the first-valley statistics are derived in the accompanying code and CSV tables.
https://navcen.uscg.gov/gps-technical-references
IIR/IIR-M Appendix B data:
https://navcen.uscg.gov/sites/default/files/ppt/gps/AppBAntennaPanelPatterns.pptx
IIF example (SVN62):
https://navcen.uscg.gov/sites/default/files/pdf/gps/sigspec/SSC%20PA%20Final%20Assess%20199%20%282023%2004-10%29_Doc%20-%20GPS%20IIF%20SVN62%20Antenna%20Patterns%20-%20L1%2C%20L2%2C%20and%20L5.xlsx
GPS III example (SVN74):
https://navcen.uscg.gov/sites/default/files/pdf/gps/GPS_ZIP/GPS_III_SVN74_EC_Antenna_Patterns_Directivity.zip
Observed GPS patterns not available in that public set were not substituted: L1 missing {gps_missing_svns['GPS L1']}; L5 missing {gps_missing_svns['GPS L5']}.

GPS first-valley diagnostic: each azimuth cut is interpolated to 1-deg spacing and smoothed with a 7-deg triangular kernel. The first local minimum after 8 deg is retained only when it is at least 3 dB below the preceding peak and followed by at least a 1 dB rise within 15 deg. Cuts are summarized with equal SVN weight. GPS L1 uses {gps_valleys['GPS L1']['n_profiles']:,} cuts from {gps_valleys['GPS L1']['n_sv']} SVNs; GPS L5 uses {gps_valleys['GPS L5']['n_profiles']:,} cuts from {gps_valleys['GPS L5']['n_sv']} SVNs.

Galileo patterns: European Commission JRC Galileo Reference Antenna Pattern (GRAP) FOC population model and JRC135110 technical report.
https://joint-research-centre.ec.europa.eu/projects-and-activities/galileo-reference-antenna-pattern-model_en
https://publications.jrc.ec.europa.eu/repository/bitstream/JRC135110/JRC135110_01.pdf

LuGRE paper: https://doi.org/10.33012/navi.756
Nominal Earth-limb references are constellation-specific: 13.8 deg for GPS and 12.4 deg for Galileo. These geometric limb angles are not main-lobe boundaries.
"""
    methods_path.write_text(methods, encoding="utf-8")
    return caption_path, methods_path


def main() -> None:
    core.configure_style()
    plt.rcParams.update({"font.size": 7.2, "axes.titlesize": 7.6})

    point_df = load_previous_points()
    raw_windows, independent_windows = core.load_windows()
    prn_svn = core.load_prn_svn_map()

    summaries: dict[str, core.PatternSummary] = {}
    gps_patterns: dict[
        str, dict[int, list[tuple[np.ndarray, np.ndarray]]]
    ] = {}
    gps_valleys: dict[str, dict[str, object]] = {}
    gps_observed_svns: dict[str, list[int]] = {}
    gps_missing_svns: dict[str, list[int]] = {}
    for signal in ("GPS L1", "GPS L5"):
        patterns, observed_svns, missing_svns = load_gps_pattern_set(
            signal, raw_windows, prn_svn
        )
        gps_patterns[signal] = patterns
        summaries[signal] = core.summarize_satellite_patterns(
            patterns,
            "NAVCEN measured SV patterns; equal weight per patterned SVN",
        )
        gps_valleys[signal] = summarize_gps_first_valleys(patterns)
        gps_observed_svns[signal] = observed_svns
        gps_missing_svns[signal] = missing_svns
    summaries["Galileo E1"] = core.load_galileo_summary("Galileo E1")
    summaries["Galileo E5a"] = core.load_galileo_summary("Galileo E5a")

    e1_raw = core.load_grap_band(core.GRAP_DIR / "GRAP_File_E1_.xlsx")
    e5a_raw = core.load_grap_band(core.GRAP_DIR / "GRAP_File_E5a_.xlsx")
    e1_valley = first_valley_summary(e1_raw)
    e5a_valley = first_valley_summary(e5a_raw)

    fig = plt.figure(figsize=(8.35, 7.85))
    outer = fig.add_gridspec(
        2,
        1,
        left=0.073,
        right=0.985,
        bottom=0.115,
        top=0.930,
        height_ratios=[1.03, 1.34],
        hspace=0.38,
    )
    ax_a, ax_box, observed_min, observed_max = draw_previous_panel_a(
        fig, outer[0], point_df
    )

    lower = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[1],
        width_ratios=[0.90, 1.45],
        wspace=0.24,
    )
    ax_b = fig.add_subplot(lower[0, 0])
    draw_previous_panel_b(
        ax_b,
        observed_min,
        observed_max,
        gps_valleys["GPS L1"],
        gps_valleys["GPS L5"],
        e1_valley,
        e5a_valley,
    )
    axes_c, legend_c = core.draw_panel_c(
        fig, lower[0, 1], summaries, point_df
    )

    lower_left = ax_b.get_position().x0
    lower_right = axes_c[1].get_position().x1
    for axis in (ax_a, ax_box):
        position = axis.get_position()
        axis.set_position(
            [lower_left, position.y0, lower_right - lower_left, position.height]
        )

    fig.text(
        ax_b.get_position().x0,
        ax_b.get_position().y1 + 0.022,
        "b  Earth-limb and pattern-derived lobe diagnostics",
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=OLD_COLORS["ink"],
    )
    fig.text(
        axes_c[0].get_position().x0,
        axes_c[0].get_position().y1 + 0.040,
        "c  Transmit-pattern evidence by signal",
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=OLD_COLORS["ink"],
    )
    fig.legend(
        handles=legend_c,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.705, 0.034),
        ncol=3,
        columnspacing=1.15,
        handlelength=1.8,
        handletextpad=0.42,
        labelspacing=0.45,
        fontsize=5.6,
    )

    paths = save_figure(fig)
    panel_paths = save_standalone_panels(
        point_df,
        observed_min,
        observed_max,
        gps_valleys["GPS L1"],
        gps_valleys["GPS L5"],
        e1_valley,
        e5a_valley,
        summaries,
    )
    reference_layout_paths = save_reference_layout_figure(
        point_df,
        observed_min,
        observed_max,
        gps_valleys["GPS L1"],
        gps_valleys["GPS L5"],
        e1_valley,
        e5a_valley,
        summaries,
    )
    coverage_path = core.write_coverage(
        raw_windows,
        independent_windows,
        prn_svn,
        summaries,
        gps_observed_svns,
        gps_missing_svns,
    )
    envelope_path, valley_path, manifest_path = write_derived_pattern_tables(
        summaries, gps_valleys, gps_patterns
    )
    caption_path, methods_path = write_notes(
        point_df,
        independent_windows,
        observed_min,
        observed_max,
        e1_valley,
        e5a_valley,
        gps_missing_svns,
        gps_valleys,
    )
    plt.close(fig)

    print(f"Previous a/b restored; observed angle range {observed_min:.3f}-{observed_max:.3f} deg")
    for path in paths:
        print(path)
    for path in panel_paths:
        print(path)
    for path in reference_layout_paths:
        print(path)
    print(coverage_path)
    print(envelope_path)
    print(valley_path)
    print(manifest_path)
    print(caption_path)
    print(methods_path)


if __name__ == "__main__":
    main()