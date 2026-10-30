from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\Reference\Paper\2026\Nature_LuGRE")
FIGURE = ROOT / "Figure"
PROGRAM = ROOT / "Program"
PY_DEPS = PROGRAM / "python_video_deps"
MPL_DEPS = FIGURE / "combined_station_global_plus_polar_zooms_package_20260615_1730" / "_runtime_deps_mpl"
ROBUST_DIR = FIGURE / "Fig1_robust_grid_statistic_candidate_20260622"

for dep in (PY_DEPS, MPL_DEPS, PROGRAM, ROBUST_DIR):
    if dep.exists() and str(dep) not in sys.path:
        sys.path.insert(0, str(dep))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import make_fig1_robust_grid_statistic_candidates as base  # noqa: E402
import make_north_center_tracks_web as web  # noqa: E402
import plot_lugre_polar_abs_nonuniform_polarcoords as polar_map  # noqa: E402
import plot_polar_same_sat_multi_frequency_cn0_altitude as mf  # noqa: E402


OUT = FIGURE / "main_text_panel_png_exports_20260630"
SOURCE_OUT = OUT / "Fig1_source_redrawn_no_label_png"
TRACK_OUT = SOURCE_OUT / "track_cn0_altitude_panels"
POLAR_OUT = SOURCE_OUT / "polar_track_maps"
SELECTED_OUT = OUT / "selected_G01_E21_main_and_extended_png"

POINTS_CACHE = (
    FIGURE
    / "main_figures_lugre_reference_update_20260622"
    / "data"
    / "Fig1_50_1000_points_cache_4fitting_new_reference_ecef.csv"
)
SUMMARY_CSV = (
    FIGURE
    / "main_figures_lugre_reference_update_20260622"
    / "Fig1_replace_1_OP76G01_replace_6_OP38E25_new_reference_ecef_summary.csv"
)

LAYER_ALPHA = 0.17
SPINE_COLOR = "#334155"


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_CSV)
    summary["display_label"] = summary["display_label"].astype(str)
    summary["time_start"] = pd.to_datetime(summary["time_start"], errors="coerce")
    summary["time_end"] = pd.to_datetime(summary["time_end"], errors="coerce")
    return summary


def load_points() -> pd.DataFrame:
    points = base.normalize_points(pd.read_csv(POINTS_CACHE))
    if "freq_group" not in points.columns:
        points["freq_group"] = points["signal_name"].map(mf.freq_group)
    return points


def op_id_from_uid(uid: str) -> str:
    parts = str(uid).split("|")
    if len(parts) >= 2:
        return parts[1]
    raise ValueError(f"Cannot parse op_id from group_uid={uid!r}")


def filter_group(points: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    op_id = op_id_from_uid(row["group_uid"])
    mask = (
        points["phase"].astype(str).eq(str(row["phase"]))
        & points["op_id"].astype(str).eq(op_id)
        & points["sat"].astype(str).eq(str(row["sat"]))
    )
    if pd.notna(row["time_start"]) and pd.notna(row["time_end"]):
        mask &= points["time_utc"].between(row["time_start"], row["time_end"])
    group = points.loc[mask].copy()
    if group.empty:
        raise RuntimeError(f"No Fig.1 source rows matched {row['display_label']} {row['group_uid']}")
    group["freq_group"] = group["signal_name"].map(mf.freq_group)
    return group.sort_values(["rx_gps_seconds", "signal_name", "segment_id"])


def selected_groups(points: pd.DataFrame, summary: pd.DataFrame) -> list[tuple[pd.Series, pd.DataFrame]]:
    rows = []
    for _, row in summary.sort_values("display_label", key=lambda s: s.astype(int)).iterrows():
        rows.append((row, filter_group(points, row)))
    return rows


def common_y_limits(groups: list[tuple[pd.Series, pd.DataFrame]]) -> tuple[float, float]:
    values = pd.concat([g["cn0_dbhz"] for _, g in groups], ignore_index=True)
    return float(np.floor(values.min() - 0.6)), float(np.ceil(values.max() + 0.6))


def pretty_track_title(row: pd.Series, include_label: bool = True, circled_label: bool = False) -> str:
    label = str(row["display_label"])
    circled = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤", "6": "⑥"}
    event_id = circled.get(label, label) if circled_label else web.event_label(label)
    prefix = f"{event_id}  " if include_label else ""
    return f"{prefix}{row['op']} {row['sat']} | {row['daynight']}"


def draw_track_panel(
    ax: plt.Axes,
    row: pd.Series,
    group: pd.DataFrame,
    y_limits: tuple[float, float],
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    include_label: bool = True,
    circled_label: bool = False,
    layer_label_box: bool = True,
) -> None:
    y_min, y_max = y_limits
    layers = mf.layer_scheme(str(row["daynight"]))
    ax.set_xscale("log")
    ax.set_xlim(50, 1000)
    ax.set_ylim(y_min, y_max)
    ax.set_facecolor("white")

    for layer_name, lo, hi, color in layers:
        ax.axvspan(lo, hi, color=color, alpha=LAYER_ALPHA, lw=0, zorder=0)
        layer_bbox = (
            {"boxstyle": "round,pad=0.06", "fc": "white", "ec": "none", "alpha": 0.70}
            if layer_label_box
            else None
        )
        ax.text(
            lo * 1.03,
            0.055,
            layer_name,
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#26323f",
            bbox=layer_bbox,
            clip_on=True,
            zorder=6,
        )

    layer_boundaries = {lo for _, lo, _, _ in layers} | {hi for _, _, hi, _ in layers}
    for boundary in sorted(layer_boundaries | {350.0}):
        if boundary == 350:
            ax.axvline(boundary, color="#64748b", ls="--", lw=1.05, alpha=0.70, zorder=1)
        elif boundary not in {50, 1000}:
            ax.axvline(boundary, color="#94a3b8", ls=":", lw=0.72, alpha=0.65, zorder=1)

    for freq, sub in group.groupby("freq_group", sort=False):
        color = mf.FREQ_COLORS.get(freq, mf.FREQ_COLORS["other"])
        sub = sub.sort_values("rx_gps_seconds")
        ax.scatter(
            sub["lugre_h_tan_km"],
            sub["cn0_dbhz"],
            s=11.0,
            color=color,
            alpha=0.86,
            edgecolors="none",
            rasterized=True,
            label=freq,
            zorder=3,
        )

    ax.set_xticks([50, 90, 150, 300, 600, 1000])
    ax.set_xticklabels(["50", "90", "150", "300", "600", "1000"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.get_offset_text().set_visible(False)
    ax.grid(True, which="major", color="#94a3b8", alpha=0.17, lw=0.58)
    ax.grid(False, which="minor")
    ax.tick_params(axis="both", which="major", labelsize=8.7, length=3.0, width=0.82, pad=1.5)
    ax.set_ylabel("C/N$_0$ (dB-Hz)" if show_ylabel else "", fontsize=9.6)
    ax.set_xlabel("Tangent-point altitude (km)" if show_xlabel else "", fontsize=9.6, labelpad=2.6)
    ax.set_title(
        pretty_track_title(row, include_label=include_label, circled_label=circled_label),
        fontsize=10.0,
        pad=5.5,
        color="#111827",
        fontfamily="DejaVu Sans" if circled_label else None,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.95)
        spine.set_edgecolor(SPINE_COLOR)


def save_png(fig: plt.Figure, out_path: Path, *, dpi: int = 700, pad: float = 0.025) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad, transparent=True)
    plt.close(fig)


def export_track_panels(groups: list[tuple[pd.Series, pd.DataFrame]]) -> list[dict]:
    TRACK_OUT.mkdir(parents=True, exist_ok=True)
    y_limits = common_y_limits(groups)
    rows: list[dict] = []

    # Combined source-redrawn panel-e equivalent.
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.95), facecolor="none")
    for idx, (ax, (row, group)) in enumerate(zip(axes.ravel(), groups)):
        draw_track_panel(
            ax,
            row,
            group,
            y_limits,
            show_ylabel=idx in {0, 3},
            show_xlabel=idx >= 3,
            include_label=True,
        )
    legend_items = [
        mpl.lines.Line2D([], [], marker="o", ls="", ms=5.0, color=mf.FREQ_COLORS["GPS L1"], label="GPS L1"),
        mpl.lines.Line2D([], [], marker="o", ls="", ms=5.0, color=mf.FREQ_COLORS["GPS L5"], label="GPS L5"),
        mpl.lines.Line2D([], [], marker="o", ls="", ms=5.0, color=mf.FREQ_COLORS["Galileo E1"], label="Galileo E1"),
        mpl.lines.Line2D([], [], marker="o", ls="", ms=5.0, color=mf.FREQ_COLORS["Galileo E5a"], label="Galileo E5a"),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.905, bottom=0.095, wspace=0.22, hspace=0.31)
    combined = TRACK_OUT / "fig1_panel_e_source_redrawn_six_tracks_no_panel_label.png"
    save_png(fig, combined, dpi=700, pad=0.035)
    rows.append({"panel": combined.name, "source": str(POINTS_CACHE), "method": "source redraw"})

    # Extended Data-ready variant: circled track identifiers and unboxed layer labels.
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.95), facecolor="none")
    for idx, (ax, (row, group)) in enumerate(zip(axes.ravel(), groups)):
        draw_track_panel(
            ax,
            row,
            group,
            y_limits,
            show_ylabel=idx in {0, 3},
            show_xlabel=idx >= 3,
            include_label=True,
            circled_label=True,
            layer_label_box=False,
        )
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.905, bottom=0.095, wspace=0.22, hspace=0.31)
    circled_combined = TRACK_OUT / "fig1_panel_e_source_redrawn_six_tracks_circled_no_layer_label_bg_no_panel_label.png"
    save_png(fig, circled_combined, dpi=700, pad=0.035)
    rows.append({"panel": circled_combined.name, "source": str(POINTS_CACHE), "method": "source redraw; circled labels; unboxed layer labels"})

    # Individual source-redrawn panels.
    for row, group in groups:
        fig, ax = plt.subplots(figsize=(3.62, 3.08), facecolor="none")
        draw_track_panel(
            ax,
            row,
            group,
            y_limits,
            show_ylabel=True,
            show_xlabel=True,
            include_label=True,
        )
        fig.subplots_adjust(left=0.185, right=0.985, top=0.880, bottom=0.165)
        safe = f"fig1_panel_e_track{int(row['display_label']):02d}_{row['op']}_{row['sat']}_{str(row['daynight']).lower()}_source_redrawn_no_panel_label.png"
        out = TRACK_OUT / safe
        save_png(fig, out, dpi=800, pad=0.02)
        rows.append({"panel": out.name, "source": str(POINTS_CACHE), "method": "source redraw"})

    return rows


def split_track_hemi(track: pd.DataFrame, hemisphere: str) -> list[pd.DataFrame]:
    lat = track["lugre_lat"].to_numpy(dtype=float)
    keep = lat >= polar_map.POLAR_LAT_DEG if hemisphere == "north" else lat <= -polar_map.POLAR_LAT_DEG
    if keep.sum() < 2:
        return []
    sub = track.loc[keep].sort_values("time_utc").copy()
    lon = sub["lugre_lon"].to_numpy(dtype=float)
    time_gap = sub["rx_gps_seconds"].diff().fillna(1.0).to_numpy(dtype=float) > 1.5
    lon_gap = np.r_[False, np.abs(np.diff(lon)) > 180.0]
    run_id = np.cumsum(time_gap | lon_gap)
    return [piece for _, piece in sub.groupby(run_id, sort=False) if len(piece) >= 2]


def draw_selected_track_on_polar(ax, track: pd.DataFrame, hemisphere: str, cmap, norm, vmax: float) -> None:
    value_col = polar_map.base.CN0_ABS_RESID_COL
    if value_col not in track.columns:
        track = web.with_cn0_residual(track)
        track[value_col] = np.abs(track[polar_map.base.CN0_RESID_COL].to_numpy(dtype=float))
    for piece in split_track_hemi(track, hemisphere):
        theta = np.deg2rad(piece["lugre_lon"].to_numpy(dtype=float))
        radius = polar_map.polar_radius(piece["lugre_lat"].to_numpy(dtype=float), hemisphere)
        if len(theta) < 2:
            continue
        xy = np.column_stack([theta, radius])
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        values_raw = piece[value_col].to_numpy(dtype=float)
        values = np.clip((values_raw[:-1] + values_raw[1:]) / 2.0, 0.0, vmax)
        ax.add_collection(
            LineCollection(
                segments,
                colors="white",
                linewidths=3.0,
                alpha=0.96,
                capstyle="round",
                joinstyle="round",
                zorder=5.8,
            )
        )
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=1.68,
            alpha=0.98,
            capstyle="round",
            joinstyle="round",
            zorder=6.2,
        )
        lc.set_array(values)
        ax.add_collection(lc)


def add_native_label(ax, track: pd.DataFrame, hemisphere: str, label: str) -> None:
    pieces = split_track_hemi(track, hemisphere)
    if not pieces:
        return
    all_parts = pd.concat(pieces, ignore_index=True)
    theta = np.deg2rad(all_parts["lugre_lon"].to_numpy(dtype=float))
    radius = polar_map.polar_radius(all_parts["lugre_lat"].to_numpy(dtype=float), hemisphere)
    ax.text(
        float(np.nanmean(theta)),
        float(np.nanmean(radius)),
        str(label),
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#111111",
        bbox={"boxstyle": "circle,pad=0.16", "fc": "white", "ec": "#111111", "lw": 0.9, "alpha": 0.97},
        zorder=8,
    )


def setup_clean_polar_axis(ax, hemisphere: str, title: str) -> None:
    polar_map.setup_axis(ax, hemisphere, title)
    ax.set_title(title, fontsize=11.6, pad=7.0)
    ax.tick_params(labelsize=8.8)
    ax.tick_params(axis="x", pad=-4.8)
    ax.tick_params(axis="y", pad=0.4)
    polar_map.draw_coastlines(ax, hemisphere)
    for text in ax.texts:
        text.set_fontsize(8.3)


def export_polar_track_maps(points: pd.DataFrame, groups: list[tuple[pd.Series, pd.DataFrame]]) -> list[dict]:
    POLAR_OUT.mkdir(parents=True, exist_ok=True)
    cmap = polar_map.colormaps()["matlab_jet_abs_bins"]
    boundaries = [value for value in polar_map.BOUNDARIES if value <= web.CN0_POLAR_CMAX_DBHZ]
    if boundaries[-1] < web.CN0_POLAR_CMAX_DBHZ:
        boundaries.append(web.CN0_POLAR_CMAX_DBHZ)
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    rows: list[dict] = []

    def draw_one(hemisphere: str, out_name: str) -> None:
        fig = plt.figure(figsize=(4.9, 4.9), facecolor="none")
        ax = fig.add_subplot(1, 1, 1, projection="polar")
        setup_clean_polar_axis(ax, hemisphere, f"{hemisphere.capitalize()} polar tangent tracks")
        polar_map.CMAX_DBHZ = web.CN0_POLAR_CMAX_DBHZ
        polar_map.add_tracks(ax, points, hemisphere, cmap, norm, linewidth=1.0)
        for row, group in groups:
            hemi = "north" if group["lugre_lat"].median() >= 0 else "south"
            if hemi != hemisphere:
                continue
            track = group[group["lugre_h_tan_km"].between(50.0, 1000.0)].copy()
            draw_selected_track_on_polar(ax, track, hemisphere, cmap, norm, web.CN0_POLAR_CMAX_DBHZ)
            add_native_label(ax, track, hemisphere, row["display_label"])
        save_png(fig, POLAR_OUT / out_name, dpi=800, pad=0.02)
        rows.append({"panel": out_name, "source": str(POINTS_CACHE), "method": "source redraw"})

    draw_one("north", "fig1_panel_c_north_polar_tracks_source_redrawn_no_panel_label.png")
    draw_one("south", "fig1_panel_c_south_polar_tracks_source_redrawn_no_panel_label.png")

    fig = plt.figure(figsize=(5.0, 9.45), facecolor="none")
    ax_n = fig.add_subplot(2, 1, 1, projection="polar")
    ax_s = fig.add_subplot(2, 1, 2, projection="polar")
    for ax, hemi, title in [
        (ax_n, "north", "North polar tangent tracks"),
        (ax_s, "south", "South polar tangent tracks"),
    ]:
        setup_clean_polar_axis(ax, hemi, title)
        polar_map.add_tracks(ax, points, hemi, cmap, norm, linewidth=1.0)
        for row, group in groups:
            group_hemi = "north" if group["lugre_lat"].median() >= 0 else "south"
            if group_hemi != hemi:
                continue
            track = group[group["lugre_h_tan_km"].between(50.0, 1000.0)].copy()
            draw_selected_track_on_polar(ax, track, hemi, cmap, norm, web.CN0_POLAR_CMAX_DBHZ)
            add_native_label(ax, track, hemi, row["display_label"])
    fig.subplots_adjust(left=0.015, right=0.985, top=0.980, bottom=0.020, hspace=0.12)
    out = POLAR_OUT / "fig1_panel_c_combined_polar_tracks_source_redrawn_no_panel_label.png"
    save_png(fig, out, dpi=800, pad=0.02)
    rows.append({"panel": out.name, "source": str(POINTS_CACHE), "method": "source redraw"})
    return rows


def copy_selected_case_replacements() -> list[dict]:
    rows = []
    replacements = {
        TRACK_OUT / "fig1_panel_e_track01_OP76_G01_night_source_redrawn_no_panel_label.png": SELECTED_OUT
        / "main_text_selected_cases"
        / "OP76_G01_north_polar_night"
        / "fig1_track_cn0_altitude_OP76_G01.png",
        TRACK_OUT / "fig1_panel_e_track04_OP38_E21_day_source_redrawn_no_panel_label.png": SELECTED_OUT
        / "main_text_selected_cases"
        / "OP38_E21_south_polar_day"
        / "fig1_track_cn0_altitude_OP38_E21.png",
        TRACK_OUT / "fig1_panel_e_track02_OP74_E36_night_source_redrawn_no_panel_label.png": SELECTED_OUT
        / "extended_data_candidate_cases"
        / "OP74_E36_north_polar_night"
        / "fig1_track_cn0_altitude_OP74_E36.png",
        TRACK_OUT / "fig1_panel_e_track03_OP76_G26_night_source_redrawn_no_panel_label.png": SELECTED_OUT
        / "extended_data_candidate_cases"
        / "OP76_G26_north_polar_night"
        / "fig1_track_cn0_altitude_OP76_G26.png",
        TRACK_OUT / "fig1_panel_e_track05_OP38_E23_day_source_redrawn_no_panel_label.png": SELECTED_OUT
        / "extended_data_candidate_cases"
        / "OP38_E23_south_polar_day"
        / "fig1_track_cn0_altitude_OP38_E23.png",
        TRACK_OUT / "fig1_panel_e_track06_OP38_E25_day_source_redrawn_no_panel_label.png": SELECTED_OUT
        / "extended_data_candidate_cases"
        / "OP38_E25_south_polar_day"
        / "fig1_track_cn0_altitude_OP38_E25.png",
    }
    for src, dst in replacements.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            rows.append({"panel": str(dst), "source": str(src), "method": "copied source-redrawn replacement"})
    return rows


def write_readme(manifest: pd.DataFrame) -> None:
    text = f"""# Fig.1 source-redrawn panel exports

These PNGs replace the earlier crop-based Fig.1 c/e exports. They are redrawn from the current Fig.1 source data and source plotting style rather than cropped from the manuscript PNG.

## Data source

- Point cache: `{POINTS_CACHE}`
- Track selection summary: `{SUMMARY_CSV}`

## Outputs

- `{POLAR_OUT}`: source-redrawn Fig.1 panel c polar track maps, without panel-letter labels.
- `{TRACK_OUT}`: source-redrawn Fig.1 panel e C/N0-altitude tracks, without panel-letter labels. Track identifiers 1-6 are retained because they identify the selected events.
- The G01/E21 selected-case package was updated with these source-redrawn track panels.

All outputs are high-resolution PNG only. No manuscript figure or original source script was overwritten.
"""
    (SOURCE_OUT / "README_fig1_source_redrawn_no_label_png.md").write_text(text, encoding="utf-8")
    manifest.to_csv(SOURCE_OUT / "fig1_source_redrawn_no_label_manifest.csv", index=False)


def main() -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "axes.linewidth": 0.9,
        }
    )
    summary = load_summary()
    points = load_points()
    groups = selected_groups(points, summary)

    manifest_rows = []
    manifest_rows.extend(export_track_panels(groups))
    manifest_rows.extend(export_polar_track_maps(points, groups))
    manifest_rows.extend(copy_selected_case_replacements())
    manifest = pd.DataFrame(manifest_rows)
    write_readme(manifest)

    print(SOURCE_OUT)
    print(TRACK_OUT)
    print(POLAR_OUT)


if __name__ == "__main__":
    main()
