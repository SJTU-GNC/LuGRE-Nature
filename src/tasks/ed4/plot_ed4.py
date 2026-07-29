from __future__ import annotations

import importlib.util
import json
import sys
from datetime import timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image


TASK_DIR = Path(__file__).resolve().parent
REPRO_ROOT = TASK_DIR.parents[2]
PACKAGE = REPRO_ROOT
DATA_DIR = REPRO_ROOT / "data" / "analysis_ready" / "ED4"
FIG_DIR = REPRO_ROOT / "outputs" / "ED4"
HELPER_CODE = TASK_DIR / "helper.py"
HELPER_REPRO_TREE = TASK_DIR / "map_assets"

RAW_CSV = DATA_DIR / "op38_e23_anomaly_focused_cn0_data.csv"
HEIGHT_CSV = DATA_DIR / "op38_e23_anomaly_focused_tangent_height.csv"
FOOTPRINT_CSV = DATA_DIR / "op38_e23_anomaly_focused_simulated_footprint.csv"
SUMMARY_CSV = DATA_DIR / "op38_e23_anomaly_focused_summary.csv"
METADATA_JSON = DATA_DIR / "op38_e23_anomaly_focused_metadata.json"

OUT_PNG = FIG_DIR / "Extended_Data_Fig_04_cn0_detrending_op38_e23.png"
OUT_PDF = None
PRIMARY_PNG = OUT_PNG
PRIMARY_PDF = None
PRIMARY_WEBP = None
BACKUP_PNG = None
BACKUP_WEBP = None
OUT_MANIFEST = None

EVENT_THRESHOLD_DB = 1.5
POLAR_SHELL_COLOR = "#9a4158"
EQUATORIAL_SHELL_COLOR = "#2d6fb7"
PANEL_HEADER_Y = 1.026
PANEL_LABEL_X = -0.112
MAP_PANEL_LABEL_X = -0.180


def set_panel_header(ax: plt.Axes, label: str, title: str, label_x: float = PANEL_LABEL_X) -> None:
    ax.text(
        label_x,
        PANEL_HEADER_Y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )
    ax.text(
        0.0,
        PANEL_HEADER_Y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=mpl.rcParams.get("axes.titlesize", 7.4),
        color="#111111",
        clip_on=False,
    )


def load_helper_module():
    sys.path.insert(0, str(HELPER_CODE.parent))
    spec = importlib.util.spec_from_file_location("op40_nature_helpers", HELPER_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module: {HELPER_CODE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "time_dt" in frame.columns:
        frame["time_dt"] = pd.to_datetime(frame["time_dt"], utc=True, errors="coerce")
    return frame


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).fillna(False)


def strict_ionosphere_spans(
    height: pd.DataFrame,
    hidden_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    hidden_by_track: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] | None = None,
    low_km: float = 50.0,
    high_km: float = 1000.0,
    max_gap_s: float = 90.0,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if height.empty:
        return spans
    h = height.copy()
    h["time_dt"] = pd.to_datetime(h["time_dt"], utc=True, errors="coerce")
    h["h_tan_km"] = pd.to_numeric(h["h_tan_km"], errors="coerce")
    reconstructed = bool_series(h["height_is_reconstructed"]) if "height_is_reconstructed" in h else pd.Series(False, index=h.index)
    for track, group_all in h.groupby("track_label", sort=False):
        in_gap = pd.Series(False, index=group_all.index)
        spans_for_track = hidden_by_track.get(str(track), hidden_spans) if hidden_by_track is not None else hidden_spans
        for start, end in spans_for_track:
            in_gap |= group_all["time_dt"].between(start, end)
        valid = group_all[group_all["h_tan_km"].between(low_km, high_km) & (~reconstructed.loc[group_all.index]) & (~in_gap)].sort_values("time_dt")
        if valid.empty:
            continue
        times = valid["time_dt"].dropna().sort_values().to_list()
        if not times:
            continue
        start = prev = times[0]
        for t in times[1:]:
            if (t - prev).total_seconds() > max_gap_s:
                spans.append((start, prev))
                start = t
            prev = t
        spans.append((start, prev))
    spans.sort(key=lambda pair: pair[0])
    return spans


def geometry_ionosphere_spans(
    height: pd.DataFrame,
    low_km: float = 50.0,
    high_km: float = 1000.0,
    max_gap_s: float = 90.0,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if height.empty:
        return spans
    h = height.copy()
    h["time_dt"] = pd.to_datetime(h["time_dt"], utc=True, errors="coerce")
    h["h_tan_km"] = pd.to_numeric(h["h_tan_km"], errors="coerce")
    valid = h[h["h_tan_km"].between(low_km, high_km)].sort_values("time_dt")
    for _, group in valid.groupby("track_label", sort=False):
        times = group["time_dt"].dropna().sort_values().to_list()
        if not times:
            continue
        start = prev = times[0]
        for t in times[1:]:
            if (t - prev).total_seconds() > max_gap_s:
                spans.append((start, prev))
                start = t
            prev = t
        spans.append((start, prev))
    spans.sort(key=lambda pair: pair[0])
    return spans


def merge_time_spans(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    join_gap_s: float = 2.0,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda pair: pair[0])
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + pd.Timedelta(seconds=join_gap_s):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def time_segments(rows: pd.DataFrame, max_gap_s: float = 90.0) -> list[pd.DataFrame]:
    if rows.empty:
        return []
    ordered = rows.dropna(subset=["time_dt"]).sort_values("time_dt").reset_index(drop=True)
    if len(ordered) < 2:
        return [ordered]
    diffs = ordered["time_dt"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)
    segment_id = np.cumsum(diffs > max_gap_s)
    return [piece for _, piece in ordered.groupby(segment_id, sort=False) if not piece.empty]


def map_extent(sim_track: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    lon = pd.to_numeric(sim_track["tp_lon_deg"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(sim_track["tp_lat_deg"], errors="coerce").to_numpy(dtype=float)
    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]
    if lon.size == 0 or lat.size == 0:
        return (-180.0, 180.0), (-90.0, 90.0)
    xlim = (max(-180.0, float(np.nanmin(lon)) - 9.0), min(180.0, float(np.nanmax(lon)) + 9.0))
    ylim = (max(-88.0, float(np.nanmin(lat)) - 5.0), min(88.0, float(np.nanmax(lat)) + 7.0))
    return xlim, ylim


def plot_footprint_with_gap_mask(
    ax: plt.Axes,
    helper,
    rows: pd.DataFrame,
    color: str,
    hidden_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    ionosphere_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    linewidth: float = 1.85,
    alpha: float = 0.88,
) -> None:
    rows = rows.dropna(subset=["time_dt", "tp_lon_deg", "tp_lat_deg"]).sort_values("time_dt").reset_index(drop=True)
    if rows.empty:
        return
    in_raw_gap = helper.mask_in_time_spans(rows["time_dt"], hidden_spans)
    bridge_flag = bool_series(rows["is_gap_bridge"]) if "is_gap_bridge" in rows else pd.Series(False, index=rows.index)
    shell = helper.mask_in_time_spans(rows["time_dt"], ionosphere_spans)
    rows = rows.assign(_bridge=(in_raw_gap.to_numpy() | bridge_flag.to_numpy()), _shell=shell.to_numpy())

    for idx in range(len(rows) - 1):
        left = rows.iloc[idx]
        right = rows.iloc[idx + 1]
        step_s = (pd.Timestamp(right["time_dt"]) - pd.Timestamp(left["time_dt"])).total_seconds()
        if not np.isfinite(step_s) or step_s > 120.0:
            continue
        shell_seg = bool(left["_shell"] and right["_shell"])
        if not shell_seg:
            continue
        ax.plot(
            [left["tp_lon_deg"], right["tp_lon_deg"]],
            [left["tp_lat_deg"], right["tp_lat_deg"]],
            color=color,
            lw=linewidth,
            ls="-",
            alpha=alpha,
            solid_capstyle="round",
            zorder=6,
        )

    for piece in time_segments(rows[rows["_bridge"]], max_gap_s=90.0):
        if len(piece) < 2:
            continue
        ax.plot(
            piece["tp_lon_deg"],
            piece["tp_lat_deg"],
            color=helper.BRIDGE_COLOR,
            lw=1.55,
            ls=(0, (5.0, 3.0)),
            alpha=0.96,
            solid_capstyle="round",
            zorder=10,
        )

    shell_rows = rows[rows["_shell"]]
    if not shell_rows.empty:
        first = shell_rows.iloc[0]
        last = shell_rows.iloc[-1]
        ax.scatter(first["tp_lon_deg"], first["tp_lat_deg"], s=15, marker="o", facecolor="white", edgecolor=color, linewidth=0.8, zorder=9)
        ax.scatter(last["tp_lon_deg"], last["tp_lat_deg"], s=17, marker=">", facecolor=color, edgecolor="white", linewidth=0.45, zorder=9)


def plot_time_pieces_local(
    ax: plt.Axes,
    rows: pd.DataFrame,
    y_col: str,
    color: str,
    *,
    linewidth: float,
    alpha: float = 1.0,
    linestyle: str | tuple[float, tuple[float, ...]] = "-",
    max_gap_s: float = 90.0,
    zorder: float = 3.0,
) -> None:
    rows = rows.dropna(subset=["time_dt", y_col]).sort_values("time_dt")
    for piece in time_segments(rows, max_gap_s=max_gap_s):
        if len(piece) < 2:
            continue
        ax.plot(
            piece["time_dt"],
            piece[y_col],
            color=color,
            lw=linewidth,
            alpha=alpha,
            ls=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )


def shell_span_labels(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    sim_track: pd.DataFrame,
) -> list[tuple[pd.Timestamp, str, str]]:
    labels: list[tuple[pd.Timestamp, str, str]] = []
    if sim_track.empty:
        return labels
    representative = sim_track[sim_track["track_label"].astype(str).eq("E23 E1C")].copy()
    if representative.empty:
        representative = sim_track.copy()
    for start, end in spans:
        rows = representative[representative["time_dt"].between(start, end)]
        if rows.empty:
            continue
        mean_abs_lat = float(pd.to_numeric(rows["tp_lat_deg"], errors="coerce").abs().mean())
        if mean_abs_lat >= 50.0:
            labels.append((start + (end - start) / 2, "polar shell\nlarger residuals", "#7f3f52"))
        else:
            labels.append((start + (end - start) / 2, "near-equatorial\nshell", "#2d6fb7"))
    return labels


def hidden_spans_by_track(raw: pd.DataFrame, helper) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if raw.empty or "cn0_is_filled" not in raw.columns:
        return spans
    for track, group in raw.groupby("track_label", sort=False):
        hidden = bool_series(group["cn0_is_filled"])
        pieces = time_segments(group[hidden], max_gap_s=2.5 * helper.FILL_STEP_S)
        spans[str(track)] = [(pd.Timestamp(piece["time_dt"].min()), pd.Timestamp(piece["time_dt"].max())) for piece in pieces if not piece.empty]
    return spans


def classified_shell_spans(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    sim_track: pd.DataFrame,
) -> list[dict[str, object]]:
    classified: list[dict[str, object]] = []
    if sim_track.empty:
        return classified
    representative = sim_track[sim_track["track_label"].astype(str).eq("E23 E1C")].copy()
    if representative.empty:
        representative = sim_track.copy()
    for start, end in spans:
        rows = representative[representative["time_dt"].between(start, end)]
        if rows.empty:
            continue
        mean_abs_lat = float(pd.to_numeric(rows["tp_lat_deg"], errors="coerce").abs().mean())
        if mean_abs_lat >= 50.0:
            classified.append({"start": start, "end": end, "kind": "polar", "label": "polar shell", "color": POLAR_SHELL_COLOR})
        else:
            classified.append({"start": start, "end": end, "kind": "equatorial", "label": "near-equatorial shell", "color": EQUATORIAL_SHELL_COLOR})
    return classified


def residual_width_text(raw: pd.DataFrame, spans: list[dict[str, object]]) -> str:
    parts: list[str] = []
    observed = raw[~bool_series(raw["cn0_is_filled"])].copy()
    observed["cn0_detrended_db"] = pd.to_numeric(observed["cn0_detrended_db"], errors="coerce")
    for item in spans:
        start = pd.Timestamp(item["start"])
        end = pd.Timestamp(item["end"])
        vals = observed.loc[observed["time_dt"].between(start, end), "cn0_detrended_db"].dropna()
        if vals.empty:
            continue
        width = float(vals.quantile(0.95) - vals.quantile(0.05))
        label = "polar" if item["kind"] == "polar" else "near-eq."
        parts.append(f"{label} {width:.1f} dB")
    if not parts:
        return ""
    return "P95-P05 width: " + "; ".join(parts)


def build_figure() -> None:
    helper = load_helper_module()
    raw = read_frame(RAW_CSV)
    height = read_frame(HEIGHT_CSV)
    sim_track = read_frame(FOOTPRINT_CSV)
    summary = pd.read_csv(SUMMARY_CSV, low_memory=False)

    raw["cn0_is_filled"] = bool_series(raw["cn0_is_filled"])
    height["height_is_reconstructed"] = bool_series(height["height_is_reconstructed"])
    sim_track["is_gap_bridge"] = bool_series(sim_track["is_gap_bridge"])

    hidden_spans = helper.merged_hidden_spans(raw)
    track_hidden_spans = hidden_spans_by_track(raw, helper)
    observed_ionosphere_spans = merge_time_spans(strict_ionosphere_spans(height, hidden_spans, track_hidden_spans))
    geometry_shell_spans = merge_time_spans(geometry_ionosphere_spans(height))
    ionosphere_spans = geometry_shell_spans
    shell_spans = classified_shell_spans(ionosphere_spans, sim_track)
    tracks = sorted(raw["track_label"].astype(str).dropna().unique(), key=helper.track_sort_key)

    start = pd.Timestamp(raw["time_dt"].min())
    end = pd.Timestamp(raw["time_dt"].max())
    land_shapes = helper.natural_earth_shapes(HELPER_REPRO_TREE)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.8,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.9,
            "legend.fontsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )

    fig = plt.figure(figsize=(7.35, 5.45), facecolor="white")
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[3.15, 1.18],
        height_ratios=[1.08, 1.00, 1.10],
        left=0.078,
        right=0.985,
        bottom=0.095,
        top=0.865,
        hspace=0.25,
        wspace=0.24,
    )
    ax_cn0 = fig.add_subplot(gs[0, 0])
    ax_det = fig.add_subplot(gs[1, 0], sharex=ax_cn0)
    ax_height = fig.add_subplot(gs[2, 0], sharex=ax_cn0)
    ax_map = fig.add_subplot(gs[:, 1])

    for ax in (ax_cn0, ax_det):
        helper.shade_ionosphere_time_spans(ax, ionosphere_spans)

    signal_handles: list[Line2D] = []
    for track in tracks:
        group = raw[raw["track_label"].astype(str).eq(track)].sort_values("time_dt")
        signal = str(group["signal_name"].iloc[0])
        color = helper.SIGNAL_COLORS.get(signal, "#5b6472")
        marker = helper.GPS_SIGNAL_MARKERS.get(signal, "o")
        hidden = bool_series(group["cn0_is_filled"])
        observed = group[~hidden]
        bridge = group[hidden]

        ax_cn0.scatter(
            observed["time_dt"],
            observed["cn0_dbhz"],
            s=5.2,
            marker=marker,
            facecolor=color,
            edgecolor="none",
            alpha=0.58,
            rasterized=True,
            zorder=3,
        )
        plot_time_pieces_local(
            ax_cn0,
            observed,
            "cn0_trend_dbhz",
            "#2f3440",
            linewidth=1.0,
            alpha=0.72,
            linestyle=(0, (4.0, 2.2)),
            max_gap_s=90.0,
            zorder=4,
        )
        plot_time_pieces_local(
            ax_cn0,
            bridge,
            "cn0_trend_dbhz",
            "#2f3440",
            linewidth=0.95,
            alpha=0.76,
            linestyle=(0, (2.2, 2.0)),
            max_gap_s=2.5 * helper.FILL_STEP_S,
            zorder=4,
        )
        ax_det.scatter(
            observed["time_dt"],
            observed["cn0_detrended_db"],
            s=5.2,
            marker=marker,
            facecolor=color,
            edgecolor="none",
            alpha=0.64,
            rasterized=True,
            zorder=3,
        )
        signal_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                ls="none",
                color=color,
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=4.0,
                label=f"E23 {signal}",
            )
        )

    set_panel_header(ax_cn0, "a", r"Raw receiver-reported $C/N_0^k$ and whole-arc baseline")
    ax_cn0.set_ylabel(r"$C/N_0^k$ (dB-Hz)")
    ax_cn0.set_ylim(17.5, 38.3)
    ax_cn0.set_yticks([20, 25, 30, 35])

    ax_det.axhline(0.0, color="#1f2328", lw=0.72, ls="-", alpha=0.92, zorder=2)
    ax_det.set_ylim(-2.0, 2.0)
    ax_det.set_yticks([-2, -1, 0, 1, 2])
    set_panel_header(ax_det, "b", r"Residual $\delta C/N_0^k=C/N_0^k-\widehat{C/N_0^k}$")
    ax_det.set_ylabel(r"$\delta C/N_0^k$ (dB)")

    ax_height.axhspan(50.0, 1000.0, facecolor="#f3dce4", edgecolor="none", alpha=0.50, zorder=0)
    for track in sorted(height["track_label"].astype(str).dropna().unique(), key=helper.track_sort_key):
        group = height[height["track_label"].astype(str).eq(track)].sort_values("time_dt")
        signal = str(group["signal_name"].iloc[0])
        color = helper.SIGNAL_COLORS.get(signal, "#5b6472")
        reconstructed = bool_series(group["height_is_reconstructed"])
        shell = pd.to_numeric(group["h_tan_km"], errors="coerce").between(50.0, 1000.0)
        in_gap = helper.mask_in_time_spans(group["time_dt"], track_hidden_spans.get(str(track), hidden_spans))
        valid = group[(~in_gap) & (~reconstructed) & shell]
        guide = group[(in_gap | reconstructed | (pd.to_numeric(group["h_tan_km"], errors="coerce") < 50.0))]
        helper.plot_time_pieces(
            ax_height,
            guide,
            "h_tan_km",
            helper.BRIDGE_COLOR,
            linewidth=0.95,
            alpha=0.84,
            linestyle=(0, (3.2, 2.2)),
            max_gap_s=2.5 * helper.FILL_STEP_S,
            zorder=3,
        )
        for piece in time_segments(valid, max_gap_s=90.0):
            if len(piece) < 2:
                continue
            ax_height.plot(
                piece["time_dt"],
                piece["h_tan_km"],
                color=color,
                lw=1.65,
                alpha=0.9,
                solid_capstyle="round",
                zorder=5,
            )

    ax_height.set_ylim(-650.0, 1040.0)
    ax_height.set_yticks([-500, 0, 250, 500, 750, 1000])
    set_panel_header(ax_height, "c", "Tangent height in the 50-1000 km shell")
    ax_height.set_ylabel("Tangent height (km)")
    ax_height.set_xlabel("UTC on 2025-03-03")
    ax_height.text(
        0.985,
        0.90,
        "50-1000 km ionosphere",
        transform=ax_height.transAxes,
        ha="right",
        va="center",
        fontsize=5.6,
        color="#7f3f52",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 1.0},
    )

    for ax in (ax_cn0, ax_det, ax_height):
        helper.style_nature_axis(ax)
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=timezone.utc))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
        ax.grid(True, axis="x", color="#e0e4e9", lw=0.42, alpha=0.62)
    plt.setp(ax_cn0.get_xticklabels(), visible=False)
    plt.setp(ax_det.get_xticklabels(), visible=False)

    xlim, ylim = map_extent(sim_track)
    helper.draw_nature_map_base(ax_map, land_shapes, xlim, ylim)
    for track in sorted(sim_track["track_label"].astype(str).dropna().unique(), key=helper.track_sort_key):
        group = sim_track[sim_track["track_label"].astype(str).eq(track)].sort_values("time_dt")
        signal = str(group["signal_name"].iloc[0])
        color = helper.SIGNAL_COLORS.get(signal, "#5b6472")
        shell_lw = 2.35 if "E1" in signal else 1.35
        shell_alpha = 0.72 if "E1" in signal else 0.88
        plot_footprint_with_gap_mask(
            ax_map,
            helper,
            group,
            color,
            track_hidden_spans.get(str(track), hidden_spans),
            ionosphere_spans,
            linewidth=shell_lw,
            alpha=shell_alpha,
        )
    ax_map.xaxis.set_major_locator(MultipleLocator(40.0))
    set_panel_header(ax_map, "d", "Tangent-point footprint", label_x=MAP_PANEL_LABEL_X)

    unique_signal_handles: list[Line2D] = []
    seen: set[str] = set()
    for handle in signal_handles:
        if handle.get_label() not in seen:
            unique_signal_handles.append(handle)
            seen.add(handle.get_label())
    handles = [
        *unique_signal_handles,
        Line2D([0], [0], color="#2f3440", lw=1.0, ls=(0, (4.0, 2.2)), label="fourth-order polynomial fit"),
        Line2D([0], [0], color=helper.BRIDGE_COLOR, lw=0.95, ls=(0, (2.2, 2.0)), label="no-fit gap guide"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.078, 0.915),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.95,
        borderaxespad=0.0,
    )

    fig.align_ylabels([ax_cn0, ax_det, ax_height])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
    print(f"wrote {PRIMARY_PNG}")
    print(f"wrote {PRIMARY_PDF}")
    print(f"wrote {PRIMARY_WEBP}")
    print(f"also wrote {OUT_PNG}")
    print(f"also wrote {OUT_PDF}")
