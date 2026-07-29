from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
REPRO_ROOT = TASK_DIR.parents[2]
PACKAGE_ROOT = TASK_DIR
DATA_DIR = REPRO_ROOT / "data" / "analysis_ready" / "ED2"
FIGURE_DIR = REPRO_ROOT / "outputs" / "ED2"
SOURCE_TABLE = DATA_DIR / "extended_data_fig7_leoro_panel_c_sources_20260617.csv"
ORIGINAL_SCRIPT = TASK_DIR / "create_global_plus_polar_zoom_station_map.py"

OUT_PNG = FIGURE_DIR / "Extended_Data_Fig_02_observing_geometry.png"
OUT_PDF = None
METOP_WINDOWS_CSV = DATA_DIR / "metop_chang2025_style_10s_L1_windows.csv.gz"
METOP_LINE_CACHE_CSV = (
    DATA_DIR / "global_plus_polar_fig3_op74_metop_occultation_line_cache.csv.gz"
)
METOP_SUMMARY_CACHE_CSV = DATA_DIR / "global_plus_polar_fig3_op74_metop_occultation_summary_cache.csv"
OP74_START_UTC = pd.Timestamp("2025-03-14T13:51:50Z")
OP74_END_UTC = pd.Timestamp("2025-03-15T12:44:56Z")

MISSION_ORDER = ["PlanetiQ", "Spire", "TerraSAR-X", "MetOp-B", "MetOp-C"]
PANEL_C_ORDER = ["PlanetiQ", "Spire", "TerraSAR-X", "MetOp-B"]
MISSION_ALIASES = {
    "planetiq": "PlanetiQ",
    "spire": "Spire",
    "fengyun": "FengYun GNOS",
    "fengyun gnos": "FengYun GNOS",
    "fy gnos": "FengYun GNOS",
    "tsx": "TerraSAR-X",
    "terrasar-x": "TerraSAR-X",
    "terrasar x": "TerraSAR-X",
    "metop-b": "MetOp-B",
    "metop b": "MetOp-B",
    "metop-c": "MetOp-C",
    "metop c": "MetOp-C",
    "metop-b/c": "MetOp-B/C",
    "metop b/c": "MetOp-B/C",
    "metop": "MetOp-B/C",
    "sentinel-6a": "Sentinel-6A",
    "sentinel 6a": "Sentinel-6A",
}
MISSION_COLORS = {
    "PlanetiQ": "#1f5aa6",
    "Spire": "#16804a",
    "TerraSAR-X": "#b51d45",
    "MetOp-B": "#6f4aa8",
    "MetOp-C": "#a78bfa",
}
PANEL_F_VERIFIED_TRACK_SOURCES = ["PlanetiQ", "Spire", "TSX", "MetOp"]
PANEL_F_COLORS = {
    "PlanetiQ": "#0f4c9a",
    "Spire": "#087a3b",
    "TSX": "#c51b4a",
    "MetOp": "#6f4aa8",
}

# Defaults are schematic only. Panel c shows mission/orbit-height context in the
# original Nature-style layout; source-table status is reported in the README.
MISSION_DEFAULTS = {
    "PlanetiQ": dict(x=1.00, altitude_km=632.0, half_range_km=14.0, label="PlanetiQ", asset="PlanetiQ", zoom=0.118, text=r"$h$~620-645 km, $i$~97.9$^\circ$", asset_y=708.0, text_y=596.0),
    "Spire": dict(x=1.22, altitude_km=550.0, half_range_km=70.0, label="Spire", asset="Spire", zoom=0.126, text=r"$h$ 400-600 km, varied $i$", asset_y=648.0, text_y=438.0),
    "TerraSAR-X": dict(x=1.47, altitude_km=514.0, half_range_km=12.0, label="TSX", asset="TSX", zoom=0.126, text=r"$h$~514 km, $i$=97.4$^\circ$", asset_y=606.0, text_y=468.0),
    "MetOp-B": dict(x=1.68, altitude_km=817.0, half_range_km=16.0, label="MetOp", asset="MetOp", zoom=0.124, text=r"$h$~817 km, $i$=98.7$^\circ$", asset_y=900.0, text_y=758.0),
    "MetOp-C": dict(x=1.68, altitude_km=817.0, half_range_km=16.0, label="MetOp", asset="MetOp", zoom=0.124, text=r"$h$~817 km, $i$=98.7$^\circ$", asset_y=900.0, text_y=758.0),
}


def load_original_module():
    sys.path.insert(0, str(ORIGINAL_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("fig7_original", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load original script: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_mission(value: object) -> str:
    raw = str(value).strip()
    key = raw.lower().replace("_", " ").replace("/", "/")
    if key in MISSION_ALIASES:
        return MISSION_ALIASES[key]
    compact = key.replace("-", " ")
    return MISSION_ALIASES.get(compact, raw)


def first_existing_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    lowered = {col.lower(): col for col in frame.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def load_source_table() -> pd.DataFrame:
    if not SOURCE_TABLE.exists():
        raise FileNotFoundError(
            "LEO-RO panel-c source table is not available yet; refusing to draw an updated panel c without verified sources. "
            f"Expected: {SOURCE_TABLE}"
        )
    table = pd.read_csv(SOURCE_TABLE, low_memory=False)
    mission_col = first_existing_column(table, ["mission", "mission_group", "source", "source_name", "platform"])
    if mission_col is None:
        raise ValueError(f"Source table must contain a mission/source/platform column. Columns: {list(table.columns)}")
    table["mission_canonical"] = table[mission_col].map(canonical_mission)
    table = table[~table["mission_canonical"].str.lower().str.contains("cosmic", na=False)].copy()
    table = table[table["mission_canonical"].isin(MISSION_ORDER)].copy()
    if table.empty:
        raise ValueError("Source table contains no supported non-COSMIC missions for panel c.")
    return table


def numeric_from_columns(rows: pd.DataFrame, names: list[str], default: float) -> float:
    col = first_existing_column(rows, names)
    if col is None:
        return default
    values = pd.to_numeric(rows[col], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.median())


def numeric_token_from_columns(rows: pd.DataFrame, names: list[str], default: float) -> float:
    col = first_existing_column(rows, names)
    if col is None:
        return default
    for value in rows[col].dropna():
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
        if match:
            return float(match.group(0))
    return default


def status_from_rows(rows: pd.DataFrame) -> str:
    col = first_existing_column(rows, ["data_status", "status", "geometry_status", "track_status"])
    if col is None:
        return "source-table listed"
    values = [str(v).strip() for v in rows[col].dropna().unique() if str(v).strip()]
    return ", ".join(values) if values else "source-table listed"


def platform_suffix(rows: pd.DataFrame) -> str:
    col = first_existing_column(rows, ["platform", "satellite", "spacecraft"])
    if col is None:
        return ""
    values = sorted({str(v).strip() for v in rows[col].dropna().unique() if str(v).strip()})
    useful = [v for v in values if v.lower() not in {"nan", "none"}]
    return " / ".join(useful[:3])


def mission_specs_from_table(table: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, str]]:
    specs: list[dict[str, object]] = []
    statuses: dict[str, str] = {}
    for mission in MISSION_ORDER:
        rows = table[table["mission_canonical"].eq(mission)]
        if rows.empty:
            continue
        status = status_from_rows(rows)
        statuses[mission] = status
        status_lower = status.lower()
        if mission not in PANEL_C_ORDER:
            continue
        base = dict(MISSION_DEFAULTS[mission])
        altitude = numeric_from_columns(rows, ["orbit_altitude_km", "spacecraft_altitude_km"], math.nan)
        if math.isfinite(altitude):
            base["altitude_km"] = altitude
        amin = numeric_from_columns(rows, ["orbit_altitude_min_km", "spacecraft_altitude_min_km"], math.nan)
        amax = numeric_from_columns(rows, ["orbit_altitude_max_km", "spacecraft_altitude_max_km"], math.nan)
        if math.isfinite(amin) and math.isfinite(amax) and amax >= amin:
            base["altitude_km"] = 0.5 * (amin + amax)
            base["half_range_km"] = max((amax - amin) / 2.0, 6.0)
        incl = numeric_token_from_columns(rows, ["inclination_deg", "inclination", "orbit_inclination_deg"], math.nan)
        if math.isfinite(incl):
            if math.isfinite(amin) and math.isfinite(amax):
                base["text"] = rf"$h$ {amin:.0f}-{amax:.0f} km, $i$={incl:.1f}$^\circ$"
            else:
                base["text"] = rf"$h$~{base['altitude_km']:.0f} km, $i$={incl:.1f}$^\circ$"
        if "missing" in status_lower:
            draw_mode = "context_missing"
        elif "inventory" in status_lower:
            draw_mode = "context_inventory"
        else:
            draw_mode = "coverage_context"
        specs.append({"mission": mission, "draw_mode": draw_mode, **base})
    return specs, statuses


def patch_orbit_panel(original, specs: list[dict[str, object]], statuses: dict[str, str]) -> None:
    def draw_ro_orbit_height_panel_v2(ax):
        layer_colors = {
            "D": "#fff3c4",
            "E": "#fbe5d6",
            "F1": "#f6b3b8",
            "F2": "#d9e8f5",
            "Topside": "#edf2f7",
        }
        layers = [("D", 50, 90), ("E", 90, 140), ("F1", 140, 200), ("F2", 200, 500), ("Topside", 500, 1000)]
        for label, y0, y1 in layers:
            ax.axhspan(y0, y1, facecolor=layer_colors[label], edgecolor="none", alpha=0.88, zorder=0)
            if label == "Topside":
                ax.text(1.93, 996.0, label, ha="right", va="top", fontsize=5.0, color="#333333", zorder=5)
            else:
                ax.text(1.91, (y0 + y1) / 2.0, label, ha="right", va="center", fontsize=5.0, color="#333333", zorder=5)
        for y in [90, 140, 200, 500, 1000]:
            ax.axhline(y, color="#343a40", lw=0.44, ls=(0, (3.2, 2.6)), alpha=0.48, zorder=1)

        altitude = np.linspace(50.0, 1000.0, 620)
        day_profile = 0.04 + 0.12 * np.exp(-((altitude - 105.0) / 18.0) ** 2) + 0.30 * np.exp(-((altitude - 165.0) / 32.0) ** 2) + 1.10 * np.exp(-((altitude - 330.0) / 135.0) ** 2) + 0.17 * np.exp(-((altitude - 650.0) / 310.0) ** 2)
        night_profile = 0.025 + 0.045 * np.exp(-((altitude - 112.0) / 25.0) ** 2) + 0.64 * np.exp(-((altitude - 345.0) / 170.0) ** 2) + 0.18 * np.exp(-((altitude - 660.0) / 360.0) ** 2)
        ax.plot(day_profile, altitude, color="#d7191c", lw=1.75, solid_capstyle="round", label="day", zorder=4)
        ax.plot(night_profile, altitude, color="#1f4e79", lw=1.55, ls=(0, (4.0, 2.4)), solid_capstyle="round", label="night", zorder=4)
        ax.text(1.06, 360, "day F2", ha="left", va="center", fontsize=4.7, color="#9f1d20", zorder=6)
        ax.text(0.56, 295, "night F", ha="left", va="center", fontsize=4.7, color="#1f4e79", zorder=6)

        asset_count = 0
        for spec in specs:
            mission = str(spec["mission"])
            x = float(spec["x"])
            altitude_km = float(spec["altitude_km"])
            plot_altitude_km = float(spec.get("plot_altitude_km", altitude_km))
            half_range = float(spec["half_range_km"])
            color = MISSION_COLORS[mission]
            y0, y1 = max(50.0, plot_altitude_km - half_range), min(1000.0, plot_altitude_km + half_range)
            if y0 <= 1000.0 and y1 >= 50.0:
                ax.plot([x, x], [y0, y1], color=color, lw=1.30, alpha=0.88, zorder=6)
                ax.scatter([x], [min(max(plot_altitude_km, 50.0), 1000.0)], s=16, color=color, edgecolors="white", linewidths=0.55, zorder=7)
            if spec.get("asset") and original.add_orbit_asset(ax, str(spec["asset"]), x, float(spec["asset_y"]), float(spec["zoom"])):
                asset_count += 1
            ax.text(x, float(spec["asset_y"]) + 34.0, str(spec["label"]), ha="center", va="bottom", fontsize=4.25, color="#222222", zorder=9)
            ax.text(x, float(spec["text_y"]), str(spec["text"]), ha="center", va="center", fontsize=3.62, color="#30343b", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.08}, zorder=9)

        ax.legend(loc="upper left", bbox_to_anchor=(0.235, 0.985), frameon=False, fontsize=4.6, handlelength=1.2, handletextpad=0.35, borderaxespad=0.0)
        ax.set_ylim(50, 1000)
        ax.set_xlim(0.0, 1.96)
        ax.set_ylabel("Altitude (km)", fontsize=6.2, labelpad=2.0)
        ax.set_xlabel(r"$N_e$ ($10^{12}$ e m$^{-3}$)", fontsize=5.9, labelpad=1.6)
        ax.set_yticks([50, 90, 140, 200, 500, 1000])
        ax.set_yticklabels(["50", "90", "140", "200", "500", "1000"], fontsize=5.2)
        ax.set_xticks([0.5, 1.0, 1.5])
        ax.set_xticklabels(["0.5", "1.0", "1.5"], fontsize=5.2)
        ax.tick_params(axis="both", length=2.2, width=0.45, pad=1.5, colors="#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_linewidth(0.48)
        ax.spines["bottom"].set_linewidth(0.48)
        ax.grid(False)
        return {
            "orbit_height_panel": "schematic_day_night_profiles_v2",
            "missions": len(specs),
            "orbit_assets": asset_count,
            "altitude_min_km": 50,
            "altitude_max_km": 1000,
            **{f"{mission}_status": status for mission, status in statuses.items()},
        }

    original.draw_ro_orbit_height_panel = draw_ro_orbit_height_panel_v2


def build_or_load_metop_line_cache(max_points_per_profile: int = 140) -> tuple[pd.DataFrame, pd.DataFrame]:
    if METOP_LINE_CACHE_CSV.exists() and METOP_SUMMARY_CACHE_CSV.exists():
        return (
            pd.read_csv(METOP_LINE_CACHE_CSV, low_memory=False),
            pd.read_csv(METOP_SUMMARY_CACHE_CSV, low_memory=False),
        )
    if not METOP_WINDOWS_CSV.exists():
        empty_lines = pd.DataFrame(columns=["source", "segment_id", "seq", "lon", "lat"])
        empty_summary = pd.DataFrame(
            [{"source": "MetOp", "events_total": 0, "events_requested": 0, "events_loaded": 0, "events_missing": 0}]
        )
        return empty_lines, empty_summary

    usecols = [
        "mission",
        "platform",
        "profile_id",
        "signal",
        "bin_start_time_utc",
        "bin_start_gps_seconds",
        "height_km",
        "tangent_lat_deg",
        "tangent_lon_deg",
        "qc_flag",
    ]
    windows = pd.read_csv(METOP_WINDOWS_CSV, usecols=usecols, low_memory=False)
    windows = windows[windows["qc_flag"].astype(str).str.startswith("ok", na=False)].copy()
    windows["bin_start_time_utc"] = pd.to_datetime(windows["bin_start_time_utc"], errors="coerce", utc=True)
    windows = windows[windows["bin_start_time_utc"].between(OP74_START_UTC, OP74_END_UTC, inclusive="both")].copy()
    for col in ["bin_start_gps_seconds", "height_km", "tangent_lat_deg", "tangent_lon_deg"]:
        windows[col] = pd.to_numeric(windows[col], errors="coerce")
    windows = windows.dropna(subset=["profile_id", "signal", "bin_start_gps_seconds", "height_km", "tangent_lat_deg", "tangent_lon_deg"])
    windows = windows[windows["height_km"].between(50.0, 1000.0, inclusive="both")].copy()

    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for index, ((profile_id, signal), group) in enumerate(windows.groupby(["profile_id", "signal"], sort=False, dropna=False)):
        group = group.sort_values("bin_start_gps_seconds").drop_duplicates(subset=["bin_start_gps_seconds"])
        if len(group) < 2:
            continue
        if len(group) > max_points_per_profile:
            take = np.linspace(0, len(group) - 1, max_points_per_profile).round().astype(int)
            group = group.iloc[np.unique(take)]
        segment_id = f"MetOp_{index:06d}"
        for seq, row in enumerate(group.itertuples(index=False)):
            rows.append(
                {
                    "source": "MetOp",
                    "segment_id": segment_id,
                    "seq": int(seq),
                    "lon": float(row.tangent_lon_deg),
                    "lat": float(row.tangent_lat_deg),
                }
            )
        event_rows.append(
            {
                "source": "MetOp",
                "profile_id": profile_id,
                "signal": signal,
                "platform": str(group["platform"].iloc[0]) if "platform" in group else "",
                "points_loaded": int(len(group)),
            }
        )

    lines = pd.DataFrame(rows, columns=["source", "segment_id", "seq", "lon", "lat"])
    events = pd.DataFrame(event_rows)
    platform_counts = events["platform"].value_counts(dropna=False).to_dict() if not events.empty and "platform" in events.columns else {}
    summary = pd.DataFrame(
        [
            {
                "source": "MetOp",
                "events_total": int(len(events)),
                "events_requested": int(windows.groupby(["profile_id", "signal"], dropna=False).ngroups),
                "events_loaded": int(len(events)),
                "events_missing": 0,
                "track_points": int(len(lines)),
                "metop_b_events": int(platform_counts.get("MetOp-B", 0)),
                "metop_c_events": int(platform_counts.get("MetOp-C", 0)),
            }
        ]
    )
    METOP_LINE_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    lines.to_csv(METOP_LINE_CACHE_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(METOP_SUMMARY_CACHE_CSV, index=False, encoding="utf-8-sig")
    return lines, summary


def patch_verified_occultation_tracks(original) -> None:
    base_loader = original.load_occultation_tracks

    def load_occultation_tracks_status_consistent() -> pd.DataFrame:
        tracks = base_loader()
        if tracks.empty or "source" not in tracks.columns:
            return tracks
        summary = tracks.attrs.get("summary")
        base_sources = ["PlanetiQ", "Spire", "TSX"]
        filtered = tracks[tracks["source"].astype(str).isin(base_sources)].copy()
        metop_lines, metop_summary = build_or_load_metop_line_cache()
        if not metop_lines.empty:
            filtered = pd.concat([filtered, metop_lines], ignore_index=True)
        if isinstance(summary, pd.DataFrame) and "source" in summary.columns:
            base_summary = summary[summary["source"].astype(str).isin(base_sources)].copy()
            filtered.attrs["summary"] = pd.concat([base_summary, metop_summary], ignore_index=True, sort=False)
        return filtered

    original.OCCULTATION_MISSION_ORDER = list(PANEL_F_VERIFIED_TRACK_SOURCES)
    original.OCCULTATION_MISSION_COLORS = dict(PANEL_F_COLORS)
    original.load_occultation_tracks = load_occultation_tracks_status_consistent

    def draw_occultation_track_panel_nature_fig3(old, ax, occultation_tracks: pd.DataFrame) -> dict[str, int]:
        original.setup_global_axis(old, ax, None, draw_grid_labels=False)
        summary = occultation_tracks.attrs.get("summary")
        group_col = "segment_id" if "segment_id" in occultation_tracks.columns else "track_id"
        segment_count = 0
        style = {
            "PlanetiQ": {"lw": 0.44, "alpha": 0.78},
            "Spire": {"lw": 0.44, "alpha": 0.78},
            "TSX": {"lw": 0.48, "alpha": 0.82},
            "MetOp": {"lw": 0.30, "alpha": 0.38},
        }
        for index, source in enumerate(original.OCCULTATION_MISSION_ORDER):
            sub = occultation_tracks[occultation_tracks["source"].eq(source)].copy()
            if sub.empty:
                continue
            segments = original.track_line_segments(sub, group_col, "lon", "lat", "seq")
            segment_count += len(segments)
            source_style = style.get(source, {"lw": 0.44, "alpha": 0.72})
            original.draw_track_segments(
                ax,
                segments,
                original.OCCULTATION_MISSION_COLORS[source],
                lw=source_style["lw"],
                alpha=source_style["alpha"],
                zorder=7 + index * 0.01,
            )
        count = original.occultation_track_count(occultation_tracks)
        out = {
            "Occultation profile tracks": count,
            "Occultation track segments": segment_count,
            "Occultation track points": int(len(occultation_tracks)),
        }
        if isinstance(summary, pd.DataFrame):
            out["summary"] = summary
        return out

    original.draw_occultation_track_panel_nature = draw_occultation_track_panel_nature_fig3


def patch_panel_titles(original) -> None:
    old_add_external_title = original.add_external_title

    def add_external_title_fig3_consistent(fig, ax, title, *args, **kwargs):
        if title == "Ionospheric layers and RO orbit heights":
            title = "Ionospheric layers and LEO-RO orbit heights"
        return old_add_external_title(fig, ax, title, *args, **kwargs)

    original.add_external_title = add_external_title_fig3_consistent


def main() -> None:
    table = load_source_table()
    specs, statuses = mission_specs_from_table(table)
    absent_from_table = [mission for mission in MISSION_ORDER if mission not in statuses]
    if absent_from_table:
        raise ValueError(f"Source table does not list required panel-c mission statuses: {absent_from_table}")
    original = load_original_module()
    original.OUT_NATURE_PNG = OUT_PNG
    original.OUT_NATURE_PDF = OUT_PDF
    patch_panel_titles(original)
    patch_orbit_panel(original, specs, statuses)
    patch_verified_occultation_tracks(original)
    png, pdf = original.build_nature_redraw()
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print("panel c missions:", ", ".join(str(spec["mission"]) for spec in specs))
    print("panel f verified track missions:", ", ".join(PANEL_F_VERIFIED_TRACK_SOURCES))
    print("statuses:", statuses)


if __name__ == "__main__":
    main()
