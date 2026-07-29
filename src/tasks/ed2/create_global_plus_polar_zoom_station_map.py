from __future__ import annotations

import importlib.util
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch, Polygon as MplPolygon
from PIL import Image, ImageDraw, ImageFont


TASK_DIR = Path(__file__).resolve().parent
REPRO_ROOT = TASK_DIR.parents[2]
PACKAGE_ROOT = TASK_DIR
DATA_DIR = REPRO_ROOT / "data" / "analysis_ready" / "ED2"
FIGURE_DIR = REPRO_ROOT / "outputs" / "ED2"
cartopy.config["pre_existing_data_dir"] = str(TASK_DIR / "cartopy_data")
cartopy.config["data_dir"] = TASK_DIR / "cartopy_data"

OLD_SCRIPT = TASK_DIR / "dependency_create_combined_polar_source_maps.py"
SUPERDARN_INVENTORY_CSV = DATA_DIR / "_not_packaged_superdarn_raw_inventory.csv"
ESWUA_ROOT = DATA_DIR / "_not_packaged_eswua_raw"
ESWUA_STATION_CSV = ESWUA_ROOT / "station_manifest.csv"
ESWUA_AVAILABILITY_CSV = ESWUA_ROOT / "availability_manifest.csv"
CHAIN_IPP_META = DATA_DIR / "_not_packaged_chain_ipp_meta.js"
LUNA_TRACK_ZIP = Path(
    DATA_DIR / "All_CTLS_recomputed_J2000_tracks_50_1000km.zip"
)
LUNA_TRACK_ZIP_MEMBER = (
    "All_CTLS_recomputed_J2000_tracks_50_1000km/"
    "data/All_CTLS_recomputed_J2000_tracks_50_1000km_tangent_points.csv"
)
LUNA_TRACK_CSV = DATA_DIR / "_not_packaged_luna_tangent_points.csv"
GLOBAL_OCCULTATION_SCRIPT = DATA_DIR / "_not_packaged_global_occultation_builder.py"

OUT_DIR = FIGURE_DIR
OUT_PNG = OUT_DIR / "combined_station_global_plus_polar_zooms.png"
OUT_PDF = OUT_DIR / "combined_station_global_plus_polar_zooms.pdf"
OUT_NATURE_PNG = OUT_DIR / "combined_station_global_plus_polar_zooms_nature.png"
OUT_NATURE_PDF = None
OUT_ISMR_CSV = DATA_DIR / "global_plus_polar_eswua_chain_ismr_stations.csv"
OUT_SUPERDARN_CSV = DATA_DIR / "global_plus_polar_current_superdarn_radars.csv"
OUT_LUNA_SUMMARY_CSV = DATA_DIR / "global_plus_polar_luna_track_summary.csv"
OUT_TRACK_SUMMARY_CSV = (
    OUT_DIR / "derived" / "global_plus_polar_track_panel_summary.csv"
)
OUT_LUNA_SAT_COLOR_CSV = DATA_DIR / "global_plus_polar_luna_satellite_colors.csv"
OUT_OP74_OCC_CACHE_CSV = (
    DATA_DIR / "global_plus_polar_op74_global_occultation_line_cache.csv.gz"
)
OUT_OP74_OCC_SUMMARY_CSV = DATA_DIR / "global_plus_polar_op74_global_occultation_summary_cache.csv"
OUT_GNSS_ROTI_CSV = DATA_DIR / "global_plus_polar_gnss_roti_stations.csv"
ORBIT_ASSET_DIR = TASK_DIR / "assets"
GNSS_ROTI_RINEX_DIR_CANDIDATES = [DATA_DIR / "_not_packaged_ground_rinex"]
GNSS_ROTI_SUMMARY_CANDIDATES = [DATA_DIR / "_not_packaged_ground_roti_summary.csv"]

POLAR_LAT_LIMIT = 60.0
SUPERDARN_SITE_LAT_LIMIT = 55.0
POLAR_CENTRAL_LONGITUDE = -90.0
ISMR_MARKER_SIZE = 18
LUNA_TRACK_COLOR = "#f59e0b"
OCCULTATION_TRACK_COLOR = "#2563eb"
LUNA_TRACK_GROUP_COLS = ["phase", "op_id", "sat", "signal_id", "segment_id"]
OCCULTATION_MISSION_ORDER = ["PlanetiQ", "Spire", "FengYun GNOS", "TSX"]
OCCULTATION_MISSION_COLORS = {
    "PlanetiQ": "#0f4c9a",
    "Spire": "#087a3b",
    "FengYun GNOS": "#b96b00",
    "TSX": "#c51b4a",
}
GEOMAG_NORTH_LAT_DEG = 80.65
GEOMAG_NORTH_LON_DEG = -72.68
GEOMAG_LAT_LINES = (-60.0, -20.0, 0.0, 20.0, 60.0)


def load_old_module():
    spec = importlib.util.spec_from_file_location("old_station_map_three_panel", OLD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_current_superdarn_radars(old) -> pd.DataFrame:
    if not SUPERDARN_INVENTORY_CSV.exists() and OUT_SUPERDARN_CSV.exists():
        return pd.read_csv(OUT_SUPERDARN_CSV, low_memory=False)
    inv = pd.read_csv(SUPERDARN_INVENTORY_CSV)
    inv["station"] = inv["code"].astype(str).str.upper()
    inv["name"] = inv["name"].astype(str)
    inv["lat"] = pd.to_numeric(inv["lat_use"], errors="coerce")
    inv["lon"] = old.normalize_lon(inv["lon_use"])
    inv["viewer_active_hours"] = pd.to_numeric(inv.get("viewer_active_hours"), errors="coerce")
    inv["viewer_cells"] = pd.to_numeric(inv.get("viewer_cells"), errors="coerce")
    inv["source_type"] = "Coherent radar"
    inv["status"] = "current viewer visible footprint"

    north = inv[
        inv["class_short"].eq("visible")
        & inv["lat"].ge(SUPERDARN_SITE_LAT_LIMIT)
        & inv["viewer_cells"].gt(0)
    ].copy()
    north["hemi_plot"] = "north"
    south = inv[
        inv["class_short"].eq("visible")
        & inv["lat"].le(-SUPERDARN_SITE_LAT_LIMIT)
        & inv["viewer_cells"].gt(0)
    ].copy()
    south["hemi_plot"] = "south"
    out = pd.concat([north, south], ignore_index=True)
    keep_cols = [
        "hemi_plot",
        "station",
        "name",
        "lat",
        "lon",
        "source_type",
        "status",
        "viewer_active_hours",
        "viewer_cells",
        "foot_lat_min",
        "foot_lat_max",
        "foot_lon_min",
        "foot_lon_max",
        "dist_median_km",
    ]
    return out[[col for col in keep_cols if col in out.columns]].dropna(subset=["lat", "lon"]).sort_values(
        ["hemi_plot", "station"]
    )


def read_chain_meta() -> list[dict[str, object]]:
    if not CHAIN_IPP_META.exists():
        return []
    text = CHAIN_IPP_META.read_text(encoding="ascii")
    payload = text.split("=", 1)[1].rsplit(";", 1)[0]
    meta = json.loads(payload)
    return list(meta.get("stations", []))


def load_updated_ismr_stations(old) -> pd.DataFrame:
    if (not ESWUA_STATION_CSV.exists() or not ESWUA_AVAILABILITY_CSV.exists()) and OUT_ISMR_CSV.exists():
        return pd.read_csv(OUT_ISMR_CSV, low_memory=False)
    stations = pd.read_csv(ESWUA_STATION_CSV)
    availability = pd.read_csv(ESWUA_AVAILABILITY_CSV)
    active = (
        availability.groupby("station_code", as_index=False)
        .agg(days_available=("available", "sum"), records=("record_count", "sum"))
        .query("days_available > 0")
    )
    eswua = stations.merge(active, left_on="code", right_on="station_code", how="inner")
    eswua["station"] = eswua["code"].astype(str)
    eswua["lat"] = pd.to_numeric(eswua["lat"], errors="coerce")
    eswua["lon"] = old.normalize_lon(eswua["lon"])
    eswua["network"] = "eSWua ISMR"
    eswua["source_type"] = "CHAIN/ISMR station"

    chain_rows = []
    for item in read_chain_meta():
        chain_rows.append(
            {
                "station": f"chain_{item['abbr']}",
                "code": f"chain_{item['abbr']}",
                "name": item.get("name", item.get("abbr", "")),
                "lat": float(item["lat"]),
                "lon": float(old.normalize_lon(float(item["lon"]))),
                "area": "CHAIN Canada",
                "network": "CHAIN Canada",
                "source_type": "CHAIN/ISMR station",
                "days_available": np.nan,
                "records": np.nan,
            }
        )
    chain = pd.DataFrame(chain_rows)

    keep_cols = [
        "station",
        "code",
        "name",
        "lat",
        "lon",
        "area",
        "network",
        "source_type",
        "days_available",
        "records",
    ]
    combined = pd.concat([eswua[keep_cols], chain[keep_cols]], ignore_index=True)
    combined = combined.dropna(subset=["lat", "lon"]).copy()
    combined["hemi_plot"] = np.where(combined["lat"].ge(0), "north", "south")
    return combined.sort_values(["hemi_plot", "network", "station"]).reset_index(drop=True)


def load_luna_tracks() -> pd.DataFrame:
    usecols = [
        "phase",
        "op_id",
        "sat",
        "signal_id",
        "signal_name",
        "segment_id",
        "time_utc",
        "rx_gps_seconds",
        "lugre_lat",
        "lugre_lon",
        "lugre_h_tan_km",
        "op_label",
        "phase_op_label",
    ]
    if LUNA_TRACK_ZIP.exists():
        with zipfile.ZipFile(LUNA_TRACK_ZIP) as archive:
            member = LUNA_TRACK_ZIP_MEMBER
            if member not in archive.namelist():
                member = next(
                    name for name in archive.namelist()
                    if name.endswith("All_CTLS_recomputed_J2000_tracks_50_1000km_tangent_points.csv")
                )
            with archive.open(member) as handle:
                df = pd.read_csv(handle, usecols=usecols, low_memory=False)
            df.attrs["source"] = str(LUNA_TRACK_ZIP)
    else:
        df = pd.read_csv(LUNA_TRACK_CSV, usecols=usecols, low_memory=False)
        df.attrs["source"] = str(LUNA_TRACK_CSV)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    for col in ["signal_id", "segment_id", "rx_gps_seconds", "lugre_lat", "lugre_lon", "lugre_h_tan_km"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["phase", "op_id", "sat", "signal_name", "op_label", "phase_op_label"]:
        df[col] = df[col].astype(str)
    df = df.dropna(subset=["time_utc", "lugre_lat", "lugre_lon", "lugre_h_tan_km"])
    df = df[df["lugre_h_tan_km"].between(50.0, 1000.0)].copy()
    return df.sort_values([*LUNA_TRACK_GROUP_COLS, "time_utc"]).reset_index(drop=True)


def load_global_occultation_module():
    source_dir = GLOBAL_OCCULTATION_SCRIPT.parent
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    spec = importlib.util.spec_from_file_location("selected_op_global_occultation_tracks", GLOBAL_OCCULTATION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def global_occultation_summary_from_stats(stats: dict[str, Counter], load_stats: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = []
    for source in OCCULTATION_MISSION_ORDER:
        counts = stats[source]
        loader = load_stats.get(source, {})
        rows.append(
            {
                "source": source,
                "events_total": int(counts["total"]),
                "events_polar": int(counts["polar"]),
                "events_mid_latitude": int(counts["mid"]),
                "events_equatorial": int(counts["equator"]),
                "events_unknown": int(counts["unknown"]),
                "events_requested": int(loader.get("requested", counts["total"])),
                "events_loaded": int(loader.get("loaded", counts["total"])),
                "events_missing": int(loader.get("missing", 0)),
            }
        )
    return pd.DataFrame(rows)


def line_cache_from_global_occultation_segments(lines_by_source: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> pd.DataFrame:
    rows = []
    for source in OCCULTATION_MISSION_ORDER:
        for line_index, (lon, lat) in enumerate(lines_by_source.get(source, [])):
            segment_id = f"{source.replace(' ', '_')}_{line_index:06d}"
            for seq, (lon_value, lat_value) in enumerate(zip(lon, lat)):
                rows.append(
                    {
                        "source": source,
                        "segment_id": segment_id,
                        "seq": seq,
                        "lon": float(lon_value),
                        "lat": float(lat_value),
                    }
                )
    return pd.DataFrame(rows, columns=["source", "segment_id", "seq", "lon", "lat"])


def load_occultation_tracks() -> pd.DataFrame:
    if OUT_OP74_OCC_CACHE_CSV.exists() and OUT_OP74_OCC_SUMMARY_CSV.exists():
        cached = pd.read_csv(OUT_OP74_OCC_CACHE_CSV, low_memory=False)
        cached.attrs["summary"] = pd.read_csv(OUT_OP74_OCC_SUMMARY_CSV, low_memory=False)
        return cached

    global_occ = load_global_occultation_module()
    stage_key = global_occ.normalize_stage_key("OP74")
    windows = global_occ.op.load_full_op_windows()
    start, end = windows[stage_key]

    lines_by_source: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    stats: dict[str, Counter] = defaultdict(Counter)
    load_stats: dict[str, dict[str, int]] = {}

    commercial_events = global_occ.load_commercial_events(start, end)
    fy_events = global_occ.load_fengyun_events(start, end)
    global_occ.collect_commercial_tracks(
        commercial_events,
        lines_by_source,
        stats,
        max_points=260,
        height_min_km=-5.0,
        height_max_km=160.0,
    )
    for source in ["PlanetiQ", "Spire"]:
        requested = int((commercial_events["source"].astype(str) == source).sum()) if len(commercial_events) else 0
        loaded = int(stats[source]["total"])
        load_stats[source] = {"requested": requested, "loaded": loaded, "missing": max(0, requested - loaded)}
    load_stats["FengYun GNOS"] = global_occ.collect_fengyun_tracks(
        fy_events,
        lines_by_source,
        stats,
        max_points=260,
        height_min_km=-5.0,
        height_max_km=160.0,
    )
    load_stats["TSX"] = global_occ.collect_tsx_tracks(
        start,
        end,
        lines_by_source,
        stats,
        max_points=260,
        height_min_km=-5.0,
        height_max_km=160.0,
    )

    out = line_cache_from_global_occultation_segments(lines_by_source)
    for col in ["seq", "lon", "lat"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["source", "segment_id", "seq", "lon", "lat"]).copy()
    out.to_csv(OUT_OP74_OCC_CACHE_CSV, index=False, encoding="utf-8-sig")
    summary = global_occultation_summary_from_stats(stats, load_stats)
    summary.to_csv(OUT_OP74_OCC_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    out.attrs["summary"] = summary
    return out


def luna_track_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df.groupby(LUNA_TRACK_GROUP_COLS, dropna=False).ngroups)


def occultation_track_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    summary = df.attrs.get("summary")
    if isinstance(summary, pd.DataFrame) and "events_total" in summary.columns:
        return int(pd.to_numeric(summary["events_total"], errors="coerce").fillna(0).sum())
    if "event_index" in df.columns:
        return int(df["event_index"].nunique(dropna=True))
    if "track_id" in df.columns:
        return int(df["track_id"].nunique(dropna=True))
    if "segment_id" in df.columns:
        return int(df["segment_id"].nunique(dropna=True))
    return 0


def split_luna_track_chunks(sub: pd.DataFrame) -> list[pd.DataFrame]:
    sub = sub.sort_values("time_utc").dropna(subset=["lugre_lon", "lugre_lat"]).copy()
    if len(sub) < 2:
        return []
    lon = sub["lugre_lon"].to_numpy(dtype=float)
    lat = sub["lugre_lat"].to_numpy(dtype=float)
    seconds = pd.to_numeric(sub["rx_gps_seconds"], errors="coerce")
    time_gap = np.nan_to_num(seconds.diff().to_numpy(dtype=float), nan=0.0, posinf=4.0) > 3.0
    lon_delta = np.abs(np.diff(lon))
    lon_gap = np.r_[False, lon_delta > 180.0]
    lat_delta = np.abs(np.diff(lat))
    angular_gap = np.r_[False, np.hypot(np.minimum(lon_delta, 360.0 - lon_delta), lat_delta) > 12.0]
    run_id = np.cumsum(time_gap | lon_gap | angular_gap)
    return [piece for _, piece in sub.groupby(run_id, sort=False) if len(piece) >= 2]


def draw_luna_tracks(ax, tracks: pd.DataFrame, hemisphere: str | None = None, lw: float = 0.34, alpha: float = 0.26) -> int:
    if tracks.empty:
        return 0
    if hemisphere == "north":
        data = tracks[tracks["lugre_lat"].ge(POLAR_LAT_LIMIT)].copy()
    elif hemisphere == "south":
        data = tracks[tracks["lugre_lat"].le(-POLAR_LAT_LIMIT)].copy()
    else:
        data = tracks
    if data.empty:
        return 0

    plate = ccrs.PlateCarree()
    drawn = 0
    for _, sub in data.groupby(LUNA_TRACK_GROUP_COLS, sort=False, dropna=False):
        for chunk in split_luna_track_chunks(sub):
            ax.plot(
                chunk["lugre_lon"].to_numpy(dtype=float),
                chunk["lugre_lat"].to_numpy(dtype=float),
                color=LUNA_TRACK_COLOR,
                lw=lw,
                alpha=alpha,
                transform=plate,
                zorder=5,
                solid_capstyle="round",
            )
            drawn += 1
    return drawn


def write_luna_summary(tracks: pd.DataFrame) -> None:
    rows = []
    for label, subset in [
        ("global", tracks),
        ("north_lat_ge_60", tracks[tracks["lugre_lat"].ge(POLAR_LAT_LIMIT)]),
        ("south_lat_le_-60", tracks[tracks["lugre_lat"].le(-POLAR_LAT_LIMIT)]),
    ]:
        rows.append(
            {
                "view": label,
                "rows": int(len(subset)),
                "tracks": luna_track_count(subset),
                "lat_min_deg": float(subset["lugre_lat"].min()) if not subset.empty else np.nan,
                "lat_max_deg": float(subset["lugre_lat"].max()) if not subset.empty else np.nan,
                "lon_min_deg": float(subset["lugre_lon"].min()) if not subset.empty else np.nan,
                "lon_max_deg": float(subset["lugre_lon"].max()) if not subset.empty else np.nan,
                "h_min_km": float(subset["lugre_h_tan_km"].min()) if not subset.empty else np.nan,
                "h_max_km": float(subset["lugre_h_tan_km"].max()) if not subset.empty else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_LUNA_SUMMARY_CSV, index=False, encoding="utf-8-sig")


def destination_point_deg(lat_deg: float, lon_deg: float, bearing_deg: float, angular_deg: float) -> tuple[float, float]:
    lat1 = np.radians(lat_deg)
    lon1 = np.radians(lon_deg)
    bearing = np.radians(bearing_deg)
    angular = np.radians(angular_deg)
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular)
        + np.cos(lat1) * np.sin(angular) * np.cos(bearing)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat2),
    )
    lon = (np.degrees(lon2) + 540.0) % 360.0 - 180.0
    return float(np.degrees(lat2)), float(lon)


def split_lon_segments(lats: np.ndarray, lons: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(lats) < 2:
        return []
    jump = np.r_[False, np.abs(np.diff(lons)) > 180.0]
    run_id = np.cumsum(jump)
    return [
        (lats[run_id == group], lons[run_id == group])
        for group in np.unique(run_id)
        if int((run_id == group).sum()) >= 2
    ]


def track_line_segments(
    df: pd.DataFrame,
    group_cols: list[str] | str,
    lon_col: str,
    lat_col: str,
    sort_cols: list[str] | str | None = None,
) -> list[np.ndarray]:
    if df.empty:
        return []
    segments: list[np.ndarray] = []
    groups = df.groupby(group_cols, sort=False, dropna=False)
    for _, sub in groups:
        if sort_cols is not None:
            sub = sub.sort_values(sort_cols)
        lon = pd.to_numeric(sub[lon_col], errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(sub[lat_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(lon) & np.isfinite(lat)
        lon = ((lon[ok] + 540.0) % 360.0) - 180.0
        lat = lat[ok]
        if len(lon) < 2:
            continue
        for lat_piece, lon_piece in split_lon_segments(lat, lon):
            if len(lon_piece) >= 2:
                segments.append(np.column_stack([lon_piece, lat_piece]))
    return segments


def draw_track_segments(ax, segments: list[np.ndarray], color: str, lw: float, alpha: float, zorder: float) -> None:
    if not segments:
        return
    collection = LineCollection(
        segments,
        colors=color,
        linewidths=lw,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
    )
    ax.add_collection(collection)


def luna_satellite_color_map(luna_tracks: pd.DataFrame) -> dict[str, str]:
    sats = sorted(luna_tracks["sat"].dropna().astype(str).unique())
    palette: list[str] = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(cmap_name)
        palette.extend(to_hex(cmap(i)) for i in range(cmap.N))
    return {sat: palette[i % len(palette)] for i, sat in enumerate(sats)}


def draw_luna_tracks_by_satellite(ax, luna_tracks: pd.DataFrame) -> tuple[int, int]:
    colors = luna_satellite_color_map(luna_tracks)
    segment_count = 0
    for sat, sat_df in luna_tracks.groupby("sat", sort=True, dropna=False):
        segments: list[np.ndarray] = []
        for _, sub in sat_df.groupby(LUNA_TRACK_GROUP_COLS, sort=False, dropna=False):
            for chunk in split_luna_track_chunks(sub):
                lon = ((chunk["lugre_lon"].to_numpy(dtype=float) + 540.0) % 360.0) - 180.0
                lat = chunk["lugre_lat"].to_numpy(dtype=float)
                for lat_piece, lon_piece in split_lon_segments(lat, lon):
                    segments.append(np.column_stack([lon_piece, lat_piece]))
        segment_count += len(segments)
        draw_track_segments(
            ax,
            segments,
            colors.get(str(sat), LUNA_TRACK_COLOR),
            lw=0.62,
            alpha=0.82,
            zorder=6,
        )
    return len(colors), segment_count


def geomagnetic_latitude_line(mlat_deg: float, samples: int = 721) -> tuple[np.ndarray, np.ndarray]:
    colat_deg = 90.0 - float(mlat_deg)
    lats: list[float] = []
    lons: list[float] = []
    for bearing in np.linspace(0.0, 360.0, samples):
        lat, lon = destination_point_deg(GEOMAG_NORTH_LAT_DEG, GEOMAG_NORTH_LON_DEG, float(bearing), colat_deg)
        lats.append(lat)
        lons.append(lon)
    return np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)


def draw_geomagnetic_latitude_lines(ax) -> None:
    plate = ccrs.PlateCarree()
    for mlat in GEOMAG_LAT_LINES:
        lats, lons = geomagnetic_latitude_line(mlat)
        color = "#5f6670"
        lw = 0.62 if abs(mlat) < 60.0 else 0.70
        alpha = 0.46 if abs(mlat) < 60.0 else 0.54
        for lat_piece, lon_piece in split_lon_segments(lats, lons):
            ax.plot(
                lon_piece,
                lat_piece,
                color=color,
                lw=lw,
                ls=(0, (3.8, 2.8)),
                alpha=alpha,
                transform=plate,
                zorder=4.2,
            )
        label_lon = 150.0
        label_idx = int(np.nanargmin(np.abs(((lons - label_lon + 180.0) % 360.0) - 180.0)))
        ax.text(
            float(lons[label_idx]),
            float(lats[label_idx]),
            r"$\lambda_m=0^\circ$" if abs(mlat) == 0.0 else rf"$\lambda_m={mlat:+.0f}^\circ$",
            fontsize=4.6,
            color=color,
            alpha=0.76,
            transform=plate,
            zorder=4.4,
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.12},
        )


def setup_global_axis(old, ax, title: str | None = "Global station distribution", draw_grid_labels: bool = True):
    plate = ccrs.PlateCarree()
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#ffffff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#ffffff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.32, edgecolor=old.COLORS["coast"], zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.12, edgecolor="#c9c9c9", zorder=2)
    gl = ax.gridlines(
        crs=plate,
        draw_labels=draw_grid_labels,
        linewidth=0.25,
        color=old.COLORS["grid"],
        alpha=0.62,
        linestyle="-",
        zorder=1,
    )
    if draw_grid_labels:
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 4.6, "color": old.COLORS["muted"]}
        gl.ylabel_style = {"size": 4.6, "color": old.COLORS["muted"]}
    if title:
        ax.set_title(title, fontsize=8.6, pad=7, color=old.COLORS["text"])
    draw_geomagnetic_latitude_lines(ax)
    return plate


def plot_points(ax, pc, df: pd.DataFrame, marker: str, color: str, size: float, alpha: float, zorder: int) -> int:
    if df.empty:
        return 0
    edgecolor = "#1f2937" if marker == "*" else "#ffffff"
    linewidth = 0.34 if marker == "*" else 0.18
    ax.scatter(
        df["lon"],
        df["lat"],
        s=size,
        marker=marker,
        c=color,
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=linewidth,
        transform=pc,
        zorder=zorder,
    )
    return int(len(df))


def plot_ismr_points(ax, pc, ismr: pd.DataFrame, size: float, zorder: int) -> int:
    if ismr.empty:
        return 0
    plot_df = jitter_nearby_points(ismr)
    ax.scatter(
        plot_df["plot_lon"],
        plot_df["plot_lat"],
        s=size,
        marker="s",
        c="#59a14f",
        alpha=0.96,
        edgecolors="#ffffff",
        linewidths=0.18,
        transform=pc,
        zorder=zorder,
    )
    return int(len(ismr))


def draw_global_panel(
    old,
    ax,
    ground_roti: pd.DataFrame,
    ismr: pd.DataFrame,
    isr: pd.DataFrame,
    coherent: pd.DataFrame,
) -> dict[str, int]:
    pc = setup_global_axis(old, ax)
    counts = {
        "GNSS/ROTI station": plot_points(ax, pc, ground_roti, "o", old.COLORS["gnss"], 9, 0.62, 7),
        "CHAIN/ISMR station": plot_ismr_points(ax, pc, ismr, ISMR_MARKER_SIZE, 13),
        "ISR radar": plot_points(ax, pc, isr, "*", old.COLORS["isr"], 64, 0.96, 11),
        "Coherent radar": plot_points(ax, pc, coherent, "D", old.COLORS["coherent"], 28, 0.96, 10),
    }
    return counts


def draw_global_panel_nature(
    old,
    ax,
    ground_roti: pd.DataFrame,
    ismr: pd.DataFrame,
    isr: pd.DataFrame,
    coherent: pd.DataFrame,
) -> dict[str, int]:
    pc = setup_global_axis(old, ax, None, draw_grid_labels=False)
    counts = {
        "GNSS/ROTI station": plot_points(ax, pc, ground_roti, "o", old.COLORS["gnss"], 7.0, 0.55, 7),
        "CHAIN/ISMR station": plot_ismr_points(ax, pc, ismr, 12.0, 13),
        "ISR radar": plot_points(ax, pc, isr, "*", old.COLORS["isr"], 42.0, 0.94, 11),
        "Coherent radar": plot_points(ax, pc, coherent, "D", old.COLORS["coherent"], 18.0, 0.94, 10),
    }
    return counts


def draw_luna_track_panel(old, ax, luna_tracks: pd.DataFrame) -> dict[str, int]:
    setup_global_axis(old, ax, "Luna tangent-point tracks", draw_grid_labels=False)
    satellite_count, segment_count = draw_luna_tracks_by_satellite(ax, luna_tracks)
    count = luna_track_count(luna_tracks)
    ax.text(
        0.02,
        0.02,
        f"{count} tracks, {satellite_count} satellites colored by sat, 50-1000 km",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=old.COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
        zorder=20,
    )
    return {
        "Luna tangent tracks": count,
        "Luna satellites": satellite_count,
        "Luna track segments": segment_count,
        "Luna track points": int(len(luna_tracks)),
    }


def draw_luna_track_panel_nature(old, ax, luna_tracks: pd.DataFrame) -> dict[str, int]:
    setup_global_axis(old, ax, None, draw_grid_labels=False)
    satellite_count, segment_count = draw_luna_tracks_by_satellite(ax, luna_tracks)
    count = luna_track_count(luna_tracks)
    return {
        "Luna tangent tracks": count,
        "Luna satellites": satellite_count,
        "Luna track segments": segment_count,
        "Luna track points": int(len(luna_tracks)),
    }


def draw_occultation_track_panel(old, ax, occultation_tracks: pd.DataFrame) -> dict[str, int]:
    setup_global_axis(old, ax, "OP74 global LEO-RO tracks by mission", draw_grid_labels=False)
    summary = occultation_tracks.attrs.get("summary")
    event_counts = {}
    if isinstance(summary, pd.DataFrame) and {"source", "events_total"}.issubset(summary.columns):
        for _, row in summary.iterrows():
            value = pd.to_numeric(row["events_total"], errors="coerce")
            event_counts[str(row["source"])] = 0 if pd.isna(value) else int(value)
    group_col = "segment_id" if "segment_id" in occultation_tracks.columns else "track_id"
    segment_count = 0
    y0 = 0.805
    for index, source in enumerate(OCCULTATION_MISSION_ORDER):
        sub = occultation_tracks[occultation_tracks["source"].eq(source)].copy()
        if sub.empty:
            continue
        segments = track_line_segments(sub, group_col, "lon", "lat", "seq")
        segment_count += len(segments)
        color = OCCULTATION_MISSION_COLORS[source]
        draw_track_segments(ax, segments, color, lw=0.42, alpha=0.82, zorder=7 + index * 0.01)
        source_count = event_counts.get(source, int(sub[group_col].nunique()))
        ax.plot(
            [0.020, 0.055],
            [y0 + index * 0.045, y0 + index * 0.045],
            color=color,
            lw=2.1,
            transform=ax.transAxes,
            solid_capstyle="round",
            zorder=25,
        )
        ax.text(
            0.062,
            y0 + index * 0.045,
            f"{source} ({source_count:,})",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=4.8,
            color="#17211d",
            zorder=25,
        )
    count = occultation_track_count(occultation_tracks)
    ax.text(
        0.02,
        0.02,
        f"{count} OP74 global occultation events, -5-160 km",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=old.COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
        zorder=20,
    )
    return {
        "Occultation profile tracks": count,
        "Occultation track segments": segment_count,
        "Occultation track points": int(len(occultation_tracks)),
    }


def draw_occultation_track_panel_nature(old, ax, occultation_tracks: pd.DataFrame) -> dict[str, int]:
    setup_global_axis(old, ax, None, draw_grid_labels=False)
    summary = occultation_tracks.attrs.get("summary")
    group_col = "segment_id" if "segment_id" in occultation_tracks.columns else "track_id"
    segment_count = 0
    for index, source in enumerate(OCCULTATION_MISSION_ORDER):
        sub = occultation_tracks[occultation_tracks["source"].eq(source)].copy()
        if sub.empty:
            continue
        segments = track_line_segments(sub, group_col, "lon", "lat", "seq")
        segment_count += len(segments)
        draw_track_segments(
            ax,
            segments,
            OCCULTATION_MISSION_COLORS[source],
            lw=0.46,
            alpha=0.82,
            zorder=7 + index * 0.01,
        )
    count = occultation_track_count(occultation_tracks)
    out = {
        "Occultation profile tracks": count,
        "Occultation track segments": segment_count,
        "Occultation track points": int(len(occultation_tracks)),
    }
    if isinstance(summary, pd.DataFrame):
        out["summary"] = summary
    return out


def draw_current_coherent_coverage(old, ax, pc, coherent: pd.DataFrame, hemisphere: str) -> int:
    subset = coherent[coherent["hemi_plot"].eq(hemisphere)].copy()
    coverage_sites = old.merge_nearby_locations(subset, old.COHERENT_COVERAGE_MERGE_KM)
    fan_polygons: list[np.ndarray] = []
    for _, row in coverage_sites.iterrows():
        lats, lons = old.coherent_fan(float(row["lat"]), float(row["lon"]), hemisphere)
        xy = old.project_polygon(ax, pc, lats, lons)
        if len(xy) >= 3:
            fan_polygons.append(xy)
    for xy in fan_polygons:
        ax.add_patch(
            MplPolygon(
                xy,
                closed=True,
                facecolor=old.COLORS["coherent_fill"],
                edgecolor="none",
                alpha=old.COVERAGE_ALPHA["coherent_fill"],
                transform=ax.transData,
                zorder=1.5,
                clip_on=True,
            )
        )
    for xy in fan_polygons:
        ax.add_patch(
            MplPolygon(
                xy,
                closed=True,
                facecolor="none",
                edgecolor=old.COLORS["coherent"],
                alpha=old.COVERAGE_ALPHA["coherent_edge"],
                linewidth=0.34,
                transform=ax.transData,
                zorder=2.4,
                clip_on=True,
            )
        )
    return int(len(coverage_sites))


def plot_current_coherent_points(old, ax, pc, coherent: pd.DataFrame, hemisphere: str) -> int:
    subset = coherent[coherent["hemi_plot"].eq(hemisphere)].copy()
    if subset.empty:
        return 0
    ax.scatter(
        subset["lon"],
        subset["lat"],
        s=old.COHERENT_MARKER_SIZE,
        marker="D",
        c=old.COLORS["coherent"],
        alpha=0.96,
        edgecolors="#ffffff",
        linewidths=0.22,
        transform=pc,
        zorder=10,
    )
    return int(len(subset))


def polar_boundary(ax) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    verts = np.vstack([0.5 + 0.5 * np.sin(theta), 0.5 + 0.5 * np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)


def _polar_lon_label(lon_deg: int) -> str:
    lon = ((int(lon_deg) + 180) % 360) - 180
    if lon == 0:
        return r"$0^\circ$"
    if abs(lon) == 180:
        return r"$180^\circ$"
    hemi = "E" if lon > 0 else "W"
    return rf"${abs(lon)}^\circ${hemi}"


def add_polar_longitude_labels(ax) -> None:
    label_color = "#777f89"
    labels = [
        (0, 1.012, 0.500, "left", "center"),
        (45, 0.862, 0.862, "left", "bottom"),
        (90, 0.500, 1.012, "center", "bottom"),
        (135, 0.138, 0.862, "right", "bottom"),
        (180, -0.012, 0.500, "right", "center"),
        (-135, 0.138, 0.138, "right", "top"),
        (-90, 0.500, -0.032, "center", "top"),
        (-45, 0.862, 0.138, "left", "top"),
    ]
    for lon, x, y, ha, va in labels:
        ax.text(
            x,
            y,
            _polar_lon_label(lon),
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=4.9,
            color=label_color,
            clip_on=False,
            zorder=30,
        )


def setup_polar_axis_60(old, ax, hemisphere: str):
    old.POLAR_LAT_LIMIT = POLAR_LAT_LIMIT
    pc = ccrs.PlateCarree()
    if hemisphere == "north":
        ax.set_extent([-180.0, 180.0, POLAR_LAT_LIMIT, 90.0], crs=pc)
    else:
        ax.set_extent([-180.0, 180.0, -90.0, -POLAR_LAT_LIMIT], crs=pc)
        ax.set_ylim(ax.get_ylim()[::-1])
    polar_boundary(ax)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#ffffff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#ffffff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.34, edgecolor=old.COLORS["coast"], zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.12, edgecolor="#c9c9c9", zorder=2)
    gl = ax.gridlines(
        crs=pc,
        draw_labels=False,
        linewidth=0.24,
        color=old.COLORS["grid"],
        alpha=0.68,
        linestyle="-",
        zorder=1,
    )
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 45))
    gl.ylocator = plt.FixedLocator(np.arange(-80, 91, 10))
    label_color = "#777f89"
    add_polar_longitude_labels(ax)
    lat_label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.08}
    label_ha = "right" if hemisphere == "north" else "left"
    label_suffix = "N" if hemisphere == "north" else "S"
    for lat_value in (80, 70, 60):
        plot_lat = float(lat_value if hemisphere == "north" else -lat_value)
        label_lon = 160.0 if hemisphere == "north" else -125.0
        label_ha = "right" if hemisphere == "north" else "right"
        ax.text(
            label_lon,
            plot_lat,
            rf"${lat_value}^\circ${label_suffix}",
            transform=pc,
            ha=label_ha,
            va="center",
            fontsize=4.9,
            color=label_color,
            bbox=lat_label_box,
            zorder=31,
        )
    return pc


def draw_polar_panel(
    old,
    ax,
    hemisphere: str,
    ground_roti: pd.DataFrame,
    ismr: pd.DataFrame,
    isr: pd.DataFrame,
    coherent: pd.DataFrame,
) -> dict[str, int]:
    pc = setup_polar_axis_60(old, ax, hemisphere)
    coherent_coverage = draw_current_coherent_coverage(old, ax, pc, coherent, hemisphere)
    isr_coverage = old.draw_isr_coverage(ax, pc, isr, hemisphere)
    counts = old.plot_station_points(ax, pc, ground_roti, isr, coherent.iloc[0:0], hemisphere)
    replot_polar_isr_markers(ax, pc, isr, hemisphere, old)
    if hemisphere == "north":
        polar_ismr = ismr[ismr["lat"].ge(POLAR_LAT_LIMIT)].copy()
    else:
        polar_ismr = ismr[ismr["lat"].le(-POLAR_LAT_LIMIT)].copy()
    counts["CHAIN/ISMR station"] = plot_ismr_points(ax, pc, polar_ismr, ISMR_MARKER_SIZE, 13)
    counts["Coherent radar"] = plot_current_coherent_points(old, ax, pc, coherent, hemisphere)
    counts["ISR coverage"] = isr_coverage
    counts["Coherent coverage"] = coherent_coverage
    return counts


def replot_polar_isr_markers(ax, pc, isr: pd.DataFrame, hemisphere: str, old) -> None:
    if hemisphere == "north":
        subset = isr[isr["lat"].ge(POLAR_LAT_LIMIT)].copy()
    else:
        subset = isr[isr["lat"].le(-POLAR_LAT_LIMIT)].copy()
    if subset.empty:
        return
    plot_df = old.merge_nearby_locations(subset, old.ISR_COVERAGE_MERGE_KM)
    ax.scatter(
        plot_df["lon"],
        plot_df["lat"],
        s=old.ISR_MARKER_SIZE,
        marker="*",
        c=old.COLORS["isr"],
        alpha=0.98,
        edgecolors="#1f2937",
        linewidths=0.42,
        transform=pc,
        zorder=12.5,
    )


def jitter_nearby_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy().reset_index(drop=True)
    out["plot_lat"] = out["lat"].astype(float)
    out["plot_lon"] = out["lon"].astype(float)
    keys = list(zip(out["lat"].round(1), out["lon"].round(1)))
    for key in sorted(set(keys)):
        idx = [i for i, value in enumerate(keys) if value == key]
        if len(idx) <= 1:
            continue
        angles = np.linspace(0, 2 * np.pi, len(idx), endpoint=False)
        for i, angle in zip(idx, angles):
            # Small display-only offsets separate co-located ISMR receivers such as Dome C.
            out.loc[i, "plot_lat"] += 0.38 * np.sin(angle)
            out.loc[i, "plot_lon"] += 0.72 * np.cos(angle) / max(0.35, np.cos(np.radians(abs(float(out.loc[i, "lat"])))))
    return out


def replot_polar_ismr_markers(ax, pc, ismr: pd.DataFrame, hemisphere: str, old) -> None:
    if hemisphere == "north":
        subset = ismr[ismr["lat"].ge(POLAR_LAT_LIMIT)].copy()
    else:
        subset = ismr[ismr["lat"].le(-POLAR_LAT_LIMIT)].copy()
    if subset.empty:
        return
    ax.scatter(
        subset["lon"],
        subset["lat"],
        s=14,
        marker="s",
        c=old.COLORS["ismr"],
        alpha=0.96,
        edgecolors="#ffffff",
        linewidths=0.18,
        transform=pc,
        zorder=13,
    )


def _legacy_build_legend_unused(old, counts: dict[str, int]) -> list:
    return [
        Line2D([0], [0], marker="o", color="none", label=f"GNSS/ROTI ({counts.get('GNSS/ROTI station', 0)})", markerfacecolor=old.COLORS["gnss"], markeredgecolor="white", markersize=5.2),
        Line2D([0], [0], marker="s", color="none", label=f"ISMR/CHAIN ({counts.get('CHAIN/ISMR station', 0)})", markerfacecolor=old.COLORS["ismr"], markeredgecolor="white", markersize=5.2),
        Line2D([0], [0], marker="*", color="none", label=f"ISR ({counts.get('ISR radar', 0)})", markerfacecolor=old.COLORS["isr"], markeredgecolor="white", markersize=7.2),
        Line2D([0], [0], marker="D", color="none", label=f"SuperDARN ({counts.get('Coherent radar', 0)})", markerfacecolor=old.COLORS["coherent"], markeredgecolor="white", markersize=5.2),
        Line2D([0], [0], color="#0f172a", lw=1.0, ls=(0, (4.2, 3.0)), alpha=0.60, label="dipole MLAT ±20/±60"),
        Patch(facecolor=old.COLORS["isr"], edgecolor=old.COLORS["isr"], alpha=old.COVERAGE_ALPHA["isr_fill"], label="ISR nominal range"),
        Patch(facecolor=old.COLORS["coherent_fill"], edgecolor=old.COLORS["coherent"], alpha=old.COVERAGE_ALPHA["coherent_fill"], label="SuperDARN nominal fan"),
    ]


def build_legend(old, counts: dict[str, int]) -> list:
    return [
        Line2D([0], [0], marker="o", color="none", label=f"GNSS/ROTI ({counts.get('GNSS/ROTI station', 0)})", markerfacecolor=old.COLORS["gnss"], markeredgecolor="white", markersize=5.2),
        Line2D([0], [0], marker="s", color="none", label=f"ISMR/CHAIN ({counts.get('CHAIN/ISMR station', 0)})", markerfacecolor=old.COLORS["ismr"], markeredgecolor="white", markersize=5.2),
        Line2D([0], [0], marker="*", color="none", label=f"ISR ({counts.get('ISR radar', 0)})", markerfacecolor=old.COLORS["isr"], markeredgecolor="white", markersize=7.2),
        Line2D([0], [0], marker="D", color="none", label=f"SuperDARN ({counts.get('Coherent radar', 0)})", markerfacecolor=old.COLORS["coherent"], markeredgecolor="white", markersize=5.2),
        Line2D([0], [0], color="#5f6670", lw=1.0, ls=(0, (3.8, 2.8)), alpha=0.70, label=r"$\lambda_m$ -60/-20/0/+20/+60"),
        Line2D([0], [0], color="#111827", lw=1.8, alpha=0.95, label="Luna tracks by satellite"),
        Line2D([0], [0], color="#0f4c9a", lw=1.8, alpha=0.95, label="LEO-RO tracks by mission"),
        Patch(facecolor=old.COLORS["isr"], edgecolor=old.COLORS["isr"], alpha=old.COVERAGE_ALPHA["isr_fill"], label="ISR nominal range"),
        Patch(facecolor=old.COLORS["coherent_fill"], edgecolor=old.COLORS["coherent"], alpha=old.COVERAGE_ALPHA["coherent_fill"], label="SuperDARN nominal fan"),
    ]


def station_legend_handles(old, counts: dict[str, int], include_context: bool = False) -> list:
    specs = [
        ("GNSS/ROTI station", "o", old.COLORS["gnss"], 4.5, "Polar GNSS receivers"),
        ("CHAIN/ISMR station", "s", old.COLORS["ismr"], 4.5, "GNSS scintillation receivers"),
        ("ISR radar", "*", old.COLORS["isr"], 6.2, "ISR radars"),
        ("Coherent radar", "D", old.COLORS["coherent"], 4.5, "SuperDARN radars"),
    ]
    handles: list = []
    for key, marker, color, size, label in specs:
        count = int(counts.get(key, 0) or 0)
        if count <= 0:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                label=f"{label} ({count})",
                markerfacecolor=color,
                markeredgecolor="#1f2937" if marker == "*" else "white",
                markeredgewidth=0.45 if marker == "*" else 0.35,
                markersize=size,
            )
        )
    if include_context:
        if int(counts.get("ISR coverage", 0) or 0) > 0:
            handles.append(
                Patch(facecolor=old.COLORS["isr"], edgecolor=old.COLORS["isr"], alpha=old.COVERAGE_ALPHA["isr_fill"], label="ISR range")
            )
        if int(counts.get("Coherent coverage", 0) or 0) > 0:
            handles.append(
                Patch(facecolor=old.COLORS["coherent_fill"], edgecolor=old.COLORS["coherent"], alpha=old.COVERAGE_ALPHA["coherent_fill"], label="SuperDARN fan")
            )
    return handles


def add_station_legend(
    ax,
    old,
    counts: dict[str, int],
    *,
    loc: str,
    anchor: tuple[float, float],
    ncol: int = 1,
    include_context: bool = False,
    fontsize: float = 4.6,
) -> None:
    handles = station_legend_handles(old, counts, include_context=include_context)
    if not handles:
        return
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=anchor,
        frameon=False,
        fontsize=fontsize,
        ncol=ncol,
        handlelength=0.95,
        handletextpad=0.28,
        columnspacing=0.58,
        borderaxespad=0.0,
    )


def mission_daily_count_labels(occultation_tracks: pd.DataFrame) -> dict[str, str]:
    summary = occultation_tracks.attrs.get("summary")
    if not isinstance(summary, pd.DataFrame) or not {"source", "events_total"}.issubset(summary.columns):
        return {source: source for source in OCCULTATION_MISSION_ORDER}
    labels: dict[str, str] = {}
    for source in OCCULTATION_MISSION_ORDER:
        sub = summary[summary["source"].astype(str).eq(source)]
        if sub.empty:
            labels[source] = source
            continue
        value = int(pd.to_numeric(sub["events_total"], errors="coerce").fillna(0).iloc[0])
        labels[source] = f"{source} ({value:,})"
    return labels


def mission_legend_handles(occultation_tracks: pd.DataFrame | None = None) -> list:
    labels = (
        mission_daily_count_labels(occultation_tracks)
        if occultation_tracks is not None
        else {source: source for source in OCCULTATION_MISSION_ORDER}
    )
    return [
        Line2D(
            [0],
            [0],
            color=OCCULTATION_MISSION_COLORS[source],
            lw=2.0,
            alpha=0.95,
            label=labels[source],
            solid_capstyle="round",
        )
        for source in OCCULTATION_MISSION_ORDER
    ]


def add_orbit_asset(ax, asset_name: str, x: float, y: float, zoom: float) -> bool:
    path = ORBIT_ASSET_DIR / f"{asset_name}.png"
    if not path.exists():
        return False
    image = np.asarray(Image.open(path).convert("RGBA"))
    artist = AnnotationBbox(
        OffsetImage(image, zoom=zoom, resample=True),
        (x, y),
        xycoords="data",
        frameon=False,
        pad=0.0,
        zorder=10,
        box_alignment=(0.5, 0.5),
    )
    ax.add_artist(artist)
    return True


def draw_ro_orbit_height_panel(ax) -> dict[str, object]:
    layer_colors = {
        "D": "#fff5df",
        "E": "#fcebe2",
        "F1": "#f6dfe8",
        "F2": "#e7eef8",
        "Topside": "#f1f4f7",
    }
    layers = [
        ("D", 50, 90),
        ("E", 90, 140),
        ("F1", 140, 200),
        ("F2", 200, 500),
        ("Topside", 500, 1000),
    ]
    for label, y0, y1 in layers:
        ax.axhspan(y0, y1, facecolor=layer_colors[label], edgecolor="none", alpha=0.88, zorder=0)
        text_x = 1.76 if label == "Topside" else 1.78
        text_y = 905.0 if label == "Topside" else (y0 + y1) / 2.0
        ax.text(
            text_x,
            text_y,
            label,
            ha="right",
            va="center",
            fontsize=5.2,
            weight="bold" if label in {"F1", "F2"} else "normal",
            color="#333333",
            zorder=5,
        )
    for y in [90, 140, 200, 500]:
        ax.axhline(y, color="#343a40", lw=0.44, ls=(0, (3.2, 2.6)), alpha=0.48, zorder=1)

    altitude = np.linspace(50.0, 1000.0, 620)
    day_profile = (
        0.04
        + 0.12 * np.exp(-((altitude - 105.0) / 18.0) ** 2)
        + 0.30 * np.exp(-((altitude - 165.0) / 32.0) ** 2)
        + 1.10 * np.exp(-((altitude - 330.0) / 135.0) ** 2)
        + 0.17 * np.exp(-((altitude - 650.0) / 310.0) ** 2)
    )
    night_profile = (
        0.025
        + 0.045 * np.exp(-((altitude - 112.0) / 25.0) ** 2)
        + 0.64 * np.exp(-((altitude - 345.0) / 170.0) ** 2)
        + 0.18 * np.exp(-((altitude - 660.0) / 360.0) ** 2)
    )
    ax.plot(day_profile, altitude, color="#d7191c", lw=1.85, solid_capstyle="round", label="day", zorder=4)
    ax.plot(night_profile, altitude, color="#1f4e79", lw=1.65, ls=(0, (4.0, 2.4)), solid_capstyle="round", label="night", zorder=4)
    ax.text(1.06, 360, "day F2", ha="left", va="center", fontsize=4.8, color="#9f1d20", zorder=6)
    ax.text(0.55, 295, "night F", ha="left", va="center", fontsize=4.8, color="#1f4e79", zorder=6)

    mission_orbits = {
        "PlanetiQ": (0.82, 620.0, 18.0, "PlanetiQ", 0.145, "PlanetiQ", r"$h$~620 km, $i$=97.9$^\circ$", 700.0, 585.0),
        "Spire": (1.08, 550.0, 70.0, "Spire", 0.155, "Spire", r"$h$ 400-600 km, varied $i$", 685.0, 455.0),
        "FengYun GNOS": (1.40, 836.0, 24.0, "FengYun_GNOS", 0.140, "FengYun", r"$h$~836 km, $i$=98.75$^\circ$", 900.0, 790.0),
        "TSX": (1.65, 514.0, 12.0, "TSX", 0.140, "TSX", r"$h$~514 km, $i$=97.4$^\circ$", 585.0, 480.0),
    }
    asset_count = 0
    for source in OCCULTATION_MISSION_ORDER:
        x, altitude_km, half_range, asset_name, zoom, label, orbit_text, asset_y, orbit_text_y = mission_orbits[source]
        color = OCCULTATION_MISSION_COLORS[source]
        ax.plot([x, x], [altitude_km - half_range, altitude_km + half_range], color=color, lw=1.45, alpha=0.88, zorder=6)
        ax.scatter([x], [altitude_km], s=17, color=color, edgecolors="white", linewidths=0.55, zorder=7)
        if add_orbit_asset(ax, asset_name, x, asset_y, zoom):
            asset_count += 1
        ax.text(
            x,
            asset_y + 48.0,
            label,
            ha="center",
            va="bottom",
            fontsize=4.6,
            color="#222222",
            zorder=9,
        )
        ax.text(
            x,
            orbit_text_y,
            orbit_text,
            ha="center",
            va="center",
            fontsize=3.95,
            color="#30343b",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.54, "pad": 0.08},
            zorder=9,
        )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.235, 0.985),
        frameon=False,
        fontsize=4.8,
        handlelength=1.2,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    ax.set_ylim(50, 1000)
    ax.set_xlim(0.0, 1.82)
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
        "orbit_height_panel": "schematic_day_night_profiles",
        "missions": len(mission_orbits),
        "orbit_assets": asset_count,
        "altitude_min_km": 50,
        "altitude_max_km": 1000,
    }


def add_panel_label(ax, label: str, x: float = -0.055, y: float = 1.055) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        weight="bold",
        color="#202020",
        clip_on=False,
        zorder=40,
    )


def ecef_to_geodetic_deg(x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lon = math.atan2(y_m, x_m)
    p = math.hypot(x_m, y_m)
    lat = math.atan2(z_m, p * (1.0 - e2))
    height = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        height = p / max(math.cos(lat), 1.0e-15) - n
        lat = math.atan2(z_m, p * (1.0 - e2 * n / (n + height)))
    return math.degrees(lat), ((math.degrees(lon) + 540.0) % 360.0) - 180.0, height


def parse_rinex_station_header(path: Path) -> dict[str, object] | None:
    marker = path.name[:4].upper()
    xyz: tuple[float, float, float] | None = None
    monument_location: tuple[float, float, float] | None = None
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for _ in range(140):
            line = handle.readline()
            if not line:
                break
            label = line[60:].strip() if len(line) >= 60 else ""
            if label == "MARKER NAME":
                name = line[:60].strip()
                if name:
                    marker = name.split()[0].upper()
            elif label == "APPROX POSITION XYZ":
                fields = line[:60].split()
                if len(fields) >= 3:
                    try:
                        xyz = (float(fields[0]), float(fields[1]), float(fields[2]))
                    except ValueError:
                        pass
            elif "Monument location:" in line:
                fields = line.split("Monument location:", 1)[1].split()
                if len(fields) >= 3:
                    try:
                        monument_location = (float(fields[0]), float(fields[1]), float(fields[2]))
                    except ValueError:
                        pass
            elif label == "END OF HEADER":
                break
    if monument_location is not None:
        lat, lon, height = monument_location
    elif xyz is not None:
        lat, lon, height = ecef_to_geodetic_deg(*xyz)
    else:
        return None
    return {
        "station": marker.lower(),
        "name": marker,
        "lat": lat,
        "lon": lon,
        "height_m": height,
        "source_type": "GNSS/ROTI station",
        "source_file": str(path),
    }


def build_gnss_roti_station_cache_from_rinex() -> pd.DataFrame:
    rinex_dir = next((path for path in GNSS_ROTI_RINEX_DIR_CANDIDATES if path.exists()), None)
    if rinex_dir is None:
        return pd.DataFrame(columns=["station", "name", "lat", "lon", "height_m", "source_type", "days", "rows"])
    rows_by_station: dict[str, dict[str, object]] = {}
    for path in sorted(rinex_dir.glob("*.25o")):
        row = parse_rinex_station_header(path)
        if row is None:
            continue
        rows_by_station.setdefault(str(row["station"]).lower(), row)
    stations = pd.DataFrame(rows_by_station.values())
    if stations.empty:
        return pd.DataFrame(columns=["station", "name", "lat", "lon", "height_m", "source_type", "days", "rows"])

    summary_path = next((path for path in GNSS_ROTI_SUMMARY_CANDIDATES if path.exists()), None)
    if summary_path is not None:
        summary = pd.read_csv(summary_path, low_memory=False)
        summary["station_key"] = summary["station"].astype(str).str.lower()
        stations["station_key"] = stations["station"].astype(str).str.lower()
        keep = [col for col in ["station_key", "days", "rows"] if col in summary.columns]
        stations = stations.merge(summary[keep], on="station_key", how="left").drop(columns=["station_key"])
    for col in ["lat", "lon", "height_m", "days", "rows"]:
        if col in stations.columns:
            stations[col] = pd.to_numeric(stations[col], errors="coerce")
    stations = stations.dropna(subset=["lat", "lon"]).sort_values("station").reset_index(drop=True)
    OUT_GNSS_ROTI_CSV.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(OUT_GNSS_ROTI_CSV, index=False, encoding="utf-8-sig")
    return stations


def load_ground_roti_stations(old) -> pd.DataFrame:
    if OUT_GNSS_ROTI_CSV.exists():
        data = pd.read_csv(OUT_GNSS_ROTI_CSV, low_memory=False)
        data["lat"] = pd.to_numeric(data["lat"], errors="coerce")
        data["lon"] = old.normalize_lon(data["lon"])
        if "source_type" not in data.columns:
            data["source_type"] = "GNSS/ROTI station"
        return data.dropna(subset=["lat", "lon"]).copy()
    try:
        data = old.load_gnss_roti_stations().dropna(subset=["lat", "lon"])
        data.to_csv(OUT_GNSS_ROTI_CSV, index=False, encoding="utf-8-sig")
        return data
    except FileNotFoundError:
        return build_gnss_roti_station_cache_from_rinex()


def _pil_font(size: int, weight: str = "normal") -> ImageFont.FreeTypeFont:
    path = font_manager.findfont(FontProperties(family="DejaVu Sans", weight=weight))
    return ImageFont.truetype(path, size)


def _fit_image(image: Image.Image, width: int | None = None, height: int | None = None) -> Image.Image:
    if width is None and height is None:
        return image.copy()
    if width is None:
        width = round(image.width * float(height) / float(image.height))
    if height is None:
        height = round(image.height * float(width) / float(image.width))
    return image.resize((int(width), int(height)), Image.Resampling.LANCZOS)


def draw_compact_legend(draw: ImageDraw.ImageDraw, x0: int, y0: int) -> None:
    font = _pil_font(36)
    font_star = _pil_font(48, weight="bold")
    legend = [
        ("circle", (142, 121, 185), "GNSS/ROTI (220)"),
        ("square", (89, 161, 79), "ISMR/CHAIN (39)"),
        ("star", (80, 151, 190), "ISR (9)"),
        ("diamond", (225, 87, 89), "SuperDARN (12)"),
        ("line", (31, 41, 55), "dipole MLAT"),
        ("line", (17, 24, 39), "Luna tracks by satellite"),
        ("line", (15, 76, 154), "LEO-RO tracks by mission"),
        ("patch", (107, 174, 214), "ISR nominal range"),
        ("patch", (225, 87, 89), "SuperDARN nominal fan"),
    ]
    x, y = x0, y0
    for index, (kind, color, text) in enumerate(legend):
        if index == 5:
            x, y = x0, y0 + 55
        if kind == "circle":
            draw.ellipse([x, y + 8, x + 28, y + 36], fill=color, outline="white", width=2)
        elif kind == "square":
            draw.rectangle([x, y + 9, x + 29, y + 37], fill=color, outline="white", width=2)
        elif kind == "diamond":
            draw.polygon([(x + 16, y + 6), (x + 32, y + 22), (x + 16, y + 38), (x, y + 22)], fill=color)
        elif kind == "star":
            draw.text((x, y - 6), "*", font=font_star, fill=color)
        elif kind == "line":
            draw.line([x, y + 22, x + 48, y + 22], fill=color, width=7)
        elif kind == "patch":
            fill = tuple(int(0.18 * channel + 0.82 * 255) for channel in color)
            draw.rectangle([x, y + 9, x + 40, y + 34], fill=fill, outline=color, width=2)
        draw.text((x + 60, y + 3), text, font=font, fill=(35, 35, 35))
        x += int(draw.textlength(text, font=font)) + 125


def build_nature_candidate_from_render(source_png: Path = OUT_PNG) -> tuple[Path, Path]:
    if not source_png.exists():
        raise FileNotFoundError(f"Missing source render: {source_png}")
    source = Image.open(source_png).convert("RGB")
    crops = {
        "a": source.crop((55, 250, 1515, 1668)),
        "b": source.crop((1600, 250, 3095, 1668)),
        "c": source.crop((780, 1760, 2880, 2835)),
        "d": source.crop((3525, 390, 5480, 1475)),
        "e": source.crop((3525, 1515, 5480, 2675)),
    }
    for label, crop in crops.items():
        crop_draw = ImageDraw.Draw(crop)
        if label in {"d", "e"}:
            crop_draw.rectangle([0, 0, 360, 150], fill="white")
        elif label in {"a", "b"}:
            crop_draw.rectangle([0, 0, 160, 115], fill="white")

    canvas = Image.new("RGB", (5200, 3220), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = _pil_font(70, weight="bold")
    title_font = _pil_font(46)

    positions = {
        "a": (70, 245, 1330, None),
        "b": (1500, 245, 1330, None),
        "c": (170, 1705, 2760, None),
        "d": (3140, 280, 1990, None),
        "e": (3140, 1635, 1990, None),
    }
    for label, (x, y, width, height) in positions.items():
        panel = _fit_image(crops[label], width=width, height=height)
        canvas.paste(panel, (x, y))
        draw.text((x - 45, y + 20), label, font=label_font, fill=(28, 28, 28))

    title = "Global station distribution"
    title_width = draw.textlength(title, font=title_font)
    draw.text((170 + 2760 / 2 - title_width / 2, 1660), title, font=title_font, fill=(35, 35, 35))
    draw_compact_legend(draw, 360, 65)

    OUT_NATURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_NATURE_PNG, dpi=(600, 600))
    if OUT_NATURE_PDF is not None:
        canvas.save(OUT_NATURE_PDF, "PDF", resolution=600.0)
    return OUT_NATURE_PNG, OUT_NATURE_PDF


def station_count_line(label: str, counts: dict[str, int]) -> str:
    return (
        f"{label}: "
        f"GNSS/ROTI {counts.get('GNSS/ROTI station', 0)}, "
        f"ISMR/CHAIN {counts.get('CHAIN/ISMR station', 0)}, "
        f"ISR {counts.get('ISR radar', 0)}, "
        f"SuperDARN {counts.get('Coherent radar', 0)}"
    )


def add_external_title(fig: plt.Figure, ax: plt.Axes, text: str, dy: float = 0.014) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 + pos.width / 2.0,
        pos.y1 + dy,
        text,
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#222222",
    )


def add_external_note(fig: plt.Figure, ax: plt.Axes, text: str, dy: float = 0.018, fontsize: float = 5.4) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0,
        pos.y0 - dy,
        text,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="#5f6670",
    )


def add_external_center_note(
    fig: plt.Figure,
    ax: plt.Axes,
    text: str,
    dy: float = 0.014,
    fontsize: float = 5.2,
    color: str = "#5f6670",
) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 + pos.width / 2.0,
        pos.y0 - dy,
        text,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=color,
    )


def add_side_label(fig: plt.Figure, ax: plt.Axes, text: str, dx: float = 0.018) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 - dx,
        pos.y0 + pos.height / 2.0,
        text,
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.2,
        color="#222222",
    )


def add_panel_label_fig(fig: plt.Figure, ax: plt.Axes, label: str, row_y: float | None = None, dx: float = 0.020) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 - dx,
        row_y if row_y is not None else pos.y1 + 0.012,
        label,
        ha="left",
        va="top",
        fontsize=10.5,
        weight="bold",
        color="#202020",
    )


def mission_count_line(occultation_tracks: pd.DataFrame) -> str:
    summary = occultation_tracks.attrs.get("summary")
    if not isinstance(summary, pd.DataFrame) or not {"source", "events_total"}.issubset(summary.columns):
        return f"Total {occultation_track_count(occultation_tracks):,}; mean {occultation_track_count(occultation_tracks):,} events d$^{{-1}}$"
    total = int(pd.to_numeric(summary["events_total"], errors="coerce").fillna(0).sum())
    return f"Total {total:,}; mean {total:,} events d$^{{-1}}$"


def build_nature_redraw() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old = load_old_module()
    old.POLAR_LAT_LIMIT = POLAR_LAT_LIMIT
    old.GROUND_MARKER_SIZE = 8.5
    old.COHERENT_MARKER_SIZE = 24
    old.ISR_MARKER_SIZE = 58
    old.COLORS["isr"] = "#1d4ed8"
    old.COVERAGE_ALPHA["isr_fill"] = 0.12

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.linewidth": 0.45,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    ground_roti = load_ground_roti_stations(old)
    ismr = load_updated_ismr_stations(old)
    isr = old.load_isr_stations().dropna(subset=["lat", "lon"])
    coherent = load_current_superdarn_radars(old)
    luna_tracks = load_luna_tracks()
    occultation_tracks = load_occultation_tracks()

    fig = plt.figure(figsize=(10.8, 7.35), dpi=450, facecolor="white")
    ax_north = fig.add_axes(
        [0.055, 0.565, 0.260, 0.382],
        projection=ccrs.NorthPolarStereo(central_longitude=POLAR_CENTRAL_LONGITUDE),
    )
    ax_south = fig.add_axes(
        [0.370, 0.565, 0.260, 0.382],
        projection=ccrs.SouthPolarStereo(central_longitude=POLAR_CENTRAL_LONGITUDE),
    )
    ax_orbit = fig.add_axes([0.705, 0.565, 0.245, 0.342])

    global_box_y = 0.195
    global_box_h = 0.285
    global_box_w = 0.300
    ax_global = fig.add_axes([0.045, global_box_y, global_box_w, global_box_h], projection=ccrs.Robinson(central_longitude=0))
    ax_luna = fig.add_axes([0.355, global_box_y, global_box_w, global_box_h], projection=ccrs.Robinson(central_longitude=0))
    ax_occultation = fig.add_axes([0.665, global_box_y, global_box_w, global_box_h], projection=ccrs.Robinson(central_longitude=0))

    north_counts = draw_polar_panel(old, ax_north, "north", ground_roti, ismr, isr, coherent)
    south_counts = draw_polar_panel(old, ax_south, "south", ground_roti, ismr, isr, coherent)
    orbit_counts = draw_ro_orbit_height_panel(ax_orbit)
    global_counts = draw_global_panel_nature(old, ax_global, ground_roti, ismr, isr, coherent)
    luna_counts = draw_luna_track_panel_nature(old, ax_luna, luna_tracks)
    occultation_counts = draw_occultation_track_panel_nature(old, ax_occultation, occultation_tracks)

    top_label_y = 0.952
    bottom_label_y = 0.502
    add_panel_label_fig(fig, ax_north, "a", row_y=top_label_y)
    add_panel_label_fig(fig, ax_south, "b", row_y=top_label_y)
    add_panel_label_fig(fig, ax_orbit, "c", row_y=top_label_y)
    add_panel_label_fig(fig, ax_global, "d", row_y=bottom_label_y, dx=0.010)
    add_panel_label_fig(fig, ax_luna, "e", row_y=bottom_label_y, dx=0.005)
    add_panel_label_fig(fig, ax_occultation, "f", row_y=bottom_label_y, dx=-0.020)

    add_side_label(fig, ax_north, "North polar", dx=0.026)
    add_side_label(fig, ax_south, "South polar", dx=0.026)
    add_external_title(fig, ax_orbit, "Ionospheric layers and RO orbit heights", dy=0.014)
    add_external_title(fig, ax_global, "Global station distribution", dy=0.010)
    add_external_title(fig, ax_luna, "LuGRE tangent-point tracks", dy=0.010)
    add_external_title(fig, ax_occultation, "Daily LEO-RO occultation tracks", dy=0.010)

    add_station_legend(ax_north, old, north_counts, loc="upper center", anchor=(0.500, -0.064), ncol=3, include_context=True, fontsize=4.35)
    add_station_legend(ax_south, old, south_counts, loc="upper center", anchor=(0.500, -0.064), ncol=4, include_context=True, fontsize=4.05)
    ax_global.legend(
        handles=station_legend_handles(old, global_counts, include_context=False),
        loc="upper center",
        bbox_to_anchor=(0.50, -0.030),
        frameon=False,
        fontsize=4.7,
        ncol=4,
        handlelength=1.15,
        handletextpad=0.30,
        columnspacing=0.72,
        borderaxespad=0.0,
    )
    add_external_center_note(
        fig,
        ax_luna,
        f"{luna_counts['Luna tangent tracks']:,} tracks; {luna_counts['Luna satellites']:,} satellites; tangent height 50-1000 km",
        dy=0.012,
        fontsize=4.8,
        color="#222222",
    )
    ax_occultation.legend(
        handles=mission_legend_handles(occultation_tracks),
        loc="upper center",
        bbox_to_anchor=(0.50, -0.030),
        frameon=False,
        fontsize=4.25,
        ncol=4,
        handlelength=0.95,
        handletextpad=0.24,
        columnspacing=0.45,
        borderaxespad=0.0,
    )

    OUT_TRACK_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"panel": "a", "source": "North polar stations", **north_counts},
            {"panel": "b", "source": "South polar stations", **south_counts},
            {"panel": "c", "source": "Ionospheric layers and RO orbit heights", **orbit_counts},
            {"panel": "d", "source": "Global station distribution", **global_counts},
            {"panel": "e", "source": "Luna tangent-point tracks", **luna_counts},
            {"panel": "f", "source": "Daily LEO-RO occultation tracks by mission", **{k: v for k, v in occultation_counts.items() if k != "summary"}},
        ]
    ).to_csv(OUT_TRACK_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    fig.savefig(OUT_NATURE_PNG, dpi=600, bbox_inches="tight", pad_inches=0.025)
    if OUT_NATURE_PDF is not None:
        fig.savefig(OUT_NATURE_PDF, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return OUT_NATURE_PNG, OUT_NATURE_PDF


def build_original_figure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old = load_old_module()
    old.POLAR_LAT_LIMIT = POLAR_LAT_LIMIT
    old.GROUND_MARKER_SIZE = 11
    old.COHERENT_MARKER_SIZE = 36
    old.ISR_MARKER_SIZE = 90

    ground_roti = load_ground_roti_stations(old)
    ismr = load_updated_ismr_stations(old)
    isr = old.load_isr_stations().dropna(subset=["lat", "lon"])
    coherent = load_current_superdarn_radars(old)
    luna_tracks = load_luna_tracks()
    occultation_tracks = load_occultation_tracks()

    ismr.to_csv(OUT_ISMR_CSV, index=False, encoding="utf-8-sig")
    coherent.to_csv(OUT_SUPERDARN_CSV, index=False, encoding="utf-8-sig")
    OUT_TRACK_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"sat": sat, "color": color} for sat, color in luna_satellite_color_map(luna_tracks).items()]
    ).to_csv(OUT_LUNA_SAT_COLOR_CSV, index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(14.2, 7.4), dpi=420, facecolor="white")
    ax_north = fig.add_axes(
        [0.065, 0.455, 0.210, 0.405],
        projection=ccrs.NorthPolarStereo(central_longitude=POLAR_CENTRAL_LONGITUDE),
    )
    ax_south = fig.add_axes(
        [0.335, 0.455, 0.210, 0.405],
        projection=ccrs.SouthPolarStereo(central_longitude=POLAR_CENTRAL_LONGITUDE),
    )
    ax_global = fig.add_axes([0.045, 0.110, 0.580, 0.305], projection=ccrs.Robinson(central_longitude=0))
    ax_luna = fig.add_axes([0.675, 0.525, 0.285, 0.310], projection=ccrs.Robinson(central_longitude=0))
    ax_occultation = fig.add_axes([0.675, 0.170, 0.285, 0.310], projection=ccrs.Robinson(central_longitude=0))

    north_counts = draw_polar_panel(old, ax_north, "north", ground_roti, ismr, isr, coherent)
    south_counts = draw_polar_panel(old, ax_south, "south", ground_roti, ismr, isr, coherent)
    global_counts = draw_global_panel(old, ax_global, ground_roti, ismr, isr, coherent)
    luna_counts = draw_luna_track_panel(old, ax_luna, luna_tracks)
    occultation_counts = draw_occultation_track_panel(old, ax_occultation, occultation_tracks)

    add_panel_label(ax_global, "c", x=-0.045, y=1.08)
    add_panel_label(ax_luna, "d", x=-0.055, y=1.08)
    add_panel_label(ax_occultation, "e", x=-0.055, y=1.08)

    pd.DataFrame(
        [
            {"panel": "d", "source": "Luna tangent-point tracks", **luna_counts},
            {"panel": "e", "source": "OP74 LEO-RO tracks by mission", **occultation_counts},
        ]
    ).to_csv(OUT_TRACK_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    fig.legend(
        handles=build_legend(old, global_counts),
        loc="lower center",
        ncol=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.040),
        fontsize=5.9,
        handlelength=1.0,
        columnspacing=0.82,
    )

    fig.text(
        0.5,
        0.965,
        "Station coverage with Luna and LEO-RO trajectory context",
        ha="center",
        va="top",
        fontsize=10.2,
        weight="bold",
        color=old.COLORS["text"],
    )
    fig.text(
        0.5,
        0.928,
        "Left: ground/radar source distribution with 60-degree polar zooms; right: Luna tangent points and OP74 mission-colored LEO-RO tracks",
        ha="center",
        va="top",
        fontsize=5.8,
        color=old.COLORS["muted"],
    )

    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_ISMR_CSV}")
    print(f"wrote {OUT_SUPERDARN_CSV}")
    print(f"wrote {OUT_TRACK_SUMMARY_CSV}")
    print(f"wrote {OUT_LUNA_SAT_COLOR_CSV}")
    print("global counts:", global_counts)
    print("north counts:", north_counts)
    print("south counts:", south_counts)
    print("luna counts:", luna_counts)
    print("occultation counts:", occultation_counts)


def main() -> None:
    if "--rebuild-source" in sys.argv:
        build_original_figure()
    if "--from-render" in sys.argv:
        png, pdf = build_nature_candidate_from_render()
    else:
        png, pdf = build_nature_redraw()
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
