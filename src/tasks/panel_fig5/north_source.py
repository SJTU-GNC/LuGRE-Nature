from __future__ import annotations

import json
import math
import re
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "panel_ready" / "Fig5"
OUT_DIR = ROOT / "work" / "panel_fig5"

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy
import cartopy.feature as cfeature
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

cartopy.config["data_dir"] = str(ROOT / "src" / "tasks" / "ed2" / "cartopy_data")

PKG = DATA_ROOT / "north"
IRI_DIR = OUT_DIR / "iri_ne_profiles_csv_20260623"
PROFILE_CSV = IRI_DIR / "iri2016_extended_50_1000km_profiles_track145_187.csv"
ISR_MATCHED_IRI_CSV = None

PNG_OUT = OUT_DIR / "fig4_north_pair_nature_layout_candidate_v43_v40_layout_fig7_basemap.png"
PDF_OUT = OUT_DIR / "fig4_north_pair_nature_layout_candidate_v43_v40_layout_fig7_basemap.pdf"
QA_OUT = OUT_DIR / "fig4_north_pair_v43_v40_layout_fig7_basemap_QA.csv"
README_OUT = OUT_DIR / "README_fig4_north_pair_v43_v40_layout_fig7_basemap.md"
ISR_ZIP = OUT_DIR.parent / "noncoherent_isr_used_data_track145_187_20260624.zip"
ISR_OBS_MEMBER = "figure_used_context/exact_eiscat_tromso_uhf_profile_20250315_035900UTC/EISCAT_Tromso_UHF_observed_profile_20250315_035900UTC.csv"
ISR_CONTEXT_MEMBERS = {
    145: "figure_used_context/data/extracted_context/track_145/nearest_isr_profile_context.json",
    187: "figure_used_context/data/extracted_context/track_187/nearest_isr_profile_context.json",
}
SPIRE_R18_RAW_CONPHS = DATA_ROOT / "external_not_packaged" / "spire_track187_conPhs.nc"
SPIRE_R18_DUAL_SOURCE_OUT = OUT_DIR / "fig4_track187_spire_R18_L1L2_cn0_residual_source_v22.csv"
ISR_SAME_TIME_PROFILE_145 = OUT_DIR / "isr_ctx_cache" / "track145_eiscat_tromso_uhf_same_window_ne_profile.csv"

MAP_TITLES = {
    145: "A  OP76 GPS G01 | north polar night | 2025-03-15 16:24-16:29 UTC",
    187: "F  OP76 GPS G26 | north polar night | 2025-03-16 13:38-13:46 UTC",
}

LYR_SITE_LAT = 78.15338
LYR_SITE_LON = 16.07342

INK = "#111827"
MUTED = "#334155"
GRID = "#D9E2EC"
AXIS = "#111827"
SPINE = "#334155"
LUGRE = "#26364C"
LEORO = "#B45309"
CELL_RED = "#EF5C73"
SITE_RED = "#D84A4A"
GNSS_SCINT = "#43A047"
OBS_BLUE = "#0F6FAE"
IRI_ORANGE = "#D97706"
MATCHED_GREY = "#7C8798"

ROTI_CMAP = LinearSegmentedColormap.from_list(
    "roti_lighter_context",
    ["#C8DAEA", "#97C9D8", "#93D2BF", "#D9E59B", "#F2C17D", "#E79A8F"],
)
ROTI_CMAP.set_bad((1, 1, 1, 0))
SD_CMAP = LinearSegmentedColormap.from_list(
    "superdarn_power_original",
    ["#30123B", "#3154D8", "#21A2B5", "#A0DA39", "#F8E621", "#F46D43", "#B40426"],
)
ISMR_CMAP = SD_CMAP


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_csv_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as f:
                return pd.read_csv(f)
    except (KeyError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def read_json_from_zip(zip_path: Path, member: str) -> dict:
    if not zip_path.exists():
        return {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return json.loads(zf.read(member).decode("utf-8"))
    except (KeyError, json.JSONDecodeError):
        return {}


def load_isr_context_profile(track_index: int) -> tuple[pd.DataFrame, dict]:
    obs = read_csv_from_zip(ISR_ZIP, ISR_OBS_MEMBER)
    context = read_json_from_zip(ISR_ZIP, ISR_CONTEXT_MEMBERS.get(track_index, ""))
    if obs.empty:
        return obs, context
    for col in ["alt_bin_km", "ne_m3", "log10_ne"]:
        if col in obs.columns:
            obs[col] = pd.to_numeric(obs[col], errors="coerce")
    obs = obs.dropna(subset=["alt_bin_km", "log10_ne"])
    if "ne_m3" in obs.columns:
        obs = obs[obs["ne_m3"].gt(0)].copy()
    obs = obs[(obs["alt_bin_km"].ge(50)) & (obs["alt_bin_km"].le(1000))].copy()
    return obs.sort_values("alt_bin_km"), context


def load_case(track_index: int) -> dict:
    d = PKG / f"track_{track_index}"
    with (d / "superdarn_match_row.json").open("r", encoding="utf-8") as f:
        sd = json.load(f)
    with (d / "ro_focus_used.json").open("r", encoding="utf-8") as f:
        ro = json.load(f)
    isr_context_path = d / "nearest_isr_profile_context.json"
    isr_context = json.loads(isr_context_path.read_text(encoding="utf-8")) if isr_context_path.exists() else {}
    return {
        "track_index": track_index,
        "track": read_csv(d / "track_points.csv"),
        "sd": sd,
        "ro": ro,
        "fan": read_csv(d / "superdarn_fan_cells.csv"),
        "roti": read_csv(d / "roti_context_field.csv"),
        "ismr": read_csv(d / "ismr_context_field.csv"),
        "ground_roti_match": read_csv(d / "ground_roti_matches.csv"),
        "isr_context": isr_context,
    }


def polar_xy(lat, lon, lat_edge: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    r = (90.0 - lat) / (90.0 - lat_edge)
    th = np.deg2rad(lon)
    return r * np.cos(th), r * np.sin(th)


def polar_points_to_lonlat(points, *, plot_xy: bool = True) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
        return np.array([]), np.array([])
    x = arr[:, 0]
    y = arr[:, 1]
    radius = np.hypot(x, y)
    lat = 90.0 - radius
    ref_y = -y if plot_xy else y
    lon = np.rad2deg(np.arctan2(x, ref_y))
    lon = ((lon + 180.0) % 360.0) - 180.0
    ok = np.isfinite(lat) & np.isfinite(lon)
    return lat[ok], lon[ok]


def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    earth_radius_km = 6371.0
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    d = distance_km / earth_radius_km
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), ((math.degrees(lon2) + 180.0) % 360.0) - 180.0


def bearing_distance_from_station(lat_deg, lon_deg, station_lat=LYR_SITE_LAT, station_lon=LYR_SITE_LON) -> tuple[np.ndarray, np.ndarray]:
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat1 = math.radians(station_lat)
    lon1 = math.radians(station_lon)
    dlon = lon - lon1
    y = np.sin(dlon) * np.cos(lat)
    x = np.cos(lat1) * np.sin(lat) - np.sin(lat1) * np.cos(lat) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    a = np.sin((lat - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    distance = 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return bearing, distance


def scatter_polar(ax, lat, lon, **kwargs):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat >= 60.0)
    if ok.sum() == 0:
        return None
    x, y = polar_xy(lat[ok], lon[ok])
    return ax.scatter(x, y, **kwargs)


def plot_polar_line(ax, lat, lon, **kwargs) -> None:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat >= 60.0)
    if ok.sum() < 2:
        return
    lat = lat[ok]
    lon = lon[ok]
    breaks = np.flatnonzero(np.abs(np.diff(lon)) > 150.0) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(lon)]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        x, y = polar_xy(lat[s:e], lon[s:e])
        ax.plot(x, y, **kwargs)


def draw_height_layers(ax) -> None:
    ax.axhspan(50, 90, facecolor="#FFF4C7", edgecolor="none", alpha=0.30, zorder=0)
    ax.axhspan(90, 150, facecolor="#FFE4DB", edgecolor="none", alpha=0.28, zorder=0)
    ax.axhspan(150, 600, facecolor="#EAF7ED", edgecolor="none", alpha=0.36, zorder=0)
    ax.axhspan(600, 1000, facecolor="#E9F3FB", edgecolor="none", alpha=0.40, zorder=0)


def make_axis_text_bold(ax) -> None:
    ax.xaxis.label.set_color(AXIS)
    ax.yaxis.label.set_color(AXIS)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_color(INK)
    ax.title.set_fontweight("bold")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color(AXIS)
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.86)
        spine.set_visible(True)


def set_height_axis(ax, *, label: str = "Tangent height [km]", side: str = "left") -> None:
    ax.set_yscale("log")
    ax.set_ylim(50, 1000)
    ticks = [50, 90, 150, 300, 600, 1000]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(label, fontsize=7.0, color=AXIS, fontweight="bold")
    if side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", labelleft=False, labelright=True)
    else:
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        ax.tick_params(axis="y", labelleft=True, labelright=False)
    ax.grid(True, which="major", color=GRID, lw=0.50, alpha=0.72)
    ax.tick_params(labelsize=6.5, colors=AXIS, width=0.72, length=2.8)
    make_axis_text_bold(ax)


def draw_polar_frame(ax) -> None:
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(-1.055, 1.055)
    ax.set_ylim(-1.055, 1.055)
    ax.axis("off")
    lat_label_lon = 67.5
    for lat, lw, alpha in [(60, 0.95, 0.96), (70, 0.62, 0.84), (80, 0.62, 0.84)]:
        r = (90.0 - lat) / 30.0
        ax.add_patch(Circle((0, 0), r, fill=False, ec=GRID, lw=lw, alpha=alpha, zorder=3))
        tx, ty = polar_xy([lat], [lat_label_lon])
        ax.text(tx[0] * 1.018, ty[0] * 1.018, f"{lat}°N", fontsize=6.1, color=MUTED, fontweight="bold", ha="center", va="center", zorder=40)
    ax.text(0.018, 0.018, "90°N", fontsize=5.9, color=MUTED, fontweight="bold", alpha=0.85, zorder=40)
    for lon in range(-180, 181, 45):
        x, y = polar_xy([60], [lon])
        ax.plot([0, x[0]], [0, y[0]], color=GRID, lw=0.45, alpha=0.62, zorder=3)
    for lon, label in [(0, "0°"), (45, "45°E"), (90, "90°E"), (135, "135°E"), (180, "180°"), (-135, "135°W"), (-90, "90°W"), (-45, "45°W")]:
        x, y = polar_xy([59.3], [lon])
        ax.text(x[0], y[0], label, fontsize=6.1, color=MUTED, fontweight="bold", ha="center", va="center", zorder=40)


def _iter_geometry_lonlat_lines(geom):
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "LineString":
        arr = np.asarray(geom.coords, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            yield arr[:, 1], arr[:, 0]
    elif geom_type == "MultiLineString":
        for sub in geom.geoms:
            yield from _iter_geometry_lonlat_lines(sub)
    elif geom_type == "Polygon":
        arr = np.asarray(geom.exterior.coords, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            yield arr[:, 1], arr[:, 0]
    elif geom_type == "MultiPolygon":
        for sub in geom.geoms:
            yield from _iter_geometry_lonlat_lines(sub)


def draw_arctic_basemap(ax) -> int:
    """Draw Fig.7-style Natural Earth coastlines/borders in the v40 polar layout."""
    count = 0
    for feature, color, lw, alpha, zorder in [
        (cfeature.COASTLINE.with_scale("110m"), "#8C96A3", 0.36, 0.72, 2.1),
        (cfeature.BORDERS.with_scale("110m"), "#C9CED6", 0.16, 0.58, 2.0),
    ]:
        for geom in feature.geometries():
            for lat, lon in _iter_geometry_lonlat_lines(geom):
                ok = np.isfinite(lat) & np.isfinite(lon) & (lat >= 60.0)
                if ok.sum() < 2:
                    continue
                idx = np.flatnonzero(ok)
                breaks = np.where(np.diff(idx) > 1)[0] + 1
                for segment in np.split(idx, breaks):
                    if len(segment) < 2:
                        continue
                    plot_polar_line(ax, lat[segment], lon[segment], color=color, lw=lw, alpha=alpha, zorder=zorder)
                    count += 1
    return count


def draw_roti_background(ax, roti: pd.DataFrame):
    if roti.empty:
        return None
    r = roti.copy()
    for col in ["lat_center", "lon_center", "roti5_p95", "n", "stations"]:
        r[col] = pd.to_numeric(r[col], errors="coerce")
    r = r.dropna(subset=["lat_center", "lon_center", "roti5_p95"])
    r = r[r["lat_center"].ge(60.0)]
    total_cells = len(r)
    r = r[np.isfinite(r["roti5_p95"])]
    if r.empty:
        return None
    px, py = polar_xy(r["lat_center"], r["lon_center"])
    points = np.column_stack([px, py])
    values = r["roti5_p95"].to_numpy(dtype=float)
    grid_n = 420
    gx = np.linspace(-1.0, 1.0, grid_n)
    gy = np.linspace(-1.0, 1.0, grid_n)
    X, Y = np.meshgrid(gx, gy)
    R = np.hypot(X, Y)
    linear = griddata(points, values, (X, Y), method="linear")
    nearest = griddata(points, values, (X, Y), method="nearest")
    tree = cKDTree(points)
    nearest_dist, _ = tree.query(np.column_stack([X.ravel(), Y.ravel()]), k=1)
    nearest_dist = nearest_dist.reshape(X.shape)
    coverage_radius = 0.044
    coverage = (nearest_dist <= coverage_radius) & (R <= 1.0)
    Z = np.where(np.isfinite(linear), linear, nearest)
    Z = np.where(coverage, Z, np.nan)
    Z = np.clip(Z, 0, 4)
    im = ax.imshow(
        np.ma.masked_invalid(Z),
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap=ROTI_CMAP,
        norm=PowerNorm(gamma=1.25, vmin=0, vmax=4),
        alpha=0.82,
        interpolation="bilinear",
        zorder=0,
    )
    im.set_clip_path(Circle((0, 0), 1.0, transform=ax.transData))
    im.roti_total_cells = int(total_cells)
    im.roti_valid_cells = int(len(r))
    im.roti_mask_radius_polar_units = coverage_radius
    im.roti_colored_grid_fraction = float(np.isfinite(Z).sum() / (R <= 1.0).sum())
    return im


def draw_los_paths(ax, paths, color: str, center_color: str, max_paths: int, alpha: float, lw: float) -> None:
    paths = paths or []
    if not paths:
        return
    step = max(1, int(math.ceil(len(paths) / max_paths)))
    centers_lat = []
    centers_lon = []
    for path in paths[::step]:
        lat, lon = polar_points_to_lonlat(path, plot_xy=True)
        plot_polar_line(ax, lat, lon, color=color, lw=max(0.48, lw * 0.30), alpha=alpha, zorder=12)
    for path in paths:
        arr = np.asarray(path, dtype=float)
        if arr.ndim == 2 and len(arr):
            mid = arr[len(arr) // 2, :2]
            lat, lon = polar_points_to_lonlat([mid], plot_xy=True)
            if len(lat):
                centers_lat.append(lat[0])
                centers_lon.append(lon[0])
    if len(centers_lat) >= 2:
        plot_polar_line(ax, centers_lat, centers_lon, color=center_color, lw=lw, alpha=0.96, zorder=22)


def map_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=LUGRE, lw=1.8, label="LuGRE TP/LOS"),
        Line2D([0], [0], color=LEORO, lw=1.5, label="LEO-RO LOS"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#F0C64D", markeredgecolor="none", markersize=5.0, label="SuperDARN power"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#2563A6", markeredgecolor="white", markersize=7.0, label="ISR context"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=SITE_RED, markeredgecolor="white", markersize=5.0, label="SuperDARN radar site"),
    ]


def draw_map(ax, fig, case: dict, label: str) -> dict:
    draw_polar_frame(ax)
    basemap_polygons = draw_arctic_basemap(ax)
    ax.text(0.0, 1.085, MAP_TITLES.get(case["track_index"], f"{label}  Map + SuperDARN"), fontsize=8.1, weight="bold", color=INK, ha="center", va="bottom")

    roti_im = draw_roti_background(ax, case["roti"])

    ismr = case["ismr"].copy()
    ismr_scatter = None
    ismr_points_total = 0
    if not ismr.empty:
        for col in ["ipp_lat", "ipp_lon", "s4", "station_lat", "station_lon"]:
            ismr[col] = pd.to_numeric(ismr[col], errors="coerce")
        ismr = ismr.dropna(subset=["ipp_lat", "ipp_lon", "s4"])
        ismr = ismr[ismr["ipp_lat"].ge(60.0)]
        ismr_points_total = len(ismr)
        ismr_scatter = scatter_polar(
            ax,
            ismr["ipp_lat"],
            ismr["ipp_lon"],
            c=ismr["s4"],
            cmap=ISMR_CMAP,
            norm=Normalize(0, 0.5),
            s=3.4,
            alpha=0.82,
            linewidths=0,
            zorder=8,
            rasterized=True,
        )

    fan = case["fan"].copy()
    fan_plot = pd.DataFrame()
    sd_scatter = None
    if not fan.empty:
        for col in ["bmazm_deg", "range_center_km", "mean_power_db"]:
            fan[col] = pd.to_numeric(fan[col], errors="coerce")
        fan = fan.dropna(subset=["bmazm_deg", "range_center_km", "mean_power_db"])
        fan_plot = fan[fan["mean_power_db"].gt(0)].copy()
        if len(fan_plot) > 900:
            fan_plot = fan_plot.sort_values(["bmazm_deg", "range_center_km"])
            keep_idx = np.linspace(0, len(fan_plot) - 1, 900).round().astype(int)
            fan_plot = fan_plot.iloc[np.unique(keep_idx)].copy()
        for beam, group in fan_plot.groupby("bmazm_deg"):
            group = group.sort_values("range_center_km")
            if len(group) < 2:
                continue
            latlon = [destination_point(LYR_SITE_LAT, LYR_SITE_LON, float(beam), float(r)) for r in group["range_center_km"].to_numpy()]
            lat, lon = np.array(latlon).T
            plot_polar_line(ax, lat, lon, color="#C94C4C", lw=0.55, alpha=0.20, zorder=5)
        latlon = [destination_point(LYR_SITE_LAT, LYR_SITE_LON, float(b), float(r)) for b, r in zip(fan_plot["bmazm_deg"], fan_plot["range_center_km"])]
        if latlon:
            flat, flon = np.array(latlon).T
            sd_scatter = scatter_polar(
                ax,
                flat,
                flon,
                c=fan_plot["mean_power_db"],
                cmap=SD_CMAP,
                norm=Normalize(0, 12),
                marker="s",
                s=8.5,
                alpha=1.0,
                linewidths=0,
                zorder=9,
            )

    ro = case["ro"]
    draw_los_paths(ax, ro.get("lugre_los_paths_xy") or [], "#24364D", LUGRE, 54, 0.32, 2.45)
    draw_los_paths(ax, ro.get("ro_los_paths_xy") or [], "#D97706", LEORO, 42, 0.20, 1.85)

    track = case["track"]
    plot_polar_line(ax, track["lugre_lat"], track["lugre_lon"], color=LUGRE, lw=2.05, alpha=0.97, zorder=25)
    scatter_polar(ax, [track["lugre_lat"].iloc[0]], [track["lugre_lon"].iloc[0]], s=38, color=LUGRE, edgecolors="white", linewidths=0.5, zorder=27)
    scatter_polar(ax, [track["lugre_lat"].iloc[-1]], [track["lugre_lon"].iloc[-1]], s=58, marker="X", color=LUGRE, edgecolors="white", linewidths=0.5, zorder=27)

    sd = case["sd"]
    scatter_polar(ax, [LYR_SITE_LAT], [LYR_SITE_LON], s=70, marker="D", color=SITE_RED, edgecolors="white", linewidths=0.8, zorder=31)
    sx, sy = polar_xy([LYR_SITE_LAT], [LYR_SITE_LON])
    ax.text(sx[0] + 0.030, sy[0] - 0.005, "SD LYR", color=SITE_RED, fontsize=6.5, weight="bold", zorder=32)

    scatter_polar(ax, [float(sd["nearest_track_lat"])], [float(sd["nearest_track_lon"])], s=88, marker="o", facecolors="#F8FAFC", edgecolors=LUGRE, linewidths=2.0, zorder=33)
    nx, ny = polar_xy([float(sd["nearest_track_lat"])], [float(sd["nearest_track_lon"])])
    ax.text(nx[0] + 0.025, ny[0] + 0.005, "LuGRE", color=LUGRE, fontsize=6.4, weight="bold", zorder=34)

    ro_lat, ro_lon = polar_points_to_lonlat([ro.get("ro_nearest_xy", [])], plot_xy=False)
    if len(ro_lat):
        scatter_polar(ax, ro_lat, ro_lon, s=78, marker="o", color=LEORO, edgecolors="white", linewidths=0.9, zorder=32)
        rx, ry = polar_xy(ro_lat, ro_lon)
        ax.text(rx[0] + 0.025, ry[0] - 0.025, "LEO-RO", color=LEORO, fontsize=6.4, weight="bold", zorder=34)

    isr_context = case.get("isr_context") or {}
    if isr_context:
        try:
            isr_lat = float(isr_context.get("station_lat"))
            isr_lon = float(isr_context.get("station_lon"))
            scatter_polar(ax, [isr_lat], [isr_lon], s=82, marker="*", color="#2563A6", edgecolors="white", linewidths=0.55, zorder=30)
            ix, iy = polar_xy([isr_lat], [isr_lon])
            ax.text(ix[0] - 0.032, iy[0] + 0.014, "ISR ctx", color="#2563A6", fontsize=6.2, weight="bold", ha="right", zorder=34)
        except (TypeError, ValueError):
            pass

    if roti_im is not None:
        cax = ax.inset_axes([-0.062, 0.705, 0.018, 0.190])
        cb = fig.colorbar(ScalarMappable(norm=PowerNorm(gamma=1.25, vmin=0, vmax=4), cmap=ROTI_CMAP), cax=cax, orientation="vertical", ticks=[0, 2, 4])
        cb.ax.tick_params(labelsize=4.6, length=1.4, pad=1)
        cb.set_label("ROTI (TECU/min)", fontsize=5.0, labelpad=1.5, color=AXIS, fontweight="bold")
        cb.ax.set_title("", fontsize=4.6, color=AXIS, fontweight="bold", pad=1.5)
        for tick in cb.ax.get_yticklabels():
            tick.set_color(AXIS)
            tick.set_fontweight("bold")
    if sd_scatter is not None:
        cax = ax.inset_axes([-0.062, 0.475, 0.018, 0.190])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 12), cmap=SD_CMAP), cax=cax, orientation="vertical", ticks=[0, 6, 12])
        cb.ax.tick_params(labelsize=4.6, length=1.4, pad=1)
        cb.set_label("SD power (dB)", fontsize=5.0, labelpad=1.5, color=AXIS, fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_color(AXIS)
            tick.set_fontweight("bold")
    if ismr_scatter is not None:
        cax = ax.inset_axes([-0.062, 0.245, 0.018, 0.190])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 0.5), cmap=ISMR_CMAP), cax=cax, orientation="vertical", ticks=[0.0, 0.25, 0.5])
        cb.ax.tick_params(labelsize=4.6, length=1.4, pad=1)
        cb.set_label(r"ISMR $S_4$", fontsize=5.0, labelpad=1.5, color=AXIS, fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_color(AXIS)
            tick.set_fontweight("bold")
    return {
        "track_index": case["track_index"],
        "outer_latitude_deg": 60,
        "roti_background": "masked continuous interpolation from all finite roti_context_field.csv polar grid cells; unsupported areas transparent",
        "roti_total_cells_lat_ge_60": getattr(roti_im, "roti_total_cells", 0),
        "roti_valid_cells_plotted": getattr(roti_im, "roti_valid_cells", 0),
        "roti_mask_radius_polar_units": getattr(roti_im, "roti_mask_radius_polar_units", np.nan),
        "roti_colored_grid_fraction": getattr(roti_im, "roti_colored_grid_fraction", np.nan),
        "fig7_basemap_line_segments_plotted": basemap_polygons,
        "ismr_ipp_points_plotted": int(ismr_points_total),
        "ismr_station_markers_plotted": 0,
        "superdarn_fan_cells_plotted": int(len(fan_plot)) if not fan.empty else 0,
        "map_superdarn_power_cells_restored": sd_scatter is not None,
        "lyr_site_lat": LYR_SITE_LAT,
        "lyr_site_lon": LYR_SITE_LON,
        "matched_cell_lat": sd.get("nearest_cell_lat"),
        "matched_cell_lon": sd.get("nearest_cell_lon"),
        "isr_context_station": isr_context.get("station") if isr_context else "",
        "isr_context_distance_km": isr_context.get("distance_km") if isr_context else np.nan,
        "isr_context_time_gap_hr": isr_context.get("time_gap_hr") if isr_context else np.nan,
    }


def _first_scalar(value, default=np.nan):
    if isinstance(value, (tuple, list)):
        return value[0] if value else default
    return value if value is not None else default


def read_netcdf3_classic(path: Path) -> tuple[dict, dict, dict]:
    data = path.read_bytes()
    pos = 0

    def u32() -> int:
        nonlocal pos
        value = struct.unpack(">I", data[pos : pos + 4])[0]
        pos += 4
        return value

    def read_name() -> str:
        nonlocal pos
        n_char = u32()
        raw = data[pos : pos + n_char]
        pos += n_char
        pos += (-n_char) % 4
        return raw.decode("utf-8", "replace")

    def read_values(nc_type: int, count: int):
        nonlocal pos
        type_size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}[nc_type]
        n_byte = type_size * count
        raw = data[pos : pos + n_byte]
        pos += n_byte
        pos += (-n_byte) % 4
        if nc_type == 2:
            return raw.decode("utf-8", "replace").rstrip("\x00")
        fmt = {1: "b", 3: "h", 4: "i", 5: "f", 6: "d"}[nc_type]
        values = struct.unpack(">" + fmt * count, raw[:n_byte])
        return values[0] if count == 1 else values

    def read_attrs() -> dict:
        tag = u32()
        attrs = {}
        if tag == 0:
            _ = u32()
            return attrs
        if tag != 12:
            raise ValueError(f"Unexpected NetCDF attribute tag {tag} in {path}")
        n_attr = u32()
        for _ in range(n_attr):
            attr_name = read_name()
            nc_type = u32()
            count = u32()
            attrs[attr_name] = read_values(nc_type, count)
        return attrs

    magic = data[pos : pos + 4]
    pos += 4
    if magic != b"CDF\x01":
        raise ValueError(f"{path} is not a NetCDF classic CDF-1 file")
    _ = u32()

    dims = []
    tag = u32()
    if tag == 0:
        _ = u32()
    elif tag == 10:
        n_dim = u32()
        for _ in range(n_dim):
            dims.append((read_name(), u32()))
    else:
        raise ValueError(f"Unexpected NetCDF dimension tag {tag} in {path}")

    global_attrs = read_attrs()
    variables = {}
    var_attrs = {}
    tag = u32()
    if tag == 0:
        _ = u32()
    elif tag == 11:
        n_var = u32()
        for _ in range(n_var):
            var_name = read_name()
            n_dim = u32()
            dim_ids = [u32() for _ in range(n_dim)]
            attrs = read_attrs()
            nc_type = u32()
            _ = u32()
            begin = u32()
            count = dims[dim_ids[0]][1] if dim_ids else 1
            dtype = {5: ">f4", 6: ">f8", 4: ">i4", 3: ">i2", 1: ">i1"}.get(nc_type)
            if dtype is None:
                continue
            arr = np.frombuffer(data, dtype=dtype, count=count, offset=begin).astype(float)
            variables[var_name] = arr
            var_attrs[var_name] = attrs
    else:
        raise ValueError(f"Unexpected NetCDF variable tag {tag} in {path}")
    return global_attrs, variables, var_attrs


def _conphs_time_base(global_attrs: dict, var_attrs: dict) -> pd.Timestamp:
    units = str(var_attrs.get("time", {}).get("units", ""))
    m = re.search(r"seconds since (\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{1,2}):([0-9.]+)", units)
    if m:
        date, hour, minute, second_text = m.groups()
        second_float = float(second_text)
        second = int(math.floor(second_float))
        frac = second_float - second
        base = pd.Timestamp(f"{date} {int(hour):02d}:{int(minute):02d}:{second:02d}", tz="UTC")
        return base + pd.Timedelta(seconds=frac)
    second_float = float(_first_scalar(global_attrs.get("second"), 0.0))
    second = int(math.floor(second_float))
    return pd.Timestamp(
        year=int(_first_scalar(global_attrs.get("year"))),
        month=int(_first_scalar(global_attrs.get("month"))),
        day=int(_first_scalar(global_attrs.get("day"))),
        hour=int(_first_scalar(global_attrs.get("hour"))),
        minute=int(_first_scalar(global_attrs.get("minute"))),
        second=second,
        tz="UTC",
    )


def load_spire_r18_dual_frequency_profiles() -> list[pd.DataFrame]:
    cached = getattr(load_spire_r18_dual_frequency_profiles, "_cached_frames", None)
    if cached is not None:
        return [frame.copy() for frame in cached]
    if not SPIRE_R18_RAW_CONPHS.exists():
        load_spire_r18_dual_frequency_profiles._cached_frames = []
        return []

    global_attrs, variables, var_attrs = read_netcdf3_classic(SPIRE_R18_RAW_CONPHS)
    base_time = _conphs_time_base(global_attrs, var_attrs)
    time = base_time + pd.to_timedelta(variables["time"], unit="s")
    height = variables["occheight"]
    source_rows = []
    frames = []
    signal_defs = [
        ("caL1Snr", "Spire R18 L1", "#E45756", "o", "GLONASS R18 L1C"),
        ("pL2Snr", "Spire R18 L2", "#7B61FF", "D", "GLONASS R18 L2C"),
    ]
    for var_name, label, color, marker, frequency_label in signal_defs:
        if var_name not in variables:
            continue
        snr = variables[var_name]
        raw = pd.DataFrame({"time": time, "height": height, "snr": snr})
        raw = raw[np.isfinite(raw["height"]) & np.isfinite(raw["snr"])].copy()
        raw = raw[(raw["height"].between(50, 1000)) & (raw["snr"].gt(0))].copy()
        if raw.empty:
            continue
        raw["second_utc"] = raw["time"].dt.floor("s")
        one_sec = (
            raw.groupby("second_utc", as_index=False)
            .agg(
                height=("height", "mean"),
                cn0_proxy_dbhz=("snr", lambda x: 10.0 * np.log10(np.mean(np.square(x)))),
                n_samples_snr=("snr", "size"),
            )
            .copy()
        )
        one_sec = one_sec[one_sec["n_samples_snr"].ge(5)].copy()
        if len(one_sec) < 5:
            continue
        fit = np.polynomial.Chebyshev.fit(
            one_sec["height"].to_numpy(),
            one_sec["cn0_proxy_dbhz"].to_numpy(),
            deg=min(4, len(one_sec) - 1),
        )(one_sec["height"].to_numpy())
        one_sec["cn0_fit_dbhz"] = fit
        one_sec["dcn0"] = one_sec["cn0_proxy_dbhz"] - one_sec["cn0_fit_dbhz"]
        one_sec["label"] = label
        one_sec["color"] = color
        one_sec["marker"] = marker
        one_sec["signal"] = label.rsplit(" ", 1)[-1]
        one_sec["raw_variable"] = var_name
        one_sec["frequency_label"] = frequency_label
        one_sec["received_gnss_prn"] = "R18"
        one_sec["leo_spacecraft"] = "Spire S194"
        one_sec["source_file"] = str(SPIRE_R18_RAW_CONPHS)
        source_rows.append(one_sec.copy())
        frames.append(
            one_sec.rename(columns={"second_utc": "time"})[
                ["time", "height", "dcn0", "label", "color", "marker"]
            ].copy()
        )

    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_csv(SPIRE_R18_DUAL_SOURCE_OUT, index=False, encoding="utf-8-sig")
    load_spire_r18_dual_frequency_profiles._cached_frames = [frame.copy() for frame in frames]
    return frames


def profile_to_frame(profile: dict, label: str, color: str, marker: str) -> pd.DataFrame:
    n = min(len(profile.get("time", [])), len(profile.get("height", [])), len(profile.get("detrended", [])))
    if n == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "time": pd.to_datetime(profile["time"][:n], utc=True, errors="coerce"),
            "height": pd.to_numeric(pd.Series(profile["height"][:n]), errors="coerce"),
            "dcn0": pd.to_numeric(pd.Series(profile["detrended"][:n]), errors="coerce"),
            "label": label,
            "color": color,
            "marker": marker,
        }
    ).dropna(subset=["time", "height", "dcn0"])


def lugre_profiles(case: dict) -> list[pd.DataFrame]:
    frames = []
    sat = str(case["track"]["sat"].iloc[0])
    for item in case["ro"].get("lugres", [])[:4]:
        label = str(item.get("label", "LuGRE"))
        if "L1" in label:
            frames.append(profile_to_frame(item.get("profile") or {}, f"LuGRE {sat} L1", "#2F80ED", "o"))
        elif "L5" in label:
            frames.append(profile_to_frame(item.get("profile") or {}, f"LuGRE {sat} L5", "#22A766", "^"))
    return [f for f in frames if not f.empty]


def leoro_short_label(occ: dict, source: str, label: str) -> str:
    upper_label = label.upper()
    if "METOPB" in upper_label:
        return "MetOp-B G18"
    event_text = " ".join(str(x) for x in [label, occ.get("id", ""), (occ.get("meta") or {}).get("eventId", ""), (occ.get("meta") or {}).get("trackId", "")])
    prn_match = re.search(r"(?<![A-Z0-9])([GREJC]\d{2})(?=[_.\-\s]|$)", event_text.upper())
    if "SPIRE" in source.upper() and prn_match:
        return f"Spire {prn_match.group(1)}"
    return source


def leoro_profiles(case: dict) -> list[pd.DataFrame]:
    occ = case["ro"].get("occultation") or {}
    source = str(occ.get("source") or "LEO-RO")
    label = str(occ.get("label") or source)
    short = leoro_short_label(occ, source, label)
    frames = []
    event_text = " ".join(
        str(x)
        for x in [label, occ.get("id", ""), (occ.get("meta") or {}).get("eventId", ""), (occ.get("meta") or {}).get("trackId", "")]
    )
    if case.get("track_index") == 187 and "SPIRE" in source.upper() and "R18" in event_text.upper():
        dual_frames = load_spire_r18_dual_frequency_profiles()
        if dual_frames:
            return [f for f in dual_frames if not f.empty]
    channels = occ.get("channelSignalProfiles") or occ.get("_channel_signal") or {}
    if isinstance(channels, dict) and channels:
        for ch, prof in channels.items():
            frames.append(profile_to_frame(prof, f"{short} {ch}", "#E45756" if str(ch).upper() == "L1" else "#7B61FF", "o" if str(ch).upper() == "L1" else "D"))
    else:
        frames.append(profile_to_frame(occ.get("profile") or {}, short, LEORO, "D"))
    return [f for f in frames if not f.empty]


def draw_height_time(ax, case: dict, label: str) -> None:
    draw_height_layers(ax)
    profiles = lugre_profiles(case) + leoro_profiles(case)
    for df in profiles:
        ax.plot(df["time"], df["height"], color=df["color"].iloc[0], lw=1.15, alpha=0.88)
    set_height_axis(ax)
    ax.set_title(f"{label}  Height-time", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=3)
    all_times = pd.concat([df["time"] for df in profiles if "time" in df.columns], ignore_index=True).dropna() if profiles else pd.Series(dtype="datetime64[ns, UTC]")
    start = pd.to_datetime(case["sd"]["start_utc"].replace(" UTC", ""), utc=True)
    end = all_times.max() if not all_times.empty else start + pd.Timedelta(minutes=5)
    if "end_utc" in case["sd"] and case["sd"].get("end_utc"):
        end = max(end, pd.to_datetime(str(case["sd"]["end_utc"]).replace(" UTC", ""), utc=True))
    tick_start = start.floor("min")
    tick_end = end.ceil("min")
    if case.get("track_index") == 187:
        tick_end = min(tick_end, pd.Timestamp("2025-03-16 13:46:00", tz="UTC"))
    ticks = list(pd.date_range(tick_start, tick_end, freq="2min"))
    if not ticks or ticks[0] != tick_start:
        ticks.insert(0, tick_start)
    ax.set_xlim(tick_start, tick_end)
    ax.set_xticks([mdates.date2num(t.to_pydatetime()) for t in ticks])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
    ax.set_xlabel("UTC", fontsize=7.0, color=AXIS, fontweight="bold", labelpad=2)
    ax.text(0.98, 0.96, f"{start:%Y-%m-%d} | OP76 S", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=AXIS, fontweight="bold")
    make_axis_text_bold(ax)


def draw_dcn0_panel(ax, case: dict, label: str) -> None:
    draw_height_layers(ax)
    handles = []
    seen = set()
    for df in lugre_profiles(case) + leoro_profiles(case):
        ax.scatter(df["dcn0"], df["height"], s=8.5, marker=df["marker"].iloc[0], color=df["color"].iloc[0], alpha=0.66, linewidths=0)
        lab = df["label"].iloc[0]
        if lab not in seen:
            handles.append(Line2D([0], [0], marker=df["marker"].iloc[0], color="none", markerfacecolor=df["color"].iloc[0], markeredgecolor="none", markersize=4.1, label=lab))
            seen.add(lab)
    ax.axvline(0, color="#334155", lw=0.65, ls=(0, (2.5, 2.2)), alpha=0.65)
    ax.set_xlim(-6, 6)
    ax.set_xticks([-6, -3, 0, 3, 6])
    set_height_axis(ax, side="right")
    ax.set_title(f"{label}  $\\delta C/N_0$", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=3)
    ax.set_xlabel(r"$\delta C/N_0$ (dB)", fontsize=7.0, color=AXIS, fontweight="bold", labelpad=2)
    make_axis_text_bold(ax)
    ax.legend(handles=handles, loc="upper right", fontsize=6.0, frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.55)


def load_ne_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    return read_csv(PROFILE_CSV), pd.DataFrame()


def draw_ne_panel(ax, prof_all: pd.DataFrame, matched_all: pd.DataFrame, figure_track_index: int, label: str) -> dict:
    draw_height_layers(ax)
    prof = prof_all[pd.to_numeric(prof_all["figure_track_index"], errors="coerce").eq(figure_track_index)].copy()
    if "profile_kind" in prof.columns:
        prof = prof[prof["profile_kind"].eq("lugre_profile_location")].copy()
    matched = pd.DataFrame()
    for col in ["alt_km", "iri_log10_ne", "iri_ne_m3"]:
        if col in prof.columns:
            prof[col] = pd.to_numeric(prof[col], errors="coerce")
    prof = prof.dropna(subset=["alt_km", "iri_log10_ne"])
    invalid_low_alt_rows = 0
    invalid_ne_rows = 0
    if "iri_ne_m3" in prof.columns:
        invalid_ne_rows = int(prof["iri_ne_m3"].le(0).sum())
        invalid_low_alt_rows = int((prof["iri_ne_m3"].le(0) & prof["alt_km"].le(60)).sum())
        prof = prof[prof["iri_ne_m3"].gt(0)].copy()
    source_model = str(prof["iri_model"].dropna().iloc[0]) if "iri_model" in prof.columns and len(prof["iri_model"].dropna()) else "IRI-2016"
    ax.plot(prof["iri_log10_ne"], prof["alt_km"], color="#1769AA", lw=1.75, ls=(0, (3.5, 2.0)), label="IRI", zorder=4)
    isr_rows = 0
    isr_status = "isr_context_profile_unavailable"
    isr_profile_time = ""
    isr_time_gap_hr = np.nan
    isr_station = ""
    isr, isr_context = load_isr_context_profile(figure_track_index)
    if not isr.empty:
        isr_station = str(isr["station"].dropna().iloc[0]) if "station" in isr.columns and len(isr["station"].dropna()) else "EISCAT Tromso UHF"
        isr_profile_time = str(isr["time_utc"].dropna().iloc[0]) if "time_utc" in isr.columns and len(isr["time_utc"].dropna()) else str(isr_context.get("profile_time", ""))
        isr_time_gap_hr = float(isr_context.get("time_gap_hr", np.nan)) if isr_context else np.nan
        ax.plot(isr["log10_ne"], isr["alt_bin_km"], color="#B42318", lw=1.65, ls="-", label="ISR ctx (EISCAT UHF)", zorder=5)
        isr_rows = len(isr)
        isr_status = "isr_context_profile_plotted"
    set_height_axis(ax, label="Height [km]")
    ax.set_xlim(8.0, 12.0)
    ax.set_xticks([8, 10, 12])
    ax.set_title(f"{label}  $N_e(h)$", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=3)
    ax.set_xlabel(r"$\log_{10} N_e$ [m$^{-3}$]", fontsize=7.0, color=AXIS, fontweight="bold", labelpad=2)
    make_axis_text_bold(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=5.8, frameon=False, handlelength=1.6, borderaxespad=0.0)
    return {
        "figure_track_index": figure_track_index,
        "profile_source": source_model,
        "profile_rows": len(prof),
        "profile_kind_plotted": "all_rows_in_updated_iri_zip",
        "profile_csv": str(PROFILE_CSV),
        "profile_valid_alt_min_km": float(prof["alt_km"].min()) if not prof.empty else np.nan,
        "profile_valid_alt_max_km": float(prof["alt_km"].max()) if not prof.empty else np.nan,
        "profile_invalid_ne_rows_total": invalid_ne_rows,
        "profile_invalid_rows_alt_le_60km": invalid_low_alt_rows,
        "matched_iri_rows": 0,
        "observed_isr_ne_rows": isr_rows,
        "isr_ne_status": isr_status,
        "isr_context_station": isr_station,
        "isr_context_profile_time": isr_profile_time,
        "isr_context_time_gap_hr": isr_time_gap_hr,
        "isr_context_zip": str(ISR_ZIP),
    }


def draw_fan_panel(ax, case: dict, label: str) -> None:
    fan = case["fan"].copy()
    ax.set_title(f"{label}  SuperDARN fan", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=3)
    if fan.empty:
        ax.text(0.5, 0.5, "No fan table", transform=ax.transAxes, ha="center", va="center", fontsize=7, color=MUTED)
        return
    for col in ["bmazm_deg", "range_center_km", "mean_power_db"]:
        fan[col] = pd.to_numeric(fan[col], errors="coerce")
    fan = fan.dropna(subset=["bmazm_deg", "range_center_km", "mean_power_db"])
    sc = ax.scatter(np.deg2rad(fan["bmazm_deg"]), fan["range_center_km"], c=fan["mean_power_db"], cmap=SD_CMAP, norm=Normalize(0, 12), s=10, alpha=0.82, linewidths=0)
    coherent = case.get("coherent", pd.DataFrame())
    ax.set_theta_zero_location("N", offset=18)
    ax.set_theta_direction(-1)
    theta_min = float(fan["bmazm_deg"].min())
    theta_max = float(fan["bmazm_deg"].max())
    theta_pad = max(4.0, 0.06 * (theta_max - theta_min))
    ax.set_thetamin(theta_min - theta_pad)
    ax.set_thetamax(theta_max + theta_pad)
    ax.set_ylim(0, min(3500, max(800, float(fan["range_center_km"].max()) + 100)))
    range_ticks = [1000, 2000, 3000]
    range_ticks = [tick for tick in range_ticks if tick <= ax.get_ylim()[1]]
    ax.set_yticks(range_ticks)
    ax.set_yticklabels([])
    track = case["track"].copy()
    b, d = bearing_distance_from_station(track["lugre_lat"], track["lugre_lon"])
    ok = np.isfinite(b) & np.isfinite(d) & (d <= ax.get_ylim()[1])
    if ok.sum() >= 2:
        ax.plot(np.deg2rad(b[ok]), d[ok], color="#111827", lw=1.35, alpha=0.95, zorder=8, label="LuGRE TP")
    ax.grid(True, color=GRID, alpha=0.68, lw=0.55)
    ax.tick_params(axis="x", labelsize=6.2, colors=AXIS, width=0.65, length=2.5, pad=-2.0)
    ax.tick_params(axis="y", length=0)
    tick_theta_deg = theta_max + theta_pad + 5.0
    label_theta_deg = theta_max + theta_pad + 13.0

    def radial_text_angle(theta_deg: float) -> float:
        theta = np.deg2rad(theta_deg)
        r0 = ax.get_ylim()[1] * 0.35
        r1 = ax.get_ylim()[1] * 0.85
        p0 = ax.transData.transform((theta, r0))
        p1 = ax.transData.transform((theta, r1))
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        return angle

    tick_theta = np.deg2rad(tick_theta_deg)
    range_label_rotation = radial_text_angle(label_theta_deg)
    for tick in range_ticks:
        ax.text(tick_theta, tick, f"{tick}", rotation=range_label_rotation, rotation_mode="anchor", fontsize=6.1, fontweight="bold", color=AXIS, ha="left", va="center", clip_on=False)
    ax.text(np.deg2rad(label_theta_deg), ax.get_ylim()[1] * 0.66, "Range (km)", rotation=range_label_rotation, rotation_mode="anchor", ha="left", va="center", fontsize=5.9, fontweight="bold", color=AXIS, clip_on=False)
    cax = ax.inset_axes([0.09, -0.075, 0.82, 0.045])
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal", ticks=[0, 4, 8, 12])
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.set_label("Power (dB)", fontsize=6.1, color=AXIS, fontweight="bold")
    for tick in cbar.ax.get_xticklabels():
        tick.set_color(AXIS)
        tick.set_fontweight("bold")
    make_axis_text_bold(ax)


def align_fan_axis(fig, ax_fan, ax_above, ax_left) -> None:
    """Make the fan a compact panel centered under C/H and vertically balanced with D/I."""
    fig_w, fig_h = fig.get_size_inches()
    above = ax_above.get_position()
    left = ax_left.get_position()
    target_h = left.height
    target_w = target_h * fig_h / fig_w
    target_w = min(target_w, above.width * 0.94)
    target_x = above.x0 + (above.width - target_w) / 2
    target_y = left.y0
    ax_fan.set_position([target_x, target_y, target_w, target_h])


def shift_axis(ax, dx: float = 0.0, dy: float = 0.0) -> None:
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height])


def build() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
    })
    prof_all, matched_all = load_ne_profiles()
    cases = [load_case(145), load_case(187)]
    letters = [("a", "b", "c", "d", "e"), ("f", "g", "h", "i", "j")]

    fig = plt.figure(figsize=(12.95, 10.6), dpi=260, facecolor="white")
    outer = fig.add_gridspec(
        2,
        3,
        left=0.030,
        right=0.955,
        bottom=0.062,
        top=0.972,
        width_ratios=[2.05, 0.48, 0.48],
        hspace=0.112,
        wspace=0.045,
    )
    qa_rows = []
    for r, case in enumerate(cases):
        ax_map = fig.add_subplot(outer[r, 0])
        shift_axis(ax_map, dx=0.018)
        qa_rows.append(draw_map(ax_map, fig, case, letters[r][0].upper()))
        sub = outer[r, 1:].subgridspec(2, 2, hspace=0.30, wspace=0.16)
        ax_b = fig.add_subplot(sub[0, 0])
        ax_c = fig.add_subplot(sub[0, 1])
        ax_d = fig.add_subplot(sub[1, 0])
        ax_e = fig.add_subplot(sub[1, 1], projection="polar")
        align_fan_axis(fig, ax_e, ax_c, ax_d)
        draw_height_time(ax_b, case, letters[r][1].upper())
        draw_dcn0_panel(ax_c, case, letters[r][2].upper())
        ne_summary = draw_ne_panel(ax_d, prof_all, matched_all, case["track_index"], letters[r][3].upper())
        qa_rows[-1].update(ne_summary)
        draw_fan_panel(ax_e, case, letters[r][4].upper())

    fig.legend(
        handles=map_legend_handles(),
        loc="center left",
        bbox_to_anchor=(0.153, 0.550),
        frameon=True,
        facecolor="white",
        edgecolor="#E2E8F0",
        fontsize=5.15,
        ncol=1,
        handlelength=1.25,
        columnspacing=0.50,
        handletextpad=0.35,
    )

    fig.savefig(PNG_OUT, dpi=260)
    fig.savefig(PDF_OUT)
    plt.close(fig)

    pd.DataFrame(qa_rows).to_csv(QA_OUT, index=False, encoding="utf-8-sig")
    README_OUT.write_text(
        "\n".join(
            [
                "# Fig.4 north-pair v43 with v40 layout and Fig.7 basemap",
                "",
                "This version keeps the v40 layout, smoothed ROTI background and power-fan evidence. The only map-layer change is that the A/F basemap coastline/border overlay now uses the same Natural Earth source and styling logic as Extended Data Fig.7, projected into the existing v40 polar layout.",
                "",
                "## Key choices",
                "- Polar maps retain the v40 60 deg N circular layout, title placement, side colorbars and smoothed ROTI rendering.",
                "- The previous compact-JSON Arctic basemap is replaced by Fig.7-style Natural Earth coastlines and borders; no A/F axes or panel layout were converted to Cartopy.",
                "- ROTI is rendered as contextual background only where the compact ROTI grid has finite polar cells; unsupported map areas are left transparent/white. No n/station-count threshold is applied in this visual-coverage version.",
                "- ISMR scintillation IPP points are plotted for all available polar S4 samples in A/F and use the same 0-0.5 S4 normalization as their side colorbar. ISMR receiver-site symbols are intentionally not plotted as a data layer.",
                "- SuperDARN fan power cells are opaque in A/F and use the same 0-12 dB normalization as their side colorbar and E/J fan panels.",
                "- LuGRE LOS paths are retained and darkened relative to v15.",
                "- B/C/D/E are redrawn as real axes, so time labels and ticks are aligned.",
                "- D/I panels use the updated IRI table `iri2016_extended_50_1000km_profiles_track145_187.csv` plus the EISCAT Tromso UHF observed context profile from `noncoherent_isr_used_data_track145_187_20260624.zip`. The ISR profile is context-only: QA records the profile time and time gap to each LuGRE track.",
                "- The right-side panels are narrowed relative to v19; A/F map styling and data layers are otherwise preserved.",
                "- A/F map titles now report OP, GNSS family and PRN in the requested order, polar-night context and UTC time window.",
                "- F title displays the full plotted context through 13:46 UTC for consistency with G. The LuGRE/SuperDARN matched-event window from `superdarn_match_row.json` is 2025-03-16 13:38:04-13:42:14 UTC.",
                "- A/F map latitude labels are placed between the 45 E and 90 E meridians to avoid overlap with the outer longitude labels.",
                "- Map panels include separate compact side colorbars for ROTI, SuperDARN power and ISMR S4. ISMR S4 uses a 0-0.5 display range.",
                "- B/G keep tangent-height axes on the left; C/H use right-side tangent-height axes for visual balance.",
                "- D/E and I/J panel boxes share the same top and bottom extent; E/J fan colorbar labels are positioned to visually match the D/I x-axis labels.",
                "- E/J range labels are reduced to 1000/2000/3000 km and placed outside the fan edge; the `Range (km)` label is placed on a separate parallel radial guide to avoid text overlap.",
                "- D/I use a dashed blue IRI curve and a solid red `ISR ctx (EISCAT UHF)` context curve, with the legend moved to the left-side empty region to avoid curve overlap.",
                "- The shared map legend is moved into the left inter-panel gap and kept inside the colorbar left boundary.",
                "- Track 187 H panel now uses the received GLONASS occultation PRN and frequency labels: `Spire R18 L1` from `caL1Snr` and `Spire R18 L2` from `pL2Snr`. The raw file metadata gives `occsatId=18`, `freq1=L1C`, and `freq2=L2C`.",
                "- The Spire L1/L2 SNR values are converted to 1 s C/N0-proxy using `10 log10(mean(SNR^2))`, then detrended with a fourth-order Chebyshev profile fit over tangent height, matching the Fig.3 C/N0-proxy residual convention.",
                "",
                f"PNG: `{PNG_OUT}`",
                f"PDF: `{PDF_OUT}`",
                f"QA: `{QA_OUT}`",
                f"Spire R18 dual-frequency source table: `{SPIRE_R18_DUAL_SOURCE_OUT}`",
            ]
        ),
        encoding="utf-8",
    )
    print(PNG_OUT)
    print(PDF_OUT)
    print(QA_OUT)


if __name__ == "__main__":
    build()
