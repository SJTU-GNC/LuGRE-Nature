from __future__ import annotations

import math
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon


TASK_DIR = Path(__file__).resolve().parent
REPRO_ROOT = TASK_DIR.parents[2]
DATA_DIR = REPRO_ROOT / "data" / "analysis_ready" / "ED2"
ROOT = DATA_DIR
BASE_ROOT = DATA_DIR / "_not_packaged_ground_roti_raw"
CHAIN_STATIONS = DATA_DIR / "_not_packaged_chain_stations.csv"
COHERENT_RADARS = DATA_DIR / "_not_packaged_coherent_radar_status.csv"
OUT = ROOT / "combined_station_polar_maps.png"

POLAR_LAT_LIMIT = 60.0
POLAR_CENTRAL_LONGITUDE = -90.0
EARTH_RADIUS_KM = 6371.0

COLORS = {
    "land": "#ffffff",
    "ocean": "#ffffff",
    "coast": "#7a7a7a",
    "grid": "#c9c9c9",
    "text": "#222222",
    "muted": "#666666",
    "gnss": "#8e79b9",
    "ismr": "#59a14f",
    "isr": "#6baed6",
    "coherent": "#e15759",
    "coherent_fill": "#e15759",
}

COVERAGE_ALPHA = {
    "isr_fill": 0.16,
    "isr_edge": 0.3,
    "coherent_fill": 0.065,
    "coherent_edge": 0.18,
}

ISR_COVERAGE_MERGE_KM = 80.0
COHERENT_COVERAGE_MERGE_KM = 120.0
GROUND_MARKER_SIZE = 14
ISR_MARKER_SIZE = 108
COHERENT_MARKER_SIZE = 42

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.55,
        "axes.edgecolor": "#333333",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "hatch.linewidth": 0.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def normalize_lon(values) -> pd.Series:
    lon = pd.to_numeric(values, errors="coerce")
    return ((lon + 180) % 360) - 180


def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    lat1 = math.radians(float(lat_deg))
    lon1 = math.radians(float(lon_deg))
    bearing = math.radians(float(bearing_deg))
    angular = float(distance_km) / EARTH_RADIUS_KM
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    lon = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
    return math.degrees(lat2), lon


def great_circle_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(float(lat1))
    lat2_rad = math.radians(float(lat2))
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def circular_lon_mean(lons: pd.Series) -> float:
    radians = np.radians(pd.to_numeric(lons, errors="coerce").dropna().to_numpy(dtype=float))
    if len(radians) == 0:
        return float("nan")
    lon = math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean()))
    return (lon + 540.0) % 360.0 - 180.0


def merge_nearby_locations(df: pd.DataFrame, max_distance_km: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    remaining = list(df.index)
    rows = []
    while remaining:
        seed = remaining[0]
        group = {seed}
        changed = True
        while changed:
            changed = False
            for idx in list(remaining):
                if idx in group:
                    continue
                row = df.loc[idx]
                if any(
                    great_circle_distance_km(row["lat"], row["lon"], df.loc[g, "lat"], df.loc[g, "lon"])
                    <= max_distance_km
                    for g in group
                ):
                    group.add(idx)
                    changed = True
        remaining = [idx for idx in remaining if idx not in group]

        part = df.loc[sorted(group)]
        representative = part.iloc[0].copy()
        representative["lat"] = pd.to_numeric(part["lat"], errors="coerce").mean()
        representative["lon"] = circular_lon_mean(part["lon"])
        rows.append(representative)

    return pd.DataFrame(rows).reset_index(drop=True)


def geodesic_circle(lat: float, lon: float, radius_km: float, steps: int = 181) -> tuple[list[float], list[float]]:
    lats: list[float] = []
    lons: list[float] = []
    for bearing in np.linspace(0, 360, steps):
        la, lo = destination_point(lat, lon, float(bearing), radius_km)
        lats.append(la)
        lons.append(lo)
    return lats, lons


def coherent_fan(
    lat: float,
    lon: float,
    hemisphere: str,
    inner_km: float = 650.0,
    outer_km: float = 3000.0,
    half_width_deg: float = 28.0,
    steps: int = 32,
) -> tuple[list[float], list[float]]:
    center_bearing = 0.0 if hemisphere == "north" else 180.0
    left = center_bearing - half_width_deg
    right = center_bearing + half_width_deg

    lats: list[float] = [lat]
    lons: list[float] = [lon]
    for distance in np.linspace(inner_km, outer_km, steps):
        la, lo = destination_point(lat, lon, left, float(distance))
        lats.append(la)
        lons.append(lo)
    for bearing in np.linspace(left, right, steps):
        la, lo = destination_point(lat, lon, float(bearing), outer_km)
        lats.append(la)
        lons.append(lo)
    for distance in np.linspace(outer_km, inner_km, steps):
        la, lo = destination_point(lat, lon, right, float(distance))
        lats.append(la)
        lons.append(lo)
    return lats, lons


def project_polygon(ax, pc, lats: list[float], lons: list[float]) -> np.ndarray:
    lon_arr = np.asarray(lons, dtype=float)
    lat_arr = np.asarray(lats, dtype=float)
    projected = ax.projection.transform_points(pc, lon_arr, lat_arr)[:, :2]
    return projected[np.isfinite(projected).all(axis=1)]


def load_gnss_roti_stations() -> pd.DataFrame:
    coords = pd.read_csv(BASE_ROOT / r"06_logs\logs\station_coordinates_downloaded_gt60.csv")
    stats = pd.read_csv(BASE_ROOT / r"03_results\roti_results\polar_roti_1p0_summary\summary_station.csv")
    coords["station_key"] = coords["station"].astype(str).str.lower()
    stats["station_key"] = stats["station"].astype(str).str.lower()
    merged = coords.merge(stats[["station_key", "days", "rows"]], on="station_key", how="left")
    merged["source_type"] = "GNSS/ROTI station"
    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
    merged["lon"] = normalize_lon(merged["lon"])
    return merged[["station", "name", "lat", "lon", "source_type", "days", "rows"]]


def load_chain_stations() -> pd.DataFrame:
    if not CHAIN_STATIONS.exists():
        return pd.DataFrame(columns=["station", "name", "lat", "lon", "source_type", "days", "rows"])
    df = pd.read_csv(CHAIN_STATIONS)
    df = df.rename(columns={"abbr": "station", "lon_e": "lon"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = normalize_lon(df["lon"])
    df["source_type"] = "CHAIN/ISMR station"
    df["days"] = np.nan
    df["rows"] = np.nan
    return df[["station", "name", "lat", "lon", "source_type", "days", "rows"]]


def load_isr_stations() -> pd.DataFrame:
    rows = [
        ("Resolute Bay North", "Resolute Bay North", 74.72955, -94.90576, "downloaded"),
        ("Poker Flat", "Poker Flat", 65.13, -147.471, "downloaded"),
        ("EISCAT Tromso VHF", "EISCAT Tromso VHF", 69.6, 19.2, "downloaded"),
        ("EISCAT Tromso UHF", "EISCAT Tromso UHF", 69.583, 19.21, "downloaded"),
        ("Chatanika", "Chatanika", 65.103, -147.45, "checked"),
        ("Sondrestrom", "Sondrestrom", 67.0, -51.0, "checked"),
        ("EISCAT Svalbard", "EISCAT Svalbard", 78.09, 16.02, "checked"),
        ("EISCAT Kiruna Rx", "EISCAT Kiruna Rx", 67.9, 20.4, "receiver"),
        ("EISCAT Sodankyla Rx", "EISCAT Sodankyla Rx", 67.4, 26.6, "receiver"),
    ]
    df = pd.DataFrame(rows, columns=["station", "name", "lat", "lon", "status"])
    df["lon"] = normalize_lon(df["lon"])
    df["source_type"] = "ISR radar"
    df["days"] = np.nan
    df["rows"] = np.nan
    return df[["station", "name", "lat", "lon", "source_type", "days", "rows", "status"]]


def load_coherent_radars() -> pd.DataFrame:
    if not COHERENT_RADARS.exists():
        return pd.DataFrame(columns=["station", "name", "lat", "lon", "source_type", "status", "files", "fitacf_hours"])
    df = pd.read_csv(COHERENT_RADARS)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = normalize_lon(df["lon"])
    df["files"] = pd.to_numeric(df.get("files"), errors="coerce")
    df["fitacf_hours"] = pd.to_numeric(df.get("fitacf_hours"), errors="coerce")
    polar = df[df["lat"].abs() >= POLAR_LAT_LIMIT].copy()
    polar = polar[polar["kind"].astype(str).str.contains("SuperDARN|HF|CSR", case=False, na=False)]
    polar["station"] = polar["id"].astype(str)
    polar["source_type"] = "Coherent radar"
    return polar[["station", "name", "lat", "lon", "source_type", "kind", "status", "download", "files", "fitacf_hours"]]


def polar_boundary(ax) -> None:
    theta = np.linspace(0, 2 * math.pi, 240)
    verts = np.vstack([0.5 + 0.5 * np.sin(theta), 0.5 + 0.5 * np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)


def setup_axis(ax, hemisphere: str) -> ccrs.PlateCarree:
    pc = ccrs.PlateCarree()
    if hemisphere == "north":
        ax.set_extent([-180, 180, POLAR_LAT_LIMIT, 90], crs=pc)
        title = "Northern polar cap (lat >= 60 deg)"
        panel = "a"
    else:
        ax.set_extent([-180, 180, -90, -POLAR_LAT_LIMIT], crs=pc)
        ax.set_ylim(ax.get_ylim()[::-1])
        title = "Southern polar cap (lat <= -60 deg)"
        panel = "b"

    polar_boundary(ax)
    ax.set_title(title, fontsize=8.4, weight="normal", loc="center", pad=18, color=COLORS["text"])
    ax.text(
        -0.035,
        1.03,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        weight="bold",
        color=COLORS["text"],
    )
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=COLORS["ocean"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=COLORS["land"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.36, edgecolor=COLORS["coast"], zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.15, edgecolor="#bdbdbd", zorder=2)
    gl = ax.gridlines(
        crs=pc,
        draw_labels=False,
        linewidth=0.26,
        color=COLORS["grid"],
        alpha=0.72,
        linestyle="-",
        zorder=1,
    )
    gl.xlocator = plt.FixedLocator([-90, 0, 45, 90, 145, 180])
    gl.ylocator = plt.FixedLocator(np.arange(-80, 91, 10))
    diag_label_r = 0.54
    ax.text(1.018, 0.5, r"$0^\circ$", transform=ax.transAxes, ha="left", va="center", fontsize=5.65, color=COLORS["muted"], clip_on=False, zorder=30)
    ax.text(-0.018, 0.5, r"$180^\circ$", transform=ax.transAxes, ha="right", va="center", fontsize=5.65, color=COLORS["muted"], clip_on=False, zorder=30)
    ax.text(0.5, 1.015, r"$90^\circ\mathrm{E}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.65, color=COLORS["muted"], clip_on=False, zorder=30)
    ax.text(0.5, -0.035, r"$90^\circ\mathrm{W}$", transform=ax.transAxes, ha="center", va="top", fontsize=5.65, color=COLORS["muted"], clip_on=False, zorder=30)
    ax.text(0.5 + diag_label_r * math.cos(math.radians(45)), 0.5 + diag_label_r * math.sin(math.radians(45)), r"$45^\circ\mathrm{E}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.35, color=COLORS["muted"], clip_on=False, zorder=30)
    ax.text(0.5 + diag_label_r * math.cos(math.radians(145)), 0.5 + diag_label_r * math.sin(math.radians(145)), r"$145^\circ\mathrm{E}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.35, color=COLORS["muted"], clip_on=False, zorder=30)
    lat_label_lon = 160.0
    lat_label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 0.1}
    for lat_value in (80, 70, 60):
        plot_lat = float(lat_value if hemisphere == "north" else -lat_value)
        ax.text(
            lat_label_lon,
            plot_lat,
            rf"${lat_value}^\circ$",
            transform=pc,
            ha="right",
            va="center",
            fontsize=5.15,
            color=COLORS["muted"],
            bbox=lat_label_box,
            zorder=31,
        )
    return pc


def hemisphere_subset(df: pd.DataFrame, hemisphere: str) -> pd.DataFrame:
    if df.empty:
        return df
    if hemisphere == "north":
        return df[df["lat"] >= POLAR_LAT_LIMIT].copy()
    return df[df["lat"] <= -POLAR_LAT_LIMIT].copy()


def draw_isr_coverage(ax, pc, isr: pd.DataFrame, hemisphere: str) -> int:
    subset = hemisphere_subset(isr, hemisphere)
    coverage_sites = merge_nearby_locations(subset, ISR_COVERAGE_MERGE_KM)
    for _, row in coverage_sites.iterrows():
        lats, lons = geodesic_circle(float(row["lat"]), float(row["lon"]), 550.0)
        ax.fill(lons, lats, color=COLORS["isr"], alpha=COVERAGE_ALPHA["isr_fill"], transform=pc, zorder=4, linewidth=0)
        ax.plot(lons, lats, color=COLORS["isr"], alpha=COVERAGE_ALPHA["isr_edge"], linewidth=0.36, transform=pc, zorder=5)
    return int(len(coverage_sites))


def draw_coherent_coverage(ax, pc, coherent: pd.DataFrame, hemisphere: str) -> int:
    subset = hemisphere_subset(coherent, hemisphere)
    coverage_sites = merge_nearby_locations(subset, COHERENT_COVERAGE_MERGE_KM)
    fan_polygons: list[np.ndarray] = []
    for _, row in coverage_sites.iterrows():
        lats, lons = coherent_fan(float(row["lat"]), float(row["lon"]), hemisphere)
        xy = project_polygon(ax, pc, lats, lons)
        if len(xy) >= 3:
            fan_polygons.append(xy)

    for xy in fan_polygons:
        ax.add_patch(
            MplPolygon(
                xy,
                closed=True,
                facecolor=COLORS["coherent_fill"],
                edgecolor="none",
                alpha=COVERAGE_ALPHA["coherent_fill"],
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
                edgecolor=COLORS["coherent"],
                alpha=COVERAGE_ALPHA["coherent_edge"],
                linewidth=0.34,
                transform=ax.transData,
                zorder=2.4,
                clip_on=True,
            )
        )
    return int(len(coverage_sites))


def plot_station_points(ax, pc, stations: pd.DataFrame, isr: pd.DataFrame, coherent: pd.DataFrame, hemisphere: str) -> dict[str, int]:
    subset = hemisphere_subset(stations, hemisphere)
    counts: dict[str, int] = {"ground": int(len(subset))}

    styles = {
        "GNSS/ROTI station": ("o", COLORS["gnss"], GROUND_MARKER_SIZE, 0.72),
        "CHAIN/ISMR station": ("s", COLORS["ismr"], GROUND_MARKER_SIZE, 0.86),
    }
    for source_type, (marker, color, size, alpha) in styles.items():
        part = subset[subset["source_type"] == source_type]
        if part.empty:
            continue
        ax.scatter(
            part["lon"],
            part["lat"],
            s=size,
            marker=marker,
            c=color,
            alpha=alpha,
            edgecolors="#ffffff",
            linewidths=0.16,
            transform=pc,
            zorder=7,
        )
        counts[source_type] = int(len(part))

    isr_subset = hemisphere_subset(isr, hemisphere)
    if not isr_subset.empty:
        isr_points = merge_nearby_locations(isr_subset, ISR_COVERAGE_MERGE_KM)
        ax.scatter(
            isr_points["lon"],
            isr_points["lat"],
            s=ISR_MARKER_SIZE,
            marker="*",
            c=COLORS["isr"],
            alpha=0.96,
            edgecolors="#ffffff",
            linewidths=0.3,
            transform=pc,
            zorder=11,
        )
    counts["ISR radar"] = int(len(isr_subset))

    coherent_subset = hemisphere_subset(coherent, hemisphere)
    if not coherent_subset.empty:
        coherent_points = merge_nearby_locations(coherent_subset, COHERENT_COVERAGE_MERGE_KM)
        ax.scatter(
            coherent_points["lon"],
            coherent_points["lat"],
            s=COHERENT_MARKER_SIZE,
            marker="D",
            c=COLORS["coherent"],
            alpha=0.96,
            edgecolors="#ffffff",
            linewidths=0.22,
            transform=pc,
            zorder=10,
        )
    counts["Coherent radar"] = int(len(coherent_subset))
    return counts


def draw_panel(ax, hemisphere: str, stations: pd.DataFrame, isr: pd.DataFrame, coherent: pd.DataFrame) -> dict[str, int]:
    pc = setup_axis(ax, hemisphere)
    coherent_count = draw_coherent_coverage(ax, pc, coherent, hemisphere)
    isr_count = draw_isr_coverage(ax, pc, isr, hemisphere)
    counts = plot_station_points(ax, pc, stations, isr, coherent, hemisphere)
    counts["ISR coverage"] = isr_count
    counts["Coherent coverage"] = coherent_count
    return counts


def make_legend_items(counts: dict[str, int]) -> list:
    return [
        Line2D([0], [0], marker="o", color="none", label=f"GNSS/ROTI ({counts.get('GNSS/ROTI station', 0)})", markerfacecolor=COLORS["gnss"], markeredgecolor="white", markersize=5.5),
        Line2D([0], [0], marker="s", color="none", label=f"CHAIN/ISMR ({counts.get('CHAIN/ISMR station', 0)})", markerfacecolor=COLORS["ismr"], markeredgecolor="white", markersize=5.5),
        Line2D([0], [0], marker="*", color="none", label=f"ISR ({counts.get('ISR radar', 0)})", markerfacecolor=COLORS["isr"], markeredgecolor="white", markersize=7.4),
        Line2D([0], [0], marker="D", color="none", label=f"Coherent radar ({counts.get('Coherent radar', 0)})", markerfacecolor=COLORS["coherent"], markeredgecolor="white", markersize=5.5),
        Patch(facecolor=COLORS["isr"], edgecolor=COLORS["isr"], alpha=COVERAGE_ALPHA["isr_fill"], label=f"ISR range ({counts.get('ISR coverage', 0)})"),
        Patch(
            facecolor=COLORS["coherent_fill"],
            edgecolor=COLORS["coherent"],
            alpha=COVERAGE_ALPHA["coherent_fill"],
            label=f"Coherent coverage ({counts.get('Coherent coverage', 0)})",
        ),
    ]


def add_panel_legend_and_counts(ax, counts: dict[str, int]) -> None:
    ax.legend(
        handles=make_legend_items(counts),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.125),
        bbox_transform=ax.transAxes,
        fontsize=5.2,
        handlelength=1.05,
        columnspacing=0.72,
        labelspacing=0.45,
    )


def main() -> None:
    station_points = pd.concat(
        [load_gnss_roti_stations(), load_chain_stations()],
        ignore_index=True,
    ).dropna(subset=["lat", "lon"])
    station_points = station_points[station_points["lat"].abs() >= POLAR_LAT_LIMIT].copy()

    isr = load_isr_stations().dropna(subset=["lat", "lon"])
    coherent = load_coherent_radars().dropna(subset=["lat", "lon"])

    fig = plt.figure(figsize=(7.55, 5.05), dpi=430, facecolor="white")
    north_ax = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo(central_longitude=POLAR_CENTRAL_LONGITUDE))
    south_ax = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=POLAR_CENTRAL_LONGITUDE))

    north_counts = draw_panel(north_ax, "north", station_points, isr, coherent)
    south_counts = draw_panel(south_ax, "south", station_points, isr, coherent)
    add_panel_legend_and_counts(north_ax, north_counts)
    add_panel_legend_and_counts(south_ax, south_counts)

    fig.text(
        0.5,
        0.975,
        "Polar source locations and nominal radar coverage",
        ha="center",
        va="top",
        fontsize=9.0,
        weight="bold",
        color=COLORS["text"],
    )
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.255, top=0.855, wspace=0.17)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
