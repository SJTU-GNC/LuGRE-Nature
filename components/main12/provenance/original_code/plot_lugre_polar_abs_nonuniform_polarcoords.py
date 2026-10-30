from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

import plot_lugre_ionosphere_tangent_tracks_by_phase as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
POLAR_LAT_DEG = 60.0
MAP_LAT_MIN = 60.0
CMAX_DBHZ = 4.0
BOUNDARIES = [0.0, 0.01, 0.025, 0.063, 0.109, 0.170, 0.268, 0.410, 0.615, 0.929, 1.374, 2.203, 4.0]
OUTPUT_PREFIX = "lugre_polar_tangent_tracks_abs_nonuniform_polarcoords"
GEODETIC = ccrs.PlateCarree()


def true_blue_red() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "true_blue_red_abs",
        ["#0000ff", "#0077ff", "#00c8ff", "#ffffff", "#fff176", "#ff9800", "#ff0000"],
        N=len(BOUNDARIES) - 1,
    )


def parula_like() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "parula_like_abs",
        ["#352a87", "#0f5cdd", "#1485d4", "#06a7c6", "#4db866", "#c8c945", "#f9fb0e"],
        N=len(BOUNDARIES) - 1,
    )


def true_blue_red_gradient() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "true_blue_red_abs_gradient",
        ["#0000ff", "#0077ff", "#00c8ff", "#ffffff", "#fff176", "#ff9800", "#ff0000"],
        N=256,
    )


def parula_like_gradient() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "parula_like_abs_gradient",
        ["#352a87", "#0f5cdd", "#1485d4", "#06a7c6", "#4db866", "#c8c945", "#f9fb0e"],
        N=256,
    )


def colorbar_colormap(variant: str) -> mpl.colors.Colormap:
    if variant == "matlab_jet_abs_bins":
        return mpl.colormaps["jet"]
    if variant == "matlab_hsv_abs_bins":
        return mpl.colormaps["hsv"]
    if variant == "matlab_turbo_abs_bins":
        return mpl.colormaps["turbo"]
    if variant == "spectral_nebula_abs_bins":
        return mpl.colormaps["nipy_spectral"]
    if variant == "parula_like_abs_bins":
        return parula_like_gradient()
    if variant == "true_blue_red_abs_bins":
        return true_blue_red_gradient()
    return mpl.colormaps["viridis"]


def colormaps() -> dict[str, mpl.colors.Colormap]:
    return {
        "matlab_jet_abs_bins": mpl.colormaps["jet"].resampled(len(BOUNDARIES) - 1),
        "matlab_hsv_abs_bins": mpl.colormaps["hsv"].resampled(len(BOUNDARIES) - 1),
        "matlab_turbo_abs_bins": mpl.colormaps["turbo"].resampled(len(BOUNDARIES) - 1),
        "parula_like_abs_bins": parula_like(),
        "spectral_nebula_abs_bins": mpl.colormaps["nipy_spectral"].resampled(len(BOUNDARIES) - 1),
        "true_blue_red_abs_bins": true_blue_red(),
    }


def wrap_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def iter_lines(geometry):
    geom_type = geometry.geom_type
    if geom_type == "LineString":
        yield np.asarray(geometry.coords)
    elif geom_type == "MultiLineString":
        for line in geometry.geoms:
            yield np.asarray(line.coords)
    elif geom_type == "Polygon":
        yield np.asarray(geometry.exterior.coords)
        for interior in geometry.interiors:
            yield np.asarray(interior.coords)
    elif geom_type == "MultiPolygon":
        for poly in geometry.geoms:
            yield np.asarray(poly.exterior.coords)
            for interior in poly.interiors:
                yield np.asarray(interior.coords)


def polar_radius(lat: np.ndarray, hemisphere: str) -> np.ndarray:
    if hemisphere == "north":
        return 90.0 - lat
    return 90.0 + lat


def circular_boundary(ax: plt.Axes) -> None:
    theta = np.linspace(0, 2 * np.pi, 256)
    center = np.array([0.5, 0.5])
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + center
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)


def setup_cartopy_axis(ax: plt.Axes, hemisphere: str, title: str) -> None:
    if hemisphere == "north":
        ax.set_extent([-180, 180, MAP_LAT_MIN, 90], crs=GEODETIC)
        pole_lat = 90.0
        lat_labels = [(60, "60N"), (70, "70N"), (80, "80N")]
        pole_label = "90N"
    else:
        ax.set_extent([-180, 180, -90, -MAP_LAT_MIN], crs=GEODETIC)
        pole_lat = -90.0
        lat_labels = [(-60, "60S"), (-70, "70S"), (-80, "80S")]
        pole_label = "90S"

    circular_boundary(ax)
    ax.set_facecolor("white")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="white", edgecolor="none", zorder=0.2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.62, edgecolor="#7e8b96", zorder=0.8)

    gl = ax.gridlines(
        crs=GEODETIC,
        draw_labels=False,
        linewidth=0.55,
        color="#9fb3c7",
        alpha=0.36,
        linestyle="-",
        xlocs=np.arange(-180, 181, 45),
        ylocs=np.arange(-90, 91, 10),
    )
    _ = gl

    for lon in [-180, -135, -90, -45, 0, 45, 90, 135]:
        lat = MAP_LAT_MIN + 0.9 if hemisphere == "north" else -(MAP_LAT_MIN + 0.9)
        label = "180" if abs(lon) == 180 else ("0E" if lon == 0 else f"{abs(lon)}{'E' if lon > 0 else 'W'}")
        ax.text(
            lon,
            lat,
            label,
            transform=GEODETIC,
            ha="center",
            va="center",
            fontsize=10,
            color="#1f2937",
            bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.62},
            zorder=6,
        )

    label_lon = 42.0
    for lat, label in lat_labels:
        ax.text(
            label_lon,
            lat,
            label,
            transform=GEODETIC,
            ha="center",
            va="center",
            fontsize=10,
            color="#1f2937",
            bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.62},
            zorder=6,
        )
    ax.text(
        0,
        pole_lat,
        pole_label,
        transform=GEODETIC,
        ha="center",
        va="center",
        fontsize=10,
        color="#1f2937",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.62},
        zorder=6,
    )
    ax.set_title(title, pad=18, fontsize=14)


def setup_axis(ax: plt.Axes, hemisphere: str, title: str) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.0, 90.0 - MAP_LAT_MIN)
    ax.set_facecolor("white")
    ax.set_title(title, pad=18, fontsize=14)
    ax.grid(True, color="#b7c0ca", linestyle="-", linewidth=0.9, alpha=0.30)
    degs = np.arange(0, 360, 45)
    labels = ["0°", "45°E", "90°E", "135°E", "180°", "135°W", "90°W", "45°W"]
    ax.set_thetagrids(degs, labels=labels)
    ax.set_yticks([10, 20, 30])
    if hemisphere == "north":
        ax.set_yticklabels(["80°N", "70°N", "60°N"])
        ax.text(0, 0, "90°N", ha="center", va="center", fontsize=8.5, color="#26323f", zorder=5)
    else:
        ax.set_yticklabels(["80°S", "70°S", "60°S"])
        ax.text(0, 0, "90°S", ha="center", va="center", fontsize=8.5, color="#26323f", zorder=5)
    for deg in degs:
        theta = np.deg2rad(deg)
        ax.plot([theta, theta], [0, 90 - MAP_LAT_MIN], color="#8fa3b5", lw=0.7, ls="-", alpha=0.24, zorder=0)


def kept_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = []
    ends = []
    in_run = False
    start = 0
    for idx, keep in enumerate(mask):
        if keep and not in_run:
            start = idx
            in_run = True
        elif not keep and in_run:
            starts.append(start)
            ends.append(idx)
            in_run = False
    if in_run:
        starts.append(start)
        ends.append(len(mask))
    return [(start, end) for start, end in zip(starts, ends) if end - start >= 2]


def split_geo_piece(lon: np.ndarray, lat: np.ndarray, hemisphere: str) -> list[tuple[np.ndarray, np.ndarray]]:
    keep = lat >= (MAP_LAT_MIN - 0.6) if hemisphere == "north" else lat <= -(MAP_LAT_MIN - 0.6)
    pieces = []
    for start, end in kept_runs(keep):
        lon_run = wrap_lon(lon[start:end])
        lat_run = lat[start:end]
        jumps = np.where(np.abs(np.diff(lon_run)) > 40.0)[0] + 1
        starts = np.r_[0, jumps]
        ends = np.r_[jumps, len(lon_run)]
        for sub_start, sub_end in zip(starts, ends):
            if sub_end - sub_start >= 2:
                pieces.append((lon_run[sub_start:sub_end], lat_run[sub_start:sub_end]))
    return pieces


def draw_land(ax: plt.Axes, hemisphere: str) -> None:
    land_path = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    reader = shpreader.Reader(land_path)
    for record in reader.records():
        for coords in iter_lines(record.geometry):
            if coords.size == 0:
                continue
            lon = coords[:, 0].astype(float)
            lat = coords[:, 1].astype(float)
            for lon_piece, lat_piece in split_geo_piece(lon, lat, hemisphere):
                if len(lon_piece) < 3:
                    continue
                radius = polar_radius(lat_piece, hemisphere)
                if np.nanmax(np.abs(np.diff(radius))) > 12.0:
                    continue
                ax.fill(
                    np.deg2rad(lon_piece),
                    radius,
                    color="white",
                    alpha=1.0,
                    linewidth=0,
                    zorder=0.18,
                )


def draw_coastlines(ax: plt.Axes, hemisphere: str) -> None:
    draw_land(ax, hemisphere)
    coast_path = shpreader.natural_earth(resolution="50m", category="physical", name="coastline")
    reader = shpreader.Reader(coast_path)
    for record in reader.records():
        for coords in iter_lines(record.geometry):
            if coords.size == 0:
                continue
            lon = coords[:, 0].astype(float)
            lat = coords[:, 1].astype(float)
            for lon_piece, lat_piece in split_geo_piece(lon, lat, hemisphere):
                radius = polar_radius(lat_piece, hemisphere)
                if np.nanmax(np.abs(np.diff(radius))) > 12.0:
                    continue
                ax.plot(
                    np.deg2rad(lon_piece),
                    radius,
                    color="#7e8b96",
                    lw=0.74,
                    alpha=0.88,
                    zorder=0.5,
                )


def split_track(track: pd.DataFrame, hemisphere: str) -> list[pd.DataFrame]:
    lat = track["lugre_lat"].to_numpy(dtype=float)
    keep = lat >= POLAR_LAT_DEG if hemisphere == "north" else lat <= -POLAR_LAT_DEG
    if keep.sum() < 2:
        return []
    sub = track.loc[keep].sort_values("time_utc").copy()
    lon = sub["lugre_lon"].to_numpy(dtype=float)
    time_gap = sub["rx_gps_seconds"].diff().fillna(1.0).to_numpy(dtype=float) > 1.5
    lon_gap = np.r_[False, np.abs(np.diff(lon)) > 180.0]
    run_id = np.cumsum(time_gap | lon_gap)
    return [piece for _, piece in sub.groupby(run_id, sort=False) if len(piece) >= 2]


def add_tracks(ax: plt.Axes, df: pd.DataFrame, hemisphere: str, cmap, norm, linewidth: float) -> int:
    count = 0
    for _, track in df.sort_values("time_utc").groupby(base.TRACK_KEYS, dropna=False):
        pieces = split_track(track, hemisphere)
        if pieces:
            count += 1
        for piece in pieces:
            lon = piece["lugre_lon"].to_numpy(dtype=float)
            lat = piece["lugre_lat"].to_numpy(dtype=float)
            theta = np.deg2rad(lon)
            radius = polar_radius(lat, hemisphere)
            xy = np.column_stack([theta, radius])
            values_raw = np.clip(piece[base.CN0_ABS_RESID_COL].to_numpy(dtype=float), 0.0, CMAX_DBHZ)
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            values = np.clip((values_raw[:-1] + values_raw[1:]) / 2.0, 0.0, CMAX_DBHZ)
            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=linewidth,
                alpha=1.0,
                capstyle="round",
                joinstyle="round",
                rasterized=True,
                zorder=4.2,
            )
            lc.set_array(values)
            ax.add_collection(lc)
    return count


def add_tracks_cartopy(ax: plt.Axes, df: pd.DataFrame, hemisphere: str, cmap, norm, linewidth: float) -> int:
    count = 0
    for _, track in df.sort_values("time_utc").groupby(base.TRACK_KEYS, dropna=False):
        pieces = split_track(track, hemisphere)
        if pieces:
            count += 1
        for piece in pieces:
            for segment in base.split_dateline(piece):
                if len(segment) < 2:
                    continue
                xy = segment[["lugre_lon", "lugre_lat"]].to_numpy(dtype=float)
                line_segments = np.stack([xy[:-1], xy[1:]], axis=1)
                values_raw = np.clip(segment[base.CN0_ABS_RESID_COL].to_numpy(dtype=float), 0.0, CMAX_DBHZ)
                values = np.clip((values_raw[:-1] + values_raw[1:]) / 2.0, 0.0, CMAX_DBHZ)
                lc = LineCollection(
                    line_segments,
                    cmap=cmap,
                    norm=norm,
                    linewidths=linewidth,
                    alpha=0.98,
                    capstyle="round",
                    joinstyle="round",
                    transform=GEODETIC,
                    rasterized=True,
                    zorder=4,
                )
                lc.set_array(values)
                ax.add_collection(lc)
    return count


def add_stats_box(ax: plt.Axes, tracks: int, points: int) -> None:
    ax.text(
        0.02,
        0.02,
        f"tracks={tracks}\npoints={points:,}",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#ccd5dd", "alpha": 0.88, "boxstyle": "round,pad=0.25"},
        zorder=6,
    )


def add_colorbar(fig: plt.Figure, axes, cmap, norm, variant: str) -> None:
    gradient_cmap = colorbar_colormap(variant)
    gradient_norm = mpl.colors.Normalize(vmin=0.0, vmax=CMAX_DBHZ)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=gradient_norm, cmap=gradient_cmap),
        ax=axes,
        shrink=0.82,
        pad=0.05,
    )
    cbar.set_label(f"|Detrended zero-mean C/N0| (dB), {variant}")
    cbar.set_ticks([0.0, 0.5, 1.0, 2.0, 4.0])
    cbar.set_ticklabels(["0", "0.5", "1", "2", ">=4"])


def polar_counts(df: pd.DataFrame, hemisphere: str) -> tuple[int, int]:
    if hemisphere == "north":
        sub = df[df["lugre_lat"] >= POLAR_LAT_DEG]
    else:
        sub = df[df["lugre_lat"] <= -POLAR_LAT_DEG]
    tracks = sub.groupby(base.TRACK_KEYS, dropna=False).ngroups if len(sub) else 0
    return tracks, len(sub)


def write_variant(df: pd.DataFrame, variant: str, cmap: mpl.colors.Colormap) -> None:
    norm = mpl.colors.BoundaryNorm(BOUNDARIES, cmap.N, clip=True)
    fig = plt.figure(figsize=(15, 7.8))
    fig.patch.set_facecolor("white")
    axes = [
        fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo()),
        fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo()),
    ]
    setup_cartopy_axis(axes[0], "north", f"North polar tangent tracks (lat >= {POLAR_LAT_DEG:.0f}N)")
    setup_cartopy_axis(axes[1], "south", f"South polar tangent tracks (lat <= {POLAR_LAT_DEG:.0f}S)")
    linewidth = 1.35 if variant != "matlab_hsv_abs_bins" else 1.22
    north_tracks_drawn = add_tracks_cartopy(axes[0], df, "north", cmap, norm, linewidth=linewidth)
    south_tracks_drawn = add_tracks_cartopy(axes[1], df, "south", cmap, norm, linewidth=linewidth)
    _, north_points = polar_counts(df, "north")
    _, south_points = polar_counts(df, "south")
    add_stats_box(axes[0], north_tracks_drawn, north_points)
    add_stats_box(axes[1], south_tracks_drawn, south_points)

    add_colorbar(fig, axes, cmap, norm, variant)
    total_tracks = df.groupby(base.TRACK_KEYS, dropna=False).ngroups
    polar_tracks = df[df["lugre_lat"].abs() >= POLAR_LAT_DEG].groupby(base.TRACK_KEYS, dropna=False).ngroups
    fig.suptitle(
        f"LuGRE polar ionosphere tangent tracks | h_tan 50-1000 km | tracks={total_tracks}, polar={polar_tracks}",
        y=0.98,
        fontsize=16,
    )
    out = OUT_DIR / f"{OUTPUT_PREFIX}_{variant}.png"
    fig.savefig(out, dpi=360, bbox_inches="tight")
    plt.close(fig)


def write_html(variants: list[str]) -> None:
    sections = []
    for variant in variants:
        img = f"{OUTPUT_PREFIX}_{variant}.png"
        sections.append(
            f"""
<section>
  <h2>{variant}</h2>
  <a href="{img}">PNG</a>
  <img src="{img}" alt="{variant}">
</section>"""
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LUGRE polar tangent tracks, polar-coordinate labels, clean colorbar ticks</title>
<style>
body {{ margin:0; font-family:Arial, sans-serif; color:#1f2937; background:#fff; }}
header {{ padding:18px 24px 8px; }}
h1 {{ margin:0 0 6px; font-size:22px; }}
.meta {{ color:#57606a; font-size:14px; line-height:1.55; }}
section {{ padding:10px 18px 28px; border-top:1px solid #d0d7de; }}
section h2 {{ margin:8px 0; font-size:18px; }}
a {{ color:#0f766e; font-weight:700; text-decoration:none; }}
img {{ width:100%; display:block; border:1px solid #d0d7de; border-radius:8px; margin-top:8px; }}
</style>
</head>
<body>
<header>
  <h1>LUGRE polar tangent tracks, polar-coordinate labels</h1>
  <div class="meta">Tracks and map extent use only |lat| >= 60 deg tangent points. The map style follows lugre_s_phase_polar_tangent_tracks_log_blue.png: 0E at the right, longitude labels every 30 deg, and latitude rings centered at 90N/90S. Colorbar uses clean labels while keeping the same nonuniform color bins.</div>
</header>
{''.join(sections)}
</body>
</html>"""
    (OUT_DIR / f"{OUTPUT_PREFIX}_colormap_variants.html").write_text(html, encoding="utf-8")


def main() -> None:
    df = base.read_ionosphere_points()
    names = []
    for name, cmap in colormaps().items():
        write_variant(df, name, cmap)
        names.append(name)
        print(name)
    write_html(names)
    print(OUT_DIR / f"{OUTPUT_PREFIX}_colormap_variants.html")


if __name__ == "__main__":
    main()
