from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT


DATA = ROOT / "data" / "panel_ready" / "Fig1"
NATURAL_EARTH = Path(__file__).resolve().parent / "natural_earth_110m"
GRID_DEG = 5.0
LAT_EDGES = np.arange(-90.0, 90.0 + GRID_DEG, GRID_DEG)
LON_EDGES = np.arange(-180.0, 180.0 + GRID_DEG, GRID_DEG)
X_LIM = (0.0, 6.0)
Y_LIM = (0.0, 60.0)
MAGNETIC_POLE_LAT_DEG = 80.65
MAGNETIC_POLE_LON_DEG = -72.68
BANDS = [
    ("Equatorial\n($|\\lambda_m| < 20^\\circ$)", "Equatorial", "#0891b2"),
    ("Mid-latitude\n($20^\\circ \\leq |\\lambda_m| < 60^\\circ$)", "Mid-latitude", "#7c3aed"),
    ("Polar\n($|\\lambda_m| \\geq 60^\\circ$)", "Polar", "#dc2626"),
]


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.78,
            "savefig.transparent": True,
        }
    )


def _smooth_hist(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins, density=False)
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    hist = np.convolve(hist.astype(float), kernel / kernel.sum(), mode="same")
    if hist.max(initial=0) > 0:
        hist /= hist.max()
    return (edges[:-1] + edges[1:]) / 2.0, hist


def _close_density(
    coord: np.ndarray, density: np.ndarray, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    return np.r_[low, coord, high], np.r_[0.0, density, 0.0]


def _draw_correlation_panel(
    fig: plt.Figure,
    slot,
    title: str,
    sub: pd.DataFrame,
    color: str,
    show_ylabel: bool,
) -> int:
    inner = slot.subgridspec(
        2,
        2,
        height_ratios=[0.40, 1.0],
        width_ratios=[1.0, 0.25],
        hspace=0.035,
        wspace=0.040,
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])

    x_all = pd.to_numeric(sub["metric_cn0"], errors="coerce").to_numpy(float)
    y_all = pd.to_numeric(sub["metric_roti"], errors="coerce").to_numpy(float)
    shown = (
        np.isfinite(x_all)
        & np.isfinite(y_all)
        & (x_all >= X_LIM[0])
        & (x_all <= X_LIM[1])
        & (y_all >= Y_LIM[0])
        & (y_all <= Y_LIM[1])
    )
    x = x_all[shown]
    y = y_all[shown]
    n = int(len(x))

    ax.scatter(
        x,
        y,
        s=14.0,
        color=color,
        alpha=0.76,
        edgecolors="white",
        linewidths=0.28,
        zorder=3,
    )
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([0, 15, 30, 45, 60])
    ax.set_xlabel(r"P95 $|\delta C/N_0|$ (dB)", fontsize=8.8, labelpad=2.0)
    if show_ylabel:
        ax.set_ylabel("P95 1-min ROTI\n(TECU/min)", fontsize=8.8, labelpad=3.4)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis="both", labelsize=7.6, width=0.72, length=2.8, pad=1.4)
    ax.grid(True, color="#cbd5e1", lw=0.62, alpha=0.92, zorder=0)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_linewidth(0.78)
        spine.set_edgecolor("#374151")
    ax.text(
        0.965,
        0.965,
        f"N={n}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.45,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.12",
            "fc": "white",
            "ec": "#cbd5e1",
            "lw": 0.50,
            "alpha": 0.90,
        },
        zorder=6,
    )

    xb, xd = _smooth_hist(x, np.linspace(*X_LIM, 38))
    xb, xd = _close_density(xb, xd, X_LIM[0], X_LIM[1])
    ax_top.fill_between(xb, 0, xd * 0.66, color=color, alpha=0.68, lw=0)
    ax_top.plot(xb, xd * 0.66, color=color, lw=1.05)
    ax_top.set_xlim(X_LIM[0] - 0.18, X_LIM[1] + 0.18)
    ax_top.set_ylim(-0.055, 1.0)
    ax_top.axis("off")
    ax_top.text(
        0.56,
        0.54,
        title,
        transform=ax_top.transAxes,
        ha="center",
        va="center",
        fontsize=8.9,
        color="#111827",
    )

    yb, yd = _smooth_hist(y, np.linspace(*Y_LIM, 44))
    yb, yd = _close_density(yb, yd, Y_LIM[0], Y_LIM[1])
    ax_right.fill_betweenx(yb, 0, yd * 0.70, color=color, alpha=0.68, lw=0)
    ax_right.plot(yd * 0.70, yb, color=color, lw=1.05)
    ax_right.set_ylim(Y_LIM[0] - 1.8, Y_LIM[1] + 1.8)
    ax_right.set_xlim(-0.06, 1.0)
    ax_right.axis("off")
    return n


def _render_b(output: Path) -> dict[str, object]:
    table = pd.read_csv(DATA / "fig1_b_common_grid_cells.csv")
    fig = plt.figure(figsize=(8.15, 2.78), facecolor="none")
    gs = fig.add_gridspec(
        1, 3, left=0.075, right=0.985, bottom=0.150, top=0.895, wspace=0.10
    )
    counts: dict[str, int] = {}
    for idx, (title, band, color) in enumerate(BANDS):
        counts[band] = _draw_correlation_panel(
            fig,
            gs[0, idx],
            title,
            table[table["lat_group"].eq(band)],
            color,
            idx == 0,
        )
    fig.savefig(output, dpi=600, bbox_inches=None, pad_inches=0.0, transparent=True)
    plt.close(fig)
    expected = {"Equatorial": 100, "Mid-latitude": 201, "Polar": 199}
    if counts != expected:
        raise RuntimeError(f"Fig1_B grid-cell counts changed: {counts} != {expected}")
    return {
        "input": str((DATA / "fig1_b_common_grid_cells.csv").relative_to(ROOT)),
        "grid_cell_counts_in_display_limits": counts,
        "expected_counts": expected,
        "status": "passed",
    }


def _cells_to_grid(cells: pd.DataFrame) -> np.ndarray:
    z = np.full((len(LAT_EDGES) - 1, len(LON_EDGES) - 1), np.nan, dtype=float)
    for row in cells.itertuples(index=False):
        lat_i = int(row.lat_bin)
        lon_i = int(row.lon_bin)
        if 0 <= lat_i < z.shape[0] and 0 <= lon_i < z.shape[1]:
            z[lat_i, lon_i] = float(row.metric)
    return z


def _magnetic_latitude(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    pole_lat = np.deg2rad(MAGNETIC_POLE_LAT_DEG)
    pole_lon = np.deg2rad(MAGNETIC_POLE_LON_DEG)
    dot = (
        np.cos(lat) * np.cos(lon) * np.cos(pole_lat) * np.cos(pole_lon)
        + np.cos(lat) * np.sin(lon) * np.cos(pole_lat) * np.sin(pole_lon)
        + np.sin(lat) * np.sin(pole_lat)
    )
    return np.rad2deg(np.arcsin(np.clip(dot, -1.0, 1.0)))


def _coastline_geometries():
    shp = NATURAL_EARTH / "ne_110m_coastline.shp"
    if not shp.is_file():
        raise FileNotFoundError(f"Bundled coastline is missing: {shp}")
    return list(shpreader.Reader(shp).geometries())


def _add_magnetic_latitude_contours(ax) -> None:
    lon = np.linspace(-180.0, 180.0, 361)
    lat = np.linspace(-89.5, 89.5, 180)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mlat = _magnetic_latitude(lat_grid, lon_grid)
    ax.contour(
        lon_grid,
        lat_grid,
        mlat,
        levels=[-60, -20, 0, 20, 60],
        colors="#66717d",
        linewidths=0.82,
        linestyles="--",
        alpha=0.90,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    label_lon = -171.0
    sample_lat = np.linspace(-89.0, 89.0, 4000)
    sample_lon = np.full_like(sample_lat, label_lon)
    sample_mlat = _magnetic_latitude(sample_lat, sample_lon)
    for level in [-60, -20, 0, 20, 60]:
        label_lat = float(sample_lat[np.argmin(np.abs(sample_mlat - level))])
        ax.text(
            label_lon,
            label_lat,
            rf"$\lambda_m={level}^\circ$",
            transform=ccrs.PlateCarree(),
            fontsize=6.7,
            color="#5b6571",
            ha="left",
            va="center",
            bbox={"fc": "white", "ec": "none", "alpha": 0.72, "pad": 0.3},
            zorder=7,
        )


def _render_map(
    output: Path,
    table_name: str,
    *,
    vmax: float,
    colorbar_label: str,
    ticks: list[float],
) -> dict[str, object]:
    table_path = DATA / table_name
    cells = pd.read_csv(table_path)
    grid = _cells_to_grid(cells)
    cmap = mpl.colormaps["jet"].copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(7.35, 3.18), facecolor="none")
    ax = fig.add_axes([0.015, 0.045, 0.835, 0.91], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor("white")
    ax.pcolormesh(
        LON_EDGES,
        LAT_EDGES,
        np.ma.masked_invalid(grid),
        cmap=cmap,
        norm=norm,
        shading="flat",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.add_geometries(
        _coastline_geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#a2afbf",
        linewidth=0.38,
        zorder=5,
    )
    _add_magnetic_latitude_contours(ax)
    ax.spines["geo"].set_edgecolor("#111111")
    ax.spines["geo"].set_linewidth(0.85)

    cax = fig.add_axes([0.872, 0.175, 0.021, 0.65])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        ticks=ticks,
    )
    tick_labels = [f"{value:g}" for value in ticks]
    tick_labels[-1] = f">={ticks[-1]:g}"
    cb.ax.set_yticklabels(tick_labels)
    cb.set_label(colorbar_label, fontsize=8.4, labelpad=6.0)
    cb.ax.tick_params(labelsize=7.2, length=2.1, width=0.55)
    cb.outline.set_linewidth(0.55)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)

    finite = pd.to_numeric(cells["metric"], errors="coerce").dropna()
    if finite.empty:
        raise RuntimeError(f"{table_name} has no finite metric values")
    return {
        "input": str(table_path.relative_to(ROOT)),
        "grid_cells": int(len(finite)),
        "metric_min": float(finite.min()),
        "metric_max": float(finite.max()),
        "display_range": [0.0, vmax],
        "magnetic_coordinate_model": (
            f"centered dipole pole {MAGNETIC_POLE_LAT_DEG} N, "
            f"{MAGNETIC_POLE_LON_DEG} E; exactly fitted to packaged mlat values"
        ),
        "status": "passed",
    }


def render(panel_id: str, output: Path) -> dict[str, object]:
    _configure_style()
    if panel_id == "Fig1_B":
        return _render_b(output)
    if panel_id == "Fig1_C":
        return _render_map(
            output,
            "fig1_c_cn0_grid_cells.csv",
            vmax=4.0,
            colorbar_label=r"P95 $|\delta C/N_0|$ (dB)",
            ticks=[0, 1, 2, 3, 4],
        )
    if panel_id == "Fig1_D":
        return _render_map(
            output,
            "fig1_d_roti_grid_cells.csv",
            vmax=30.0,
            colorbar_label="P95 1-min ROTI\n(TECU/min)",
            ticks=[0, 5, 10, 15, 20, 25, 30],
        )
    raise RuntimeError(f"Unsupported Fig1 panel: {panel_id}")
