from __future__ import annotations

from pathlib import Path

import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from .common import ROOT


DATA = ROOT / "data" / "panel_ready" / "Fig2"
NATURAL_EARTH = Path(__file__).resolve().parent / "natural_earth_110m"
POINTS = DATA / "fig2_polar_points.csv"
SUMMARY = DATA / "fig2_selected_tracks.csv"
POLAR_LAT_DEG = 60.0
LAYER_ALPHA = 0.17
SPINE_COLOR = "#334155"
BOUNDARIES = [
    0.0,
    0.01,
    0.025,
    0.063,
    0.109,
    0.170,
    0.268,
    0.410,
    0.615,
    0.929,
    1.374,
    2.203,
    4.0,
]
FREQ_COLORS = {
    "GPS L1": "#0b2e8a",
    "GPS L5": "#8b0000",
    "Galileo E1": "#0000ff",
    "Galileo E5a": "#ff0000",
    "other": "#475569",
}
SIGNAL_ENCODING = {
    "GPS L1": {
        "constellation": "GPS",
        "role": "primary signal",
        "color_hex": FREQ_COLORS["GPS L1"],
    },
    "GPS L5": {
        "constellation": "GPS",
        "role": "secondary signal",
        "color_hex": FREQ_COLORS["GPS L5"],
    },
    "Galileo E1": {
        "constellation": "Galileo",
        "role": "primary signal",
        "color_hex": FREQ_COLORS["Galileo E1"],
    },
    "Galileo E5a": {
        "constellation": "Galileo",
        "role": "secondary signal",
        "color_hex": FREQ_COLORS["Galileo E5a"],
    },
}
DAY_LAYERS = [
    ("D", 50.0, 90.0, "#f3e6c8"),
    ("E", 90.0, 150.0, "#c9ddf2"),
    ("F1", 150.0, 200.0, "#d7c9ee"),
    ("F2", 200.0, 600.0, "#d7c9ee"),
    ("Topside", 600.0, 1000.0, "#ead0da"),
]
NIGHT_LAYERS = [
    ("E", 90.0, 150.0, "#c9ddf2"),
    ("F", 150.0, 600.0, "#d7c9ee"),
    ("Topside", 600.0, 1000.0, "#ead1d8"),
]
TRACK_KEYS = ["phase", "op_id", "sat", "signal_id", "segment_id"]


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "axes.linewidth": 0.9,
            "savefig.transparent": True,
        }
    )


def _freq_group(signal_name: str) -> str:
    value = str(signal_name)
    if "GPS" in value and "L1" in value:
        return "GPS L1"
    if "GPS" in value and "L5" in value:
        return "GPS L5"
    if "Galileo" in value and "E1" in value:
        return "Galileo E1"
    if "Galileo" in value and "E5" in value:
        return "Galileo E5a"
    return "other"


def _load_points() -> pd.DataFrame:
    points = pd.read_csv(POINTS)
    points["time_utc"] = pd.to_datetime(points["time_utc"], errors="coerce")
    points["freq_group"] = points["signal_name"].map(_freq_group)
    points["abs_delta_cn0_db"] = pd.to_numeric(
        points["plot_cn0_detrended_db"], errors="coerce"
    ).abs()
    return points.sort_values(TRACK_KEYS + ["rx_gps_seconds"])


def _load_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY)
    summary["display_label"] = summary["display_label"].astype(int)
    summary["time_start"] = pd.to_datetime(summary["time_start"], errors="raise")
    summary["time_end"] = pd.to_datetime(summary["time_end"], errors="raise")
    return summary.sort_values("display_label")


def _op_id_from_uid(uid: str) -> str:
    parts = str(uid).split("|")
    if len(parts) < 2:
        raise RuntimeError(f"Cannot parse op_id from group_uid={uid!r}")
    return parts[1]


def _selected_groups(
    points: pd.DataFrame, summary: pd.DataFrame
) -> list[tuple[pd.Series, pd.DataFrame]]:
    groups: list[tuple[pd.Series, pd.DataFrame]] = []
    for _, row in summary.iterrows():
        mask = (
            points["phase"].astype(str).eq(str(row["phase"]))
            & points["op_id"].astype(str).eq(_op_id_from_uid(row["group_uid"]))
            & points["sat"].astype(str).eq(str(row["sat"]))
            & points["time_utc"].between(row["time_start"], row["time_end"])
        )
        group = points.loc[mask].copy()
        if group.empty:
            raise RuntimeError(
                f"No packaged rows matched selected event {row['display_label']}"
            )
        groups.append(
            (
                row,
                group.sort_values(["rx_gps_seconds", "signal_name", "segment_id"]),
            )
        )
    return groups


def _polar_radius(lat: np.ndarray, hemisphere: str) -> np.ndarray:
    return 90.0 - lat if hemisphere == "north" else 90.0 + lat


def _kept_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, keep in enumerate(mask):
        if keep and start is None:
            start = idx
        elif not keep and start is not None:
            if idx - start >= 2:
                runs.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= 2:
        runs.append((start, len(mask)))
    return runs


def _split_geo_piece(
    lon: np.ndarray, lat: np.ndarray, hemisphere: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    keep = (
        lat >= POLAR_LAT_DEG - 0.6
        if hemisphere == "north"
        else lat <= -(POLAR_LAT_DEG - 0.6)
    )
    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in _kept_runs(keep):
        lon_run = ((lon[start:end] + 180.0) % 360.0) - 180.0
        lat_run = lat[start:end]
        jumps = np.where(np.abs(np.diff(lon_run)) > 40.0)[0] + 1
        starts = np.r_[0, jumps]
        ends = np.r_[jumps, len(lon_run)]
        for sub_start, sub_end in zip(starts, ends):
            if sub_end - sub_start >= 2:
                pieces.append(
                    (lon_run[sub_start:sub_end], lat_run[sub_start:sub_end])
                )
    return pieces


def _iter_lines(geometry):
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
        for polygon in geometry.geoms:
            yield np.asarray(polygon.exterior.coords)
            for interior in polygon.interiors:
                yield np.asarray(interior.coords)


def _shape_path(name: str) -> Path:
    return NATURAL_EARTH / f"ne_110m_{name}.shp"


def _draw_land_and_coastlines(ax, hemisphere: str) -> None:
    land_path = _shape_path("land")
    coast_path = _shape_path("coastline")
    if not land_path.is_file() or not coast_path.is_file():
        raise FileNotFoundError("Bundled Natural Earth 110m land/coastline is missing")

    for record in shpreader.Reader(land_path).records():
        for coords in _iter_lines(record.geometry):
            if coords.size == 0:
                continue
            for lon_piece, lat_piece in _split_geo_piece(
                coords[:, 0].astype(float), coords[:, 1].astype(float), hemisphere
            ):
                if len(lon_piece) < 3:
                    continue
                radius = _polar_radius(lat_piece, hemisphere)
                if np.nanmax(np.abs(np.diff(radius))) <= 12.0:
                    ax.fill(
                        np.deg2rad(lon_piece),
                        radius,
                        color="white",
                        alpha=1.0,
                        linewidth=0,
                        zorder=0.18,
                    )
    for record in shpreader.Reader(coast_path).records():
        for coords in _iter_lines(record.geometry):
            if coords.size == 0:
                continue
            for lon_piece, lat_piece in _split_geo_piece(
                coords[:, 0].astype(float), coords[:, 1].astype(float), hemisphere
            ):
                radius = _polar_radius(lat_piece, hemisphere)
                if np.nanmax(np.abs(np.diff(radius))) <= 12.0:
                    ax.plot(
                        np.deg2rad(lon_piece),
                        radius,
                        color="#7e8b96",
                        lw=0.74,
                        alpha=0.88,
                        zorder=0.5,
                    )


def _setup_polar_axis(ax, hemisphere: str) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.0, 30.0)
    ax.set_facecolor("white")
    ax.set_title(
        f"{hemisphere.capitalize()} polar tangent tracks", fontsize=11.6, pad=7.0
    )
    ax.grid(True, color="#b7c0ca", linestyle="-", linewidth=0.9, alpha=0.30)
    degrees = np.arange(0, 360, 45)
    labels = ["0°", "45°E", "90°E", "135°E", "180°", "135°W", "90°W", "45°W"]
    ax.set_thetagrids(degrees, labels=labels)
    ax.set_yticks([10, 20, 30])
    suffix = "N" if hemisphere == "north" else "S"
    ax.set_yticklabels([f"80°{suffix}", f"70°{suffix}", f"60°{suffix}"])
    ax.text(
        0,
        0,
        f"90°{suffix}",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#26323f",
        zorder=5,
    )
    for degree in degrees:
        theta = np.deg2rad(degree)
        ax.plot(
            [theta, theta],
            [0, 30],
            color="#8fa3b5",
            lw=0.7,
            ls="-",
            alpha=0.24,
            zorder=0,
        )
    ax.tick_params(labelsize=8.8)
    ax.tick_params(axis="x", pad=-4.8)
    ax.tick_params(axis="y", pad=0.4)
    _draw_land_and_coastlines(ax, hemisphere)


def _split_track(track: pd.DataFrame, hemisphere: str) -> list[pd.DataFrame]:
    lat = track["lugre_lat"].to_numpy(dtype=float)
    keep = lat >= POLAR_LAT_DEG if hemisphere == "north" else lat <= -POLAR_LAT_DEG
    if keep.sum() < 2:
        return []
    sub = track.loc[keep].sort_values("time_utc").copy()
    lon = sub["lugre_lon"].to_numpy(dtype=float)
    time_gap = (
        sub["rx_gps_seconds"].diff().fillna(1.0).to_numpy(dtype=float) > 1.5
    )
    lon_gap = np.r_[False, np.abs(np.diff(lon)) > 180.0]
    run_id = np.cumsum(time_gap | lon_gap)
    return [piece for _, piece in sub.groupby(run_id, sort=False) if len(piece) >= 2]


def _add_tracks(
    ax,
    points: pd.DataFrame,
    hemisphere: str,
    cmap,
    norm,
    *,
    linewidth: float,
    white_underlay: bool = False,
) -> int:
    count = 0
    for _, track in points.sort_values("time_utc").groupby(TRACK_KEYS, dropna=False):
        pieces = _split_track(track, hemisphere)
        if pieces:
            count += 1
        for piece in pieces:
            theta = np.deg2rad(piece["lugre_lon"].to_numpy(dtype=float))
            radius = _polar_radius(
                piece["lugre_lat"].to_numpy(dtype=float), hemisphere
            )
            xy = np.column_stack([theta, radius])
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            raw = piece["abs_delta_cn0_db"].to_numpy(dtype=float)
            values = np.clip((raw[:-1] + raw[1:]) / 2.0, 0.0, 4.0)
            if white_underlay:
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
            collection = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=linewidth,
                alpha=1.0,
                capstyle="round",
                joinstyle="round",
                rasterized=True,
                zorder=6.2 if white_underlay else 4.2,
            )
            collection.set_array(values)
            ax.add_collection(collection)
    return count


def _add_selected_label(
    ax, track: pd.DataFrame, hemisphere: str, label: str
) -> None:
    pieces = _split_track(track, hemisphere)
    if not pieces:
        return
    combined = pd.concat(pieces, ignore_index=True)
    theta = np.deg2rad(combined["lugre_lon"].to_numpy(dtype=float))
    radius = _polar_radius(combined["lugre_lat"].to_numpy(dtype=float), hemisphere)
    ax.text(
        float(np.nanmean(theta)),
        float(np.nanmean(radius)),
        label,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#111111",
        bbox={
            "boxstyle": "circle,pad=0.16",
            "fc": "white",
            "ec": "#111111",
            "lw": 0.9,
            "alpha": 0.97,
        },
        zorder=8,
    )


def _render_polar(panel_id: str, output: Path) -> dict[str, object]:
    hemisphere = "north" if panel_id == "Fig2_A" else "south"
    points = _load_points()
    summary = _load_summary()
    groups = _selected_groups(points, summary)
    cmap = mpl.colormaps["jet"].resampled(len(BOUNDARIES) - 1)
    norm = mpl.colors.BoundaryNorm(BOUNDARIES, cmap.N, clip=True)

    fig = plt.figure(figsize=(4.9, 4.9), facecolor="none")
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    _setup_polar_axis(ax, hemisphere)
    track_count = _add_tracks(
        ax, points, hemisphere, cmap, norm, linewidth=1.0
    )
    selected_labels: list[int] = []
    for row, group in groups:
        group_hemi = "north" if group["lugre_lat"].median() >= 0 else "south"
        if group_hemi != hemisphere:
            continue
        _add_tracks(
            ax,
            group,
            hemisphere,
            cmap,
            norm,
            linewidth=1.68,
            white_underlay=True,
        )
        _add_selected_label(ax, group, hemisphere, str(row["display_label"]))
        selected_labels.append(int(row["display_label"]))
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

    hemisphere_rows = int(
        (
            points["lugre_lat"].ge(60.0)
            if hemisphere == "north"
            else points["lugre_lat"].le(-60.0)
        ).sum()
    )
    if hemisphere_rows <= 0 or track_count <= 0:
        raise RuntimeError(f"No {hemisphere} polar tracks were rendered")
    return {
        "input": str(POINTS.relative_to(ROOT)),
        "hemisphere": hemisphere,
        "point_rows": hemisphere_rows,
        "track_groups": track_count,
        "highlighted_event_labels": selected_labels,
        "status": "passed",
    }


def _layer_scheme(daynight: str):
    return DAY_LAYERS if str(daynight) == "Day" else NIGHT_LAYERS


def _draw_track_panel(
    ax,
    row: pd.Series,
    group: pd.DataFrame,
    y_limits: tuple[float, float],
) -> None:
    y_min, y_max = y_limits
    layers = _layer_scheme(str(row["daynight"]))
    ax.set_xscale("log")
    ax.set_xlim(50, 1000)
    ax.set_ylim(y_min, y_max)
    ax.set_facecolor("white")

    for layer_name, low, high, color in layers:
        ax.axvspan(low, high, color=color, alpha=LAYER_ALPHA, lw=0, zorder=0)
        ax.text(
            low * 1.03,
            0.055,
            layer_name,
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#26323f",
            bbox={
                "boxstyle": "round,pad=0.06",
                "fc": "white",
                "ec": "none",
                "alpha": 0.70,
            },
            clip_on=True,
            zorder=6,
        )
    boundaries = {low for _, low, _, _ in layers} | {
        high for _, _, high, _ in layers
    }
    for boundary in sorted(boundaries | {350.0}):
        if boundary == 350.0:
            ax.axvline(
                boundary,
                color="#64748b",
                ls="--",
                lw=1.05,
                alpha=0.70,
                zorder=1,
            )
        elif boundary not in {50.0, 1000.0}:
            ax.axvline(
                boundary,
                color="#94a3b8",
                ls=":",
                lw=0.72,
                alpha=0.65,
                zorder=1,
            )

    for frequency, sub in group.groupby("freq_group", sort=False):
        ax.scatter(
            sub["lugre_h_tan_km"],
            sub["cn0_dbhz"],
            s=11.0,
            color=FREQ_COLORS.get(frequency, FREQ_COLORS["other"]),
            alpha=0.86,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )
    ax.set_xticks([50, 90, 150, 300, 600, 1000])
    ax.set_xticklabels(["50", "90", "150", "300", "600", "1000"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.get_offset_text().set_visible(False)
    ax.grid(True, which="major", color="#94a3b8", alpha=0.17, lw=0.58)
    ax.grid(False, which="minor")
    ax.tick_params(
        axis="both", which="major", labelsize=8.7, length=3.0, width=0.82, pad=1.5
    )
    ax.set_ylabel("C/N$_0$ (dB-Hz)", fontsize=9.6)
    ax.set_xlabel("Tangent-point altitude (km)", fontsize=9.6, labelpad=2.6)
    ax.set_title(
        f"{int(row['display_label'])}  {row['op']} {row['sat']} | {row['daynight']}",
        fontsize=10.0,
        pad=5.5,
        color="#111827",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)
        spine.set_edgecolor(SPINE_COLOR)


def _render_selected_track(
    panel_id: str, output: Path, event_label: int
) -> dict[str, object]:
    points = _load_points()
    summary = _load_summary()
    groups = _selected_groups(points, summary)
    all_cn0 = pd.concat([group["cn0_dbhz"] for _, group in groups], ignore_index=True)
    y_limits = (
        float(np.floor(all_cn0.min() - 0.6)),
        float(np.ceil(all_cn0.max() + 0.6)),
    )
    selected_row, selected_group = groups[event_label - 1]
    if int(selected_row["display_label"]) != event_label:
        raise RuntimeError(f"Selected-event ordering changed for {panel_id}")

    fig, ax = plt.subplots(figsize=(3.62, 3.08), facecolor="none")
    _draw_track_panel(ax, selected_row, selected_group, y_limits)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.880, bottom=0.165)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

    signal_counts = {
        str(name): int(count)
        for name, count in selected_group["freq_group"].value_counts().items()
    }
    return {
        "input": str(POINTS.relative_to(ROOT)),
        "selection_metadata": str(SUMMARY.relative_to(ROOT)),
        "event_label": event_label,
        "op": str(selected_row["op"]),
        "satellite": str(selected_row["sat"]),
        "daynight": str(selected_row["daynight"]),
        "point_rows": int(len(selected_group)),
        "signal_counts": signal_counts,
        "signal_color_mapping": SIGNAL_ENCODING,
        "wgs84_tangent_height_range_km": [
            float(selected_group["lugre_h_tan_km"].min()),
            float(selected_group["lugre_h_tan_km"].max()),
        ],
        "status": "passed",
    }


def render(panel_id: str, output: Path) -> dict[str, object]:
    _configure_style()
    if panel_id in {"Fig2_A", "Fig2_B"}:
        return _render_polar(panel_id, output)
    if panel_id.startswith("Fig2_C"):
        event_label = int(panel_id.removeprefix("Fig2_C"))
        if not 1 <= event_label <= 6:
            raise RuntimeError(f"Unsupported Fig2 selected track: {panel_id}")
        return _render_selected_track(panel_id, output, event_label)
    raise RuntimeError(f"Unsupported Fig2 panel: {panel_id}")
