from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "panel_ready" / "Fig5"
SOURCE_ROOT = TASK_DIR
SOURCE_SCRIPT = TASK_DIR / "v37_source.py"
OUTPUT_ROOT = ROOT / "work" / "panel_fig5"
FIGURE_DIR = OUTPUT_ROOT / "figures"
DATA_DIR = OUTPUT_ROOT / "data"
REPORT_DIR = OUTPUT_ROOT / "reports"

VELOCITY_LIMIT_MS = 2000.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v37 = load_module("fig5_v37_velocity_source", SOURCE_SCRIPT)
north = v37.north
south = v37.south

# Loading the accepted source first registers its bundled H-drive plotting
# dependencies. All rendering below therefore uses the same Python runtime and
# plotting stack as the source figure.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import PathCollection
from matplotlib.transforms import Bbox

# Reuse the accepted power-panel sequential palette exactly.
VELOCITY_CMAP = north.SD_CMAP


def velocity_case(case: dict) -> dict:
    """Map mean absolute velocity onto the source code's 0-12 colour coordinate."""
    updated = copy.copy(case)
    fan = case["fan"].copy()
    velocity = pd.to_numeric(fan["mean_abs_velocity_ms"], errors="coerce")
    clipped = velocity.clip(0.0, VELOCITY_LIMIT_MS)
    scaled = 12.0 * clipped / VELOCITY_LIMIT_MS
    # The north map source retains cells with mean_power_db > 0. An epsilon
    # preserves finite velocity cells at the zero colour limit.
    fan["mean_power_db"] = scaled.clip(0.001, 11.999)
    fan["_mean_abs_velocity_ms"] = velocity
    updated["fan"] = fan
    return updated


def replace_horizontal_velocity_colourbar(ax, mappable) -> None:
    if not ax.child_axes:
        raise RuntimeError("Expected the source fan function to create an inset colourbar.")
    cax = ax.child_axes[-1]
    cax.clear()
    cbar = plt.colorbar(
        mappable,
        cax=cax,
        orientation="horizontal",
        ticks=[0, 500, 1000, 1500, 2000],
    )
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.set_label(
        r"Mean $|V_{\rm LOS}|$ (m/s)",
        fontsize=6.1,
        color=north.AXIS,
        fontweight="bold",
    )
    for tick in cbar.ax.get_xticklabels():
        tick.set_color(north.AXIS)
        tick.set_fontweight("bold")


def midpoint_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if centres.size < 2:
        return np.array([centres[0] - 1.62, centres[0] + 1.62])
    interior = 0.5 * (centres[:-1] + centres[1:])
    first = centres[0] - 0.5 * (centres[1] - centres[0])
    last = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return np.concatenate(([first], interior, [last]))


def draw_filled_beam_gate_cells(ax, case: dict, hemisphere: str):
    fan = (
        case["fan"].copy()
        if hemisphere == "north"
        else south.prepare_fir_fan_bearings(case["fan"])
    )
    bearing_col = "bmazm_deg" if hemisphere == "north" else "_bearing_deg"
    numeric_cols = [
        "beam",
        "range_gate",
        bearing_col,
        "range_start_km",
        "range_width_km",
        "mean_abs_velocity_ms",
    ]
    for col in numeric_cols:
        fan[col] = pd.to_numeric(fan[col], errors="coerce")
    fan = fan.dropna(
        subset=[
            "beam",
            "range_gate",
            bearing_col,
            "range_start_km",
            "range_width_km",
        ]
    )

    beams = np.array(sorted(fan["beam"].unique()), dtype=float)
    gates = np.array(sorted(fan["range_gate"].unique()), dtype=float)
    bearings = (
        fan.groupby("beam")[bearing_col].median().reindex(beams).to_numpy(dtype=float)
    )
    theta_edges = np.deg2rad(midpoint_edges(bearings))

    range_meta = fan.groupby("range_gate")[["range_start_km", "range_width_km"]].median()
    range_meta = range_meta.reindex(gates)
    range_starts = range_meta["range_start_km"].to_numpy(dtype=float)
    range_widths = range_meta["range_width_km"].to_numpy(dtype=float)
    range_edges = np.concatenate(
        ([range_starts[0]], range_starts + range_widths)
    )

    values = (
        fan.pivot_table(
            index="range_gate",
            columns="beam",
            values="mean_abs_velocity_ms",
            aggfunc="mean",
        )
        .reindex(index=gates, columns=beams)
        .to_numpy(dtype=float)
    )
    cell_cmap = copy.copy(VELOCITY_CMAP)
    cell_cmap.set_bad("#E8EDF3")
    mesh = ax.pcolormesh(
        theta_edges,
        range_edges,
        np.ma.masked_invalid(values),
        cmap=cell_cmap,
        norm=Normalize(0.0, VELOCITY_LIMIT_MS),
        shading="flat",
        edgecolors="none",
        linewidth=0,
        antialiased=False,
        rasterized=True,
        zorder=1,
    )
    return mesh


def crop_axes(fig, ax, output: Path, dpi: int = 600) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer)]
    boxes.extend(child.get_tightbbox(renderer) for child in ax.child_axes)
    bbox = Bbox.union([box for box in boxes if box is not None])
    bbox = bbox.expanded(1.035, 1.035)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=0.01,
        transparent=True,
    )


def draw_velocity_fan(ax, case: dict, hemisphere: str) -> None:
    if hemisphere == "north":
        north.draw_fan_panel(ax, case, "")
    else:
        v37.draw_south_fan_panel_6500km(ax, case, "")
    for collection in list(ax.collections):
        if isinstance(collection, PathCollection):
            collection.remove()
    mesh = draw_filled_beam_gate_cells(ax, case, hemisphere)
    ax.set_title(
        "SuperDARN fan",
        loc="left",
        fontsize=8.6,
        fontweight="bold",
        color=north.INK,
        pad=3,
    )
    replace_horizontal_velocity_colourbar(ax, mesh)


def draw_velocity_map(ax, fig, case: dict, hemisphere: str) -> dict:
    draw_func = north.draw_map if hemisphere == "north" else south.draw_south_map
    qa = v37.draw_map_without_duplicate_colorbars(draw_func, ax, fig, case, "")
    ax.add_patch(
        north.Circle(
            (0, 0),
            1.0,
            fill=False,
            edgecolor=north.SPINE,
            linewidth=1.05,
            alpha=0.98,
            zorder=90,
            clip_on=False,
        )
    )
    qa["outer_60deg_boundary_redrawn"] = True
    return qa


def save_vertical_velocity_colourbar() -> None:
    fig = plt.figure(figsize=(0.62, 2.15), dpi=260, facecolor="none")
    cax = fig.add_axes([0.18, 0.06, 0.18, 0.88])
    cb = fig.colorbar(
        ScalarMappable(
            norm=Normalize(0.0, VELOCITY_LIMIT_MS),
            cmap=VELOCITY_CMAP,
        ),
        cax=cax,
        orientation="vertical",
        ticks=[0, 500, 1000, 1500, 2000],
    )
    cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
    cb.set_label(
        r"SD mean $|V_{\rm LOS}|$ (m/s)",
        fontsize=5.6,
        labelpad=1.8,
        color=north.AXIS,
        fontweight="bold",
    )
    for tick in cb.ax.get_yticklabels():
        tick.set_color(north.AXIS)
        tick.set_fontweight("bold")
    fig.savefig(
        FIGURE_DIR
        / "SuperDARN_mean_abs_velocity_filled_cells_shared_vertical_colorbar.png",
        dpi=600,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)


def source_summary(role: str, case: dict, source_path: str) -> dict:
    velocity = pd.to_numeric(
        case["fan"]["mean_abs_velocity_ms"], errors="coerce"
    ).dropna()
    return {
        "role": role,
        "track_index": case["track_index"],
        "radar": str(case["fan"]["radar"].dropna().iloc[0]) if len(case["fan"]) else "",
        "source_csv": source_path,
        "fan_rows": len(case["fan"]),
        "finite_velocity_rows": len(velocity),
        "mean_abs_velocity_min_ms": velocity.min(),
        "mean_abs_velocity_p01_ms": velocity.quantile(0.01),
        "mean_abs_velocity_median_ms": velocity.median(),
        "mean_abs_velocity_p99_ms": velocity.quantile(0.99),
        "mean_abs_velocity_max_ms": velocity.max(),
        "fraction_mean_abs_velocity_gt_2000": float((velocity > 2000).mean()),
        "colour_limit_ms": VELOCITY_LIMIT_MS,
    }


def main() -> None:
    for directory in (FIGURE_DIR, DATA_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )
    north.SD_CMAP = VELOCITY_CMAP
    south.SD_CMAP = VELOCITY_CMAP

    north.MAP_TITLES = {
        145: "a  OP76 GPS G01 | north polar night | 2025-03-15 16:24-16:29 UTC",
        187: "f  OP76 GPS G26 | north polar night | 2025-03-16 13:38-13:46 UTC",
    }
    south.SOUTH_MAP_TITLES = {
        9: "k  OP38 Galileo E21 | south polar day | 2025-03-03 12:23-12:32 UTC",
        17: "p  OP38 Galileo E23 | south polar day | 2025-03-03 14:06-14:13 UTC",
    }

    cases = {
        "Fig5_main_north_track145": (
            velocity_case(v37.with_pfisr_site_context(north.load_case(145))),
            "north",
            str(
                north.PKG
                / "data"
                / "extracted_context"
                / "track_145"
                / "superdarn_fan_cells.csv"
            ),
        ),
        "ExtendedDataFig5_north_track187": (
            velocity_case(v37.with_pfisr_site_context(north.load_case(187))),
            "north",
            str(
                north.PKG
                / "data"
                / "extracted_context"
                / "track_187"
                / "superdarn_fan_cells.csv"
            ),
        ),
        "Fig5_main_south_track009": (
            velocity_case(south.load_south_case(9)),
            "south",
            str(
                south.SOUTH_DATA
                / "superdarn"
                / "track_009_fir_event_segment_fan.csv"
            ),
        ),
        "ExtendedDataFig5_south_track017": (
            velocity_case(south.load_south_case(17)),
            "south",
            str(
                south.SOUTH_DATA
                / "superdarn"
                / "track_017_fir_event_segment_fan.csv"
            ),
        ),
    }

    fig = plt.figure(figsize=(16.40, 10.25), dpi=260, facecolor="none")
    map_positions = {
        "Fig5_main_north_track145": [0.232, 0.535, 0.242, 0.395],
        "ExtendedDataFig5_north_track187": [0.232, 0.075, 0.242, 0.395],
        "Fig5_main_south_track009": [0.509, 0.535, 0.242, 0.395],
        "ExtendedDataFig5_south_track017": [0.509, 0.075, 0.242, 0.395],
    }
    fan_positions = {
        "Fig5_main_north_track145": [0.118, 0.555, 0.086, 0.165],
        "ExtendedDataFig5_north_track187": [0.118, 0.070, 0.086, 0.165],
        "Fig5_main_south_track009": [0.884, 0.555, 0.086, 0.165],
        "ExtendedDataFig5_south_track017": [0.884, 0.070, 0.086, 0.165],
    }

    axes_by_role = {}
    qa_rows = []
    for role, (case, hemisphere, source_path) in cases.items():
        ax_map = fig.add_axes(map_positions[role])
        qa = draw_velocity_map(ax_map, fig, case, hemisphere)
        crop_axes(
            fig,
            ax_map,
            FIGURE_DIR / f"{role}_map_mean_abs_velocity_60deg_border.png",
        )

        ax_fan = fig.add_axes(fan_positions[role], projection="polar")
        draw_velocity_fan(ax_fan, case, hemisphere)
        crop_axes(
            fig,
            ax_fan,
            FIGURE_DIR
            / f"{role}_SuperDARN_fan_mean_abs_velocity_filled_cells.png",
        )
        axes_by_role[role] = (ax_map, ax_fan)

        row = source_summary(role, case, source_path)
        row.update(
            {
                "hemisphere": hemisphere,
                "fan_theta_min_deg": ax_fan.get_thetamin(),
                "fan_theta_max_deg": ax_fan.get_thetamax(),
                "fan_range_max_km": ax_fan.get_ylim()[1],
                "map_superdarn_cells_plotted": qa.get(
                    "superdarn_fan_cells_plotted",
                    qa.get("superdarn_fan_power_cells_plotted"),
                ),
            }
        )
        qa_rows.append(row)

    plt.close(fig)
    save_vertical_velocity_colourbar()

    pd.DataFrame(qa_rows).to_csv(
        DATA_DIR / "Fig5_ExtendedDataFig5_SuperDARN_mean_abs_velocity_source_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (DATA_DIR / "source_manifest.json").write_text(
        json.dumps(
            {
                "source_script": str(SOURCE_SCRIPT),
                "observable": "mean_abs_velocity_ms",
                "velocity_definition": "mean absolute SuperDARN Doppler LOS velocity in each source beam-range cell",
                "colour_limit_ms": VELOCITY_LIMIT_MS,
                "outputs": sorted(str(path) for path in FIGURE_DIR.glob("*.png")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Fig.5 and Extended Data Fig.5 SuperDARN filled-cell mean absolute velocity subpanels",
                "",
                "Only the SuperDARN observable was changed. The source event, radar, beam/range geometry, LuGRE tangent track, map layers, axes, typography and panel proportions follow the accepted v37 source.",
                "",
                "The plotted observable is `mean_abs_velocity_ms`, not backscatter power. All four panels use the accepted power-panel sequential colour map with a common 0 to 2000 m/s range. Values above that range are saturated at the upper colour limit.",
                "",
                "Each fan is rendered as contiguous beam-by-range-gate cells. Cells without a finite velocity estimate are filled light grey rather than interpolated. Units are shown as m/s.",
                "",
                "All four polar maps redraw the complete 60-degree latitude boundary above the data layers: 60 degrees north for the northern cases and 60 degrees south for the southern cases.",
                "",
                "The fan PNGs include their horizontal mean-absolute-velocity colourbars. The map PNGs omit per-map colourbars, matching the accepted shared-colourbar layout. A replacement vertical SuperDARN mean-absolute-velocity colourbar is exported separately.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
