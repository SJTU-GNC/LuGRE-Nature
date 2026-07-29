from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MPL_CONFIG_ROOT = ROOT / "work" / "matplotlib"
MPL_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_ROOT)

from PIL import Image

from . import velocity_source as velocity


DATA_ROOT = ROOT / "data" / "panel_ready" / "Fig5"
REFERENCE_ROOT = ROOT / "reference" / "panels" / "Fig5"
OUTPUT_ROOT = ROOT / "outputs"
WORK_ROOT = ROOT / "work" / "panel_fig5"
QA_ROOT = ROOT / "work" / "qa"

v37 = velocity.v37
plt = velocity.plt

PANEL_METADATA = {
    "A": {
        "case": "north_track145",
        "method": "mean_abs_velocity_20260727_map",
        "inputs": [
            "north/track_145/track_points.csv",
            "north/track_145/superdarn_match_row.json",
            "north/track_145/ro_focus_used.json",
            "north/track_145/superdarn_fan_cells.csv",
            "north/track_145/roti_context_field.csv",
            "north/track_145/ismr_context_field.csv",
            "north/track_145/ground_roti_matches.csv",
            "common/yunyao_profiles/matched_profiles.csv",
            "common/yunyao_profiles/track_145_Y038_E05/ionprf_position_time_ne_profile.csv",
        ],
    },
    "B": {
        "case": "north_track145",
        "method": "v37_height_time",
        "inputs": ["north/track_145/track_points.csv", "north/track_145/ro_focus_used.json"],
    },
    "C": {
        "case": "north_track145",
        "method": "v37_delta_cn0",
        "inputs": [
            "north/track_145/ro_focus_used.json",
            "common/leoro_rolling_points.csv",
            "common/leoro_highrate_points.csv",
        ],
    },
    "D": {
        "case": "north_track145",
        "method": "v37_yunyao_ne_with_pfisr",
        "inputs": [
            "common/yunyao_profiles/matched_profiles.csv",
            "common/yunyao_profiles/track_145_Y038_E05/ionprf_position_time_ne_profile.csv",
            "common/yunyao_profiles/track_145_Y038_E05/iri_profile_50_1000km_median_location.csv",
            "common/pfisr_profiles.csv",
        ],
    },
    "E": {
        "case": "north_track145",
        "method": "mean_abs_velocity_20260727_fan",
        "inputs": ["north/track_145/superdarn_fan_cells.csv"],
    },
    "F": {
        "case": "south_track009",
        "method": "mean_abs_velocity_20260727_map",
        "inputs": [
            "south/track.csv",
            "south/ro_focus.json",
            "south/selected_superdarn_row.csv",
            "south/selected_superdarn_station.csv",
            "south/fan_cells.csv",
            "south/fan_context.csv",
            "south/roti_context_field.csv",
            "south/ismr_context_field.csv",
            "south/lugre_los_paths.csv",
            "common/yunyao_profiles/matched_profiles.csv",
            "common/yunyao_profiles/track_009_Y040_C19/ionprf_position_time_ne_profile.csv",
        ],
    },
    "G": {
        "case": "south_track009",
        "method": "v37_height_time",
        "inputs": ["south/track.csv", "south/ro_focus.json"],
    },
    "H": {
        "case": "south_track009",
        "method": "v37_delta_cn0",
        "inputs": [
            "south/ro_focus.json",
            "common/leoro_rolling_points.csv",
            "common/leoro_highrate_points.csv",
        ],
    },
    "I": {
        "case": "south_track009",
        "method": "v37_yunyao_ne",
        "inputs": [
            "common/yunyao_profiles/matched_profiles.csv",
            "common/yunyao_profiles/track_009_Y040_C19/ionprf_position_time_ne_profile.csv",
            "common/yunyao_profiles/track_009_Y040_C19/iri_profile_50_1000km_median_location.csv",
        ],
    },
    "J": {
        "case": "south_track009",
        "method": "mean_abs_velocity_20260727_fan",
        "inputs": ["south/fan_cells.csv", "south/selected_superdarn_station.csv"],
    },
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )
    velocity.north.SD_CMAP = velocity.VELOCITY_CMAP
    velocity.south.SD_CMAP = velocity.VELOCITY_CMAP
    velocity.north.MAP_TITLES = {
        145: "a  OP76 GPS G01 | north polar night | 2025-03-15 16:24-16:29 UTC",
    }
    velocity.south.SOUTH_MAP_TITLES = {
        9: "k  OP38 Galileo E21 | south polar day | 2025-03-03 12:23-12:32 UTC",
    }


def _clean_titles(fig) -> None:
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            title = ax.get_title(loc=loc)
            if title:
                title = re.sub(r"^\s*[a-tA-T]\s+", "", title).strip()
                ax.set_title(title, loc=loc)
        for text in ax.texts:
            value = text.get_text()
            if isinstance(value, str):
                text.set_text(
                    re.sub(
                        r"^\s*[a-tA-T]\s+(?=OP|Height|SuperDARN|\$|N_)",
                        "",
                        value,
                    ).strip()
                )


def _single_axis_figure(kind: str):
    if kind == "dcn0":
        fig = plt.figure(figsize=(2.35, 2.45), facecolor="white")
        ax = fig.add_axes([0.120, 0.150, 0.735, 0.770])
    else:
        fig = plt.figure(figsize=(2.35, 2.45), facecolor="white")
        ax = fig.add_axes([0.165, 0.150, 0.765, 0.770])
    return fig, ax


def _north_case() -> dict:
    return v37.with_pfisr_site_context(v37.north.load_case(145))


def _south_case() -> dict:
    return v37.south.load_south_case(9)


def _render_velocity_panel(panel: str, output: Path) -> None:
    if panel in {"A", "E"}:
        case = velocity.velocity_case(_north_case())
        hemisphere = "north"
    else:
        case = velocity.velocity_case(_south_case())
        hemisphere = "south"

    fig = plt.figure(figsize=(16.40, 10.25), dpi=260, facecolor="none")
    try:
        if panel in {"A", "F"}:
            position = (
                [0.232, 0.535, 0.242, 0.395]
                if panel == "A"
                else [0.509, 0.535, 0.242, 0.395]
            )
            ax = fig.add_axes(position)
            velocity.draw_velocity_map(ax, fig, case, hemisphere)
        else:
            position = (
                [0.118, 0.555, 0.086, 0.165]
                if panel == "E"
                else [0.884, 0.555, 0.086, 0.165]
            )
            ax = fig.add_axes(position, projection="polar")
            velocity.draw_velocity_fan(ax, case, hemisphere)
        velocity.crop_axes(fig, ax, output, dpi=600)
    finally:
        plt.close(fig)


def _render_v37_panel(panel: str, output: Path) -> None:
    kind = "dcn0" if panel in {"C", "H"} else "height"
    fig, ax = _single_axis_figure(kind)
    try:
        if panel in {"B", "C", "D"}:
            case = _north_case()
        else:
            case = _south_case()

        if panel == "B":
            v37.north.draw_height_time(ax, case, "")
        elif panel == "C":
            v37.draw_dcn0_panel_with_highrate_background(
                ax,
                case,
                "",
                panel="c",
                lugre_func=v37.north.lugre_profiles,
                leoro_func=v37.north.leoro_profiles,
            )
        elif panel == "D":
            v37.draw_yunyao_ne_panel(ax, 145, "", include_pfisr_context=True)
        elif panel == "G":
            v37.south.draw_south_height_time(ax, case, "")
        elif panel == "H":
            v37.draw_dcn0_panel_with_highrate_background(
                ax,
                case,
                "",
                panel="m",
                lugre_func=v37.south.south_lugre_profiles,
                leoro_func=v37.south.south_leoro_profiles,
            )
        elif panel == "I":
            v37.draw_yunyao_ne_panel(ax, 9, "", include_pfisr_context=False)
        else:
            raise ValueError(f"Unsupported v37 panel: {panel}")

        _clean_titles(fig)
        fig.savefig(
            output,
            dpi=600,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.035,
        )
    finally:
        plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compare_images(output: Path, reference: Path) -> dict:
    with Image.open(output) as out_image, Image.open(reference) as ref_image:
        out_rgba = out_image.convert("RGBA")
        ref_rgba = ref_image.convert("RGBA")
        dimensions_match = out_rgba.size == ref_rgba.size
        pixel_exact = dimensions_match and out_rgba.tobytes() == ref_rgba.tobytes()
        return {
            "output_dimensions_px": list(out_rgba.size),
            "reference_dimensions_px": list(ref_rgba.size),
            "dimensions_match": dimensions_match,
            "pixel_exact": pixel_exact,
            "comparison_class": (
                "pixel_exact" if pixel_exact else "scientific_reproduction_only"
            ),
        }


def _clear_own_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_dir():
            raise RuntimeError(
                f"Unexpected subdirectory in panel output; refusing recursive removal: {path}"
            )
        path.unlink()


def run(panel: str) -> Path:
    panel = str(panel).strip().upper()
    if panel not in PANEL_METADATA:
        raise ValueError(f"Unknown Fig. 5 panel {panel!r}; choose A through J.")

    _configure_style()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    QA_ROOT.mkdir(parents=True, exist_ok=True)
    temp_root = WORK_ROOT / "temporary_outputs"
    temp_root.mkdir(parents=True, exist_ok=True)

    output_dir = OUTPUT_ROOT / f"Fig5_{panel}"
    output = output_dir / f"Fig5_{panel}.png"
    reference = REFERENCE_ROOT / f"Fig5_{panel}.png"
    temp = temp_root / f"Fig5_{panel}_{os.getpid()}.png"

    _clear_own_output(output_dir)
    temp.unlink(missing_ok=True)
    try:
        if panel in {"A", "E", "F", "J"}:
            _render_velocity_panel(panel, temp)
        else:
            _render_v37_panel(panel, temp)
        if not temp.exists() or temp.stat().st_size == 0:
            raise RuntimeError(f"Renderer did not create a valid PNG for Fig5_{panel}.")
        temp.replace(output)
    except Exception:
        temp.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise

    input_rows = []
    for relative in PANEL_METADATA[panel]["inputs"]:
        path = DATA_ROOT / relative
        input_rows.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path) if path.is_file() else None,
            }
        )
    if not all(row["exists"] for row in input_rows):
        output.unlink(missing_ok=True)
        missing = [row["path"] for row in input_rows if not row["exists"]]
        raise FileNotFoundError(f"Missing panel input(s): {missing}")
    if not reference.is_file():
        output.unlink(missing_ok=True)
        raise FileNotFoundError(f"Missing reference PNG: {reference}")

    qa = {
        "panel": f"Fig5_{panel}",
        "case": PANEL_METADATA[panel]["case"],
        "method": PANEL_METADATA[panel]["method"],
        "reproduction_level": "analysis-ready numeric inputs to source-code redraw",
        "raw_telemetry_to_panel_complete": False,
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        "reference": str(reference.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": _sha256(output),
        "reference_sha256": _sha256(reference),
        "inputs": input_rows,
    }
    qa.update(_compare_images(output, reference))
    qa_path = QA_ROOT / f"Fig5_{panel}.json"
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Fig5_{panel}: {qa['comparison_class']} -> "
        f"{output.relative_to(ROOT)}"
    )
    return output
