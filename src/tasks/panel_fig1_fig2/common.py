from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "panel_fig1_fig2"
QA_ROOT = ROOT / "work" / "qa" / "panel_fig1_fig2"
WORK.mkdir(parents=True, exist_ok=True)
QA_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "mplconfig"))
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True


PANEL_META: dict[str, dict[str, str]] = {
    "Fig1_B": {
        "figure": "Fig1",
        "level": "analysis-ready grid-cell reproduction",
        "exactness": (
            "Scientific values and panel semantics are reproduced from the packaged "
            "artifact-screened common-grid table; the manually assembled manuscript "
            "raster is not claimed to be byte-identical."
        ),
    },
    "Fig1_C": {
        "figure": "Fig1",
        "level": "analysis-ready grid-cell reproduction",
        "exactness": (
            "Scientific values and Robinson-map semantics are reproduced from the "
            "packaged artifact-screened grid table; coastline/font rasterization may "
            "differ from the manually assembled manuscript raster."
        ),
    },
    "Fig1_D": {
        "figure": "Fig1",
        "level": "analysis-ready grid-cell reproduction",
        "exactness": (
            "Scientific values and Robinson-map semantics are reproduced from the "
            "packaged ROTI grid table; coastline/font rasterization may differ from "
            "the manually assembled manuscript raster."
        ),
    },
    "Fig2_A": {
        "figure": "Fig2",
        "level": "analysis-ready point-track reproduction",
        "exactness": (
            "North-polar tracks and highlighted events are redrawn from packaged "
            "WGS84 point data; the package uses its bundled Natural Earth coastline."
        ),
    },
    "Fig2_B": {
        "figure": "Fig2",
        "level": "analysis-ready point-track reproduction",
        "exactness": (
            "South-polar tracks and highlighted events are redrawn from packaged "
            "WGS84 point data; the package uses its bundled Natural Earth coastline."
        ),
    },
}
for _idx in range(1, 7):
    PANEL_META[f"Fig2_C{_idx}"] = {
        "figure": "Fig2",
        "level": "analysis-ready selected-track reproduction",
        "exactness": (
            "C/N0 samples, WGS84 tangent heights, frequency colours and ionospheric "
            "layer context are redrawn from the packaged selected-event point data."
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def output_paths(panel_id: str) -> tuple[Path, Path]:
    output_dir = ROOT / "outputs" / panel_id
    return output_dir / f"{panel_id}.png", QA_ROOT / f"{panel_id}.json"


def reference_path(panel_id: str) -> Path:
    figure = PANEL_META[panel_id]["figure"]
    return ROOT / "reference" / "panels" / figure / f"{panel_id}.png"


def clear_own_output(panel_id: str) -> tuple[Path, Path]:
    png_path, json_path = output_paths(panel_id)
    output_dir = png_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.unlink(missing_ok=True)
    return png_path, json_path


def validate_png(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Renderer did not create a non-empty PNG: {path}")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise RuntimeError(f"Output is not a PNG: {path}")
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if width < 64 or height < 64:
        raise RuntimeError(f"PNG dimensions are unexpectedly small: {width}x{height}")
    return {
        "bytes": path.stat().st_size,
        "width_px": width,
        "height_px": height,
        "sha256": sha256(path),
    }


def compare_to_reference(generated: Path, reference: Path) -> dict[str, object]:
    if not reference.is_file():
        return {"available": False, "reference": str(reference.relative_to(ROOT))}
    with Image.open(generated) as left_image, Image.open(reference) as right_image:
        left_rgba = left_image.convert("RGBA")
        right_rgba = right_image.convert("RGBA")
        left_shape = left_rgba.size
        right_shape = right_rgba.size
        background = Image.new("RGBA", left_rgba.size, "white")
        left_rgb = Image.alpha_composite(background, left_rgba).convert("RGB")
        background = Image.new("RGBA", right_rgba.size, "white")
        right_rgb = Image.alpha_composite(background, right_rgba).convert("RGB")
        left = _content_normalized_array(left_rgb)
        right = _content_normalized_array(right_rgb)
    mean_abs = float(np.abs(left - right).mean() / 255.0)
    left_gray = left.mean(axis=2).ravel()
    right_gray = right.mean(axis=2).ravel()
    correlation = float(np.corrcoef(left_gray, right_gray)[0, 1])
    return {
        "available": True,
        "reference": str(reference.relative_to(ROOT)),
        "reference_sha256": sha256(reference),
        "byte_identical": sha256(generated) == sha256(reference),
        "generated_size_px": list(left_shape),
        "reference_size_px": list(right_shape),
        "normalized_mean_absolute_error": mean_abs,
        "normalized_luminance_correlation": correlation,
        "comparison_method": (
            "non-white content bounding boxes independently fitted to a 256 px "
            "white canvas; intended as a rough visual QA metric, not an equivalence test"
        ),
    }


def _content_normalized_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    content = np.any(rgb < 250, axis=2)
    if content.any():
        rows, cols = np.where(content)
        box = (
            int(cols.min()),
            int(rows.min()),
            int(cols.max()) + 1,
            int(rows.max()) + 1,
        )
        image = image.crop(box)
    contained = ImageOps.contain(image, (248, 248), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (256, 256), "white")
    offset = ((256 - contained.width) // 2, (256 - contained.height) // 2)
    canvas.paste(contained, offset)
    return np.asarray(canvas, dtype=np.float32)


def run_panel(panel_id: str) -> None:
    if panel_id not in PANEL_META:
        raise RuntimeError(f"Unknown panel task: {panel_id}")

    png_path, json_path = clear_own_output(panel_id)
    try:
        if panel_id.startswith("Fig1_"):
            from .fig1 import render
        else:
            from .fig2 import render

        scientific_checks = render(panel_id, png_path)
        png_validation = validate_png(png_path)
        visual_comparison = compare_to_reference(png_path, reference_path(panel_id))
        sidecar = {
            "panel_id": panel_id,
            **PANEL_META[panel_id],
            "output": str(png_path.relative_to(ROOT)),
            "png_validation": png_validation,
            "scientific_checks": scientific_checks,
            "visual_reference_comparison": visual_comparison,
        }
        json_path.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        output_children = list(png_path.parent.iterdir())
        if output_children != [png_path]:
            raise RuntimeError(
                f"{panel_id} must leave exactly one PNG and no other files in "
                f"{png_path.parent}; found {output_children}"
            )
        print(f"{panel_id}: {png_path}")
        print(f"validation: {json_path}")
    except Exception:
        png_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise
