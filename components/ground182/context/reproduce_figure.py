"""Reproduce the supplied six-panel figure from its frozen v5 inputs."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / "mpl_cache"))
sys.dont_write_bytecode = True

import cartopy
import matplotlib
import numpy as np
import pandas as pd
import PIL
from PIL import Image

cartopy.config["pre_existing_data_dir"] = DATA / "cartopy"
cartopy.config["data_dir"] = RESULTS / "cartopy_cache"
cartopy.config["cache_dir"] = RESULTS / "cartopy_tmp"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    script = ROOT / "code" / "create_global_plus_polar_zoom_station_map_v5_fig3_consistent_time_matched.py"
    spec = importlib.util.spec_from_file_location("frozen_v5", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source: {script}")
    v5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v5)
    v5.SOURCE_TABLE = DATA / "extended_data_fig7_leoro_panel_c_sources_20260617.csv"
    table = v5.load_source_table()
    specs, statuses = v5.mission_specs_from_table(table)
    missing = [name for name in v5.MISSION_ORDER if name not in statuses]
    if missing:
        raise ValueError(f"Missing source-table mission status: {missing}")
    original = v5.load_original_module()
    original.OUT_DIR = ROOT
    original.OUT_NATURE_PNG = ROOT / "Extended_Data_Observing_context.png"
    original.OUT_NATURE_PDF = ROOT / "Extended_Data_Observing_context.pdf"
    original.OUT_TRACK_SUMMARY_CSV = RESULTS / "panel_summary_R3.csv"

    # Freeze the plotted station inventory, independent of connected live sources.
    original.load_ground_roti_stations = lambda old: pd.read_csv(
        DATA / "global_plus_polar_gnss_roti_stations.csv", low_memory=False
    )
    original.load_updated_ismr_stations = lambda old: pd.read_csv(
        DATA / "global_plus_polar_eswua_chain_ismr_stations.csv", low_memory=False
    )
    original.load_current_superdarn_radars = lambda old: pd.read_csv(
        DATA / "global_plus_polar_current_superdarn_radars.csv", low_memory=False
    )
    v5.patch_panel_titles(original)
    v5.patch_orbit_panel(original, specs, statuses)
    v5.patch_verified_occultation_tracks(original)
    png, _ = original.build_nature_redraw()

    summary = pd.read_csv(original.OUT_TRACK_SUMMARY_CSV)
    expected = {
        "a": {"GNSS/ROTI station": 131, "CHAIN/ISMR station": 17, "ISR radar": 9, "Coherent radar": 7},
        "b": {"GNSS/ROTI station": 83, "CHAIN/ISMR station": 5, "ISR radar": 0, "Coherent radar": 5},
        "d": {"GNSS/ROTI station": 214, "CHAIN/ISMR station": 39, "ISR radar": 9, "Coherent radar": 12},
        "e": {"Luna tangent tracks": 527, "Luna satellites": 41},
        "f": {"Occultation profile tracks": 4196},
    }
    checks = {}
    for panel, fields in expected.items():
        row = summary.loc[summary["panel"].eq(panel)].iloc[0]
        for field, count in fields.items():
            actual = int(row[field])
            checks[f"{panel}: {field}"] = {"actual": actual, "expected": count}
            if actual != count:
                raise ValueError(f"Unexpected {panel}/{field}: {actual} != {count}")

    with Image.open(png) as current, Image.open(ROOT / "reference" / "original_v5.png") as reference:
        current_rgb = np.asarray(current.convert("RGB"))
        reference_rgb = np.asarray(reference.convert("RGB"))
        identical = current_rgb.shape == reference_rgb.shape and np.array_equal(current_rgb, reference_rgb)
        comparison = {
            "generated_size_px": list(current.size),
            "reference_size_px": list(reference.size),
            "dpi": list(current.info.get("dpi", ())),
            "pixels_identical": bool(identical),
        }
        if current_rgb.shape == reference_rgb.shape:
            difference = np.abs(current_rgb.astype(np.int16) - reference_rgb.astype(np.int16))
            comparison["mean_absolute_RGB_difference_0_255"] = float(difference.mean())
            comparison["fraction_pixels_different"] = float(np.any(difference != 0, axis=2).mean())

    qa = {
        "historical_figure": "v5 Fig.3-consistent OP74-matched six-panel observing context",
        "purpose": "Reproduce the supplied figure; station inventory is historical context.",
        "revision": "R3 desktop reproduction, 2026-10-26",
        "software": {"python": sys.version, **{
            module.__name__: module.__version__ for module in [matplotlib, cartopy, np, pd, PIL]
        }},
        "count_checks": checks,
        "reference_comparison": comparison,
        "output": png.name,
    }
    (RESULTS / "reproduction_QA_R3.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
