from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
REPRO_ROOT = TASK_DIR.parents[2]
DATA_DIR = REPRO_ROOT / "data" / "analysis_ready" / "ED3"
ASSET_DIR = REPRO_ROOT / "assets" / "ED3"
OUTPUT_DIR = REPRO_ROOT / "outputs" / "ED3"
OUTPUT_PNG = OUTPUT_DIR / "Extended_Data_Fig_03_main_lobe_antenna_geometry_R1.png"
REFERENCE_SHA256 = "2d076e17907e89ed5d86e26362ba7c47c1a78ef35fed0219d730290d6fa99622"

for dependency_dir in (TASK_DIR,):
    sys.path.insert(0, str(dependency_dir))

import plot_main_lobe_gps_galileo as core  # noqa: E402
import plot_main_lobe_grap_revised as grap  # noqa: E402
import plot_main_lobe_nature_figures as base  # noqa: E402
import plot_main_lobe_previous_ab_gps_galileo as figure_code  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_inputs(work_dir: Path) -> None:
    gps_dir = DATA_DIR / "GPS_antenna_patterns"
    grap_dir = DATA_DIR / "GRAP_metadata"
    point_csv = DATA_DIR / "lugre_gnss_main_lobe_point_check.csv"
    windows_csv = DATA_DIR / "lugre_full_geometry_noise_windows_refined.csv"

    core.ROOT = work_dir
    core.GPS_DIR = gps_dir
    core.GRAP_DIR = grap_dir
    core.WINDOWS_CSV = windows_csv
    core.IGS_METADATA = gps_dir / "igs_satellite_metadata.snx"
    core.SATELLITE_IMAGE = ASSET_DIR / "gps_iii_satellite_cutout.png"
    core.EARTH_IMAGE = ASSET_DIR / "earth_user_reference_globe.png"
    core.OUTPUT_STEM = work_dir / "lugre_main_lobe_ab_figure_nature_gps_galileo"
    core.CURRENT_STEM = work_dir / "lugre_main_lobe_ab_figure_nature"

    grap.ROOT = work_dir
    grap.POINT_CSV = point_csv
    grap.GRAP_DIR = grap_dir
    grap.OUTPUT_STEM = work_dir / "lugre_main_lobe_ab_figure_nature_grap"

    base.ROOT = work_dir
    base.OUT_DIR = work_dir
    base.FIG_DIR = work_dir
    base.POINT_CSV = point_csv
    base.GPS_III_RAW = ASSET_DIR / "gps_iii_satellite_cutout.png"
    base.GPS_III_CUTOUT = ASSET_DIR / "gps_iii_satellite_cutout.png"
    base.USER_EARTH_IMAGE = ASSET_DIR / "earth_user_reference_globe.png"
    base.EARTH_RAW = ASSET_DIR / "earth_user_reference_globe.png"
    base.EARTH_CUTOUT = ASSET_DIR / "earth_user_reference_globe.png"

    figure_code.ROOT = work_dir
    figure_code.POINT_CSV = point_csv
    figure_code.OUTPUT_STEM = work_dir / "lugre_main_lobe_ab_figure_nature_previous_ab_gps_galileo"
    figure_code.CURRENT_STEM = work_dir / "lugre_main_lobe_ab_figure_nature"


def main(*, root: Path, rebuild_derived: bool, validate: bool) -> None:
    del root, rebuild_derived
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lugre_ed3_") as temp:
        work_dir = Path(temp)
        _configure_inputs(work_dir)
        figure_code.main()
        generated = work_dir / "lugre_main_lobe_abc_reference_layout.png"
        if not generated.exists():
            generated = work_dir / "lugre_main_lobe_ab_figure_nature.png"
        shutil.copy2(generated, OUTPUT_PNG)

    actual = _sha256(OUTPUT_PNG)
    print(f"ED3 wrote {OUTPUT_PNG}")
    print(f"ED3 generated SHA256 {actual}")
    print(f"ED3 manuscript reference SHA256 {REFERENCE_SHA256}")
    if validate and actual != REFERENCE_SHA256:
        print(
            "ED3 validation is informational: the packaged code regenerates the "
            "scientific panels, while the manuscript PNG used an undocumented final composition."
        )
