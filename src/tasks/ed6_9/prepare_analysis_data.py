from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PACKAGE_ROOT / "data" / "analysis_ready" / "ED6_9"

TARGET_OPS = {"38_0", "40_0", "74_0", "76_0", "77_0", "77_1", "78_1"}
EXPECTED_ROWS = {
    "38_0": 259_128,
    "40_0": 100_108,
    "74_0": 214_137,
    "76_0": 235_014,
    "77_0": 34_414,
    "77_1": 28_427,
    "78_1": 9_416,
}

SOURCES = {
    "observations": (
        SOURCE_ROOT
        / "Figure"
        / "Supplementary"
        / "phase_story"
        / "data"
        / "all_heights_master_S_track_library_cache.csv"
    ),
    "ecef_geometry": (
        SOURCE_ROOT
        / "Figure"
        / "geometry_reference_update_20260622"
        / "data"
        / "phase_story_new_reference_tangent_geometry_ecef.csv"
    ),
    "wgs84_geometry": (
        SOURCE_ROOT
        / "Figure"
        / "wgs84_tangent_height_update_20260701"
        / "data"
        / "phase_story_new_reference_tangent_geometry_wgs84_compact.csv"
    ),
}

OUTPUTS = {
    "observations": DATA_ROOT / "observations_s_phase.csv.gz",
    "ecef_geometry": DATA_ROOT / "tangent_geometry_ecef_s_phase.csv.gz",
    "wgs84_geometry": DATA_ROOT / "tangent_geometry_wgs84_s_phase.csv.gz",
}

ECEF_COLUMNS = [
    "phase",
    "op_id",
    "sat",
    "signal_id",
    "signal_name",
    "segment_id",
    "time_utc",
    "rx_gps_seconds",
    "lugre_lat",
    "lugre_lon",
    "lugre_h_tan_km",
    "lugre_tangent_lambda",
    "lugre_tangent_lambda_clip",
    "coordinate_source",
]

WGS84_COLUMNS = [
    "phase",
    "op_id",
    "sat",
    "signal_id",
    "signal_name",
    "segment_id",
    "time_utc",
    "rx_gps_seconds",
    "lugre_lat_wgs84",
    "lugre_lon_wgs84",
    "lugre_h_tan_wgs84_km",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def count_target_rows(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phase = str(row.get("phase", ""))
            op_id = str(row.get("op_id", ""))
            if phase == "S" and op_id in TARGET_OPS:
                counts[op_id] += 1
    return counts


def extract_columns(source: Path, target: Path, columns: list[str]) -> Counter[str]:
    temporary = target.with_suffix(target.suffix + ".tmp")
    counts: Counter[str] = Counter()
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as src:
            reader = csv.DictReader(src)
            missing = [column for column in columns if column not in (reader.fieldnames or [])]
            if missing:
                raise RuntimeError(f"{source.name} is missing columns: {missing}")

            with temporary.open("w", encoding="utf-8", newline="") as dst:
                writer = csv.DictWriter(
                    dst,
                    fieldnames=columns,
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in reader:
                    phase = str(row.get("phase", ""))
                    op_id = str(row.get("op_id", ""))
                    if phase != "S" or op_id not in TARGET_OPS:
                        continue
                    writer.writerow({column: row.get(column, "") for column in columns})
                    counts[op_id] += 1
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return counts


def assert_counts(label: str, counts: Counter[str]) -> None:
    actual = {op_id: int(counts.get(op_id, 0)) for op_id in sorted(TARGET_OPS)}
    if actual != EXPECTED_ROWS:
        raise RuntimeError(f"{label} row counts differ: expected {EXPECTED_ROWS}, got {actual}")


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source tables:\n" + "\n".join(missing))

    if not OUTPUTS["observations"].is_file():
        shutil.copy2(SOURCES["observations"], OUTPUTS["observations"])
    observation_counts = count_target_rows(OUTPUTS["observations"])
    assert_counts("observations", observation_counts)

    if OUTPUTS["ecef_geometry"].is_file():
        ecef_counts = count_target_rows(OUTPUTS["ecef_geometry"])
    else:
        ecef_counts = extract_columns(
            SOURCES["ecef_geometry"],
            OUTPUTS["ecef_geometry"],
            ECEF_COLUMNS,
        )
    assert_counts("ECEF geometry", ecef_counts)

    if OUTPUTS["wgs84_geometry"].is_file():
        wgs84_counts = count_target_rows(OUTPUTS["wgs84_geometry"])
    else:
        wgs84_counts = extract_columns(
            SOURCES["wgs84_geometry"],
            OUTPUTS["wgs84_geometry"],
            WGS84_COLUMNS,
        )
    assert_counts("WGS84 geometry", wgs84_counts)

    result = {
        "data_level": "analysis-ready",
        "row_counts_by_op": EXPECTED_ROWS,
        "files": {
            label: {
                "path": str(path.relative_to(PACKAGE_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for label, path in OUTPUTS.items()
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
