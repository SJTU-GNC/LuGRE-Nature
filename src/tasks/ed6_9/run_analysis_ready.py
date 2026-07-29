from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "data" / "analysis_ready" / "ED6_9"
CARTOPY_DATA = TASK_ROOT / "cartopy_data"

OBSERVATIONS = DATA_ROOT / "observations_s_phase.csv.gz"
ECEF_GEOMETRY = DATA_ROOT / "tangent_geometry_ecef_s_phase.csv.gz"
WGS84_GEOMETRY = DATA_ROOT / "tangent_geometry_wgs84_s_phase.csv.gz"
BUILDER_SCRIPT = TASK_ROOT / "wgs84_support.py"
RAW_OBSERVATION_BUILDER = (
    PACKAGE_ROOT
    / "src"
    / "raw_pipeline"
    / "lugre"
    / "build_observations_s_phase.py"
)
REBUILT_OBSERVATIONS = (
    PACKAGE_ROOT
    / "work"
    / "derived"
    / "lugre_raw"
    / "observations_s_phase_from_raw.csv"
)
REBUILT_OBSERVATIONS_QA = REBUILT_OBSERVATIONS.parent / "qa_summary.json"
EXPECTED_REBUILT_OBSERVATIONS_SHA256 = (
    "85AFD8079B68496F99E6DEB2B7D26C7BC1458B5CDEB08D21E5F9B144DED8049C"
)
_RAW_REBUILD_QA_CACHE: dict[str, object] | None = None

MIN_DT_S = 0.5
MAX_DT_S = 1.5
WINDOW_S = 60.0
MIN_WINDOW_SAMPLES = 10

TASKS = {
    "ED6": {
        "group": "S2",
        "ops": ["38_0"],
        "output": "Extended_Data_Fig_06_s_phase_day1_op38_R1.png",
        "reference_sha256": "A38E7B98191B6769C2C899E481E84B29B499689FA0F2435AA30DB5EA3359BC51",
    },
    "ED7": {
        "group": "S3",
        "ops": ["40_0"],
        "output": "Extended_Data_Fig_07_s_phase_day2_op40_R1.png",
        "reference_sha256": "8962E6BB854DFF1C08A6A4C04642CE829BFE46CD736FE8EEFCA02EE5E38D5029",
    },
    "ED8": {
        "group": "S4",
        "ops": ["74_0"],
        "output": "Extended_Data_Fig_08_s_phase_day3_op74_R1.png",
        "reference_sha256": "018AE8BA38FDF85A473982A1BA72742BC47EA3DC1EBF868DE0831D83E70A53B3",
    },
    "ED9": {
        "group": "S5",
        "ops": ["76_0", "77_0", "77_1", "78_1"],
        "output": "Extended_Data_Fig_09_s_phase_day4_op76_op78_R1.png",
        "reference_sha256": "DD0EEB05C42727757C2EA796D6CBD4C9C7BD3157759A33CBBAE80C7EF88B457C",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rolling_std_min_count(
    time_s: np.ndarray,
    values: np.ndarray,
    window_s: float,
    min_count: int,
) -> np.ndarray:
    """Trailing time-window sample standard deviation used by the R1 figures."""
    out = np.full(len(values), np.nan, dtype=float)
    left = 0
    count = 0
    total = 0.0
    total2 = 0.0
    for right in range(len(values)):
        value = values[right]
        if np.isfinite(value):
            count += 1
            total += value
            total2 += value * value
        while left <= right and time_s[right] - time_s[left] > window_s:
            old_value = values[left]
            if np.isfinite(old_value):
                count -= 1
                total -= old_value
                total2 -= old_value * old_value
            left += 1
        if count >= min_count:
            variance = (total2 - total * total / count) / (count - 1)
            out[right] = math.sqrt(max(0.0, variance))
    return out


def add_trailing_rolling_roti(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing 60-s ROTI from the analysis-ready ROT column."""
    out = df.copy()
    output_column = "roti_1min_independent"
    out[output_column] = np.nan
    required = [
        "phase",
        "op_id",
        "sat",
        "signal_id",
        "segment_id",
        "time_utc",
        "rot_tecu_per_min",
    ]
    if any(column not in out.columns for column in required) or out.empty:
        return out

    key_columns = ["phase", "op_id", "sat", "signal_id", "segment_id"]
    ordered = out.sort_values(key_columns + ["time_utc"], kind="mergesort")

    for _, group in ordered.groupby(key_columns, sort=False, dropna=False):
        indices = group.index.to_numpy()
        time = pd.to_datetime(group["time_utc"], utc=True, errors="coerce")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        time_s = (time - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
        rot = pd.to_numeric(group["rot_tecu_per_min"], errors="coerce").to_numpy(float)
        finite = np.isfinite(time_s) & np.isfinite(rot)

        start = 0
        while start < len(group):
            while start < len(group) and not finite[start]:
                start += 1
            if start >= len(group):
                break

            end = start + 1
            while end < len(group):
                dt = time_s[end] - time_s[end - 1]
                if not finite[end] or dt < MIN_DT_S or dt > MAX_DT_S:
                    break
                end += 1

            if end - start >= MIN_WINDOW_SAMPLES:
                rolling = rolling_std_min_count(
                    time_s[start:end],
                    rot[start:end],
                    WINDOW_S,
                    MIN_WINDOW_SAMPLES,
                )
                out.loc[indices[start:end], output_column] = rolling
            start = end

    return out


def patch_roti_title(function, roti_column: str):
    def wrapped(ax, frame, y_column, title, *args, **kwargs):
        if y_column == roti_column:
            title = "ROTI 60-s rolling"
        return function(ax, frame, y_column, title, *args, **kwargs)

    return wrapped


def validate_inputs(observations: Path) -> None:
    missing = [
        str(path)
        for path in (observations, ECEF_GEOMETRY, WGS84_GEOMETRY, BUILDER_SCRIPT)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing packaged dependencies:\n" + "\n".join(missing))
    if not CARTOPY_DATA.is_dir():
        raise FileNotFoundError(f"Missing packaged Cartopy data: {CARTOPY_DATA}")


def read_task_rows(path: Path, ops: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        op_id = chunk["op_id"].astype(str)
        phase = chunk["phase"].astype(str)
        selected = chunk.loc[phase.eq("S") & op_id.isin(ops)].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError(f"No requested S-phase rows found in {path}")
    return pd.concat(frames, ignore_index=True)


def load_task_observations(
    mod,
    ops: set[str],
    observations: Path,
) -> pd.DataFrame:
    frame = read_task_rows(observations, ops)
    frame["phase"] = frame["phase"].astype(str)
    frame["op_id"] = frame["op_id"].astype(str)
    frame["sat"] = frame["sat"].astype(str)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
    numeric_columns = [
        "rx_gps_seconds",
        "signal_id",
        "segment_id",
        "cn0_dbhz",
        "plot_cn0_detrended_db",
        "stec_tecu",
        "rot_tecu_per_min",
        "roti_5min",
        "lugre_lat",
        "lugre_lon",
        "lugre_h_tan_km",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["phase", "op_id", "sat", "time_utc"]).sort_values(
        ["phase", "op_id", "sat", "signal_id", "segment_id", "time_utc"]
    )
    return add_trailing_rolling_roti(frame)


def load_task_ecef_geometry(ops: set[str]) -> pd.DataFrame:
    frame = read_task_rows(ECEF_GEOMETRY, ops)
    frame["phase"] = frame["phase"].astype(str)
    frame["op_id"] = frame["op_id"].astype(str)
    frame["sat"] = frame["sat"].astype(str)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
    numeric_columns = [
        "rx_gps_seconds",
        "signal_id",
        "segment_id",
        "lugre_lat",
        "lugre_lon",
        "lugre_h_tan_km",
        "lugre_tangent_lambda",
        "lugre_tangent_lambda_clip",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(
        subset=["phase", "op_id", "sat", "time_utc", "lugre_lat", "lugre_lon", "lugre_h_tan_km"]
    ).sort_values(["phase", "op_id", "sat", "signal_id", "segment_id", "time_utc"])


def _load_validated_rebuilt_observations() -> dict[str, object]:
    if not REBUILT_OBSERVATIONS.is_file():
        raise RuntimeError(
            f"Rebuilt observations are missing: {REBUILT_OBSERVATIONS}"
        )
    if not REBUILT_OBSERVATIONS_QA.is_file():
        raise RuntimeError(
            f"Rebuilt-observation QA is missing: {REBUILT_OBSERVATIONS_QA}"
        )

    qa = json.loads(REBUILT_OBSERVATIONS_QA.read_text(encoding="utf-8"))
    if qa.get("status") != "pass":
        raise RuntimeError(
            "Raw-observation QA did not pass:\n"
            + json.dumps(qa, indent=2, sort_keys=True)
        )
    actual_rebuilt_hash = sha256(REBUILT_OBSERVATIONS)
    if (
        actual_rebuilt_hash != EXPECTED_REBUILT_OBSERVATIONS_SHA256
        or qa.get("output_sha256") != EXPECTED_REBUILT_OBSERVATIONS_SHA256
    ):
        raise RuntimeError(
            "Raw-observation rebuild SHA256 mismatch: "
            f"actual={actual_rebuilt_hash}, "
            f"qa={qa.get('output_sha256')}, "
            f"expected={EXPECTED_REBUILT_OBSERVATIONS_SHA256}"
        )
    return qa


def rebuild_observations_from_raw() -> dict[str, object]:
    global _RAW_REBUILD_QA_CACHE
    if _RAW_REBUILD_QA_CACHE is not None:
        return _RAW_REBUILD_QA_CACHE

    reuse_validated = os.environ.get(
        "LUGRE_REUSE_VALIDATED_RAW_REBUILD", ""
    ).strip() == "1"
    if not reuse_validated:
        if not RAW_OBSERVATION_BUILDER.is_file():
            raise FileNotFoundError(
                f"Missing raw-observation builder: {RAW_OBSERVATION_BUILDER}"
            )
        raw_builder = load_module(
            RAW_OBSERVATION_BUILDER,
            "lugre_raw_observation_builder",
        )
        return_code = int(raw_builder.main([]))
        if return_code != 0:
            raise RuntimeError(
                f"Raw-observation builder failed with exit code {return_code}"
            )

    qa = _load_validated_rebuilt_observations()
    _RAW_REBUILD_QA_CACHE = qa
    return qa


def build_one(
    task: str,
    output: Path,
    *,
    rebuild_derived: bool = False,
) -> dict[str, object]:
    task = task.upper()
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")

    raw_rebuild_qa: dict[str, object] | None = None
    observations = OBSERVATIONS
    if rebuild_derived:
        raw_rebuild_qa = rebuild_observations_from_raw()
        observations = REBUILT_OBSERVATIONS
    validate_inputs(observations)

    with tempfile.TemporaryDirectory(prefix=f"lugre_{task.lower()}_") as temporary:
        temporary_root = Path(temporary)
        os.environ.setdefault("MPLCONFIGDIR", str(temporary_root / "mplconfig"))

        import cartopy

        cartopy.config["data_dir"] = CARTOPY_DATA

        builder = load_module(BUILDER_SCRIPT, f"lugre_{task.lower()}_wgs84_support")
        builder.WGS84_GEOMETRY = WGS84_GEOMETRY
        task_ops = set(str(op_id) for op_id in TASKS[task]["ops"])
        builder.WANTED_OPS = task_ops
        mod = builder.load_phase_story_module()

        mod.ROOT = DATA_ROOT
        mod.S_CACHE = observations
        mod.RECOMPUTED_TANGENT_CACHE = ECEF_GEOMETRY
        mod.LINE_GEOMETRY_CSV = DATA_ROOT / "unused_line_geometry.csv"
        mod.OUT_DIR = temporary_root / "figures"
        mod.OUT_DIR.mkdir(parents=True, exist_ok=True)

        mod.add_independent_minute_roti = add_trailing_rolling_roti
        mod.add_time_panel = patch_roti_title(
            mod.add_time_panel,
            mod.ROTI_INDEPENDENT_COL,
        )
        mod.add_datetime_time_panel = patch_roti_title(
            mod.add_datetime_time_panel,
            mod.ROTI_INDEPENDENT_COL,
        )

        mod.setup_style()
        builder.install_htan_only_altitude_panels(mod)

        s_frame = load_task_observations(mod, task_ops, observations)
        tangent_frame = load_task_ecef_geometry(task_ops)
        tangent_frame, geometry_qa = builder.attach_wgs84_geometry(tangent_frame)

        group_code = TASKS[task]["group"]
        groups = [
            dict(group)
            for group in builder.candidate_groups(mod)
            if str(group["figure"]) == group_code
        ]
        if len(groups) != 1:
            raise RuntimeError(f"Expected one group for {task}, found {len(groups)}")
        group = groups[0]
        group["out_stem"] = str(group["out_stem"]).replace(
            "_wgs84_htan_map_candidate",
            "_wgs84_htan_map_rolling60s_roti_candidate",
        )

        if len(group["ops"]) > 1:
            result = mod.plot_s_compact_group(s_frame, tangent_frame, group)
        else:
            result = mod.plot_s_group(s_frame, tangent_frame, group)

        generated = mod.OUT_DIR / str(result["png"])
        if not generated.is_file():
            raise RuntimeError(f"Plot did not create {generated}")

        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.with_name(output.name + ".tmp")
        try:
            shutil.copy2(generated, staged)
            from PIL import Image

            with Image.open(staged) as image:
                image.verify()
            staged.replace(output)
        finally:
            if staged.exists():
                staged.unlink()

        actual_hash = sha256(output)
        reference_hash = str(TASKS[task]["reference_sha256"])
        return {
            "task": task,
            "data_level": (
                "raw-rebuilt-observations-with-processed-fallbacks"
                if rebuild_derived
                else "packaged-analysis-ready"
            ),
            "observations": str(observations),
            "raw_observation_qa_status": (
                raw_rebuild_qa.get("status")
                if raw_rebuild_qa is not None
                else None
            ),
            "raw_observation_end_to_end_status": (
                raw_rebuild_qa.get("end_to_end_status")
                if raw_rebuild_qa is not None
                else None
            ),
            "geometry_source": "packaged-analysis-ready-fallback",
            "output": str(output),
            "points": int(result["points"]),
            "windows": str(result["windows"]),
            "geometry_qa_rows": int(len(geometry_qa)),
            "sha256": actual_hash,
            "reference_sha256": reference_hash,
            "matches_reference_sha256": actual_hash == reference_hash,
        }


def default_output(task: str) -> Path:
    config = TASKS[task.upper()]
    return PACKAGE_ROOT / "outputs" / task.upper() / str(config["output"])


def cli(fixed_task: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute one LuGRE Extended Data Figure from packaged tables, "
            "optionally rebuilding observations from raw telemetry first."
        )
    )
    if fixed_task is None:
        parser.add_argument("--figure", required=True, choices=sorted(TASKS))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help=(
            "Rebuild S-phase observations from packaged raw telemetry before "
            "plotting; tangent geometry remains a documented packaged fallback."
        ),
    )
    args = parser.parse_args()

    task = fixed_task or args.figure
    output = args.output or default_output(task)
    result = build_one(
        task,
        output,
        rebuild_derived=args.rebuild_derived,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
