#!/usr/bin/env python3
"""Rebuild the S-phase LuGRE observation table from raw telemetry.

The raw TLM files are authoritative for receiver time, satellite/signal
identity, C/N0, pseudorange, and accumulated Doppler.  A dual-frequency
carrier/code combination is also used to recompute STEC, ROT, and ROTI where
the result agrees with the accepted processed table.

Some accepted fields cannot be reconstructed from the raw material currently
present in this package:

* the exact robust fourth-order C/N0 detrending implementation;
* the precise-orbit tangent geometry generation chain;
* a small set of manually/implicitly joined carrier-continuity boundaries.

For those rows and fields, the accepted processed table is retained as an
explicit fallback.  QA and provenance files make every fallback count visible;
the script never silently presents a copied processed value as raw-recomputed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RAW_TLM_DIR = PACKAGE_ROOT / "data" / "raw" / "lugre" / "L0" / "TLM"
REFERENCE_CSV = (
    PACKAGE_ROOT
    / "data"
    / "analysis_ready"
    / "ED6_9"
    / "observations_s_phase.csv.gz"
)
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "work" / "derived" / "lugre_raw"

OUTPUT_COLUMNS = [
    "phase",
    "op_id",
    "sat",
    "signal_id",
    "segment_id",
    "time_utc",
    "rx_gps_seconds",
    "cn0_dbhz",
    "plot_cn0_detrended_db",
    "stec_tecu",
    "rot_tecu_per_min",
    "roti_5min",
    "lugre_lat",
    "lugre_lon",
    "lugre_h_tan_km",
    "daynight",
]

KEY_COLUMNS = ["phase", "op_id", "sat", "signal_id", "rx_key"]
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
NUMBER_PATTERN = rf"(?:{FLOAT_PATTERN}|[-+]?(?:nan|inf(?:inity)?))"
RX_TIME_RE = re.compile(rf"\brxTime:\s*(?P<rx>{FLOAT_PATTERN})")
MEASURE_RE = re.compile(
    rf"\bsvid:\s*(?P<svid>\d+)"
    rf"\s+prRaw:\s*(?P<pr>{NUMBER_PATTERN})"
    rf"\s+cn0:\s*(?P<cn0>{NUMBER_PATTERN})"
    rf"\s+signalId:\s*(?P<signal>\d+)"
    rf"\s+fdRaw:\s*(?P<fd>{NUMBER_PATTERN})"
    rf"\s+accDoppler:\s*(?P<acc>{NUMBER_PATTERN})",
    re.IGNORECASE,
)
OP_RE = re.compile(r"_S_OP(?P<op>\d+_\d+)\.txt(?:\.gz)?$", re.IGNORECASE)

GPS_EPOCH = pd.Timestamp("1980-01-06T00:00:00Z")
GPS_MINUS_UTC_SECONDS = 18.0
FREQ_HIGH_HZ = 1_575_420_000.0
FREQ_LOW_HZ = 1_176_450_000.0
STEC_FACTOR = (
    1.0
    / (40.3 * (1.0 / FREQ_LOW_HZ**2 - 1.0 / FREQ_HIGH_HZ**2))
    / 1.0e16
)

STEC_TOLERANCE = 1.0e-8
ROT_TOLERANCE = 1.0e-7
ROTI_TOLERANCE = 1.0e-7


@dataclass(frozen=True)
class SourceFile:
    path: Path
    op_id: str


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def discover_sources(selected_ops: set[str] | None) -> list[SourceFile]:
    sources: list[SourceFile] = []
    for path in sorted(RAW_TLM_DIR.glob("TLM_RAW_*_S_OP*.txt.gz")):
        match = OP_RE.search(path.name)
        if match is None:
            continue
        op_id = match.group("op")
        if selected_ops is None or op_id in selected_ops:
            sources.append(SourceFile(path=path, op_id=op_id))

    if not sources:
        scope = "all operations" if selected_ops is None else sorted(selected_ops)
        raise FileNotFoundError(
            f"No S-phase TLM raw text files found for {scope} under {RAW_TLM_DIR}"
        )

    found_ops = {source.op_id for source in sources}
    if selected_ops is not None and found_ops != selected_ops:
        missing = sorted(selected_ops - found_ops)
        raise FileNotFoundError(f"Missing raw TLM file(s) for operation(s): {missing}")
    return sources


def signal_to_satellite(signal_id: int, svid: int) -> str:
    if signal_id in (0, 1):
        return f"G{svid:02d}"
    if signal_id in (2, 3):
        return f"E{svid:02d}"
    raise ValueError(f"Unsupported signalId={signal_id} for svid={svid}")


def parse_tlm_file(source: SourceFile) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[tuple[object, ...]] = []
    line_count = 0
    empty_measure_lines = 0
    malformed_measure_lines = 0
    nonfinite_measure_rows = 0

    opener = gzip.open if source.path.suffix.lower() == ".gz" else Path.open
    with opener(source.path, "rt", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            line_count += 1
            rx_match = RX_TIME_RE.search(line)
            if rx_match is None:
                continue
            rx_seconds = float(rx_match.group("rx"))
            matches = list(MEASURE_RE.finditer(line))
            stated_measure_count = len(re.findall(r"\bsvid:", line))
            if stated_measure_count != len(matches):
                malformed_measure_lines += 1
                continue
            if not matches:
                if re.search(r"\bmeasures:\s*\[\s*\]", line):
                    empty_measure_lines += 1
                continue

            for match in matches:
                svid = int(match.group("svid"))
                signal_id = int(match.group("signal"))
                pr_raw = float(match.group("pr"))
                cn0 = float(match.group("cn0"))
                acc_doppler = float(match.group("acc"))
                if not np.isfinite([pr_raw, cn0, acc_doppler]).all():
                    nonfinite_measure_rows += 1
                    continue
                rows.append(
                    (
                        "S",
                        source.op_id,
                        signal_to_satellite(signal_id, svid),
                        signal_id,
                        rx_seconds,
                        cn0,
                        pr_raw,
                        acc_doppler,
                        source.path.name,
                        line_number,
                    )
                )

    if malformed_measure_lines:
        raise ValueError(
            f"{source.path.name}: {malformed_measure_lines} non-empty measures "
            "line(s) could not be parsed completely"
        )

    frame = pd.DataFrame.from_records(
        rows,
        columns=[
            "phase",
            "op_id",
            "sat",
            "signal_id",
            "rx_gps_seconds",
            "cn0_dbhz",
            "pr_raw_m",
            "acc_doppler_m",
            "raw_source_file",
            "raw_line_number",
        ],
    )
    manifest_row = {
        "op_id": source.op_id,
        "file": source.path.relative_to(PACKAGE_ROOT).as_posix(),
        "bytes": source.path.stat().st_size,
        "sha256": sha256_file(source.path),
        "text_lines": line_count,
        "empty_measure_lines": empty_measure_lines,
        "nonfinite_measure_rows_skipped": nonfinite_measure_rows,
        "parsed_measure_rows": len(frame),
    }
    return frame, manifest_row


def parse_raw_sources(
    sources: Iterable[SourceFile],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for source in sources:
        print(f"[raw] parsing {source.path.name}", flush=True)
        frame, manifest_row = parse_tlm_file(source)
        frames.append(frame)
        manifest_rows.append(manifest_row)

    raw = pd.concat(frames, ignore_index=True)
    raw["rx_key"] = raw["rx_gps_seconds"].round(6)
    raw.sort_values(
        ["phase", "op_id", "sat", "signal_id", "rx_gps_seconds"],
        kind="mergesort",
        inplace=True,
        ignore_index=True,
    )

    group_columns = ["phase", "op_id", "sat", "signal_id"]
    elapsed = raw.groupby(group_columns, sort=False)["rx_gps_seconds"].diff()
    starts_segment = elapsed.isna() | elapsed.gt(10.0)
    raw["segment_id"] = (
        starts_segment.groupby(
            [raw[column] for column in group_columns], sort=False
        )
        .cumsum()
        .astype("int64")
    )

    utc = GPS_EPOCH + pd.to_timedelta(
        raw["rx_gps_seconds"] - GPS_MINUS_UTC_SECONDS, unit="s"
    )
    raw["time_utc"] = utc.dt.floor("s").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    duplicate_count = int(raw.duplicated(KEY_COLUMNS, keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"Raw telemetry contains {duplicate_count} rows on duplicated keys "
            f"{KEY_COLUMNS}"
        )

    manifest = pd.DataFrame(manifest_rows)
    return raw, manifest


def read_reference(selected_ops: set[str] | None) -> pd.DataFrame:
    if not REFERENCE_CSV.is_file():
        raise FileNotFoundError(f"Accepted processed table not found: {REFERENCE_CSV}")

    reference = pd.read_csv(REFERENCE_CSV, low_memory=False)
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in reference]
    if missing_columns:
        raise ValueError(
            f"Accepted processed table is missing columns: {missing_columns}"
        )
    if selected_ops is not None:
        reference = reference[reference["op_id"].astype(str).isin(selected_ops)].copy()
    reference["op_id"] = reference["op_id"].astype(str)
    reference["rx_key"] = reference["rx_gps_seconds"].round(6)
    reference["_reference_order"] = np.arange(len(reference), dtype=np.int64)

    duplicate_count = int(reference.duplicated(KEY_COLUMNS, keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"Accepted table contains {duplicate_count} rows on duplicated keys "
            f"{KEY_COLUMNS}"
        )
    return reference


def compare_key_coverage(
    raw: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    coverage = reference[KEY_COLUMNS].merge(
        raw[KEY_COLUMNS],
        how="outer",
        on=KEY_COLUMNS,
        indicator=True,
        validate="one_to_one",
    )
    counts = {
        "both": int((coverage["_merge"] == "both").sum()),
        "reference_only": int((coverage["_merge"] == "left_only").sum()),
        "raw_only": int((coverage["_merge"] == "right_only").sum()),
    }
    if counts["reference_only"] or counts["raw_only"]:
        raise ValueError(
            "Raw/reference key coverage is not one-to-one: "
            f"{json.dumps(counts, sort_keys=True)}"
        )

    merged = reference.merge(
        raw,
        how="inner",
        on=KEY_COLUMNS,
        suffixes=("_reference", "_raw"),
        validate="one_to_one",
        sort=False,
    )
    merged.sort_values("_reference_order", inplace=True, ignore_index=True)
    return merged, counts


def build_stec_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    high = raw[raw["signal_id"].isin([0, 2])].copy()
    high["low_signal_id"] = high["signal_id"] + 1
    low = raw[raw["signal_id"].isin([1, 3])][
        [
            "phase",
            "op_id",
            "sat",
            "signal_id",
            "rx_key",
            "segment_id",
            "pr_raw_m",
            "acc_doppler_m",
        ]
    ].copy()
    low.rename(
        columns={
            "signal_id": "low_signal_id",
            "segment_id": "low_segment_id",
            "pr_raw_m": "low_pr_raw_m",
            "acc_doppler_m": "low_acc_doppler_m",
        },
        inplace=True,
    )

    pair_keys = ["phase", "op_id", "sat", "low_signal_id", "rx_key"]
    pairs = high.merge(
        low,
        how="inner",
        on=pair_keys,
        validate="one_to_one",
        suffixes=("", "_unused"),
    )
    pairs.rename(
        columns={
            "segment_id": "high_segment_id",
            "pr_raw_m": "high_pr_raw_m",
            "acc_doppler_m": "high_acc_doppler_m",
        },
        inplace=True,
    )
    pairs["pair_arc_id"] = (
        pairs["phase"].astype(str)
        + "|"
        + pairs["op_id"].astype(str)
        + "|"
        + pairs["sat"].astype(str)
        + "|sig"
        + pairs["signal_id"].astype(str)
        + "|high"
        + pairs["high_segment_id"].astype(str)
        + "|low"
        + pairs["low_segment_id"].astype(str)
    )
    pairs["code_stec_tecu"] = (
        pairs["low_pr_raw_m"] - pairs["high_pr_raw_m"]
    ) * STEC_FACTOR
    pairs["phase_stec_tecu"] = (
        pairs["high_acc_doppler_m"] - pairs["low_acc_doppler_m"]
    ) * STEC_FACTOR
    pairs["arc_bias_tecu"] = (
        pairs["code_stec_tecu"] - pairs["phase_stec_tecu"]
    ).groupby(pairs["pair_arc_id"], sort=False).transform("median")
    pairs["candidate_stec_tecu"] = (
        pairs["phase_stec_tecu"] + pairs["arc_bias_tecu"]
    )
    return pairs[
        KEY_COLUMNS
        + [
            "pair_arc_id",
            "high_segment_id",
            "low_segment_id",
            "code_stec_tecu",
            "phase_stec_tecu",
            "arc_bias_tecu",
            "candidate_stec_tecu",
        ]
    ]


def choose_candidate_or_fallback(
    candidate: pd.Series,
    reference: pd.Series,
    tolerance: float,
) -> tuple[pd.Series, pd.Series]:
    both = candidate.notna() & reference.notna()
    agrees = both & (candidate - reference).abs().le(tolerance)
    selected = reference.copy()
    selected.loc[agrees] = candidate.loc[agrees]

    source = pd.Series("not_applicable", index=reference.index, dtype="object")
    source.loc[agrees] = "raw_recomputed"
    source.loc[reference.notna() & candidate.isna()] = (
        "processed_fallback_missing_candidate"
    )
    source.loc[both & ~agrees] = "processed_fallback_mismatch"
    source.loc[reference.isna() & candidate.notna()] = "reference_scope_excluded"
    return selected, source


def build_rot_candidates(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid = frame["pair_arc_id"].notna() & frame["stec_tecu"].notna()
    work = frame.loc[
        valid, ["pair_arc_id", "rx_gps_seconds", "stec_tecu"]
    ].copy()
    work.sort_values(
        ["pair_arc_id", "rx_gps_seconds"], kind="mergesort", inplace=True
    )
    grouped = work.groupby("pair_arc_id", sort=False)
    delta_stec = grouped["stec_tecu"].diff()
    delta_seconds = grouped["rx_gps_seconds"].diff()
    candidate = delta_stec * 60.0 / delta_seconds
    candidate.loc[delta_seconds.le(0.0)] = np.nan
    result.loc[work.index] = candidate
    return result


def build_roti_candidates(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid = frame["pair_arc_id"].notna() & frame["rot_tecu_per_min"].notna()
    work = frame.loc[
        valid, ["pair_arc_id", "rx_gps_seconds", "rot_tecu_per_min"]
    ].copy()
    work.sort_values(
        ["pair_arc_id", "rx_gps_seconds"], kind="mergesort", inplace=True
    )
    for _, group in work.groupby("pair_arc_id", sort=False):
        time_index = pd.to_datetime(
            group["rx_gps_seconds"].to_numpy(), unit="s", origin="unix"
        )
        values = pd.Series(
            group["rot_tecu_per_min"].to_numpy(), index=time_index
        )
        rolling = (
            values.rolling("300s", min_periods=2, closed="both")
            .std(ddof=1)
            .to_numpy()
        )
        result.loc[group.index] = rolling
    return result


def mismatch_detail(
    frame: pd.DataFrame,
    field: str,
    candidate_column: str,
    reference_column: str,
    source_column: str,
    tolerance: float,
) -> pd.DataFrame:
    candidate = frame[candidate_column]
    reference = frame[reference_column]
    both = candidate.notna() & reference.notna()
    mismatch = (
        (both & (candidate - reference).abs().gt(tolerance))
        | (candidate.notna() & reference.isna())
        | (candidate.isna() & reference.notna())
    )
    columns = [
        "phase",
        "op_id",
        "sat",
        "signal_id",
        "segment_id",
        "time_utc",
        "rx_gps_seconds",
        "pair_arc_id",
    ]
    detail = frame.loc[mismatch, columns].copy()
    detail["field"] = field
    detail["candidate"] = candidate.loc[mismatch]
    detail["reference"] = reference.loc[mismatch]
    detail["abs_error"] = (
        candidate.loc[mismatch] - reference.loc[mismatch]
    ).abs()
    detail["selection"] = frame.loc[mismatch, source_column]
    detail["reason"] = np.select(
        [
            candidate.loc[mismatch].notna() & reference.loc[mismatch].isna(),
            candidate.loc[mismatch].isna() & reference.loc[mismatch].notna(),
        ],
        ["reference_scope_excluded", "raw_candidate_missing"],
        default="candidate_reference_mismatch",
    )
    return detail


def compare_columns(
    output: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    numeric_tolerances = {
        "signal_id": 0.0,
        "segment_id": 0.0,
        "rx_gps_seconds": 0.0,
        "cn0_dbhz": 0.0,
        "plot_cn0_detrended_db": 0.0,
        "stec_tecu": STEC_TOLERANCE,
        "rot_tecu_per_min": ROT_TOLERANCE,
        "roti_5min": ROTI_TOLERANCE,
        "lugre_lat": 0.0,
        "lugre_lon": 0.0,
        "lugre_h_tan_km": 0.0,
    }
    rows: list[dict[str, object]] = []
    for column in OUTPUT_COLUMNS:
        left = output[column]
        right = reference[column]
        null_mask_mismatch = int((left.isna() != right.isna()).sum())
        both = left.notna() & right.notna()
        row: dict[str, object] = {
            "column": column,
            "rows": len(output),
            "output_non_null": int(left.notna().sum()),
            "reference_non_null": int(right.notna().sum()),
            "null_mask_mismatches": null_mask_mismatch,
        }
        if column in numeric_tolerances:
            tolerance = numeric_tolerances[column]
            difference = (
                pd.to_numeric(left.loc[both])
                - pd.to_numeric(right.loc[both])
            ).abs()
            row.update(
                {
                    "comparison": "numeric",
                    "tolerance": tolerance,
                    "value_mismatches": int(difference.gt(tolerance).sum()),
                    "max_abs_error": (
                        float(difference.max()) if len(difference) else 0.0
                    ),
                }
            )
        else:
            different = left.loc[both].astype(str) != right.loc[both].astype(str)
            row.update(
                {
                    "comparison": "exact_string",
                    "tolerance": 0.0,
                    "value_mismatches": int(different.sum()),
                    "max_abs_error": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def provenance_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_fixed(
        field: str, method: str, source: str, count: int, note: str
    ) -> None:
        rows.append(
            {
                "field": field,
                "method": method,
                "source_class": source,
                "rows": int(count),
                "note": note,
            }
        )

    raw_note = "Parsed or deterministically derived from packaged TLM_RAW text."
    for field in [
        "phase",
        "op_id",
        "sat",
        "signal_id",
        "segment_id",
        "time_utc",
        "rx_gps_seconds",
        "cn0_dbhz",
    ]:
        add_fixed(field, "raw telemetry", "raw_recomputed", len(frame), raw_note)

    for field, source_column, method in [
        (
            "stec_tecu",
            "_stec_source",
            "dual-frequency code/carrier STEC with per-pair-arc median bias",
        ),
        (
            "rot_tecu_per_min",
            "_rot_source",
            "successive STEC difference divided by elapsed minutes",
        ),
        (
            "roti_5min",
            "_roti_source",
            "inclusive trailing 300-second sample standard deviation of ROT",
        ),
    ]:
        counts = frame[source_column].value_counts(dropna=False)
        for source, count in counts.items():
            add_fixed(
                field,
                method,
                str(source),
                int(count),
                "Candidate retained only when it agrees with the accepted "
                "processed table within the declared tolerance.",
            )

    add_fixed(
        "plot_cn0_detrended_db",
        "accepted processed fallback",
        "processed_fallback",
        int(frame["plot_cn0_detrended_db"].notna().sum()),
        "Exact robust_poly4_median60s implementation is not present.",
    )
    for field in ["lugre_lat", "lugre_lon", "lugre_h_tan_km"]:
        add_fixed(
            field,
            "accepted processed fallback",
            "processed_fallback",
            int(frame[field].notna().sum()),
            "Exact precise-orbit tangent geometry chain is not present.",
        )
    add_fixed(
        "daynight",
        "accepted processed fallback",
        "not_applicable",
        int(frame["daynight"].isna().sum()),
        "The accepted S-phase table contains no populated day/night values.",
    )
    return pd.DataFrame(rows)


def build_observations(
    raw: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, int]]:
    merged, coverage = compare_key_coverage(raw, reference)

    output = pd.DataFrame(index=merged.index)
    output["phase"] = merged["phase"]
    output["op_id"] = merged["op_id"]
    output["sat"] = merged["sat"]
    output["signal_id"] = merged["signal_id"]
    output["segment_id"] = merged["segment_id_raw"].astype("int64")
    output["time_utc"] = merged["time_utc_raw"]
    output["rx_gps_seconds"] = merged["rx_gps_seconds_raw"]
    output["cn0_dbhz"] = merged["cn0_dbhz_raw"]
    for column in [
        "plot_cn0_detrended_db",
        "stec_tecu",
        "rot_tecu_per_min",
        "roti_5min",
        "lugre_lat",
        "lugre_lon",
        "lugre_h_tan_km",
        "daynight",
    ]:
        output[column] = merged[column]

    pairs = build_stec_candidates(raw)
    output["rx_key"] = output["rx_gps_seconds"].round(6)
    output = output.merge(
        pairs,
        how="left",
        on=KEY_COLUMNS,
        validate="one_to_one",
        sort=False,
    )

    output["_reference_stec"] = output["stec_tecu"]
    output["stec_tecu"], output["_stec_source"] = choose_candidate_or_fallback(
        output["candidate_stec_tecu"],
        output["_reference_stec"],
        STEC_TOLERANCE,
    )

    output["_reference_rot"] = output["rot_tecu_per_min"]
    output["_candidate_rot"] = build_rot_candidates(output)
    output["rot_tecu_per_min"], output["_rot_source"] = (
        choose_candidate_or_fallback(
            output["_candidate_rot"],
            output["_reference_rot"],
            ROT_TOLERANCE,
        )
    )

    output["_reference_roti"] = output["roti_5min"]
    output["_candidate_roti"] = build_roti_candidates(output)
    output["roti_5min"], output["_roti_source"] = (
        choose_candidate_or_fallback(
            output["_candidate_roti"],
            output["_reference_roti"],
            ROTI_TOLERANCE,
        )
    )

    details = {
        "stec_mismatch_details.csv": mismatch_detail(
            output,
            "stec_tecu",
            "candidate_stec_tecu",
            "_reference_stec",
            "_stec_source",
            STEC_TOLERANCE,
        ),
        "rot_mismatch_details.csv": mismatch_detail(
            output,
            "rot_tecu_per_min",
            "_candidate_rot",
            "_reference_rot",
            "_rot_source",
            ROT_TOLERANCE,
        ),
        "roti_mismatch_details.csv": mismatch_detail(
            output,
            "roti_5min",
            "_candidate_roti",
            "_reference_roti",
            "_roti_source",
            ROTI_TOLERANCE,
        ),
    }
    return output, details, coverage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the S-phase observation table from packaged raw LuGRE "
            "telemetry and emit explicit QA/provenance records."
        )
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        metavar="OP_ID",
        help="Optional operation subset, for example: --ops 38_0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def write_validation_report(
    path: Path,
    summary: dict[str, object],
    raw_manifest: pd.DataFrame,
) -> None:
    fallback = summary["processed_fallback_rows"]
    recomputed = summary["raw_recomputed_rows"]
    mismatches = summary["candidate_mismatch_rows"]
    skipped_nonfinite = int(
        raw_manifest["nonfinite_measure_rows_skipped"].sum()
    )
    text = f"""# Validation Report

## Overall assessment: PASS WITH CAVEATS

The written CSV passed all declared column checks against the accepted
processed S-phase observation table. It is a partial raw-data reconstruction,
not a completely independent scientific recomputation.

- Raw/reference/output rows: {summary["raw_rows"]:,} /
  {summary["reference_rows"]:,} / {summary["output_rows"]:,}
- One-to-one key matches: {summary["key_coverage"]["both"]:,}
- Raw-only keys: {summary["key_coverage"]["raw_only"]:,}
- Reference-only keys: {summary["key_coverage"]["reference_only"]:,}
- Written-output column QA failures: {summary["column_qa_failures"]:,}
- Non-finite raw measurements explicitly skipped: {skipped_nonfinite:,}

## Raw recomputation coverage

- STEC: {recomputed["stec_tecu"]:,} rows
- ROT: {recomputed["rot_tecu_per_min"]:,} rows
- ROTI: {recomputed["roti_5min"]:,} rows

The candidate audit contains {mismatches["stec"]:,} STEC,
{mismatches["rot"]:,} ROT, and {mismatches["roti"]:,} ROTI disagreement or
scope rows. The selected processed fallbacks are {fallback["stec_tecu"]:,},
{fallback["rot_tecu_per_min"]:,}, and {fallback["roti_5min"]:,} rows,
respectively. Candidate-only rows excluded by the accepted reference scope are
audited but are not counted as copied fallback values.

## Independent-reproduction blockers

1. All {fallback["plot_cn0_detrended_db"]:,} detrended C/N0 values use the
   accepted processed fallback because the exact `robust_poly4_median60s`
   implementation and boundary rules are absent.
2. All {fallback["lugre_h_tan_km"]:,} tangent-geometry rows use the accepted
   processed fallback because the exact precise-orbit interpolation,
   frame/time conversion, and tangent-point generator are absent.
3. A small number of dual-frequency carrier-continuity joins are not encoded
   in the available raw files or scripts; every affected row is listed in the
   mismatch detail CSVs.

## Evidence

- Raw file hashes and parse counts: `raw_input_manifest.csv`
- Per-column checks: `column_comparison.csv`
- Per-field source counts: `field_provenance.csv`
- Candidate discrepancies: `*_mismatch_details.csv`
- Machine-readable summary: `qa_summary.json`
- Written output SHA256: `{summary["output_sha256"]}`
- Accepted reference SHA256: `{summary["reference_sha256"]}`
"""
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_ops = set(args.ops) if args.ops else None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    sources = discover_sources(selected_ops)
    raw, raw_manifest = parse_raw_sources(sources)
    print(f"[raw] parsed {len(raw):,} measurement rows", flush=True)
    reference = read_reference(selected_ops)
    output, detail_tables, coverage = build_observations(raw, reference)

    output_path = output_dir / "observations_s_phase_from_raw.csv"
    output[OUTPUT_COLUMNS].to_csv(output_path, index=False)
    written_output = pd.read_csv(output_path, low_memory=False)
    reference_aligned = reference.sort_values(
        "_reference_order", kind="mergesort"
    ).reset_index(drop=True)
    comparison = compare_columns(
        written_output[OUTPUT_COLUMNS], reference_aligned[OUTPUT_COLUMNS]
    )
    provenance = provenance_summary(output)

    raw_manifest.to_csv(output_dir / "raw_input_manifest.csv", index=False)
    comparison.to_csv(output_dir / "column_comparison.csv", index=False)
    provenance.to_csv(output_dir / "field_provenance.csv", index=False)
    for filename, detail in detail_tables.items():
        detail.to_csv(output_dir / filename, index=False)

    elapsed_seconds = time.perf_counter() - started
    hard_failures = int(
        comparison["null_mask_mismatches"].sum()
        + comparison["value_mismatches"].sum()
    )
    summary = {
        "status": "pass" if hard_failures == 0 else "fail",
        "end_to_end_status": "partial_with_explicit_processed_fallbacks",
        "fully_independent_from_processed_inputs": False,
        "scope": sorted(selected_ops) if selected_ops is not None else "all_s_phase",
        "raw_rows": int(len(raw)),
        "reference_rows": int(len(reference)),
        "output_rows": int(len(output)),
        "key_coverage": coverage,
        "column_qa_failures": hard_failures,
        "raw_recomputed_rows": {
            "stec_tecu": int((output["_stec_source"] == "raw_recomputed").sum()),
            "rot_tecu_per_min": int(
                (output["_rot_source"] == "raw_recomputed").sum()
            ),
            "roti_5min": int(
                (output["_roti_source"] == "raw_recomputed").sum()
            ),
        },
        "processed_fallback_rows": {
            "plot_cn0_detrended_db": int(
                output["plot_cn0_detrended_db"].notna().sum()
            ),
            "stec_tecu": int(
                output["_stec_source"].str.startswith("processed_fallback").sum()
            ),
            "rot_tecu_per_min": int(
                output["_rot_source"].str.startswith("processed_fallback").sum()
            ),
            "roti_5min": int(
                output["_roti_source"].str.startswith("processed_fallback").sum()
            ),
            "lugre_lat": int(output["lugre_lat"].notna().sum()),
            "lugre_lon": int(output["lugre_lon"].notna().sum()),
            "lugre_h_tan_km": int(output["lugre_h_tan_km"].notna().sum()),
        },
        "candidate_mismatch_rows": {
            filename.removesuffix("_mismatch_details.csv"): int(len(detail))
            for filename, detail in detail_tables.items()
        },
        "tolerances": {
            "stec_tecu": STEC_TOLERANCE,
            "rot_tecu_per_min": ROT_TOLERANCE,
            "roti_5min": ROTI_TOLERANCE,
        },
        "reference_file": REFERENCE_CSV.relative_to(PACKAGE_ROOT).as_posix(),
        "reference_sha256": sha256_file(REFERENCE_CSV),
        "output_file": output_path.relative_to(PACKAGE_ROOT).as_posix(),
        "output_sha256": sha256_file(output_path),
        "raw_nonfinite_measure_rows_skipped": int(
            raw_manifest["nonfinite_measure_rows_skipped"].sum()
        ),
        "independent_reproduction_blockers": [
            "Exact robust_poly4_median60s detrending implementation absent",
            "Exact precise-orbit tangent geometry generator absent",
            "Carrier-continuity join metadata absent for a small set of rows",
        ],
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    (output_dir / "qa_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_validation_report(
        output_dir / "validation_report.md", summary, raw_manifest
    )

    print(
        f"[qa] {summary['status'].upper()}: {len(output):,} rows; "
        f"column mismatches={hard_failures}; output={output_path}",
        flush=True,
    )
    return 0 if hard_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
