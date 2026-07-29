from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent

PREV_SCRIPT = TASK_DIR / "build_cn0qc.py"

OUT = ROOT / "outputs" / "Fig4"
DATA_OUT = OUT / "derived"
FIG_OUT = OUT
REPORT_OUT = OUT / "reports"

PNG_OUT = FIG_OUT / "Fig4.png"
PDF_OUT = FIG_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_tail_mask_candidate.pdf"
SUMMARY_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_tail_mask_source_summary.csv"
QA_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_tail_mask_QA.csv"
TAIL_SEGMENTS_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_tail_suspect_segment_cells.csv"
BEFORE_AFTER_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_tail_mask_before_after_by_lat.csv"
ENVELOPE_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_tail_mask_midlat_low_height_envelope.csv"
ALL_LAT_ENVELOPE_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_tail_mask_all_lat_height_envelope.csv"
LUGRE_PRE_MASK_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_pre_tail_mask_rows_compact.csv"
LUGRE_MASKED_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_tail_masked_lugre_rows_used_compact.csv"
README_OUT = REPORT_OUT / "README_Fig4_panel_a_lugre_wgs84_cn0QC_tail_mask_update.md"

CELL_HEIGHT_BIN_KM = 32.0
CELL_MIN_N = 8
CELL_P95_MIN_DB = 1.5
TAIL_FRACTION_THRESHOLD = 0.30
SEGMENT_TAIL_MIN_N = 5
SEGMENT_CELL_MEDIAN_BIAS_THRESHOLD_DB = 1.0


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantile_or_nan(values: pd.Series, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return float(np.nanquantile(arr, q)) if arr.size else np.nan


def add_height_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h = pd.to_numeric(out["lugre_h_tan_km"], errors="coerce")
    idx = np.floor((h - 50.0) / CELL_HEIGHT_BIN_KM)
    out["tail_cell_height_bin_lo_km"] = 50.0 + idx * CELL_HEIGHT_BIN_KM
    out["tail_cell_height_bin_hi_km"] = out["tail_cell_height_bin_lo_km"] + CELL_HEIGHT_BIN_KM
    return out


def lat_stats(df: pd.DataFrame, lat_order: list[str], stage: str) -> pd.DataFrame:
    rows = []
    for lat in lat_order:
        sub = df[df["lat_group"].eq(lat)]
        rows.append(
            {
                "mask_stage": stage,
                "lat_group": lat,
                "n": int(len(sub)),
                "p50_abs_delta_db": quantile_or_nan(sub["abs_cn0_detrended_db"], 0.50),
                "p90_abs_delta_db": quantile_or_nan(sub["abs_cn0_detrended_db"], 0.90),
                "p95_abs_delta_db": quantile_or_nan(sub["abs_cn0_detrended_db"], 0.95),
                "height_p50_km": quantile_or_nan(sub["lugre_h_tan_km"], 0.50),
                "height_p95_km": quantile_or_nan(sub["lugre_h_tan_km"], 0.95),
            }
        )
    return pd.DataFrame(rows)


def envelope_table(df: pd.DataFrame, lat: str) -> pd.DataFrame:
    sub = df[
        df["lat_group"].eq(lat)
        & df["cn0_detrend_zero_mean_dbhz"].between(-10, 10)
        & df["cn0_dbhz"].between(15, 55)
    ].copy()
    sub = add_height_bin(sub)
    rows = []
    for lo, g in sub.groupby("tail_cell_height_bin_lo_km"):
        vals = pd.to_numeric(g["cn0_detrend_zero_mean_dbhz"], errors="coerce").dropna().to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals >= -10) & (vals <= 10)]
        if vals.size < CELL_MIN_N:
            continue
        amp = np.abs(vals)
        rows.append(
            {
                "height_bin_lo_km": float(lo),
                "height_bin_hi_km": float(lo + CELL_HEIGHT_BIN_KM),
                "height_center_km": float(lo + CELL_HEIGHT_BIN_KM / 2.0),
                "n": int(vals.size),
                "p50_abs_delta_db": float(np.nanpercentile(amp, 50)),
                "p90_abs_delta_db": float(np.nanpercentile(amp, 90)),
                "p95_abs_delta_db": float(np.nanpercentile(amp, 95)),
                "signed_median_delta_db": float(np.nanmedian(vals)),
            }
        )
    out = pd.DataFrame(rows).sort_values("height_bin_lo_km")
    for col in ["p50_abs_delta_db", "p90_abs_delta_db", "p95_abs_delta_db"]:
        out[col + "_smoothed_as_plotted"] = out[col].rolling(3, center=True, min_periods=1).median()
    return out


def all_lat_envelope_table(df: pd.DataFrame, lat_order: list[str], stage: str) -> pd.DataFrame:
    frames = []
    for lat in lat_order:
        env = envelope_table(df, lat)
        env["lat_group"] = lat
        env["mask_stage"] = stage
        frames.append(env)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_tail_sensitivity_mask(qc: pd.DataFrame, base, qa_rows: list[dict]) -> pd.DataFrame:
    qc = add_height_bin(qc)
    qc["tail_artifact_suspect_segment_cell"] = False

    cell_cols = ["lat_group", "tail_cell_height_bin_lo_km", "tail_cell_height_bin_hi_km"]
    seg_cols = ["op_id", "sat", "signal_name", "segment_id"]
    rows = []
    for cell_key, g in qc.groupby(cell_cols, dropna=False):
        vals = pd.to_numeric(g["abs_cn0_detrended_db"], errors="coerce").dropna().to_numpy(float)
        if vals.size < CELL_MIN_N:
            continue
        cell_p95 = float(np.nanpercentile(vals, 95))
        tail = g[pd.to_numeric(g["abs_cn0_detrended_db"], errors="coerce").ge(cell_p95)].copy()
        if cell_p95 < CELL_P95_MIN_DB or tail.empty:
            continue
        for seg_key, s in g.groupby(seg_cols, dropna=False):
            stail = s[pd.to_numeric(s["abs_cn0_detrended_db"], errors="coerce").ge(cell_p95)].copy()
            if len(stail) < SEGMENT_TAIL_MIN_N:
                continue
            segment_cell_median = float(np.nanmedian(pd.to_numeric(s["cn0_detrend_zero_mean_dbhz"], errors="coerce")))
            segment_tail_median = float(np.nanmedian(pd.to_numeric(stail["cn0_detrend_zero_mean_dbhz"], errors="coerce")))
            tail_fraction = len(stail) / len(tail)
            suspect = (
                tail_fraction >= TAIL_FRACTION_THRESHOLD
                and (
                    abs(segment_cell_median) > SEGMENT_CELL_MEDIAN_BIAS_THRESHOLD_DB
                    or abs(segment_tail_median) > SEGMENT_CELL_MEDIAN_BIAS_THRESHOLD_DB
                )
            )
            if suspect:
                row = dict(zip(cell_cols, cell_key))
                row.update(dict(zip(seg_cols, seg_key)))
                row.update(
                    {
                        "cell_n": int(len(g)),
                        "cell_p95_abs_delta_db": cell_p95,
                        "cell_tail_n": int(len(tail)),
                        "segment_cell_n": int(len(s)),
                        "segment_tail_n": int(len(stail)),
                        "segment_tail_fraction": float(tail_fraction),
                        "segment_cell_median_delta_db": segment_cell_median,
                        "segment_tail_median_delta_db": segment_tail_median,
                        "segment_cell_p95_abs_delta_db": quantile_or_nan(s["abs_cn0_detrended_db"], 0.95),
                    }
                )
                rows.append(row)

    suspect = pd.DataFrame(rows)
    if suspect.empty:
        suspect = pd.DataFrame(columns=cell_cols + seg_cols)
    suspect.to_csv(TAIL_SEGMENTS_OUT, index=False)

    if not suspect.empty:
        flag_keys = suspect[cell_cols + seg_cols].drop_duplicates()
        qc = qc.merge(
            flag_keys.assign(tail_artifact_suspect_segment_cell=True),
            on=cell_cols + seg_cols,
            how="left",
            suffixes=("", "_flag"),
        )
        qc["tail_artifact_suspect_segment_cell"] = qc["tail_artifact_suspect_segment_cell_flag"].fillna(False).astype(bool)
        qc = qc.drop(columns=["tail_artifact_suspect_segment_cell_flag"])

    before = lat_stats(qc, base.LAT_ORDER, "before_tail_mask")
    masked = qc[~qc["tail_artifact_suspect_segment_cell"]].copy()
    after = lat_stats(masked, base.LAT_ORDER, "after_tail_mask")
    comparison = pd.concat([before, after], ignore_index=True)
    comparison.to_csv(BEFORE_AFTER_OUT, index=False)

    env_before = envelope_table(qc, "Mid-latitude")
    env_before["mask_stage"] = "before_tail_mask"
    env_after = envelope_table(masked, "Mid-latitude")
    env_after["mask_stage"] = "after_tail_mask"
    pd.concat([env_before, env_after], ignore_index=True).to_csv(ENVELOPE_OUT, index=False)
    pd.concat(
        [
            all_lat_envelope_table(qc, base.LAT_ORDER, "before_tail_mask"),
            all_lat_envelope_table(masked, base.LAT_ORDER, "after_tail_mask"),
        ],
        ignore_index=True,
    ).to_csv(ALL_LAT_ENVELOPE_OUT, index=False)

    qa_rows.extend(
        [
            {"metric": "tail_cell_height_bin_km", "value": CELL_HEIGHT_BIN_KM},
            {"metric": "tail_cell_min_n", "value": CELL_MIN_N},
            {"metric": "tail_cell_p95_min_db", "value": CELL_P95_MIN_DB},
            {"metric": "tail_fraction_threshold", "value": TAIL_FRACTION_THRESHOLD},
            {"metric": "tail_segment_tail_min_n", "value": SEGMENT_TAIL_MIN_N},
            {"metric": "tail_segment_abs_median_threshold_db", "value": SEGMENT_CELL_MEDIAN_BIAS_THRESHOLD_DB},
            {"metric": "tail_suspect_segment_cells_total", "value": int(len(suspect))},
            {"metric": "tail_rows_flagged_total", "value": int(qc["tail_artifact_suspect_segment_cell"].sum())},
            {"metric": "tail_rows_retained_total", "value": int(len(masked))},
        ]
    )
    for lat in base.LAT_ORDER:
        qa_rows.append(
            {
                "metric": f"tail_suspect_segment_cells_{lat}",
                "value": int(suspect["lat_group"].eq(lat).sum()) if not suspect.empty else 0,
            }
        )
        qa_rows.append(
            {
                "metric": f"tail_rows_flagged_{lat}",
                "value": int(qc.loc[qc["lat_group"].eq(lat), "tail_artifact_suspect_segment_cell"].sum()),
            }
        )

    compact_cols = [
        "phase",
        "op_id",
        "sat",
        "signal_name",
        "segment_id",
        "time_utc",
        "rx_gps_seconds",
        "cn0_dbhz",
        "cn0_detrend_zero_mean_dbhz",
        "abs_cn0_detrended_db",
        "lugre_lat",
        "lugre_lon",
        "lugre_h_tan_km",
        "mlat_igrf_dip",
        "lat_group",
        "height_layer",
        "tail_cell_height_bin_lo_km",
        "tail_cell_height_bin_hi_km",
        "tail_artifact_suspect_segment_cell",
    ]
    masked[[c for c in compact_cols if c in masked.columns]].to_csv(LUGRE_MASKED_OUT, index=False)
    return masked


def write_readme(summary: pd.DataFrame, qa: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lugre = summary[summary["panel"].eq("a LuGRE")]
    text = f"""# Fig4 panel-a LuGRE WGS84 + C/N0 QC + P95-tail sensitivity mask

This candidate applies one uniform P95-tail dominance sensitivity QC across all magnetic latitude bands and all WGS84 tangent-height cells in panel a.

The motivation is not to hand-edit one latitude band. The scientific concern is that a percentile envelope can be biased when the highest-amplitude tail in a latitude-height cell is dominated by one OP/satellite/signal/segment whose local residual baseline is shifted away from zero. That condition is treated identically for Equatorial, Mid-latitude and Polar cells.

## Tail-mask definition

- Cell: magnetic latitude band x {CELL_HEIGHT_BIN_KM:.0f} km WGS84 tangent-height bin.
- For each cell, compute the cell P95 of `abs(delta C/N0)`.
- A segment-cell is flagged when:
  - cell P95 >= {CELL_P95_MIN_DB:.1f} dB;
  - the segment contributes at least {SEGMENT_TAIL_MIN_N} points to that P95 tail;
  - the segment contributes >= {TAIL_FRACTION_THRESHOLD:.2f} of the cell P95-tail points;
  - the segment-cell median or tail median has abs(delta C/N0) > {SEGMENT_CELL_MEDIAN_BIAS_THRESHOLD_DB:.1f} dB.

Only flagged OP/satellite/signal/segment rows inside the affected cell are excluded. Figure layout and all styling are unchanged.

## Outputs

- PNG: `{PNG_OUT}`
- PDF: `{PDF_OUT}`
- Source summary: `{SUMMARY_OUT}`
- QA: `{QA_OUT}`
- Tail suspect segment-cells: `{TAIL_SEGMENTS_OUT}`
- Before/after by magnetic latitude band: `{BEFORE_AFTER_OUT}`
- All-latitude height-envelope before/after: `{ALL_LAT_ENVELOPE_OUT}`
- Mid-latitude low-height envelope before/after: `{ENVELOPE_OUT}`
- Masked LuGRE rows used: `{LUGRE_MASKED_OUT}`

## Panel-a LuGRE counts after tail mask

{lugre[["lat_group", "n", "p50", "p90", "p95"]].to_string(index=False) if not lugre.empty else "No LuGRE rows found."}

## Before/after summary

{comparison.to_string(index=False)}

## QA

{qa.to_string(index=False)}
"""
    README_OUT.write_text(text, encoding="utf-8")


def main() -> None:
    for path in [OUT, DATA_OUT, FIG_OUT, REPORT_OUT]:
        path.mkdir(parents=True, exist_ok=True)
    if not PREV_SCRIPT.exists():
        raise FileNotFoundError(PREV_SCRIPT)

    prev = import_module(PREV_SCRIPT, "fig4_panel_a_wgs84_cn0qc_previous_tail_mask")
    prev.LUGRE_USED_OUT = LUGRE_PRE_MASK_OUT

    current = prev.import_module(prev.CURRENT_SCRIPT, "fig4_current_source_right_axis_fixed_cn0qc_tail_mask")
    sens = current.import_module(current.SENSITIVITY_SCRIPT, "fig3_leoro_rolling_sensitivity_wgs84_fig4_cn0qc_tail_mask")
    base = sens.load_base_module()
    current.patch_panel_a_right_axis_label(base)

    base.PNG_OUT = PNG_OUT
    base.PDF_OUT = PDF_OUT
    base.SUMMARY_OUT = SUMMARY_OUT
    base.README_OUT = README_OUT
    base.LUGRE_CACHE = prev.QC_LUGRE_CACHE
    base.LUGRE_4FITTING_XLSX = prev.QC_LUGRE_CACHE

    qa_rows: list[dict] = []

    def load_lugre_points_masked() -> pd.DataFrame:
        pre = prev.load_lugre_points_qc(base, qa_rows)
        return apply_tail_sensitivity_mask(pre, base, qa_rows)

    base.load_lugre_points = load_lugre_points_masked

    lugre = base.load_lugre_points()
    events = base.load_leoro_events()
    ground = base.load_ground_ismr()
    rolling = sens.build_rolling_cache(base)

    summary = sens.build_summary(base, lugre, rolling, events, ground, prev.METHOD_LABEL, prev.SOURCE_NOTE, SUMMARY_OUT)
    summary.loc[summary["panel"].eq("a LuGRE"), "source_file"] = str(prev.QC_LUGRE_CACHE)
    summary.loc[summary["panel"].eq("a LuGRE"), "notes"] = (
        "S phase only; GPS L1 and Galileo E1; WGS84 ellipsoidal tangent height; "
        "hybrid C/N0-quality QC; IGRF dip-latitude bands joined from WGS84 cache; "
        "P95-tail sensitivity excludes OP/sat/signal/segment rows that dominate a high-P95 cell tail "
        f"(tail fraction >= {TAIL_FRACTION_THRESHOLD:.2f}, cell P95 >= {CELL_P95_MIN_DB:.1f} dB, "
        f"abs segment-cell or tail median delta C/N0 > {SEGMENT_CELL_MEDIAN_BIAS_THRESHOLD_DB:.1f} dB)."
    )
    summary.to_csv(SUMMARY_OUT, index=False)

    qa = pd.DataFrame(qa_rows)
    qa.to_csv(QA_OUT, index=False)
    comparison = pd.read_csv(BEFORE_AFTER_OUT)

    base.draw_figure(lugre, rolling, events, ground)
    write_readme(summary, qa, comparison)

    print(PNG_OUT)
    print(PDF_OUT)
    print(SUMMARY_OUT)
    print(QA_OUT)
    print(TAIL_SEGMENTS_OUT)
    print(BEFORE_AFTER_OUT)
    print(ALL_LAT_ENVELOPE_OUT)
    print(ENVELOPE_OUT)
    print(LUGRE_MASKED_OUT)
    print(README_OUT)


if __name__ == "__main__":
    main()
