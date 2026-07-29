from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "analysis_ready" / "Fig4"

CURRENT_SCRIPT = TASK_DIR / "current_figure.py"
QC_LUGRE_CACHE = INPUT / "lugre_cn0_qc_points.csv.gz"
WGS84_GEOM_CACHE = INPUT / "lugre_wgs84_magnetic_geometry.csv.gz"

OUT = ROOT / "outputs" / "Fig4" / "derived" / "cn0qc"
DATA_OUT = OUT
FIG_OUT = OUT
REPORT_OUT = OUT / "reports"

PNG_OUT = FIG_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_candidate.png"
PDF_OUT = FIG_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_candidate.pdf"
SUMMARY_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_source_summary.csv"
QA_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_QA.csv"
LUGRE_USED_OUT = DATA_OUT / "Fig4_panel_a_lugre_wgs84_cn0QC_lugre_rows_used_compact.csv"
README_OUT = REPORT_OUT / "README_Fig4_panel_a_lugre_wgs84_cn0QC_update.md"

CURRENT_FORMAL_FIG = ROOT / "outputs" / "Fig4" / "Fig4.png"

METHOD_LABEL = "1 Hz median-SNR C/N0 proxy minus centered dynamic rolling-median baseline"
SOURCE_NOTE = (
    "LEO-RO panel a uses SNR-derived delta C/N0 proxy residuals from the rolling-median/local-median baseline; "
    "FengYun is excluded; 0.05 Hz high-pass is retained only as sensitivity/backup."
)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_merge_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rx_gps_seconds_key"] = pd.to_numeric(out["rx_gps_seconds"], errors="coerce").round(6)
    out["segment_id_key"] = pd.to_numeric(out["segment_id"], errors="coerce").astype("Int64")
    return out


def load_lugre_points_qc(base, qa_rows: list[dict]) -> pd.DataFrame:
    qc = pd.read_csv(QC_LUGRE_CACHE)
    qc_before = len(qc)

    if "cn0_quality_qc_keep" in qc.columns:
        keep = qc["cn0_quality_qc_keep"].astype("boolean").fillna(False)
        qc = qc.loc[keep].copy()
    qc_after_flag = len(qc)

    geom_cols = [
        "phase",
        "op_id",
        "sat",
        "signal_name",
        "segment_id",
        "rx_gps_seconds",
        "mlat_igrf_dip",
        "lat_group",
        "height_layer",
    ]
    geom = pd.read_csv(WGS84_GEOM_CACHE, usecols=geom_cols)
    geom = prepare_merge_key(geom)
    geom = geom.drop_duplicates(["phase", "op_id", "sat", "signal_name", "segment_id_key", "rx_gps_seconds_key"])

    qc = prepare_merge_key(qc)
    qc = qc.merge(
        geom[
            [
                "phase",
                "op_id",
                "sat",
                "signal_name",
                "segment_id_key",
                "rx_gps_seconds_key",
                "mlat_igrf_dip",
                "lat_group",
                "height_layer",
            ]
        ],
        on=["phase", "op_id", "sat", "signal_name", "segment_id_key", "rx_gps_seconds_key"],
        how="left",
    )
    after_merge = len(qc)
    matched_lat_group = int(qc["lat_group"].notna().sum())

    # The original Fig.4 panel-a renderer expects this exact residual column.
    if "plot_cn0_detrended_db" in qc.columns:
        qc["cn0_detrend_zero_mean_dbhz"] = pd.to_numeric(qc["plot_cn0_detrended_db"], errors="coerce")
    qc["cn0_detrend_zero_mean_abs_dbhz"] = qc["cn0_detrend_zero_mean_dbhz"].abs()
    qc["abs_cn0_detrended_db"] = qc["cn0_detrend_zero_mean_abs_dbhz"]

    for col in ["lugre_h_tan_km", "cn0_dbhz", "cn0_detrend_zero_mean_dbhz", "abs_cn0_detrended_db"]:
        qc[col] = pd.to_numeric(qc[col], errors="coerce")

    qc = qc[qc["signal_name"].isin(base.LUGRE_SIGNAL_KEEP)].copy()
    after_signal = len(qc)
    qc = qc[qc["phase"].astype(str).eq("S")].copy()
    after_phase = len(qc)
    qc = base.restrict_to_window(qc, "time_utc")
    after_window = len(qc)
    qc = qc[qc["lugre_h_tan_km"].between(50.0, 1000.0)].copy()
    after_shell = len(qc)
    qc = qc[qc["lat_group"].isin(base.LAT_ORDER)].copy()
    after_lat = len(qc)
    qc = qc.dropna(subset=["lugre_h_tan_km", "lat_group", "cn0_detrend_zero_mean_dbhz", "cn0_dbhz"]).copy()
    after_dropna = len(qc)

    qa_rows.extend(
        [
            {"metric": "qc_cache_rows_all_signals", "value": qc_before},
            {"metric": "after_cn0_quality_qc_keep_flag", "value": qc_after_flag},
            {"metric": "after_wgs84_geom_merge_rows", "value": after_merge},
            {"metric": "rows_with_joined_magnetic_lat_group", "value": matched_lat_group},
            {"metric": "after_signal_GPSL1_GalileoE1", "value": after_signal},
            {"metric": "after_phase_S", "value": after_phase},
            {"metric": "after_sphase_time_window", "value": after_window},
            {"metric": "after_wgs84_50_1000_km_shell", "value": after_shell},
            {"metric": "after_magnetic_lat_group_filter", "value": after_lat},
            {"metric": "after_required_value_dropna", "value": after_dropna},
        ]
    )
    for lat in base.LAT_ORDER:
        sub = qc[qc["lat_group"].eq(lat)]
        qa_rows.append({"metric": f"panel_a_lugre_n_{lat}", "value": int(len(sub))})
        if not sub.empty:
            qa_rows.append({"metric": f"panel_a_lugre_height_p50_km_{lat}", "value": float(np.nanpercentile(sub["lugre_h_tan_km"], 50))})
            qa_rows.append({"metric": f"panel_a_lugre_height_p95_km_{lat}", "value": float(np.nanpercentile(sub["lugre_h_tan_km"], 95))})
            qa_rows.append({"metric": f"panel_a_abs_delta_p95_db_{lat}", "value": float(np.nanpercentile(sub["abs_cn0_detrended_db"], 95))})

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
        "secondary_detrend_method",
        "cn0_quality_qc_keep",
    ]
    available = [col for col in compact_cols if col in qc.columns]
    qc[available].to_csv(LUGRE_USED_OUT, index=False)
    return qc


def write_readme(summary: pd.DataFrame, qa: pd.DataFrame) -> None:
    lugre = summary[summary["panel"].eq("a LuGRE")]
    text = f"""# Fig4 panel-a LuGRE WGS84 + C/N0-quality QC update

This candidate redraws the current LaTeX `Fig4.png` source figure while changing only the LuGRE data used in panel a.

## Source code and data

- Current source script reused: `{CURRENT_SCRIPT}`
- Original formal figure reference: `{CURRENT_FORMAL_FIG}`
- QC LuGRE point cache: `{QC_LUGRE_CACHE}`
- WGS84 magnetic latitude source cache: `{WGS84_GEOM_CACHE}`

## What changed

- LuGRE panel-a samples use the Main Figure Agent hybrid C/N0-quality QC point cache.
- The retained rows pass point C/N0, segment-median C/N0 and centered 60-s local-median C/N0 >= 25 dB-Hz.
- LuGRE panel-a residuals use the QC cache field `plot_cn0_detrended_db` as the plotted `δC/N0`.
- WGS84 ellipsoidal tangent height is retained and the 50-1000 km shell is re-applied.
- Magnetic latitude bands are joined from the WGS84 geodetic/IGRF cache.

## What did not change

- Figure type, layout, titles, legends, ticks, axes, color palette, fonts, density style and line style are unchanged.
- Panel-a LEO-RO source and rolling-median C/N0 proxy processing are unchanged.
- Panels b and c are unchanged in data source and style.
- No file in `Nature_LuGRE_Latex` was overwritten.

## Outputs

- PNG: `{PNG_OUT}`
- PDF: `{PDF_OUT}`
- Source summary: `{SUMMARY_OUT}`
- QA: `{QA_OUT}`
- Compact LuGRE rows used in panel a: `{LUGRE_USED_OUT}`

## Panel-a LuGRE counts after WGS84 + C/N0 QC filtering

{lugre[["lat_group", "n", "p50", "p90", "p95"]].to_string(index=False) if not lugre.empty else "No LuGRE rows found."}

## Cache QA

{qa.to_string(index=False)}
"""
    README_OUT.write_text(text, encoding="utf-8")


def main() -> None:
    for path in [OUT, DATA_OUT, FIG_OUT, REPORT_OUT]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [CURRENT_SCRIPT, QC_LUGRE_CACHE, WGS84_GEOM_CACHE]:
        if not path.exists():
            raise FileNotFoundError(path)

    current = import_module(CURRENT_SCRIPT, "fig4_current_source_right_axis_fixed_cn0qc")
    sens = current.import_module(current.SENSITIVITY_SCRIPT, "fig3_leoro_rolling_sensitivity_wgs84_fig4_cn0qc")
    base = sens.load_base_module()
    current.patch_panel_a_right_axis_label(base)

    base.PNG_OUT = PNG_OUT
    base.PDF_OUT = PDF_OUT
    base.SUMMARY_OUT = SUMMARY_OUT
    base.README_OUT = README_OUT
    base.LUGRE_CACHE = QC_LUGRE_CACHE
    base.LUGRE_4FITTING_XLSX = QC_LUGRE_CACHE

    qa_rows: list[dict] = []
    base.load_lugre_points = lambda: load_lugre_points_qc(base, qa_rows)

    lugre = base.load_lugre_points()
    events = base.load_leoro_events()
    ground = base.load_ground_ismr()
    rolling = sens.build_rolling_cache(base)

    summary = sens.build_summary(base, lugre, rolling, events, ground, METHOD_LABEL, SOURCE_NOTE, SUMMARY_OUT)
    summary.loc[summary["panel"].eq("a LuGRE"), "source_file"] = str(QC_LUGRE_CACHE)
    summary.loc[summary["panel"].eq("a LuGRE"), "notes"] = (
        "S phase only; GPS L1 and Galileo E1; WGS84 ellipsoidal tangent height; "
        "hybrid C/N0-quality QC; plotted delta C/N0 from QC cache plot_cn0_detrended_db; "
        "IGRF dip-latitude band joined from WGS84 cache; WGS84 50-1000 km shell."
    )
    summary.to_csv(SUMMARY_OUT, index=False)

    qa = pd.DataFrame(qa_rows)
    qa.to_csv(QA_OUT, index=False)

    base.draw_figure(lugre, rolling, events, ground)
    write_readme(summary, qa)

    print(PNG_OUT)
    print(PDF_OUT)
    print(SUMMARY_OUT)
    print(QA_OUT)
    print(LUGRE_USED_OUT)
    print(README_OUT)


if __name__ == "__main__":
    main()
