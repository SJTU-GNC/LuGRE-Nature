from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "outputs" / "Fig6"
DATA = ROOT / "data" / "analysis_ready" / "Fig6"

BASE_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = BASE_DIR / "base_figure.py"

OLD_RA_SOURCE = (
    DATA / "response_ratio_source.csv"
)
OLD_LOS_SOURCE = (
    DATA / "projected_scale_source.csv"
)
OLD_LOS_SUMMARY = (
    DATA / "projected_scale_summary.csv"
)

LATEST_POINTS = (
    DATA / "lugre_points.csv.gz"
)
LATEST_WINDOWS = (
    DATA / "lugre_event_windows.csv"
)

OUT_RA_SOURCE = OUTDIR / "fig6_ra_vs_fresnel_source_summary_latest_lugre.csv"
OUT_LOS_SOURCE = OUTDIR / "fig6_los_scale_fresnel_match_source_latest_lugre_event_filtered.csv"
OUT_LOS_SUMMARY = OUTDIR / "fig6_los_scale_fresnel_match_summary_latest_lugre_event_filtered.csv"
OUT_SOURCE_SUMMARY = OUTDIR / "fig6_fresnel_latest_lugre_source_summary.csv"
OUT_README = OUTDIR / "README_fig6_fresnel_latest_lugre_candidate.md"
OUT_PNG = OUTDIR / "Fig6.png"
OUT_PDF = OUTDIR / "Fig6_fresnel_latest_lugre_candidate_rx_RA_label_fixed_bc_swapped.pdf"


def load_base_module():
    spec = importlib.util.spec_from_file_location("fig5_v23_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recompute_lugre_ra() -> dict[str, float]:
    points = pd.read_csv(LATEST_POINTS, usecols=["lat_group", "abs_delta_cn0_db"])
    rows: list[dict[str, object]] = []
    for group in ["Equatorial", "Mid-lat", "Polar"]:
        values = pd.to_numeric(points.loc[points["lat_group"].eq(group), "abs_delta_cn0_db"], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        rows.append(
            {
                "lat_group": group,
                "n_points": int(values.size),
                "p50_abs_delta_cn0_db": float(np.nanpercentile(values, 50)),
                "p90_abs_delta_cn0_db": float(np.nanpercentile(values, 90)),
                "p95_abs_delta_cn0_db": float(np.nanpercentile(values, 95)),
                "p99_abs_delta_cn0_db": float(np.nanpercentile(values, 99)),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(OUTDIR / "fig6_lugre_latest_abs_delta_cn0_p95_by_latitude.csv", index=False)

    p95 = stats.set_index("lat_group")["p95_abs_delta_cn0_db"]
    return {
        "lugre_polar_p95": float(p95["Polar"]),
        "lugre_equatorial_p95": float(p95["Equatorial"]),
        "lugre_ra": float(p95["Polar"] / p95["Equatorial"]),
        "lugre_midlat_p95": float(p95["Mid-lat"]),
        "lugre_n_equatorial": int(stats.set_index("lat_group").loc["Equatorial", "n_points"]),
        "lugre_n_midlat": int(stats.set_index("lat_group").loc["Mid-lat", "n_points"]),
        "lugre_n_polar": int(stats.set_index("lat_group").loc["Polar", "n_points"]),
    }


def write_updated_ra_source(latest: dict[str, float]) -> pd.DataFrame:
    ra = pd.read_csv(OLD_RA_SOURCE)
    mask = ra["label"].eq("LuGRE lunar link")
    if mask.sum() != 1:
        raise RuntimeError("Expected exactly one LuGRE lunar link row in RA source.")
    ra.loc[mask, "R_A_polar_over_equatorial"] = latest["lugre_ra"]
    ra.loc[mask, "polar_p95"] = latest["lugre_polar_p95"]
    ra.loc[mask, "equatorial_p95"] = latest["lugre_equatorial_p95"]
    ra.to_csv(OUT_RA_SOURCE, index=False)
    return ra


def write_updated_los_sources() -> dict[str, float]:
    windows = pd.read_csv(
        LATEST_WINDOWS,
        usecols=["track_uid", "window_idx", "is_event", "dominant_frequency_hz"],
    )
    event_freqs = windows.loc[windows["is_event"].astype(bool)].copy()
    event_freqs["track_uid"] = event_freqs["track_uid"].astype(str)
    event_freqs["dominant_frequency_hz"] = pd.to_numeric(event_freqs["dominant_frequency_hz"], errors="coerce")
    event_freqs = event_freqs.replace([np.inf, -np.inf], np.nan).dropna(subset=["dominant_frequency_hz"])
    event_freqs = event_freqs[event_freqs["dominant_frequency_hz"] > 0].copy()
    event_freqs = event_freqs.drop_duplicates(["track_uid", "window_idx", "dominant_frequency_hz"])

    # Keep the original radar match/velocity metadata, but replace the LuGRE
    # event frequency with the latest QC'd event-window dominant frequency.
    los = pd.read_csv(OLD_LOS_SOURCE)
    los["track_uid"] = los["track_uid"].astype(str)
    match_cols = [c for c in los.columns if c not in {"f_hz", "L_LOS_km", "latest_lugre_event_track_keep"}]
    matches = los[match_cols].drop_duplicates().copy()
    merged = matches.merge(event_freqs, on="track_uid", how="inner")
    merged["f_hz"] = merged["dominant_frequency_hz"].astype(float)
    merged["velocity_ms"] = pd.to_numeric(merged["velocity_ms"], errors="coerce")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["velocity_ms", "f_hz"])
    merged["L_LOS_km"] = merged["velocity_ms"].abs() / merged["f_hz"] / 1000.0
    merged["latest_lugre_event_track_keep"] = True
    # Preserve the original column order where possible.
    ordered = [c for c in los.columns if c in merged.columns]
    extra = [c for c in merged.columns if c not in ordered]
    filtered = merged[ordered + extra].copy()
    filtered.to_csv(OUT_LOS_SOURCE, index=False)

    old_summary = pd.read_csv(OLD_LOS_SUMMARY).set_index("metric")["value"].astype(float).to_dict()
    compact_mask = filtered["match_class"].astype(str).str.contains("strict", case=False, na=False)
    compact = filtered.loc[compact_mask].copy()
    lugre_l1 = float(old_summary["lugre_l1_RF_km"])
    lugre_l5 = float(old_summary["lugre_l5_RF_km"])
    summary_rows = [
        {"metric": "n_all", "value": float(len(filtered))},
        {"metric": "n_compact", "value": float(len(compact))},
        {"metric": "median_all_km", "value": float(np.nanmedian(filtered["L_LOS_km"]))},
        {"metric": "median_compact_km", "value": float(np.nanmedian(compact["L_LOS_km"]))},
        {"metric": "lugre_l1_RF_km", "value": lugre_l1},
        {"metric": "lugre_l5_RF_km", "value": lugre_l5},
        {"metric": "lugre_neighbourhood_low_km", "value": 0.5 * lugre_l1},
        {"metric": "lugre_neighbourhood_high_km", "value": 2.0 * lugre_l5},
        {
            "metric": "all_fraction_lugre_neighbourhood",
            "value": float(((filtered["L_LOS_km"] >= 0.5 * lugre_l1) & (filtered["L_LOS_km"] <= 2.0 * lugre_l5)).mean()),
        },
        {
            "metric": "compact_fraction_lugre_neighbourhood",
            "value": float(((compact["L_LOS_km"] >= 0.5 * lugre_l1) & (compact["L_LOS_km"] <= 2.0 * lugre_l5)).mean()),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(OUT_LOS_SUMMARY, index=False)
    return {r["metric"]: float(r["value"]) for r in summary_rows}


def polish_response_panel_rf_labels(ax, base):
    for child in list(ax.get_children()):
        if child.__class__.__name__ == "FancyArrowPatch":
            child.remove()
    for txt in list(ax.texts):
        if txt.get_text() == "" and getattr(txt, "arrow_patch", None) is not None:
            txt.arrow_patch.set_color("#aab4bd")
            txt.arrow_patch.set_alpha(0.75)
            txt.arrow_patch.set_linewidth(1.05)
            continue
        if txt.get_text() == "ground/LEO\n$R_F$":
            txt.set_text("Ground $R_F$")
            txt.set_position((0.255, 3.04))
            txt.set_ha("left")
            txt.set_va("top")
            txt.set_fontsize(6.8)
        elif txt.get_text() == "LuGRE\n$R_F$":
            txt.set_text("LuGRE $R_F$")
            txt.set_position((1.88, 3.04))
            txt.set_ha("left")
            txt.set_va("top")
            txt.set_fontsize(6.8)
        elif "response changes from" in txt.get_text():
            txt.set_text("near-Earth and LuGRE ratios\non the same Fresnel-scale axis")
            txt.set_position((0.58, 1.35))
            txt.set_color("#687584")
            txt.set_fontsize(6.6)
            txt.set_ha("left")
            txt.set_va("center")
    ax.text(0.43, 3.04, "LEO-RO $R_F$", color=base.COL_LEO, fontsize=6.8, ha="left", va="top")


def polish_scale_map_rx_label(ax):
    ax.set_ylabel(r"Receiver distance, $D_{\rm rx}$ (km)", fontsize=8)


def polish_ratio_panel_ra_labels(ax):
    for txt in list(ax.texts):
        match = re.fullmatch(r"(\d+\.\d{2})x", txt.get_text())
        if match:
            txt.set_text(r"$R_A = " + match.group(1) + r"$")


def render_figure():
    base = load_base_module()
    base.RA_SOURCE = OUT_RA_SOURCE
    base.LOS_SOURCE = OUT_LOS_SOURCE
    base.LOS_SUMMARY = OUT_LOS_SUMMARY

    original_panel_label = base.panel_label
    original_outside_title = base.outside_title
    label_map = {"b": "a", "c": "c", "d": "b", "e": "d"}
    title_map = {
        "Polar/equatorial amplitude response versus Fresnel scale": "Response versus Fresnel scale",
        "LuGRE-SuperDARN projected structure scale": "Projected structure scale",
    }

    def mapped_panel_label(ax, label: str):
        original_panel_label(ax, label_map.get(label, label))

    def mapped_outside_title(ax, title: str):
        original_outside_title(ax, title_map.get(title, title))

    base.panel_label = mapped_panel_label
    base.outside_title = mapped_outside_title

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
        }
    )

    fig = plt.figure(figsize=(7.25, 5.85), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.52],
        left=0.075,
        right=0.985,
        top=0.935,
        bottom=0.090,
        wspace=0.30,
        hspace=0.39,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    base.draw_scale_map(ax_a)
    polish_scale_map_rx_label(ax_a)
    ax_b = fig.add_subplot(gs[0, 1])
    base.draw_ra(ax_b)
    polish_response_panel_rf_labels(ax_b, base)
    ax_c = fig.add_subplot(gs[1, 0])
    base.draw_ratio(ax_c)
    polish_ratio_panel_ra_labels(ax_c)
    ax_d = fig.add_subplot(gs[1, 1])
    base.draw_los_hist(ax_d)

    fig.savefig(OUT_PNG, dpi=450)
    plt.close(fig)


def write_summary(latest_ra: dict[str, float], los_summary: dict[str, float]) -> None:
    old_ra = pd.read_csv(OLD_RA_SOURCE).set_index("label")
    old_lugre = old_ra.loc["LuGRE lunar link"]
    rows = [
        {"metric": "base_script", "value": str(BASE_SCRIPT)},
        {"metric": "old_RA_source", "value": str(OLD_RA_SOURCE)},
        {"metric": "updated_RA_source", "value": str(OUT_RA_SOURCE)},
        {"metric": "latest_LuGRE_points_source", "value": str(LATEST_POINTS)},
        {"metric": "latest_LuGRE_windows_source", "value": str(LATEST_WINDOWS)},
        {"metric": "old_lugre_RA", "value": float(old_lugre["R_A_polar_over_equatorial"])},
        {"metric": "new_lugre_RA", "value": latest_ra["lugre_ra"]},
        {"metric": "old_lugre_polar_p95_db", "value": float(old_lugre["polar_p95"])},
        {"metric": "new_lugre_polar_p95_db", "value": latest_ra["lugre_polar_p95"]},
        {"metric": "old_lugre_equatorial_p95_db", "value": float(old_lugre["equatorial_p95"])},
        {"metric": "new_lugre_equatorial_p95_db", "value": latest_ra["lugre_equatorial_p95"]},
        {"metric": "old_LOS_source", "value": str(OLD_LOS_SOURCE)},
        {"metric": "updated_LOS_source", "value": str(OUT_LOS_SOURCE)},
        {"metric": "new_LOS_n_all", "value": los_summary["n_all"]},
        {"metric": "new_LOS_n_compact", "value": los_summary["n_compact"]},
        {"metric": "new_LOS_median_all_km", "value": los_summary["median_all_km"]},
        {"metric": "new_LOS_median_compact_km", "value": los_summary["median_compact_km"]},
        {"metric": "new_LOS_fraction_lugre_neighbourhood", "value": los_summary["all_fraction_lugre_neighbourhood"]},
        {"metric": "new_LOS_fraction_compact_lugre_neighbourhood", "value": los_summary["compact_fraction_lugre_neighbourhood"]},
        {"metric": "output_png", "value": str(OUT_PNG)},
        {"metric": "output_pdf", "value": str(OUT_PDF)},
    ]
    with OUT_SOURCE_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    OUT_README.write_text(
        "# Fig.6 Fresnel mechanism latest-LuGRE candidate\n\n"
        "This candidate preserves the four-panel Fig.6/Fig.5 visual style and updates only the LuGRE-dependent statistics.\n\n"
        "Updated LuGRE source: S-phase only, WGS84 tangent height, baseline C/N0 >= 25 dB-Hz quality control, "
        "hybrid L5/E5 detrending where needed, and the detrend-artifact sensitivity mask from the 2026-07-02 Fig.1/Fig.3 workflow.\n\n"
        f"- LuGRE P95 |delta C/N0| polar/equatorial ratio: old {float(old_lugre['R_A_polar_over_equatorial']):.3f}, "
        f"new {latest_ra['lugre_ra']:.3f}.\n"
        f"- LuGRE P95 values: Polar {latest_ra['lugre_polar_p95']:.3f} dB; "
        f"Equatorial {latest_ra['lugre_equatorial_p95']:.3f} dB; Mid-latitude {latest_ra['lugre_midlat_p95']:.3f} dB.\n"
        f"- Projected-scale source filtered to track_uids that remain events in the latest 120-s LuGRE event-window table: "
        f"n_all={los_summary['n_all']:.0f}, n_compact={los_summary['n_compact']:.0f}; "
        f"median all={los_summary['median_all_km']:.2f} km, median compact={los_summary['median_compact_km']:.2f} km.\n\n"
        "Label-only polish in this version: panel a uses `D_rx` to match the Fig.1 schematic notation, "
        "the amplitude-ratio labels are written as `R_A = x`, and the displayed b/c panel labels are swapped "
        "without moving any panel content.\n\n"
        "Panel d keeps the compact nearest-cell SuperDARN LOS-velocity context inherited from the Fig.5/Fig.6 source package, "
        "with the LuGRE event-frequency term updated to the latest event table. A stricter raw-FITACF event-window sensitivity check "
        "is provided separately in `fig6_los_scale_strict_event_window_velocity_panel.png`; it has much smaller coverage and should not "
        "be mixed with this compact-context panel in the caption.\n\n"
        "Ground and LEO-RO response-ratio rows were not recalculated in this candidate. "
        "`L_LOS=|V_LOS|/f` remains a LOS-projected SuperDARN scale consistency diagnostic, not a full 2-D irregularity-size inversion.\n",
        encoding="utf-8",
    )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    latest_ra = recompute_lugre_ra()
    write_updated_ra_source(latest_ra)
    los_summary = write_updated_los_sources()
    render_figure()
    write_summary(latest_ra, los_summary)
    print(
        json.dumps(
            {
                "png": str(OUT_PNG),
                "pdf": str(OUT_PDF),
                "source_summary": str(OUT_SOURCE_SUMMARY),
                "readme": str(OUT_README),
                "new_lugre_ra": latest_ra["lugre_ra"],
                "new_los_n_all": los_summary["n_all"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
