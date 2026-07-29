from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = PACKAGE_ROOT / "outputs"
FIGURES = OUT_ROOT
DATA = PACKAGE_ROOT / "data" / "analysis_ready" / "ED6_9"
REPORTS = PACKAGE_ROOT / "manifest" / "fragments"

SOURCE_SCRIPT = Path(__file__).resolve().with_name("phase_story_plot.py")
WGS84_GEOMETRY = DATA / "tangent_geometry_wgs84_s_phase.csv.gz"
WANTED_OPS = {"38_0", "40_0", "74_0", "76_0", "77_0", "77_1", "78_1"}

FORMAL_STEM_MAP = {
    "S2_S_Phase_Day1_OP38": "Extended_Data_Fig_06_s_phase_day1_op38",
    "S3_S_Phase_Day2_OP40": "Extended_Data_Fig_07_s_phase_day2_op40",
    "S4_S_Phase_Day3_OP74": "Extended_Data_Fig_08_s_phase_day3_op74",
    "S5_S_Phase_Day4_OP76_OP77_OP77_1_OP78_1": "Extended_Data_Fig_09_s_phase_day4_op76_op78",
}


def ensure_dirs() -> None:
    for path in (FIGURES, DATA, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def load_phase_story_module():
    spec = importlib.util.spec_from_file_location("phase_story_current_ecef", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source plotting script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def time_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def load_wgs84_geometry() -> pd.DataFrame:
    usecols = [
        "phase",
        "op_id",
        "sat",
        "signal_id",
        "segment_id",
        "time_utc",
        "rx_gps_seconds",
        "lugre_lat_wgs84",
        "lugre_lon_wgs84",
        "lugre_h_tan_wgs84_km",
    ]
    wanted_ops = WANTED_OPS
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(WGS84_GEOMETRY, usecols=usecols, chunksize=250_000, low_memory=False):
        chunk["phase"] = chunk["phase"].astype(str)
        chunk["op_id"] = chunk["op_id"].astype(str)
        sub = chunk[chunk["phase"].eq("S") & chunk["op_id"].isin(wanted_ops)].copy()
        if not sub.empty:
            frames.append(sub)
    if not frames:
        raise RuntimeError(f"No S-phase WGS84 rows found in {WGS84_GEOMETRY}")
    wgs = pd.concat(frames, ignore_index=True)
    wgs["sat"] = wgs["sat"].astype(str)
    wgs["signal_id"] = pd.to_numeric(wgs["signal_id"], errors="coerce").astype("Int64")
    wgs["segment_id"] = pd.to_numeric(wgs["segment_id"], errors="coerce").astype("Int64")
    wgs["time_key"] = time_key(wgs["time_utc"])
    for col in ["lugre_lat_wgs84", "lugre_lon_wgs84", "lugre_h_tan_wgs84_km"]:
        wgs[col] = pd.to_numeric(wgs[col], errors="coerce")
    wgs = wgs.dropna(subset=["time_key", "lugre_lat_wgs84", "lugre_lon_wgs84", "lugre_h_tan_wgs84_km"])
    return wgs.drop_duplicates(["phase", "op_id", "sat", "signal_id", "segment_id", "time_key"], keep="first")


def attach_wgs84_geometry(tangent_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wgs = load_wgs84_geometry()
    out = tangent_df.copy()
    out["phase"] = out["phase"].astype(str)
    out["op_id"] = out["op_id"].astype(str)
    out["sat"] = out["sat"].astype(str)
    out["signal_id"] = pd.to_numeric(out["signal_id"], errors="coerce").astype("Int64")
    out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce").astype("Int64")
    out["time_key"] = time_key(out["time_utc"])
    merged = out.merge(
        wgs[
            [
                "phase",
                "op_id",
                "sat",
                "signal_id",
                "segment_id",
                "time_key",
                "lugre_lat_wgs84",
                "lugre_lon_wgs84",
                "lugre_h_tan_wgs84_km",
            ]
        ],
        on=["phase", "op_id", "sat", "signal_id", "segment_id", "time_key"],
        how="left",
        validate="m:1",
    )
    merged["lugre_lat_original"] = merged["lugre_lat"]
    merged["lugre_lon_original"] = merged["lugre_lon"]
    merged["lugre_h_tan_km_original"] = merged["lugre_h_tan_km"]
    merged["lugre_h_tan_km_for_altitude_wgs84"] = merged["lugre_h_tan_wgs84_km"]
    matched_all = (
        merged["lugre_lat_wgs84"].notna()
        & merged["lugre_lon_wgs84"].notna()
        & merged["lugre_h_tan_wgs84_km"].notna()
    )
    merged.loc[matched_all, "lugre_lat"] = merged.loc[matched_all, "lugre_lat_wgs84"]
    merged.loc[matched_all, "lugre_lon"] = merged.loc[matched_all, "lugre_lon_wgs84"]
    merged.loc[matched_all, "lugre_h_tan_km"] = merged.loc[matched_all, "lugre_h_tan_wgs84_km"]

    stats = []
    for op_id, sub in merged[merged["phase"].eq("S")].groupby("op_id", sort=True):
        matched = (
            sub["lugre_lat_wgs84"].notna()
            & sub["lugre_lon_wgs84"].notna()
            & sub["lugre_h_tan_wgs84_km"].notna()
        )
        old_h = pd.to_numeric(sub.loc[matched, "lugre_h_tan_km_original"], errors="coerce")
        new_h = pd.to_numeric(sub.loc[matched, "lugre_h_tan_wgs84_km"], errors="coerce")
        old_lat = pd.to_numeric(sub.loc[matched, "lugre_lat_original"], errors="coerce")
        old_lon = pd.to_numeric(sub.loc[matched, "lugre_lon_original"], errors="coerce")
        new_lat = pd.to_numeric(sub.loc[matched, "lugre_lat_wgs84"], errors="coerce")
        new_lon = pd.to_numeric(sub.loc[matched, "lugre_lon_wgs84"], errors="coerce")
        diff = new_h - old_h
        lat_diff = new_lat - old_lat
        lon_diff = ((new_lon - old_lon + 180.0) % 360.0) - 180.0
        stats.append(
            {
                "op_id": op_id,
                "rows": int(len(sub)),
                "matched_wgs84_rows": int(matched.sum()),
                "unmatched_rows": int((~matched).sum()),
                "height_diff_median_km": float(diff.median()) if len(diff) else None,
                "height_diff_abs_p95_km": float(diff.abs().quantile(0.95)) if len(diff) else None,
                "lat_diff_abs_p95_deg": float(lat_diff.abs().quantile(0.95)) if len(lat_diff) else None,
                "lon_diff_abs_p95_deg": float(lon_diff.abs().quantile(0.95)) if len(lon_diff) else None,
                "old_shell_rows": int(old_h.between(50, 1000).sum()) if len(old_h) else 0,
                "wgs84_shell_rows": int(new_h.between(50, 1000).sum()) if len(new_h) else 0,
                "shell_membership_changed_rows": int((old_h.between(50, 1000) != new_h.between(50, 1000)).sum()) if len(new_h) else 0,
            }
        )
    return merged, pd.DataFrame(stats)


def install_htan_only_altitude_panels(mod) -> None:
    def altitude_values(df: pd.DataFrame) -> pd.Series:
        if "lugre_h_tan_km_for_altitude_wgs84" in df.columns:
            return pd.to_numeric(df["lugre_h_tan_km_for_altitude_wgs84"], errors="coerce")
        return pd.to_numeric(df["lugre_h_tan_km"], errors="coerce")

    def add_altitude_panel_wgs84(ax, df: pd.DataFrame, colors: dict[str, str]) -> None:
        ymin = 50.0
        ymax = 1000.0
        mod.add_altitude_layers(ax, ymax)
        for sat, sub in df.groupby("sat", sort=False):
            y = altitude_values(sub)
            mask = sub["plot_x"].notna() & y.notna() & (y >= ymin) & (y <= ymax)
            if not mask.any():
                continue
            ax.scatter(
                sub.loc[mask, "plot_x"],
                y.loc[mask],
                s=1.25,
                color=colors.get(str(sat), "#64748b"),
                alpha=0.52,
                edgecolors="none",
                rasterized=True,
            )
        for boundary in [90, 150, 350, 600, 1000]:
            if ymin < boundary < ymax:
                style = "--" if boundary == 350 else ":"
                ax.axhline(boundary, color="#94a3b8", ls=style, lw=0.55, alpha=0.72)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([50, 90, 150, 350, 600, 1000])
        ax.set_yticklabels(["50", "90", "150", "350", "600", "1000"])
        ax.yaxis.set_minor_formatter(mod.mpl.ticker.NullFormatter())
        ax.set_ylabel("h$_{tan}$\n(km)")
        ax.set_xlabel("UTC time", labelpad=-2.0)
        ax.grid(True, color="#cbd5e1", alpha=0.32, lw=0.48)
        ax.tick_params(axis="both", which="major", labelsize=6.4, length=2.0, width=0.5, pad=1.0)

    def add_datetime_altitude_panel_wgs84(ax, df: pd.DataFrame, colors: dict[str, str]) -> None:
        ymin = 50.0
        ymax = 1000.0
        mod.add_altitude_layers(ax, ymax)
        for sat, sub in df.groupby("sat", sort=False):
            y = altitude_values(sub)
            mask = sub["time_utc"].notna() & y.notna() & (y >= ymin) & (y <= ymax)
            if not mask.any():
                continue
            ax.scatter(
                sub.loc[mask, "time_utc"],
                y.loc[mask],
                s=1.25,
                color=colors.get(str(sat), "#64748b"),
                alpha=0.52,
                edgecolors="none",
                rasterized=True,
            )
        for boundary in [90, 150, 350, 600, 1000]:
            if ymin < boundary < ymax:
                style = "--" if boundary == 350 else ":"
                ax.axhline(boundary, color="#94a3b8", ls=style, lw=0.55, alpha=0.72)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([50, 90, 150, 350, 600, 1000])
        ax.set_yticklabels(["50", "90", "150", "350", "600", "1000"])
        ax.yaxis.set_minor_formatter(mod.mpl.ticker.NullFormatter())
        ax.set_ylabel("h$_{tan}$\n(km)")
        ax.set_xlabel("UTC time", labelpad=-2.0)
        ax.grid(True, color="#cbd5e1", alpha=0.32, lw=0.48)
        ax.tick_params(axis="both", which="major", labelsize=6.4, length=2.0, width=0.5, pad=1.0)

    mod.add_altitude_panel = add_altitude_panel_wgs84
    mod.add_datetime_altitude_panel = add_datetime_altitude_panel_wgs84


def candidate_groups(mod):
    groups = []
    for group in mod.S_GROUPS:
        new_group = dict(group)
        formal_stem = FORMAL_STEM_MAP[str(group["out_stem"])]
        new_group["out_stem"] = formal_stem + "_wgs84_htan_map_candidate"
        groups.append(new_group)
    return groups


def write_readme(manifest: list[dict[str, str]], qa: pd.DataFrame) -> None:
    readme = REPORTS / "README_sphase_wgs84_htan_map_20260702.md"
    qa_text = qa.to_csv(index=False).strip()
    readme.write_text(
        "\n".join(
            [
                "# Extended Data Fig. 6-9 S-phase WGS84 htan + map candidates",
                "",
                "Scope: the tangent-height panel and the map/track panel use WGS84 geodetic latitude/longitude and WGS84 ellipsoidal height/shell mask.",
                "",
                "- Visual/layout source: the current ECEF/geographic S-phase plotting script used for the LaTeX figures.",
                "- C/N0, detrended C/N0, STEC, ROT, ROTI, titles, legends, colors, background bands and layout are unchanged.",
                "- The map panel now uses WGS84 geodetic tangent-point latitude/longitude and the WGS84 50-1000 km shell mask.",
                "- WGS84 source: phase_story_new_reference_tangent_geometry_wgs84_compact.csv.",
                "",
                "Outputs:",
                *[f"- {row['figure']}: `{row['png']}`, `{row['pdf']}`" for row in manifest],
                "",
                "QA summary by OP:",
                "",
                "```csv",
                qa_text,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    mod = load_phase_story_module()
    phase_story_root = PROJECT / "Figure" / "Supplementary" / "phase_story"
    mod.ROOT = phase_story_root
    mod.DATA_XLSX = phase_story_root / "data" / "lugre_phase_story_all_heights_final.xlsx"
    mod.CTL_CACHE = phase_story_root / "data" / "all_heights_master_CTL_cache.csv"
    mod.S_TRACK_LIBRARY = phase_story_root / "s_phase_track_library"
    mod.S_TRACK_MANIFEST = mod.S_TRACK_LIBRARY / "manifest.csv"
    mod.S_CACHE = phase_story_root / "data" / "all_heights_master_S_track_library_cache.csv"
    mod.RECOMPUTED_TANGENT_CACHE = (
        PROJECT
        / "Figure"
        / "geometry_reference_update_20260622"
        / "data"
        / "phase_story_new_reference_tangent_geometry_ecef.csv"
    )
    mod.OUT_DIR = FIGURES
    mod.setup_style()
    install_htan_only_altitude_panels(mod)

    s_df = mod.load_s_rows()
    tangent_df = mod.load_recomputed_tangent_rows()
    tangent_df, qa = attach_wgs84_geometry(tangent_df)
    qa.to_csv(DATA / "sphase_wgs84_htan_map_match_qa.csv", index=False, encoding="utf-8-sig")

    rows: list[dict[str, str]] = []
    for group in candidate_groups(mod):
        if len(group["ops"]) > 1:
            result = mod.plot_s_compact_group(s_df, tangent_df, group)
        else:
            result = mod.plot_s_group(s_df, tangent_df, group)
        result["png"] = str(FIGURES / result["png"])
        result["pdf"] = str(FIGURES / result["pdf"])
        result["wgs84_height_source"] = str(WGS84_GEOMETRY)
        result["geometry_update_scope"] = "htan panel plus map/track WGS84 lat/lon/height shell"
        rows.append(result)

    manifest_path = DATA / "sphase_wgs84_htan_map_candidate_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    source_manifest = {
        "source_plot_script": str(SOURCE_SCRIPT),
        "wgs84_geometry": str(WGS84_GEOMETRY),
        "output_dir": str(FIGURES),
        "scope": "Altitude-panel htan values and map/track geometry use WGS84 geodetic lat/lon plus WGS84 ellipsoidal height. C/N0, detrended C/N0, STEC, ROT, ROTI, titles, legends, colors, background bands and layout are unchanged.",
    }
    (DATA / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    write_readme(rows, qa)
    print(pd.DataFrame(rows)[["figure", "title", "png", "pdf"]].to_string(index=False))
    print(f"qa={DATA / 'sphase_wgs84_htan_map_match_qa.csv'}")


if __name__ == "__main__":
    main()
