from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


TASK_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = TASK_DIR / "receiver_regime_base.py"

OUT_DIR = ROOT / "outputs" / "Fig4" / "derived" / "leoro_sensitivity"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
REPORT_DIR = OUT_DIR / "reports"

ROLLING_CACHE = (
    ROOT
    / "data"
    / "analysis_ready"
    / "Fig4"
    / "leoro_rolling_median_points.csv.gz"
)
HP05_CACHE = DATA_DIR / "fig3_leoro_cn0_hp05_points.csv"

HP05_PNG = FIG_DIR / "fig3_receiver_regime_cn0_hp05_sensitivity.png"
HP05_PDF = FIG_DIR / "fig3_receiver_regime_cn0_hp05_sensitivity.pdf"
HP05_SUMMARY = DATA_DIR / "fig3_receiver_regime_cn0_hp05_sensitivity_summary.csv"
HP05_README = REPORT_DIR / "README_fig3_cn0_hp05_sensitivity.md"

ROLLING_PNG = FIG_DIR / "fig3_receiver_regime_cn0_rolling_median_sensitivity.png"
ROLLING_PDF = FIG_DIR / "fig3_receiver_regime_cn0_rolling_median_sensitivity.pdf"
ROLLING_SUMMARY = DATA_DIR / "fig3_receiver_regime_cn0_rolling_median_sensitivity_summary.csv"
ROLLING_README = REPORT_DIR / "README_fig3_cn0_rolling_median_sensitivity.md"

TRACK_EXAMPLE_PNG = FIG_DIR / "leoro_cn0_detrend_method_track_examples.png"
TRACK_EXAMPLE_CSV = DATA_DIR / "leoro_cn0_detrend_method_track_examples_source.csv"
METHOD_SUMMARY_CSV = DATA_DIR / "leoro_cn0_detrend_method_summary.csv"


def load_base_module():
    spec = importlib.util.spec_from_file_location("fig3_v9_hp05", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TASK_DIR))
    spec.loader.exec_module(module)
    return module


def rolling_window(n: int) -> int:
    if n <= 21:
        return max(5, n if n % 2 == 1 else n - 1)
    window = min(61, max(9, (n // 9) * 2 + 1))
    return window if window % 2 == 1 else window + 1


def rolling_median_baseline(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    window = rolling_window(len(values))
    min_periods = max(3, window // 4)
    series = pd.Series(values)
    baseline = series.rolling(window, center=True, min_periods=min_periods).median()
    fallback_window = min(7, max(3, len(values)))
    baseline = baseline.fillna(series.rolling(fallback_window, center=True, min_periods=1).median())
    return baseline.to_numpy(float)


def normalize_mission(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["mission"] = out["mission"].replace({"MetOp-B": "MetOp", "MetOp-C": "MetOp"})
    return out


def event_map(base) -> pd.DataFrame:
    event_cols = ["mission", "source_file", "latitude_band_magnetic", "geomagnetic_dip_lat_deg"]
    out = (
        pd.read_csv(base.LEORO_EVENT_SOURCE, usecols=event_cols)
        .dropna(subset=["mission", "source_file", "latitude_band_magnetic"])
        .drop_duplicates(["mission", "source_file"])
    )
    out["mission"] = out["mission"].replace({"MetOp-B": "MetOp", "MetOp-C": "MetOp"})
    return out


def build_hp05_cache(base) -> pd.DataFrame:
    if HP05_CACHE.exists():
        df = pd.read_csv(HP05_CACHE)
        df["time_utc"] = base.parse_utc_mixed(df["time_utc"])
        return df

    existing = base.load_leoro_windows().copy()
    existing = normalize_mission(existing)
    existing["cn0_proxy_method"] = "1 Hz median-SNR C/N0 proxy minus zero-phase 0.05 Hz Butterworth high-pass background"
    HP05_CACHE.parent.mkdir(parents=True, exist_ok=True)
    existing.to_csv(HP05_CACHE, index=False)
    return existing


def build_rolling_cache(base) -> pd.DataFrame:
    if ROLLING_CACHE.exists():
        df = pd.read_csv(ROLLING_CACHE)
        df["time_utc"] = base.parse_utc_mixed(df["time_utc"])
        return df

    usecols = [
        "mission",
        "platform",
        "source_file",
        "profile_id",
        "satellite_id",
        "snr_variable",
        "sample_time_utc",
        "height_km",
        "n_highrate_samples",
        "cn0_1hz_median_dbhz",
        "filter_segment_id",
        "usable_for_cn0_highpass",
        "cn0_filter_qc_flag",
    ]
    frames: list[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(base.CN0_HP_SOURCE, usecols=usecols, chunksize=500000, low_memory=False), 1):
        t = base.parse_utc_mixed(chunk["sample_time_utc"])
        h = pd.to_numeric(chunk["height_km"], errors="coerce")
        cn0 = pd.to_numeric(chunk["cn0_1hz_median_dbhz"], errors="coerce")
        ns = pd.to_numeric(chunk["n_highrate_samples"], errors="coerce")
        usable = chunk["usable_for_cn0_highpass"].astype("boolean").fillna(False)
        mask = (
            chunk["mission"].isin(base.MISSIONS)
            & t.notna()
            & t.between(base.WINDOW_START_UTC, base.WINDOW_END_UTC, inclusive="both")
            & h.between(50, 1000)
            & cn0.between(15, 90)
            & ns.ge(1)
            & usable
        )
        if mask.any():
            part = pd.DataFrame(
                {
                    "mission": chunk.loc[mask, "mission"].astype(str),
                    "platform": chunk.loc[mask, "platform"].astype(str),
                    "profile_id": chunk.loc[mask, "profile_id"].astype(str),
                    "source_file": chunk.loc[mask, "source_file"].astype(str),
                    "satellite_id": chunk.loc[mask, "satellite_id"].astype(str),
                    "snr_variable": chunk.loc[mask, "snr_variable"].astype(str),
                    "filter_segment_id": pd.to_numeric(chunk.loc[mask, "filter_segment_id"], errors="coerce").fillna(-1).astype(int),
                    "signal": "L1",
                    "time_utc": t[mask],
                    "tangent_height_km": h[mask],
                    "cn0_proxy_dbhz": cn0[mask],
                    "n_samples_snr": ns[mask],
                }
            )
            frames.append(part)
        if i % 5 == 0:
            kept = sum(len(x) for x in frames)
            print(f"rolling-cache read chunks {i}; kept rows={kept:,}", flush=True)

    if not frames:
        out = pd.DataFrame(columns=base.LEORO_POINT_COLUMNS)
        out.to_csv(ROLLING_CACHE, index=False)
        return out

    data = pd.concat(frames, ignore_index=True)
    data = normalize_mission(data)
    data = data.sort_values(["profile_id", "filter_segment_id", "time_utc"]).reset_index(drop=True)

    baseline = np.full(len(data), np.nan, dtype=float)
    group_cols = ["profile_id", "filter_segment_id"]
    for _, idx in data.groupby(group_cols, sort=False).groups.items():
        pos = np.asarray(idx, dtype=int)
        values = data.loc[pos, "cn0_proxy_dbhz"].to_numpy(float)
        if len(values) >= 3:
            baseline[pos] = rolling_median_baseline(values)
    data["cn0_fit_dbhz"] = baseline
    data["delta_cn0_db"] = data["cn0_proxy_dbhz"] - data["cn0_fit_dbhz"]
    data["source_kind"] = "leo_ro_1hz_dynamic_rolling_median_cn0_proxy"
    data["cn0_proxy_method"] = "1 Hz median-SNR C/N0 proxy minus centered dynamic rolling-median baseline"
    data = data[np.isfinite(data["delta_cn0_db"]) & data["delta_cn0_db"].between(-20, 20)].copy()

    emap = event_map(base)
    data = data.merge(emap, on=["mission", "source_file"], how="left")
    data = data[
        data["latitude_band_magnetic"].isin(base.LAT_ORDER)
        & data["tangent_height_km"].between(50, 1000)
        & data["delta_cn0_db"].between(-20, 20)
    ].copy()
    out = data[base.LEORO_POINT_COLUMNS].copy()
    ROLLING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ROLLING_CACHE, index=False)
    return out


def build_summary(base, lugre: pd.DataFrame, windows: pd.DataFrame, events: pd.DataFrame, ground: pd.DataFrame, method_label: str, source_note: str, out_path: Path) -> pd.DataFrame:
    rows = []
    for lat in base.LAT_ORDER:
        sub = lugre[lugre["lat_group"].eq(lat)]
        rows.append(
            {
                "panel": "a LuGRE",
                "observable": "abs_delta_cn0_db",
                "lat_group": lat,
                "n": int(sub["abs_cn0_detrended_db"].notna().sum()),
                "p50": base.quantile_or_nan(sub["abs_cn0_detrended_db"], 0.50),
                "p90": base.quantile_or_nan(sub["abs_cn0_detrended_db"], 0.90),
                "p95": base.quantile_or_nan(sub["abs_cn0_detrended_db"], 0.95),
                "source_file": str(base.LUGRE_4FITTING_XLSX),
                "notes": f"S phase only, restricted to {base.WINDOW_LABEL}; latest 4-fitting LuGRE source.",
            }
        )
        subw = windows[windows["latitude_band_magnetic"].eq(lat)]
        cn0_abs = subw["delta_cn0_db"].abs()
        rows.append(
            {
                "panel": "a LEO-RO",
                "observable": "abs_snr_derived_cn0_proxy_residual_db",
                "lat_group": lat,
                "n": int(cn0_abs.notna().sum()),
                "p50": base.quantile_or_nan(cn0_abs, 0.50),
                "p90": base.quantile_or_nan(cn0_abs, 0.90),
                "p95": base.quantile_or_nan(cn0_abs, 0.95),
                "source_file": str(base.CN0_HP_SOURCE),
                "notes": f"{method_label}; {source_note}; S-phase time window; geomagnetic dip latitude bands joined from {base.LEORO_EVENT_SOURCE}.",
            }
        )
        sube = events[events["latitude_band_magnetic"].eq(lat)]
        for obs, col in [("event_S4_P90", "s4_p90"), ("event_sigma_phi_rad_P90", "sigma_phi_rad_p90")]:
            rows.append(
                {
                    "panel": "b LEO-RO",
                    "observable": obs,
                    "lat_group": lat,
                    "n": int(sube[col].notna().sum()),
                    "p50": base.quantile_or_nan(sube[col], 0.50),
                    "p90": base.quantile_or_nan(sube[col], 0.90),
                    "p95": base.quantile_or_nan(sube[col], 0.95),
                    "source_file": str(base.LEORO_EVENT_SOURCE),
                    "notes": "PlanetiQ/Spire/TSX/MetOp-B/C only; event-level P90.",
                }
            )
        subg = ground[ground["lat_group_magnetic_approx"].eq(lat)]
        subg = subg[subg["s4"].between(0, 1) & subg["sigma_phi"].between(0, 1)]
        for obs, col in [("ground_station_minute_S4", "s4"), ("ground_station_minute_sigma_phi_rad", "sigma_phi")]:
            rows.append(
                {
                    "panel": "c Ground ISMR",
                    "observable": obs,
                    "lat_group": lat,
                    "n": int(subg[col].notna().sum()),
                    "p50": base.quantile_or_nan(subg[col], 0.50),
                    "p90": base.quantile_or_nan(subg[col], 0.90),
                    "p95": base.quantile_or_nan(subg[col], 0.95),
                    "source_file": str(base.GROUND_ISMR),
                    "notes": f"Ground station-minute context restricted to {base.WINDOW_LABEL}.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def write_method_readme(path: Path, png: Path, pdf: Path, summary: Path, method_label: str, source_note: str) -> None:
    text = f"""# Fig.3 LEO-RO C/N0 detrending sensitivity

Generated by `make_fig3_leoro_cn0_detrend_sensitivity.py`.

## Outputs

- `{png.name}`
- `{pdf.name}`
- `{summary.name}`

## LEO-RO C/N0 method

{method_label}

{source_note}

The 1 Hz C/N0 proxy starts from high-rate SNR and uses:

`C/N0_1Hz = 20 log10(median(SNR within each UTC second))`

Panel a changes only the LEO-RO C/N0 residual detrending method. LuGRE panel-a data, panel-b LEO-RO S4/sigma_phi, and panel-c ground ISMR data are unchanged from the v9 Fig.3 candidate.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def generate_figures(base, lugre: pd.DataFrame, events: pd.DataFrame, ground: pd.DataFrame, hp05: pd.DataFrame, rolling: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    variants = [
        (
            "hp05",
            hp05,
            HP05_PNG,
            HP05_PDF,
            HP05_SUMMARY,
            HP05_README,
            "1 Hz median-SNR C/N0 proxy minus zero-phase 6th-order Butterworth high-pass residual",
            "The plotted residual column is `delta_cn0_hp_0p05_db` with a 0.05 Hz cutoff.",
        ),
        (
            "rolling",
            rolling,
            ROLLING_PNG,
            ROLLING_PDF,
            ROLLING_SUMMARY,
            ROLLING_README,
            "1 Hz median-SNR C/N0 proxy minus centered dynamic rolling-median baseline",
            "The rolling window follows the track-example package: min(61 s, about 2/9 of the usable segment length), with short-window edge fallback.",
        ),
    ]
    for _, windows, png, pdf, summary_path, readme_path, label, note in variants:
        base.PNG_OUT = png
        base.PDF_OUT = pdf
        base.SUMMARY_OUT = summary_path
        summary = build_summary(base, lugre, windows, events, ground, label, note, summary_path)
        base.draw_figure(lugre, windows, events, ground)
        write_method_readme(readme_path, png, pdf, summary_path, label, note)
        print(png)
        print(pdf)
        print(summary_path)


def pick_profiles(base, hp05: pd.DataFrame, rolling: pd.DataFrame) -> list[str]:
    merged = (
        hp05[["profile_id", "mission", "delta_cn0_db"]]
        .assign(abs_hp=lambda x: x["delta_cn0_db"].abs())
        .groupby(["mission", "profile_id"], as_index=False)
        .agg(hp_p95=("abs_hp", lambda x: float(np.nanpercentile(x, 95))), hp_n=("abs_hp", "size"))
    )
    roll = (
        rolling[["profile_id", "delta_cn0_db"]]
        .assign(abs_roll=lambda x: x["delta_cn0_db"].abs())
        .groupby("profile_id", as_index=False)
        .agg(roll_p95=("abs_roll", lambda x: float(np.nanpercentile(x, 95))), roll_n=("abs_roll", "size"))
    )
    merged = merged.merge(roll, on="profile_id", how="inner")
    chosen: list[str] = []
    for mission in ["PlanetiQ", "Spire", "TSX", "MetOp"]:
        sub = merged[merged["mission"].eq(mission) & merged["hp_n"].ge(25) & merged["roll_n"].ge(25)].copy()
        if sub.empty:
            continue
        sub["score"] = sub[["hp_p95", "roll_p95"]].max(axis=1)
        chosen.append(str(sub.sort_values("score", ascending=False).iloc[0]["profile_id"]))
    metop_track145 = "gnssro_metop_ucar_l1b_2023.1110_metopb-G18-202503151625"
    if metop_track145 in set(hp05["profile_id"].astype(str)) and metop_track145 not in chosen:
        chosen[-1:] = [metop_track145]
    return chosen


def load_profiles_for_examples(base, profile_ids: list[str]) -> pd.DataFrame:
    usecols = [
        "mission",
        "platform",
        "source_file",
        "profile_id",
        "satellite_id",
        "sample_time_utc",
        "height_km",
        "cn0_1hz_median_dbhz",
        "cn0_background_hp_0p05_dbhz",
        "delta_cn0_hp_0p05_db",
        "filter_segment_id",
        "usable_for_cn0_highpass",
    ]
    frames = []
    wanted = set(profile_ids)
    for chunk in pd.read_csv(base.CN0_HP_SOURCE, usecols=usecols, chunksize=500000, low_memory=False):
        mask = chunk["profile_id"].astype(str).isin(wanted)
        if mask.any():
            frames.append(chunk.loc[mask].copy())
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["time_utc"] = base.parse_utc_mixed(data["sample_time_utc"])
    data["mission"] = data["mission"].replace({"MetOp-B": "MetOp", "MetOp-C": "MetOp"})
    for col in ["height_km", "cn0_1hz_median_dbhz", "cn0_background_hp_0p05_dbhz", "delta_cn0_hp_0p05_db"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["usable_for_cn0_highpass"] = data["usable_for_cn0_highpass"].astype("boolean").fillna(False)
    data = data[data["usable_for_cn0_highpass"] & data["height_km"].between(50, 1000) & data["cn0_1hz_median_dbhz"].between(15, 90)].copy()
    data = data.sort_values(["profile_id", "filter_segment_id", "time_utc"]).reset_index(drop=True)
    data["rolling_background_dbhz"] = np.nan
    for _, idx in data.groupby(["profile_id", "filter_segment_id"], sort=False).groups.items():
        pos = np.asarray(idx, dtype=int)
        values = data.loc[pos, "cn0_1hz_median_dbhz"].to_numpy(float)
        if len(values) >= 3:
            data.loc[pos, "rolling_background_dbhz"] = rolling_median_baseline(values)
    data["delta_cn0_rolling_db"] = data["cn0_1hz_median_dbhz"] - data["rolling_background_dbhz"]
    return data


def plot_track_examples(base, hp05: pd.DataFrame, rolling: pd.DataFrame) -> None:
    profiles = pick_profiles(base, hp05, rolling)
    data = load_profiles_for_examples(base, profiles)
    if data.empty:
        return
    TRACK_EXAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(TRACK_EXAMPLE_CSV, index=False)

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})
    rows = len(profiles)
    fig, axes = plt.subplots(rows, 2, figsize=(9.0, max(2.2 * rows, 4.5)), dpi=240, sharex=False)
    if rows == 1:
        axes = np.asarray([axes])
    colors = {"cn0": "#111827", "hp": "#2563eb", "roll": "#d97706", "hp_delta": "#1d4ed8", "roll_delta": "#b45309"}
    for r, profile_id in enumerate(profiles):
        sub = data[data["profile_id"].eq(profile_id)].copy()
        if sub.empty:
            continue
        t = (sub["time_utc"] - sub["time_utc"].min()).dt.total_seconds().to_numpy(float)
        mission = str(sub["mission"].iloc[0])
        sat = str(sub["satellite_id"].iloc[0])
        ax0, ax1 = axes[r, 0], axes[r, 1]
        ax0.plot(t, sub["cn0_1hz_median_dbhz"], color=colors["cn0"], lw=0.9, label="1 Hz C/N0")
        ax0.plot(t, sub["cn0_background_hp_0p05_dbhz"], color=colors["hp"], lw=0.85, label="HP 0.05 background")
        ax0.plot(t, sub["rolling_background_dbhz"], color=colors["roll"], lw=0.85, label="Rolling-median background")
        ax0.set_ylabel("C/N0 (dB-Hz)")
        ax0.set_title(f"{mission} {sat} | {profile_id}", fontsize=7.5, loc="left", pad=3)
        ax1.axhline(0, color="#6b7280", lw=0.55, alpha=0.55)
        ax1.plot(t, sub["delta_cn0_hp_0p05_db"], color=colors["hp_delta"], lw=0.85, label="HP 0.05 residual")
        ax1.plot(t, sub["delta_cn0_rolling_db"], color=colors["roll_delta"], lw=0.85, label="Rolling residual")
        ax1.set_ylabel(r"$\delta C/N_0$ (dB)")
        for ax in [ax0, ax1]:
            ax.grid(True, color="#d8dee9", lw=0.45, alpha=0.65)
            ax.tick_params(labelsize=6.5, length=2.0)
            ax.set_xlabel("time from first 1 Hz sample (s)")
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#9ca3af")
    axes[0, 0].legend(loc="upper left", fontsize=6.2, frameon=False, ncol=3)
    axes[0, 1].legend(loc="upper left", fontsize=6.2, frameon=False, ncol=2)
    fig.suptitle("LEO-RO C/N0 detrending sensitivity: high-pass vs rolling median", fontsize=10.0, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    TRACK_EXAMPLE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRACK_EXAMPLE_PNG, dpi=300)
    plt.close(fig)
    print(TRACK_EXAMPLE_PNG)


def write_method_comparison_summary(base, hp05: pd.DataFrame, rolling: pd.DataFrame) -> None:
    rows = []
    for name, frame in [("hp05", hp05), ("rolling_median", rolling)]:
        for mission, sub in frame.groupby("mission"):
            rows.append(
                {
                    "method": name,
                    "mission": mission,
                    "n_points": int(len(sub)),
                    "n_profiles": int(sub["profile_id"].nunique()),
                    "abs_delta_p50_db": base.quantile_or_nan(sub["delta_cn0_db"].abs(), 0.50),
                    "abs_delta_p90_db": base.quantile_or_nan(sub["delta_cn0_db"].abs(), 0.90),
                    "abs_delta_p95_db": base.quantile_or_nan(sub["delta_cn0_db"].abs(), 0.95),
                }
            )
        for lat, sub in frame.groupby("latitude_band_magnetic"):
            rows.append(
                {
                    "method": name,
                    "mission": f"ALL_{lat}",
                    "n_points": int(len(sub)),
                    "n_profiles": int(sub["profile_id"].nunique()),
                    "abs_delta_p50_db": base.quantile_or_nan(sub["delta_cn0_db"].abs(), 0.50),
                    "abs_delta_p90_db": base.quantile_or_nan(sub["delta_cn0_db"].abs(), 0.90),
                    "abs_delta_p95_db": base.quantile_or_nan(sub["delta_cn0_db"].abs(), 0.95),
                }
            )
    out = pd.DataFrame(rows)
    METHOD_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(METHOD_SUMMARY_CSV, index=False)
    print(METHOD_SUMMARY_CSV)


def main() -> None:
    base = load_base_module()
    for path in [FIG_DIR, DATA_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    lugre = base.load_lugre_points()
    events = base.load_leoro_events()
    ground = base.load_ground_ismr()
    hp05 = build_hp05_cache(base)
    rolling = build_rolling_cache(base)
    generate_figures(base, lugre, events, ground, hp05, rolling)
    plot_track_examples(base, hp05, rolling)
    write_method_comparison_summary(base, hp05, rolling)


if __name__ == "__main__":
    main()
