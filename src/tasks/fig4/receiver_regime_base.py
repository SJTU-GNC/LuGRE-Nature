from __future__ import annotations

from pathlib import Path
import sys
import re
import tarfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "analysis_ready" / "Fig4"
OUT = ROOT / "outputs" / "Fig4" / "derived" / "receiver_regime"
OUT.mkdir(parents=True, exist_ok=True)
try:
    import ppigrf  # type: ignore
except ImportError:
    ppigrf = None


LUGRE_4FITTING_XLSX = INPUT / "lugre_cn0_qc_points.csv.gz"
LUGRE_CACHE = INPUT / "lugre_cn0_qc_points.csv.gz"
GROUND_ISMR = INPUT / "ground_ismr_station_minutes.csv.gz"
GROUND_CACHE = OUT / "fig3_ground_ismr_station_geomag_bands_cache.csv"

LEORO_ROOT = INPUT
LEORO_EVENT_SOURCE = INPUT / "leoro_event_summary.csv"
LEORO_CONPHS_WINDOWS = INPUT / "unused_conphs_windows.csv"
LEORO_CONPHS_CN0_WINDOWS = INPUT / "unused_conphs_cn0_windows.csv"
LEORO_METOP_WINDOWS = INPUT / "unused_metop_windows.csv"
CN0_HP_ROOT = INPUT
CN0_HP_SOURCE = INPUT / "leoro_rolling_median_points.csv.gz"
CN0_HP_PROFILE_SUMMARY = INPUT / "unused_cn0_profile_summary.csv"
FENGYUN_WINDOWS = INPUT / "unused_fengyun_windows.csv"

LEORO_BASE_WINDOWS_CACHE = OUT / "fig3_leoro_all5_sphase_cn0_1s_points_hp05_20260619_cache.csv"
LEORO_WINDOWS_CACHE = OUT / "fig3_leoro_all5_plus_fengyun_amp_cn0_context_cache.csv"
LEORO_CONPHS_POINT_CN0_CACHE = OUT / "fig3_leoro_conphs_sphase_cn0_1s_points_cache.csv"
LEORO_METOP_CN0_CACHE = OUT / "fig3_leoro_metop_sphase_cn0_1s_points_cache.csv"
FENGYUN_CN0_CACHE = OUT / "fig3_leoro_fengyun_sparse1hz_cn0_context_cache.csv"
LEORO_EVENTS_CACHE = OUT / "fig3_leoro_all5_sphase_event_pairs_cache_v2.csv"

PNG_OUT = OUT / "fig3_receiver_regime_comparison_merged_height_lugre_l1e1_v9_hp05_no_fengyun.png"
PDF_OUT = OUT / "fig3_receiver_regime_comparison_merged_height_lugre_l1e1_v9_hp05_no_fengyun.pdf"
SUMMARY_OUT = OUT / "fig3_receiver_regime_comparison_merged_height_lugre_l1e1_v9_hp05_no_fengyun_source_summary.csv"
README_OUT = OUT / "README_figure3_lugre_l1e1_v9_hp05_no_fengyun.md"

WINDOW_START_UTC = pd.Timestamp("2025-03-03T06:47:50Z")
WINDOW_END_UTC = pd.Timestamp("2025-03-16T22:30:24.999999Z")
WINDOW_LABEL = "2025-03-03 06:47:50 UTC to 2025-03-16 22:30:24 UTC"

LAT_ORDER = ["Equatorial", "Mid-latitude", "Polar"]
LUGRE_SIGNAL_KEEP = ["GPS L1", "Galileo E1"]
MISSIONS = ["PlanetiQ", "Spire", "TSX", "MetOp-B", "MetOp-C"]
MISSION_PLOT_ORDER = ["PlanetiQ", "Spire", "TSX", "MetOp"]
CN0_MISSION_PLOT_ORDER = ["PlanetiQ", "Spire", "TSX", "MetOp"]
MISSION_COLORS = {
    "PlanetiQ": "#1f77b4",
    "Spire": "#2ca02c",
    "TSX": "#7b3294",
    "MetOp-B": "#d62728",
    "MetOp-C": "#d62728",
    "MetOp": "#d62728",
    "FengYun": "#C58A1F",
}
CONPHS_SIGNAL_MAP = {"caL1Snr": "L1_CA", "pL1Snr": "L1_P"}
LEORO_POINT_COLUMNS = [
    "mission",
    "profile_id",
    "source_file",
    "signal",
    "time_utc",
    "latitude_band_magnetic",
    "geomagnetic_dip_lat_deg",
    "tangent_height_km",
    "delta_cn0_db",
    "cn0_proxy_dbhz",
    "cn0_fit_dbhz",
    "n_samples_snr",
    "source_kind",
    "cn0_proxy_method",
]

INK = "#1F2937"
MUTED = "#5F6B7A"
LIGHT_MUTED = "#8B95A1"
GRID = "#ECEFF1"
SPINE = "#AAB4BF"
LUGRE_COLOR = "#2F6FA3"
LEORO_COLOR = "#B84A4A"
B_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "event_dark_to_red",
    ["#240039", "#12306E", "#1766A6", "#229C8D", "#71BF44", "#F0A51D", "#D6422F"],
)
GROUND_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "ground_viridis_like",
    ["#341052", "#334F9B", "#1E93B3", "#28B98D", "#A8D84B", "#FFF06A"],
)


def magnetic_dip_latitude(lon_deg: np.ndarray, lat_deg: np.ndarray, height_km: np.ndarray) -> np.ndarray:
    out = np.full(lat_deg.shape, np.nan, dtype=float)
    date = pd.Timestamp("2025-02-15")
    for start in range(0, len(out), 20000):
        end = min(start + 20000, len(out))
        be, bn, bu = ppigrf.igrf(lon_deg[start:end], lat_deg[start:end], height_km[start:end], date)
        horizontal = np.hypot(np.asarray(be).reshape(-1), np.asarray(bn).reshape(-1))
        inclination_rad = np.arctan2(-np.asarray(bu).reshape(-1), horizontal)
        out[start:end] = np.degrees(np.arctan(np.tan(inclination_rad) / 2.0))
    return out


def latitude_group_from_mlat(mlat: float) -> str | None:
    if not np.isfinite(mlat):
        return None
    a = abs(float(mlat))
    if a < 20:
        return "Equatorial"
    if a < 60:
        return "Mid-latitude"
    if a <= 90:
        return "Polar"
    return None


def height_layer(height_km: float) -> str | None:
    if 50 <= height_km < 150:
        return "D/E"
    if 150 <= height_km < 300:
        return "F1"
    if 300 <= height_km < 600:
        return "F2"
    if 600 <= height_km <= 1000:
        return "Topside"
    return None


def restrict_to_window(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    t = parse_utc_mixed(df[time_col])
    return df[t.between(WINDOW_START_UTC, WINDOW_END_UTC, inclusive="both")].copy()


def parse_utc_mixed(values: pd.Series | pd.Index | np.ndarray) -> pd.Series:
    try:
        return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, utc=True, errors="coerce")


def quantile_or_nan(values: pd.Series | np.ndarray, q: float) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return float(np.quantile(arr, q)) if arr.size else np.nan


def load_lugre_points() -> pd.DataFrame:
    if LUGRE_CACHE.exists():
        df = pd.read_csv(LUGRE_CACHE)
        df = df[df["signal_name"].isin(LUGRE_SIGNAL_KEEP)].copy()
        return restrict_to_window(df[df["phase"].eq("S")].copy(), "time_utc")
    cols = [
        "phase",
        "time_utc",
        "lugre_lat",
        "lugre_lon",
        "lugre_h_tan_km",
        "cn0_dbhz",
        "cn0_detrend_zero_mean_dbhz",
    ]
    df = pd.read_excel(LUGRE_4FITTING_XLSX, sheet_name="figure_ionosphere_points", usecols=cols)
    for col in ["lugre_lat", "lugre_lon", "lugre_h_tan_km", "cn0_dbhz", "cn0_detrend_zero_mean_dbhz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["lugre_lat", "lugre_lon", "lugre_h_tan_km", "cn0_detrend_zero_mean_dbhz"]).copy()
    df = df[df["lugre_h_tan_km"].between(50, 1000)].copy()
    df["mlat_igrf_dip"] = magnetic_dip_latitude(
        df["lugre_lon"].to_numpy(float), df["lugre_lat"].to_numpy(float), df["lugre_h_tan_km"].to_numpy(float)
    )
    df["lat_group"] = df["mlat_igrf_dip"].map(latitude_group_from_mlat)
    df["height_layer"] = df["lugre_h_tan_km"].map(height_layer)
    df["abs_cn0_detrended_db"] = df["cn0_detrend_zero_mean_dbhz"].abs()
    df = df.dropna(subset=["lat_group", "height_layer"]).copy()
    df.to_csv(LUGRE_CACHE, index=False)
    df = df[df["signal_name"].isin(LUGRE_SIGNAL_KEEP)].copy()
    return restrict_to_window(df[df["phase"].eq("S")].copy(), "time_utc")


def load_ground_ismr() -> pd.DataFrame:
    if GROUND_CACHE.exists():
        return restrict_to_window(pd.read_csv(GROUND_CACHE), "utc_minute")
    cols = ["station", "network", "utc_minute", "S4", "sigma_phi", "latitude_band", "geomag_lat"]
    df = pd.read_csv(GROUND_ISMR, usecols=cols)
    df = df.rename(columns={"S4": "s4", "latitude_band": "lat_group_magnetic_approx"})
    for col in ["s4", "sigma_phi", "geomag_lat"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["s4", "sigma_phi", "lat_group_magnetic_approx"]).copy()
    df = restrict_to_window(df, "utc_minute")
    df.to_csv(GROUND_CACHE, index=False)
    return df


def _standardize_window_chunk(chunk: pd.DataFrame, source_kind: str) -> pd.DataFrame:
    if source_kind == "conphs":
        t = pd.to_datetime(chunk["event_time_from_name_utc"], utc=True, errors="coerce") + pd.to_timedelta(
            pd.to_numeric(chunk["bin_start_profile_s"], errors="coerce"), unit="s"
        )
        out = pd.DataFrame(
            {
                "mission": chunk["mission"],
                "profile_id": chunk["source_file"],
                "signal": chunk["signal"],
                "time_utc": t,
                "latitude_band_magnetic": chunk["latitude_band_magnetic"],
                "tangent_height_km": pd.to_numeric(chunk["tangent_height_km"], errors="coerce"),
                "s4_10s": pd.to_numeric(chunk["s4_10s"], errors="coerce"),
                "sigma_phi_10s_rad": pd.to_numeric(chunk["sigma_phi_10s_rad"], errors="coerce"),
            }
        )
    else:
        out = pd.DataFrame(
            {
                "mission": chunk["mission"],
                "profile_id": chunk["profile_id"],
                "signal": chunk["signal"],
                "time_utc": pd.to_datetime(chunk["bin_start_time_utc"], utc=True, errors="coerce"),
                "latitude_band_magnetic": chunk["latitude_band_magnetic"],
                "tangent_height_km": pd.to_numeric(chunk["height_km"], errors="coerce"),
                "s4_10s": pd.to_numeric(chunk["s4_10s"], errors="coerce"),
                "sigma_phi_10s_rad": pd.to_numeric(chunk["sigma_phi_10s_rad"], errors="coerce"),
            }
        )
    mask = (
        out["mission"].isin(MISSIONS)
        & out["latitude_band_magnetic"].isin(LAT_ORDER)
        & out["tangent_height_km"].between(50, 1000)
        & out["s4_10s"].between(0, 1)
        & out["sigma_phi_10s_rad"].between(0, 1)
        & out["time_utc"].between(WINDOW_START_UTC, WINDOW_END_UTC, inclusive="both")
    )
    return out[mask].copy()


def load_leoro_windows() -> pd.DataFrame:
    if LEORO_BASE_WINDOWS_CACHE.exists():
        base = pd.read_csv(LEORO_BASE_WINDOWS_CACHE)
        base["time_utc"] = parse_utc_mixed(base["time_utc"])
    else:
        event_cols = ["mission", "source_file", "latitude_band_magnetic", "geomagnetic_dip_lat_deg"]
        event_map = (
            pd.read_csv(LEORO_EVENT_SOURCE, usecols=event_cols)
            .dropna(subset=["mission", "source_file", "latitude_band_magnetic"])
            .drop_duplicates(["mission", "source_file"])
        )
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
            "cn0_background_hp_0p05_dbhz",
            "delta_cn0_hp_0p05_db",
            "usable_for_cn0_highpass",
            "cn0_filter_qc_flag",
        ]
        frames: list[pd.DataFrame] = []
        for i, chunk in enumerate(pd.read_csv(CN0_HP_SOURCE, usecols=usecols, chunksize=400000, low_memory=False), 1):
            t = parse_utc_mixed(chunk["sample_time_utc"])
            h = pd.to_numeric(chunk["height_km"], errors="coerce")
            d = pd.to_numeric(chunk["delta_cn0_hp_0p05_db"], errors="coerce")
            cn0 = pd.to_numeric(chunk["cn0_1hz_median_dbhz"], errors="coerce")
            ns = pd.to_numeric(chunk["n_highrate_samples"], errors="coerce")
            mask = (
                chunk["mission"].isin(MISSIONS)
                & t.between(WINDOW_START_UTC, WINDOW_END_UTC, inclusive="both")
                & h.between(50, 1000)
                & d.between(-20, 20)
                & cn0.between(15, 90)
                & ns.ge(1)
                & chunk["usable_for_cn0_highpass"].astype("boolean").fillna(False)
            )
            if mask.any():
                part = pd.DataFrame(
                    {
                        "mission": chunk.loc[mask, "mission"].astype(str),
                        "platform": chunk.loc[mask, "platform"].astype(str),
                        "profile_id": chunk.loc[mask, "profile_id"].astype(str),
                        "source_file": chunk.loc[mask, "source_file"].astype(str),
                        "signal": "L1",
                        "time_utc": t[mask],
                        "tangent_height_km": h[mask],
                        "delta_cn0_db": d[mask],
                        "cn0_proxy_dbhz": cn0[mask],
                        "cn0_fit_dbhz": pd.to_numeric(chunk.loc[mask, "cn0_background_hp_0p05_dbhz"], errors="coerce"),
                        "n_samples_snr": ns[mask],
                        "source_kind": "leo_ro_1hz_hp05_cn0_proxy",
                        "cn0_proxy_method": "1 Hz median-SNR C/N0 proxy minus zero-phase 0.05 Hz high-pass background",
                    }
                )
                frames.append(part)
            if i % 5 == 0:
                kept = sum(len(frame) for frame in frames)
                print(f"read 0.05 Hz C/N0 high-pass chunks {i}; kept rows={kept:,}", flush=True)
        if frames:
            base = pd.concat(frames, ignore_index=True)
            base = base.merge(event_map, on=["mission", "source_file"], how="left")
            base = base[
                base["latitude_band_magnetic"].isin(LAT_ORDER)
                & base["tangent_height_km"].between(50, 1000)
                & base["delta_cn0_db"].between(-20, 20)
            ].copy()
            base = base[LEORO_POINT_COLUMNS].copy()
        else:
            base = pd.DataFrame(columns=LEORO_POINT_COLUMNS)
        base.to_csv(LEORO_BASE_WINDOWS_CACHE, index=False)
    return base


def load_fengyun_cn0_context() -> pd.DataFrame:
    if FENGYUN_CN0_CACHE.exists():
        df = pd.read_csv(FENGYUN_CN0_CACHE)
        df["time_utc"] = parse_utc_mixed(df["time_utc"])
        return df
    usecols = [
        "mission",
        "source_file",
        "event_time_from_name_utc",
        "signal",
        "bin_start_profile_s",
        "bin_end_profile_s",
        "n_samples_snr",
        "cn0_proxy_dbhz_median",
        "tangent_height_km",
        "tangent_lat_deg",
        "tangent_lon_deg",
        "geomagnetic_dip_lat_deg",
        "latitude_band_magnetic",
    ]
    frames: list[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(FENGYUN_WINDOWS, usecols=usecols, chunksize=350000, low_memory=False), 1):
        h = pd.to_numeric(chunk["tangent_height_km"], errors="coerce")
        cn0 = pd.to_numeric(chunk["cn0_proxy_dbhz_median"], errors="coerce")
        ns = pd.to_numeric(chunk["n_samples_snr"], errors="coerce")
        t = pd.to_datetime(chunk["event_time_from_name_utc"], utc=True, errors="coerce") + pd.to_timedelta(
            pd.to_numeric(chunk["bin_start_profile_s"], errors="coerce"), unit="s"
        )
        mask = (
            chunk["signal"].eq("caL1")
            & chunk["latitude_band_magnetic"].isin(LAT_ORDER)
            & h.between(50, 1000)
            & cn0.notna()
            & ns.ge(8)
            & t.between(WINDOW_START_UTC, WINDOW_END_UTC, inclusive="both")
        )
        if mask.any():
            out = pd.DataFrame(
                {
                    "mission": "FengYun",
                    "profile_id": chunk.loc[mask, "source_file"].astype(str),
                    "source_file": chunk.loc[mask, "source_file"].astype(str),
                    "signal": "caL1",
                    "time_utc": t[mask],
                    "bin_start_profile_s": pd.to_numeric(chunk.loc[mask, "bin_start_profile_s"], errors="coerce"),
                    "latitude_band_magnetic": chunk.loc[mask, "latitude_band_magnetic"].astype(str),
                    "geomagnetic_dip_lat_deg": pd.to_numeric(chunk.loc[mask, "geomagnetic_dip_lat_deg"], errors="coerce"),
                    "tangent_height_km": h[mask],
                    "cn0_proxy_dbhz": cn0[mask],
                    "n_samples_snr": ns[mask].astype("Int64"),
                }
            )
            frames.append(out)
        if i % 10 == 0:
            kept = sum(len(frame) for frame in frames)
            print(f"read FengYun C/N0 chunks {i}; kept rows={kept:,}", flush=True)
    if not frames:
        empty = pd.DataFrame(columns=LEORO_POINT_COLUMNS)
        empty.to_csv(FENGYUN_CN0_CACHE, index=False)
        return empty
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["source_file", "signal", "bin_start_profile_s", "time_utc"]).reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    for (_source, _signal), g in df.groupby(["source_file", "signal"], sort=False):
        if len(g) < 20:
            continue
        x = pd.to_numeric(g["bin_start_profile_s"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(g["cn0_proxy_dbhz"], errors="coerce").to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 20:
            continue
        span = float(np.nanmax(x[good]) - np.nanmin(x[good]))
        if not np.isfinite(span) or span <= 10:
            continue
        xnorm = 2 * (x[good] - np.nanmin(x[good])) / span - 1
        degree = min(4, max(0, int(good.sum()) - 2))
        try:
            coef = np.polynomial.chebyshev.chebfit(xnorm, y[good], deg=degree)
            fit_good = np.polynomial.chebyshev.chebval(xnorm, coef)
        except Exception:
            continue
        fit = np.full_like(y, np.nan, dtype=float)
        fit[np.where(good)[0]] = fit_good
        gg = g.copy()
        gg["cn0_fit_dbhz"] = fit
        gg["delta_cn0_db"] = gg["cn0_proxy_dbhz"] - gg["cn0_fit_dbhz"]
        gg = gg[np.isfinite(gg["delta_cn0_db"]) & gg["delta_cn0_db"].between(-20, 20)].copy()
        if not gg.empty:
            rows.append(gg)
    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=LEORO_POINT_COLUMNS)
    out["source_kind"] = "fengyun_sparse_1hz_10s_cn0_proxy_context"
    out["cn0_proxy_method"] = "10s caL1 median C/N0 proxy minus fourth-order Chebyshev source-file background"
    out = out[LEORO_POINT_COLUMNS].copy()
    out.to_csv(FENGYUN_CN0_CACHE, index=False)
    return out


def profile_base_time(profile, rc_module) -> pd.Timestamp | None:
    units = str(profile.variables.get("time", {}).get("attrs", {}).get("units", ""))
    base = rc_module.parse_time_units(units)
    if base is None:
        try:
            base = datetime(
                int(float(profile.attrs.get("year"))),
                int(float(profile.attrs.get("month"))),
                int(float(profile.attrs.get("day"))),
                int(float(profile.attrs.get("hour"))),
                int(float(profile.attrs.get("minute"))),
                int(float(profile.attrs.get("second", 0))),
                tzinfo=timezone.utc,
            )
        except Exception:
            return None
    return pd.Timestamp(base)


def one_second_cn0_points(
    mission: str,
    profile_id: str,
    source_file: str,
    signal: str,
    time_s: np.ndarray,
    abs_time: pd.DatetimeIndex,
    snr: np.ndarray,
    height_km: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    min_seconds_for_fit: int = 18,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = min(len(time_s), len(abs_time), len(snr), len(height_km), len(lat_deg), len(lon_deg))
    if n <= 0:
        return rows
    t = np.asarray(time_s[:n], dtype=float)
    snr = np.asarray(snr[:n], dtype=float)
    h = np.asarray(height_km[:n], dtype=float)
    lat = np.asarray(lat_deg[:n], dtype=float)
    lon = np.asarray(lon_deg[:n], dtype=float)
    valid = (
        np.isfinite(t)
        & np.isfinite(snr)
        & (snr > 0)
        & np.isfinite(h)
        & np.isfinite(lat)
        & np.isfinite(lon)
        & (h >= 50)
        & (h <= 1000)
        & abs_time.to_series(index=np.arange(n)).between(WINDOW_START_UTC, WINDOW_END_UTC, inclusive="both").to_numpy()
    )
    if valid.sum() < 30:
        return rows
    sec = np.floor(t).astype(int)
    sec_ids = np.unique(sec[valid])
    agg = []
    for sid in sec_ids:
        local = valid & (sec == sid)
        if local.sum() < 3:
            continue
        v = snr[local]
        cn0 = 10.0 * np.log10(np.nanmean(v * v))
        agg.append(
            {
                "time_s": float(np.nanmedian(t[local])),
                "time_utc": abs_time[np.where(local)[0][0]].floor("1s"),
                "cn0_proxy_dbhz": float(cn0),
                "tangent_height_km": float(np.nanmedian(h[local])),
                "tangent_lat_deg": float(np.nanmedian(lat[local])),
                "tangent_lon_deg": float(np.nanmedian(lon[local])),
                "n_samples_snr": int(local.sum()),
            }
        )
    if len(agg) < min_seconds_for_fit:
        return rows
    points = pd.DataFrame(agg)
    x = points["time_s"].to_numpy(float)
    y = points["cn0_proxy_dbhz"].to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < min_seconds_for_fit:
        return rows
    span = float(np.nanmax(x[good]) - np.nanmin(x[good]))
    if not np.isfinite(span) or span <= 10:
        return rows
    xnorm = 2 * (x[good] - np.nanmin(x[good])) / span - 1
    degree = min(4, max(0, int(good.sum()) - 2))
    try:
        coef = np.polynomial.chebyshev.chebfit(xnorm, y[good], deg=degree)
        fit_good = np.polynomial.chebyshev.chebval(xnorm, coef)
    except Exception:
        return rows
    fit = np.full_like(y, np.nan, dtype=float)
    fit[np.where(good)[0]] = fit_good
    points["cn0_fit_dbhz"] = fit
    points["delta_cn0_db"] = points["cn0_proxy_dbhz"] - points["cn0_fit_dbhz"]
    m = np.isfinite(points["delta_cn0_db"]) & points["delta_cn0_db"].between(-20, 20)
    points = points[m].copy()
    if points.empty:
        return rows
    mlat = magnetic_dip_latitude(
        points["tangent_lon_deg"].to_numpy(float),
        points["tangent_lat_deg"].to_numpy(float),
        points["tangent_height_km"].to_numpy(float),
    )
    points["geomagnetic_dip_lat_deg"] = mlat
    points["latitude_band_magnetic"] = points["geomagnetic_dip_lat_deg"].map(latitude_group_from_mlat)
    points = points[points["latitude_band_magnetic"].isin(LAT_ORDER)].copy()
    for _, row in points.iterrows():
        rows.append(
            {
                "mission": mission,
                "profile_id": profile_id,
                "source_file": source_file,
                "signal": signal,
                "time_utc": row["time_utc"].isoformat(),
                "latitude_band_magnetic": row["latitude_band_magnetic"],
                "geomagnetic_dip_lat_deg": float(row["geomagnetic_dip_lat_deg"]),
                "tangent_height_km": float(row["tangent_height_km"]),
                "delta_cn0_db": float(row["delta_cn0_db"]),
                "cn0_proxy_dbhz": float(row["cn0_proxy_dbhz"]),
                "cn0_fit_dbhz": float(row["cn0_fit_dbhz"]),
                "n_samples_snr": int(row["n_samples_snr"]),
                "source_kind": "raw_snr_1s_cn0_proxy",
                "cn0_proxy_method": "1s_10log10_mean_snr_squared_minus_4th_order_chebyshev_profile_fit",
            }
        )
    return rows


def load_conphs_cn0_points() -> pd.DataFrame:
    if LEORO_CONPHS_POINT_CN0_CACHE.exists():
        return pd.read_csv(LEORO_CONPHS_POINT_CN0_CACHE, parse_dates=["time_utc"])
    raise FileNotFoundError(
        "The packaged LEO-RO C/N0 point cache is missing. Its original conPhs "
        "preprocessing dependency was not archived in this reviewer package."
    )

    mission_frames: list[pd.DataFrame] = []
    mission_map = {"planetiq": "PlanetiQ", "spire": "Spire", "tsx": "TSX"}
    for raw_mission, fig_mission in mission_map.items():
        mission_cache = OUT / f"fig3_leoro_conphs_{raw_mission}_sphase_cn0_1s_points_cache.csv"
        if mission_cache.exists():
            mission_frames.append(pd.read_csv(mission_cache, parse_dates=["time_utc"]))
            continue
        rows: list[dict[str, object]] = []
        archives = sorted(rc.mission_roots()[raw_mission].glob("*.tar.gz"))
        for archive in archives:
            processed = 0
            with tarfile.open(archive, "r:gz") as tf:
                for member in tf:
                    if not member.isfile() or not Path(member.name).name.startswith("conPhs_"):
                        continue
                    payload = tf.extractfile(member).read()
                    try:
                        profile = rc.read_conphs_profile(raw_mission, str(archive), member.name, payload)
                    except Exception:
                        continue
                    base = profile_base_time(profile, rc)
                    if base is None:
                        continue
                    abs_time = base + pd.to_timedelta(profile.time_s, unit="s")
                    if not abs_time.to_series(index=np.arange(len(abs_time))).between(WINDOW_START_UTC, WINDOW_END_UTC, inclusive="both").any():
                        continue
                    parsed = rc.parse_member_id(member.name)
                    profile_id = f"{fig_mission}:{parsed.get('profile_id', Path(member.name).name)}"
                    for raw_signal, signal in CONPHS_SIGNAL_MAP.items():
                        if raw_signal not in profile.snr:
                            continue
                        rows.extend(
                            one_second_cn0_points(
                                fig_mission,
                                profile_id,
                                f"{archive.name}::{member.name}",
                                signal,
                                profile.time_s,
                                abs_time,
                                profile.snr[raw_signal],
                                profile.tangent_height_km,
                                profile.tangent_lat_deg,
                                profile.tangent_lon_deg,
                            )
                        )
                    processed += 1
                    if processed % 250 == 0:
                        print(f"processed {fig_mission} {archive.name}: {processed} profiles; points={len(rows)}", flush=True)
        mission_df = pd.DataFrame(rows)
        if mission_df.empty:
            mission_df = pd.DataFrame(columns=LEORO_POINT_COLUMNS)
        mission_df.to_csv(mission_cache, index=False)
        mission_frames.append(mission_df)
    out = pd.concat(mission_frames, ignore_index=True) if mission_frames else pd.DataFrame(columns=LEORO_POINT_COLUMNS)
    if out.empty:
        out = pd.DataFrame(columns=LEORO_POINT_COLUMNS)
    out.to_csv(LEORO_CONPHS_POINT_CN0_CACHE, index=False)
    return out


def load_conphs_cn0_windows() -> pd.DataFrame:
    cols = [
        "mission",
        "profile_id",
        "source_file",
        "latitude_band_magnetic",
        "bin_start_time_utc",
        "tangent_height_km",
        "delta_cn0_db_p05",
        "delta_cn0_db_p50",
        "delta_cn0_db_p95",
        "delta_cn0_db_std",
        "n_samples_snr",
        "detrend_qc_flag",
    ]
    df = pd.read_csv(LEORO_CONPHS_CN0_WINDOWS, usecols=cols, low_memory=False)
    for col in ["tangent_height_km", "delta_cn0_db_p05", "delta_cn0_db_p50", "delta_cn0_db_p95", "delta_cn0_db_std", "n_samples_snr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time_utc"] = pd.to_datetime(df["bin_start_time_utc"], utc=True, errors="coerce")
    mask = (
        df["mission"].isin(["PlanetiQ", "Spire", "TSX"])
        & df["latitude_band_magnetic"].isin(LAT_ORDER)
        & df["tangent_height_km"].between(50, 1000)
        & df["delta_cn0_db_p50"].notna()
        & df["time_utc"].between(WINDOW_START_UTC, WINDOW_END_UTC, inclusive="both")
    )
    out = df[mask].copy()
    out["source_kind"] = "strict_v2_conphs_60s_cn0_proxy"
    out["cn0_proxy_method"] = "existing_strict_v2_60s_window_residual"
    return out[[
        "mission",
        "profile_id",
        "source_file",
        "time_utc",
        "latitude_band_magnetic",
        "tangent_height_km",
        "delta_cn0_db_p05",
        "delta_cn0_db_p50",
        "delta_cn0_db_p95",
        "delta_cn0_db_std",
        "n_samples_snr",
        "source_kind",
        "cn0_proxy_method",
    ]]


def load_metop_cn0_windows() -> pd.DataFrame:
    if LEORO_METOP_CN0_CACHE.exists():
        return pd.read_csv(LEORO_METOP_CN0_CACHE, parse_dates=["time_utc"])
    paths = collect_metop_paths()
    partial_cache = LEORO_METOP_CN0_CACHE.with_suffix(".partial.csv")
    rows: list[dict[str, object]] = []
    sys.path.insert(0, str(LEORO_ROOT / ".leo_ro_pydeps"))
    sys.path.insert(0, str(LEORO_ROOT / "leo_ro_processing"))
    import h5py  # noqa: WPS433
    import process_metop_chang2025_style_10s_l1 as metop_proc  # noqa: WPS433

    for i, p in enumerate(paths, 1):
        try:
            rows.extend(process_metop_cn0_profile(Path(p), h5py, metop_proc))
        except Exception as exc:
            if i <= 5 or i % 500 == 0:
                print(f"skip MetOp profile {i}/{len(paths)}: {Path(p).name}: {exc}", flush=True)
            continue
        if i % 100 == 0:
            print(f"processed MetOp C/N0 profiles {i}/{len(paths)}; windows={len(rows)}", flush=True)
        if i % 250 == 0 and rows:
            pd.DataFrame(rows).to_csv(partial_cache, index=False)
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=LEORO_POINT_COLUMNS)
    out.to_csv(LEORO_METOP_CN0_CACHE, index=False)
    if partial_cache.exists():
        partial_cache.unlink()
    return out


def collect_metop_paths() -> list[str]:
    cols = ["mission", "local_path", "bin_start_time_utc", "height_km", "latitude_band_magnetic"]
    paths: set[str] = set()
    for chunk in pd.read_csv(LEORO_METOP_WINDOWS, usecols=cols, chunksize=250000, low_memory=False):
        t = pd.to_datetime(chunk["bin_start_time_utc"], utc=True, errors="coerce")
        mask = (
            chunk["mission"].isin(["MetOp-B", "MetOp-C"])
            & chunk["latitude_band_magnetic"].isin(LAT_ORDER)
            & pd.to_numeric(chunk["height_km"], errors="coerce").between(50, 1000)
            & t.between(WINDOW_START_UTC, WINDOW_END_UTC, inclusive="both")
        )
        paths.update(chunk.loc[mask, "local_path"].dropna().astype(str).unique().tolist())
    return sorted(paths)


def process_metop_cn0_profile(p: Path, h5py_module, metop_proc) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not p.exists():
        return rows
    platform, _prn, _event = metop_proc.parse_file_name(p)
    with h5py_module.File(p, "r") as f:
        snr_codes = metop_proc.decode_codes(f["snr_observation_code"][()])
        phase_codes = metop_proc.decode_codes(f["phase_observation_code"][()])
        time = metop_proc.clean(f["time"][()])
        start_gps = float(np.asarray(f["start_time"][()]).reshape(-1)[0])
        snr_all = metop_proc.clean(f["snr"][()])
        phase_all = metop_proc.clean(f["excess_phase"][()])
        h_km, lat, lon, _los_s = metop_proc.tangent_from_ecf(f["receiver_orbit"][()], f["transmitter_orbit"][()])
    idx = metop_proc.choose_l1_signal(snr_codes, phase_codes, phase_all)
    if idx is None:
        return rows
    n = min(len(time), snr_all.shape[1], len(h_km), len(lat), len(lon))
    tt = np.asarray(time[:n], dtype=float)
    snr = np.asarray(snr_all[idx, :n], dtype=float)
    h = np.asarray(h_km[:n], dtype=float)
    lat = np.asarray(lat[:n], dtype=float)
    lon = np.asarray(lon[:n], dtype=float)
    abs_time = pd.Timestamp("1980-01-06T00:00:00Z") + pd.to_timedelta(start_gps + tt - metop_proc.GPS_UTC_OFFSET_S, unit="s")
    return one_second_cn0_points(
        platform,
        p.stem,
        p.name,
        "L1",
        tt,
        abs_time,
        snr,
        h,
        lat,
        lon,
        min_seconds_for_fit=18,
    )


def load_leoro_events() -> pd.DataFrame:
    if LEORO_EVENTS_CACHE.exists():
        return pd.read_csv(LEORO_EVENTS_CACHE)
    df = pd.read_csv(LEORO_EVENT_SOURCE, low_memory=False)
    for col in ["s4_p90", "sigma_phi_rad_p90", "height_median_km", "height_min_km", "height_max_km", "n_windows_ok"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    t = pd.to_datetime(df["date_start"], utc=True, errors="coerce")
    missing = t.isna()
    if missing.any():
        parsed = df.loc[missing, "source_file"].map(parse_conphs_time_from_name)
        t.loc[missing] = pd.to_datetime(parsed, utc=True, errors="coerce")
    mask = (
        df["mission"].isin(MISSIONS)
        & df["latitude_band_magnetic"].isin(LAT_ORDER)
        & df["s4_p90"].between(0, 1)
        & df["sigma_phi_rad_p90"].between(0, 1)
        & t.between(WINDOW_START_UTC, WINDOW_END_UTC, inclusive="both")
    )
    out = df[mask].copy()
    out["event_time_for_filter_utc"] = t[mask].astype(str)
    out.to_csv(LEORO_EVENTS_CACHE, index=False)
    return out


def parse_conphs_time_from_name(name: object) -> pd.Timestamp | pd.NaT:
    text = str(name)
    m = re.search(r"\.(\d{4})\.(\d{3})\.(\d{2})\.(\d{2})\.", text)
    if not m:
        return pd.NaT
    year, doy, hour, minute = map(int, m.groups())
    return pd.Timestamp(year=year, month=1, day=1, tz="UTC") + pd.Timedelta(days=doy - 1, hours=hour, minutes=minute)


def style_height_axis_cn0(ax: plt.Axes, show_y: bool) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.set_ylim(50, 1000)
    ax.set_yticks([50, 90, 150, 300, 600, 800, 1000])
    ax.set_yticklabels(["50", "90", "150", "300", "600", "800", "1000"], fontsize=6.7)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xscale("symlog", linthresh=0.3, linscale=0.36)
    ax.set_xlim(-10, 10)
    ax.set_xticks([-10, -3, -1, -0.3, 0, 0.3, 1, 3, 10])
    ax.set_xticklabels(["-10", "-3", "-1", "-0.3", "0", "0.3", "1", "3", "10"], fontsize=6.0)
    add_height_background(ax, show_y)
    ax.axvline(0, color=INK, lw=0.45, ls=(0, (2.0, 2.3)), alpha=0.42)
    finish_axis(ax, show_y)


def style_height_axis_s4(ax: plt.Axes, show_y: bool) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.set_ylim(50, 1000)
    ax.set_yticks([50, 90, 150, 300, 600, 800, 1000])
    ax.set_yticklabels(["50", "90", "150", "300", "600", "800", "1000"], fontsize=6.7)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=6.0)
    add_height_background(ax, show_y)
    finish_axis(ax, show_y)


def add_height_background(ax: plt.Axes, show_y: bool) -> None:
    spans = [
        ("D", 50, 90, "#f3e6c8"),
        ("E", 90, 150, "#c9ddf2"),
        ("F1", 150, 300, "#d7c9ee"),
        ("F2", 300, 600, "#d7c9ee"),
        ("topside", 600, 1000, "#ead0da"),
    ]
    for _name, lo, hi, color in spans:
        ax.axhspan(lo, hi, color=color, alpha=0.155, zorder=-3)
    if show_y:
        x0, x1 = ax.get_xlim()
        for name, lo, hi, _ in spans:
            ax.text(x0 + 0.03 * (x1 - x0), (lo + hi) / 2, name, fontsize=5.4, color=MUTED, alpha=0.98, va="center", ha="left")


def finish_axis(ax: plt.Axes, show_y: bool) -> None:
    ax.grid(True, color=GRID, lw=0.38, zorder=0)
    ax.tick_params(length=1.9, width=0.45, colors=INK, pad=1.5)
    if not show_y:
        ax.set_yticklabels([])
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(SPINE)
        ax.spines[spine].set_linewidth(0.48)


def centered_abs_envelope(points: pd.DataFrame, value_col: str, height_col: str) -> pd.DataFrame:
    bins = np.arange(50, 1000 + 32, 32)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = points.loc[points[height_col].between(lo, hi, inclusive="left"), value_col].dropna().to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals >= -10) & (vals <= 10)]
        if vals.size < 8:
            continue
        amp = np.abs(vals)
        q = np.nanpercentile(amp, [50, 90, 95])
        rows.append({"height_km": (lo + hi) / 2, "p50_abs": q[0], "p90_abs": q[1], "p95_abs": q[2]})
    out = pd.DataFrame(rows)
    for col in ["p50_abs", "p90_abs", "p95_abs"]:
        if col in out:
            out[col] = out[col].rolling(3, center=True, min_periods=1).median()
    return out


def quantile_envelope(points: pd.DataFrame, value_col: str, height_col: str) -> pd.DataFrame:
    bins = np.arange(50, 1000 + 32, 32)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = points.loc[points[height_col].between(lo, hi, inclusive="left"), value_col].dropna().to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 20:
            continue
        q = np.nanpercentile(vals, [50, 90, 95])
        rows.append({"height_km": (lo + hi) / 2, "p50": q[0], "p90": q[1], "p95": q[2]})
    out = pd.DataFrame(rows)
    for col in ["p50", "p90", "p95"]:
        if col in out:
            out[col] = out[col].rolling(3, center=True, min_periods=1).median()
    return out


def plot_lugre_panel(ax: plt.Axes, lugre: pd.DataFrame, lat: str) -> None:
    sub = lugre[lugre["lat_group"].eq(lat)].copy()
    sub = sub[sub["cn0_detrend_zero_mean_dbhz"].between(-10, 10) & sub["cn0_dbhz"].between(15, 55)]
    plot = sub.sample(36000, random_state=42) if len(sub) > 36000 else sub
    ax.scatter(plot["cn0_detrend_zero_mean_dbhz"], plot["lugre_h_tan_km"], s=0.78, color=LUGRE_COLOR, alpha=0.052, linewidths=0, rasterized=True, zorder=3)
    env = centered_abs_envelope(sub, "cn0_detrend_zero_mean_dbhz", "lugre_h_tan_km")
    for key, lw, alpha in [("p95_abs", 1.02, 0.76), ("p90_abs", 1.22, 0.95), ("p50_abs", 0.72, 0.46)]:
        if env.empty:
            continue
        y = env["height_km"].to_numpy(float)
        v = env[key].to_numpy(float)
        ax.plot(v, y, color=LUGRE_COLOR, lw=lw, alpha=alpha, zorder=4)
        ax.plot(-v, y, color=LUGRE_COLOR, lw=lw, alpha=alpha, zorder=4)


def mission_display_name(mission: object) -> str:
    text = str(mission)
    if text.startswith("MetOp"):
        return "MetOp"
    return text


def plot_combined_height_cn0_panel(ax: plt.Axes, lugre: pd.DataFrame, windows: pd.DataFrame, lat: str) -> None:
    lug = lugre[lugre["lat_group"].eq(lat)].copy()
    lug = lug[lug["cn0_detrend_zero_mean_dbhz"].between(-10, 10) & lug["cn0_dbhz"].between(15, 55)]
    lug_plot = lug.sample(28000, random_state=42) if len(lug) > 28000 else lug
    ax.scatter(
        lug_plot["cn0_detrend_zero_mean_dbhz"],
        lug_plot["lugre_h_tan_km"],
        s=0.62,
        color=LUGRE_COLOR,
        alpha=0.035,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    lug_env = centered_abs_envelope(lug, "cn0_detrend_zero_mean_dbhz", "lugre_h_tan_km")
    for key, lw, alpha in [("p95_abs", 1.0, 0.58), ("p90_abs", 1.18, 0.86)]:
        if lug_env.empty:
            continue
        y = lug_env["height_km"].to_numpy(float)
        v = lug_env[key].to_numpy(float)
        ax.plot(v, y, color=LUGRE_COLOR, lw=lw, alpha=alpha, zorder=4)
        ax.plot(-v, y, color=LUGRE_COLOR, lw=lw, alpha=alpha, zorder=4)

    leo = windows[windows["latitude_band_magnetic"].eq(lat)].copy()
    leo = leo[leo["delta_cn0_db"].between(-10, 10)]
    leo["mission_plot"] = leo["mission"].map(mission_display_name)
    for im, mission in enumerate(CN0_MISSION_PLOT_ORDER):
        sm = leo[leo["mission_plot"].eq(mission)]
        if sm.empty:
            continue
        plot = sm.sample(min(len(sm), 22000), random_state=83 + im)
        ax.scatter(
            plot["delta_cn0_db"],
            plot["tangent_height_km"],
            s=0.72,
            color=MISSION_COLORS.get(mission, LEORO_COLOR),
            alpha=0.075 if mission not in {"MetOp", "FengYun"} else 0.058,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
    leo_env = centered_abs_envelope(leo, "delta_cn0_db", "tangent_height_km")
    for key, lw, alpha in [("p95_abs", 0.98, 0.62), ("p90_abs", 1.16, 0.90)]:
        if leo_env.empty:
            continue
        y = leo_env["height_km"].to_numpy(float)
        v = leo_env[key].to_numpy(float)
        ax.plot(v, y, color=LEORO_COLOR, lw=lw, alpha=alpha, zorder=5)
        ax.plot(-v, y, color=LEORO_COLOR, lw=lw, alpha=alpha, zorder=5)
    ax.text(
        0.98,
        0.975,
        f"LuGRE n={len(lug):,}\nLEO-RO n={len(leo):,}",
        transform=ax.transAxes,
        fontsize=5.4,
        color=MUTED,
        va="top",
        ha="right",
        linespacing=1.15,
    )


def plot_leoro_height_cn0_panel(ax: plt.Axes, windows: pd.DataFrame, lat: str) -> None:
    sub = windows[windows["latitude_band_magnetic"].eq(lat)].copy()
    sub = sub[sub["delta_cn0_db"].between(-10, 10)]
    for im, mission in enumerate(MISSIONS):
        sm = sub[sub["mission"].eq(mission)]
        if sm.empty:
            continue
        plot = sm.sample(min(len(sm), 30000), random_state=53 + im)
        ax.scatter(
            plot["delta_cn0_db"],
            plot["tangent_height_km"],
            s=0.72,
            color=MISSION_COLORS.get(mission, LEORO_COLOR),
            alpha=0.040,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        track_ids = sm["profile_id"].dropna().drop_duplicates()
        if len(track_ids) > 0:
            chosen = track_ids.sample(min(len(track_ids), 18), random_state=83 + im)
            for (_pid, _sig), g in sm[sm["profile_id"].isin(chosen)].groupby(["profile_id", "signal"], dropna=False):
                if len(g) < 8:
                    continue
                g = g.sort_values("tangent_height_km")
                ax.plot(
                    g["delta_cn0_db"],
                    g["tangent_height_km"],
                    color=MISSION_COLORS.get(mission, LEORO_COLOR),
                    lw=0.23,
                    alpha=0.055,
                    zorder=2,
                    rasterized=True,
                )
    env = centered_abs_envelope(sub, "delta_cn0_db", "tangent_height_km")
    for key, lw, alpha in [("p95_abs", 1.02, 0.76), ("p90_abs", 1.22, 0.95), ("p50_abs", 0.72, 0.46)]:
        if env.empty:
            continue
        y = env["height_km"].to_numpy(float)
        v = env[key].to_numpy(float)
        ax.plot(v, y, color=LEORO_COLOR, lw=lw, alpha=alpha, zorder=4)
        ax.plot(-v, y, color=LEORO_COLOR, lw=lw, alpha=alpha, zorder=4)
    ax.text(0.98, 0.975, f"n={len(sub):,}", transform=ax.transAxes, fontsize=5.6, color=MUTED, va="top", ha="right")


def plot_joint_density_mission(
    ax: plt.Axes,
    df: pd.DataFrame,
    lat: str,
    x_col: str,
    y_col: str,
    *,
    label_n: str = "n",
) -> object | None:
    sub = df[df["latitude_band_magnetic"].eq(lat)].copy()
    sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])]
    sub = sub[sub[x_col].between(0, 1) & sub[y_col].between(0, 1)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color=MUTED)
        return None
    hb = ax.hexbin(
        sub[x_col].to_numpy(float),
        sub[y_col].to_numpy(float),
        gridsize=42,
        extent=(0, 1, 0, 1),
        mincnt=1,
        bins="log",
        cmap=B_DENSITY_CMAP,
        linewidths=0,
        alpha=0.82,
        zorder=1,
    )
    sub["mission_plot"] = sub["mission"].map(mission_display_name)
    for im, mission in enumerate(MISSION_PLOT_ORDER):
        sm = sub[sub["mission_plot"].eq(mission)]
        if sm.empty:
            continue
        plot = sm.sample(min(len(sm), 520), random_state=121 + im)
        ax.scatter(
            plot[x_col],
            plot[y_col],
            s=8.0,
            color=MISSION_COLORS.get(mission, INK),
            alpha=0.62,
            linewidths=0,
            rasterized=True,
            zorder=4,
        )
    ax.text(0.02, 1.015, f"{label_n}={len(sub):,}", transform=ax.transAxes, fontsize=5.9, color=MUTED, va="bottom", clip_on=False)
    return hb


def ground_station_day_p90(ground: pd.DataFrame) -> pd.DataFrame:
    g = ground.copy()
    g["date"] = pd.to_datetime(g["utc_minute"], utc=True, errors="coerce").dt.date.astype(str)
    out = (
        g.dropna(subset=["lat_group_magnetic_approx", "station", "date", "s4", "sigma_phi"])
        .groupby(["lat_group_magnetic_approx", "station", "date"], as_index=False)
        .agg(s4_p90=("s4", lambda x: quantile_or_nan(x, 0.90)), sigma_phi_p90=("sigma_phi", lambda x: quantile_or_nan(x, 0.90)), n_minutes=("s4", "size"))
        .rename(columns={"lat_group_magnetic_approx": "latitude_band_magnetic"})
    )
    return out[out["n_minutes"].ge(3)].copy()


def plot_joint_density(ax: plt.Axes, df: pd.DataFrame, lat: str, x_col: str, y_col: str, *, label_n: str = "n") -> object | None:
    sub = df[df["latitude_band_magnetic"].eq(lat)].copy()
    sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])]
    sub = sub[sub[x_col].between(0, 1) & sub[y_col].between(0, 1)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color=MUTED)
        return None
    x = np.clip(sub[x_col].to_numpy(float), 0, 1)
    y = np.clip(sub[y_col].to_numpy(float), 0, 1)
    hb = ax.hexbin(x, y, gridsize=42, extent=(0, 1, 0, 1), mincnt=1, bins="log", cmap=GROUND_DENSITY_CMAP, linewidths=0, alpha=0.98)
    ax.text(0.02, 1.015, f"{label_n}={len(sub):,}", transform=ax.transAxes, fontsize=5.9, color=MUTED, va="bottom", clip_on=False)
    return hb


def style_unit_axis(ax: plt.Axes, show_y: bool) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=6.4)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"] if show_y else [], fontsize=6.4)
    ax.tick_params(length=1.9, width=0.45, colors=INK, pad=1.5)
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(SPINE)
        ax.spines[spine].set_linewidth(0.48)


def add_right_unit_axis(ax: plt.Axes) -> None:
    ticks = np.linspace(0, 1, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=6.4)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", right=True, labelright=True, left=False, labelleft=False, length=1.9, width=0.45, colors=INK, pad=1.5)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(SPINE)
    ax.spines["right"].set_linewidth(0.48)


def add_right_height_axis(ax: plt.Axes) -> None:
    ax.set_yticks([50, 90, 150, 300, 600, 800, 1000])
    ax.set_yticklabels(["50", "90", "150", "300", "600", "800", "1000"], fontsize=6.7)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Tangent height (km)", fontsize=7.4, rotation=270, labelpad=15)
    ax.tick_params(axis="y", right=True, labelright=True, left=False, labelleft=False, length=1.9, width=0.45, colors=INK, pad=1.5)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(SPINE)
    ax.spines["right"].set_linewidth(0.48)


def build_summary(lugre: pd.DataFrame, windows: pd.DataFrame, events: pd.DataFrame, ground: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lat in LAT_ORDER:
        sub = lugre[lugre["lat_group"].eq(lat)]
        rows.append({"panel": "a LuGRE", "observable": "abs_delta_cn0_db", "lat_group": lat, "n": int(sub["abs_cn0_detrended_db"].notna().sum()), "p50": quantile_or_nan(sub["abs_cn0_detrended_db"], 0.50), "p90": quantile_or_nan(sub["abs_cn0_detrended_db"], 0.90), "p95": quantile_or_nan(sub["abs_cn0_detrended_db"], 0.95), "source_file": str(LUGRE_4FITTING_XLSX), "notes": f"S phase only, restricted to {WINDOW_LABEL}; latest 4-fitting LuGRE source."})
        subw = windows[windows["latitude_band_magnetic"].eq(lat)]
        cn0_abs = subw["delta_cn0_db"].abs()
        rows.append({"panel": "a LEO-RO", "observable": "abs_snr_derived_cn0_proxy_residual_db", "lat_group": lat, "n": int(cn0_abs.notna().sum()), "p50": quantile_or_nan(cn0_abs, 0.50), "p90": quantile_or_nan(cn0_abs, 0.90), "p95": quantile_or_nan(cn0_abs, 0.95), "source_file": str(CN0_HP_SOURCE), "notes": f"PlanetiQ/Spire/TSX/MetOp use latest 1 Hz median-SNR C/N0-proxy residuals with 0.05 Hz high-pass cutoff; FengYun excluded; S-phase time window; geomagnetic dip latitude bands joined from {LEORO_EVENT_SOURCE}."})
        sube = events[events["latitude_band_magnetic"].eq(lat)]
        for obs, col in [("event_S4_P90", "s4_p90"), ("event_sigma_phi_rad_P90", "sigma_phi_rad_p90")]:
            rows.append({"panel": "b LEO-RO", "observable": obs, "lat_group": lat, "n": int(sube[col].notna().sum()), "p50": quantile_or_nan(sube[col], 0.50), "p90": quantile_or_nan(sube[col], 0.90), "p95": quantile_or_nan(sube[col], 0.95), "source_file": str(LEORO_EVENT_SOURCE), "notes": f"Five missions {', '.join(MISSIONS)}; event-level P90; S-phase time window; Sentinel-6A excluded."})
        subg = ground[ground["lat_group_magnetic_approx"].eq(lat)]
        subg = subg[subg["s4"].between(0, 1) & subg["sigma_phi"].between(0, 1)]
        for obs, col in [("ground_station_minute_S4", "s4"), ("ground_station_minute_sigma_phi_rad", "sigma_phi")]:
            rows.append({"panel": "c Ground ISMR", "observable": obs, "lat_group": lat, "n": int(subg[col].notna().sum()), "p50": quantile_or_nan(subg[col], 0.50), "p90": quantile_or_nan(subg[col], 0.90), "p95": quantile_or_nan(subg[col], 0.95), "source_file": str(GROUND_ISMR), "notes": f"Ground station-minute context restricted to {WINDOW_LABEL}; not event matching."})
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_OUT, index=False)
    return out


def draw_figure(lugre: pd.DataFrame, windows: pd.DataFrame, events: pd.DataFrame, ground: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(7.25, 8.55), dpi=350)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.95, 1.07, 1.07], hspace=0.33, wspace=0.16, left=0.122, right=0.925, top=0.958, bottom=0.052)

    for j, lat in enumerate(LAT_ORDER):
        ax = fig.add_subplot(gs[0, j])
        plot_combined_height_cn0_panel(ax, lugre, windows, lat)
        style_height_axis_cn0(ax, j == 0)
        if j == 2:
            add_right_height_axis(ax)
        ax.set_title(lat, color=INK, fontsize=8.0, weight="bold", pad=5)
        ax.set_xlabel(r"$\delta C/N_0$ (dB)", fontsize=7.0, labelpad=2.5)
        if j == 0:
            ax.set_ylabel("Tangent height (km)", fontsize=7.4)
            ax.text(-0.28, 1.035, "a", transform=ax.transAxes, fontsize=10.6, weight="bold", color=INK)
            ax.text(-0.385, 0.5, r"LuGRE + LEO-RO $C/N_0$", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=8.0, weight="bold", color=INK)
            handles = [
                Line2D([0], [0], color=LUGRE_COLOR, lw=1.25, label="LuGRE"),
                Line2D([0], [0], color=LEORO_COLOR, lw=1.25, label="LEO-RO"),
            ]
            ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=4.9, handletextpad=0.28, borderpad=0.25, labelspacing=0.18)

    last_hb_c = None
    for j, lat in enumerate(LAT_ORDER):
        ax = fig.add_subplot(gs[1, j])
        hb = plot_joint_density_mission(ax, events, lat, "s4_p90", "sigma_phi_rad_p90", label_n="n")
        if hb is not None:
            last_hb_c = hb
        style_unit_axis(ax, j == 0)
        if j == 0:
            ax.set_ylabel(r"$\sigma_\phi$ (rad)", fontsize=7.0)
            ax.text(-0.28, 1.035, "b", transform=ax.transAxes, fontsize=10.6, weight="bold", color=INK)
            ax.text(-0.385, 0.5, r"LEO-RO $S_4$ / $\sigma_\phi$", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=8.1, weight="bold", color=INK)
            handles = [
                Line2D([0], [0], marker="o", color="none", markerfacecolor=MISSION_COLORS[m], markeredgewidth=0, markersize=3.6, label=m)
                for m in MISSION_PLOT_ORDER
            ]
            ax.legend(
                handles=handles,
                loc="upper left",
                ncol=4,
                frameon=True,
                fontsize=4.9,
                handletextpad=0.18,
                columnspacing=0.55,
                borderpad=0.22,
                labelspacing=0.12,
            )
        if j == 2:
            add_right_unit_axis(ax)
            ax.set_ylabel(r"$\sigma_\phi$ (rad)", fontsize=7.0, rotation=90, labelpad=12)
        ax.set_xlabel(r"$S_4$", fontsize=6.8, labelpad=2.5)

    last_hb_d = None
    for j, lat in enumerate(LAT_ORDER):
        ax = fig.add_subplot(gs[2, j])
        g = ground.rename(columns={"lat_group_magnetic_approx": "latitude_band_magnetic"})
        hb = plot_joint_density(ax, g, lat, "s4", "sigma_phi", label_n="n")
        if hb is not None:
            last_hb_d = hb
        style_unit_axis(ax, j == 0)
        if j == 0:
            ax.set_ylabel(r"$\sigma_\phi$ (rad)", fontsize=7.4)
            ax.set_xlabel(r"$S_4$", fontsize=7.4)
            ax.text(-0.28, 1.035, "c", transform=ax.transAxes, fontsize=10.6, weight="bold", color=INK)
            ax.text(-0.385, 0.5, "Ground ISMR $S_4$ / $\\sigma_\\phi$", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=8.0, weight="bold", color=INK)
        else:
            ax.set_xlabel(r"$S_4$", fontsize=7.4)
        if j == 2:
            add_right_unit_axis(ax)
            ax.set_ylabel(r"$\sigma_\phi$ (rad)", fontsize=7.4, rotation=90, labelpad=12)
    fig.savefig(PNG_OUT, dpi=350)
    plt.close(fig)


def write_readme(summary: pd.DataFrame) -> None:
    caption = (
        f"Receiver-regime comparison during the S-phase interval ({WINDOW_LABEL}). "
        "Panel a overlays LuGRE 4-fitting detrended C/N0 residuals for GPS L1 and Galileo E1 with available LEO-RO 1 Hz SNR-derived C/N0-proxy residuals high-pass filtered at 0.05 Hz, making the different height coverage explicit against the broader LuGRE tangent-height range. "
        "Panel b shows high-rate LEO-RO event-level S4 and sigma_phi P90 using the same mission palette, with MetOp-B and MetOp-C grouped as MetOp. Panel c shows ground ISMR station-minute S4--sigma_phi surface context. "
        "These panels compare receiver and observable regimes rather than one-to-one event validation."
    )
    readme = f"""# Fig.3 merged-height candidate without FengYun C/N0 context, LuGRE L1/E1 v6

Generated by `make_fig3_receiver_regime_comparison_merged_height_lugre_l1e1_v9_hp05_no_fengyun.py`.

## Outputs

- `fig3_receiver_regime_comparison_merged_height_lugre_l1e1_v9_hp05_no_fengyun.png`
- `fig3_receiver_regime_comparison_merged_height_lugre_l1e1_v9_hp05_no_fengyun.pdf`
- `fig3_receiver_regime_comparison_merged_height_lugre_l1e1_v9_hp05_no_fengyun_source_summary.csv`
- `README_figure3_lugre_l1e1_v9_hp05_no_fengyun.md`

## Design change

- Panel a merges the previous separate LuGRE and LEO-RO height-resolved C/N0 rows. LuGRE is restricted to `GPS L1` and `Galileo E1` from the 4-fitting source over the full 50--1000 km tangent-height range; LEO-RO is overlaid only where conPhs/atmPhs high-rate samples are available.
- FengYun is excluded from panel a in this version because its C/N0 residual processing is currently under review.
- PlanetiQ, Spire and TSX use the blue/green/red palette from the user-preferred event-level density example.
- MetOp-B and MetOp-C are displayed as one `MetOp` category in the figure. The source summary still retains the original mission names through the underlying source files.
- Panel b uses a dark-to-red logarithmic density palette; panel c uses a purple-blue-to-yellow-green density palette, following the two user-supplied style references. Colorbars are omitted from the main figure to reduce clutter.
- Panel c quality control is limited to finite station-minute values and the physical display interval 0--1 for `S_4` and `sigma_phi`; no additional P90 clipping is applied, so the tail structure remains visible.
- v2 removes the small surface-context note below panel c, adds right-side `sigma_phi` axis labels to panels b and c, and makes the panel-b mission legend horizontal.
- v3 adds `delta C/N0` x-axis labels to all panel-a columns, restores right-side y-axis ticks in the final columns of panels b and c, and reduces bottom whitespace.
- v4 adds complete panel borders, shifts the grid left to avoid right-label clipping, makes the ground row title one vertical line, and restricts LuGRE to L1/E1 for same-frequency comparison with LEO-RO L1 observables.
- v5 excludes FengYun from panel a and flips the right-side `sigma_phi` labels to the reading direction used elsewhere in the figure.
- v6 adds a right-side tangent-height axis to panel a and darkens the D/E/F/topside labels.
- v8 updates panel-a LEO-RO C/N0-proxy residuals to the latest `processed_cn0_1hz_highpass_v2_20260619` source and uses the 0.05 Hz high-pass residual column (`delta_cn0_hp_0p05_db`). LuGRE, panel b and panel c sources are unchanged from v6.

## Sources

- Panel a LuGRE: `{LUGRE_4FITTING_XLSX}`; S phase; latest 4-fitting source; `GPS L1` and `Galileo E1` only.
- Panel a LEO-RO C/N0-proxy points: `{CN0_HP_SOURCE}`; 1 Hz median-SNR-derived C/N0-proxy residual points; 0.05 Hz high-pass cutoff (`delta_cn0_hp_0p05_db`); S-phase time window; geomagnetic dip latitude bands joined from `{LEORO_EVENT_SOURCE}`.
- Panel b LEO-RO events: `{LEORO_EVENT_SOURCE}`; PlanetiQ/Spire/TSX/MetOp-B/C only; event-level `S_4` P90 versus `sigma_phi` P90.
- Panel c Ground ISMR: `{GROUND_ISMR}`; station-minute `S_4` and `sigma_phi`; retained surface-day context.

## Caption draft

{caption}

## Caveat

Panel a LEO-RO is an SNR-derived `C/N0` proxy residual in dB, not an absolute calibrated receiver `C/N0` product. For each PlanetiQ/Spire/TSX/MetOp profile/signal, high-rate SNR is aggregated to 1 Hz as `20 log10(median(SNR within 1 s))`, then the latest zero-phase 0.05 Hz high-pass processing is used before plotting signed residual points. FengYun is intentionally excluded from this version. The limited LEO-RO height extent should be read as available conPhs/atmPhs high-rate sample coverage, not as a complete electron-density profile. Panel b uses event-level P90 values because each high-rate LEO-RO event contains many 10 s windows; P90 gives a robust upper-tail event amplitude without allowing long events to dominate by contributing many raw-window points. Panel c is ground station-minute context and is not clipped to the P90; the figure only filters to the physical display range 0--1 for `S_4` and `sigma_phi`. Event-level validation belongs in Fig.4.
"""
    README_OUT.write_text(readme, encoding="utf-8")


def main() -> None:
    for path in [LUGRE_4FITTING_XLSX, LEORO_EVENT_SOURCE, LEORO_CONPHS_WINDOWS, LEORO_METOP_WINDOWS, GROUND_ISMR]:
        if not path.exists():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)
    lugre = load_lugre_points()
    windows = load_leoro_windows()
    events = load_leoro_events()
    ground = load_ground_ismr()
    summary = build_summary(lugre, windows, events, ground)
    draw_figure(lugre, windows, events, ground)
    write_readme(summary)
    print(PNG_OUT)
    print(PDF_OUT)
    print(SUMMARY_OUT)
    print(README_OUT)


if __name__ == "__main__":
    main()

