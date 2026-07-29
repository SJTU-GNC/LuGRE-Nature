from __future__ import annotations

import csv
import html
from datetime import timezone
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib as mpl
import matplotlib.dates as mdates

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
ROOT = PACKAGE_ROOT / "data" / "analysis_ready" / "ED6_9"
DATA_XLSX = ROOT / "unused_phase_story.xlsx"
CTL_CACHE = ROOT / "unused_ctl_cache.csv"
S_TRACK_LIBRARY = ROOT / "unused_s_phase_track_library"
S_TRACK_MANIFEST = S_TRACK_LIBRARY / "manifest.csv"
S_CACHE = ROOT / "observations_s_phase.csv.gz"
LINE_GEOMETRY_CSV = ROOT / "unused_line_geometry.csv"
RECOMPUTED_TANGENT_CACHE = ROOT / "tangent_geometry_ecef_s_phase.csv.gz"
OUT_DIR = PACKAGE_ROOT / "outputs"
ALL_TRACKS_STEM = "All_OPs_ECEF_geographic_tracks_50_1000km"
ALL_TRACKS_HTML = "all_ops_ecef_geographic_tracks_50_1000km.html"

EARTH_RADIUS_KM = 6378.137
DISPLAY_HOURS_PER_WINDOW = 4.0
DISPLAY_GAP_HOURS = 0.18
SHORT_WINDOW_THRESHOLD_HOURS = 0.75
SHORT_WINDOW_DISPLAY_HOURS = 1.05
S_COMPACT_MIN_DISPLAY_HOURS = 0.75
FIGSIZE = (15.4, 13.2)
GRID_HEIGHT_RATIOS = [1.05, 1.0, 1.0, 1.0, 1.05, 1.25, 2.05]

CTL_GROUPS = [
    {
        "figure": "S1",
        "title": "CTL observing windows: Phase C, Phase T, and Phase L",
        "out_stem": "S1_CTL_Phase_C_T_L_compact",
        "members": [
            ("C", "1_0"),
            ("C", "2_0"),
            ("T", "5_0"),
            ("T", "9_0"),
            ("T", "12_0"),
            ("T", "14_0"),
            ("T", "17_0"),
            ("T", "18_0"),
            ("T", "21_0"),
            ("T", "22_0"),
            ("L", "23_0"),
            ("L", "27_0"),
            ("L", "37_0"),
        ],
    },
]

S_GROUPS = [
    {
        "figure": "S2",
        "title": "S phase observing day 1, OP38",
        "out_stem": "S2_S_Phase_Day1_OP38",
        "ops": ["38_0"],
    },
    {
        "figure": "S3",
        "title": "S phase observing day 2, OP40",
        "out_stem": "S3_S_Phase_Day2_OP40",
        "ops": ["40_0"],
    },
    {
        "figure": "S4",
        "title": "S phase observing day 3, OP74",
        "out_stem": "S4_S_Phase_Day3_OP74",
        "ops": ["74_0"],
    },
    {
        "figure": "S5",
        "title": "S phase observing day 4, OP76 + OP77 + OP77_1 + OP78_1",
        "out_stem": "S5_S_Phase_Day4_OP76_OP77_OP77_1_OP78_1",
        "ops": ["76_0", "77_0", "77_1", "78_1"],
    },
]

ROTI_INDEPENDENT_COL = "roti_1min_independent"
OBSERVATION_COLUMNS = ["cn0_dbhz", "plot_cn0_detrended_db", "stec_tecu", "rot_tecu_per_min", ROTI_INDEPENDENT_COL]
CTL_CN0_LIMITS = (18.0, 48.0)
CTL_CN0_TICKS = [20, 25, 30, 35, 40, 45]
S_CN0_LIMITS = (18.0, 42.0)
S_CN0_TICKS = [20, 25, 30, 35, 40]
DETREND_LIMITS = (-3.0, 3.0)
DETREND_TICKS = [-3, -2, -1, 0, 1, 2, 3]
STEC_LIMITS = (0.0, 275.0)
STEC_TICKS = [0, 50, 100, 150, 200, 250]
ROT_LIMITS = (-40.0, 40.0)
ROT_TICKS = [-40, -20, 0, 20, 40]
ROTI_LIMITS = (0.0, 32.0)
ROTI_TICKS = [0, 10, 20, 30]
FIXED_PANEL_SCALES = {
    "plot_cn0_detrended_db": (DETREND_LIMITS, DETREND_TICKS),
    "stec_tecu": (STEC_LIMITS, STEC_TICKS),
    "rot_tecu_per_min": (ROT_LIMITS, ROT_TICKS),
    ROTI_INDEPENDENT_COL: (ROTI_LIMITS, ROTI_TICKS),
}
SPARSE_TAIL_TRIM_RULES = {
    ("L", "37_0"): {"gap": pd.Timedelta(minutes=10), "max_rows": 200, "max_duration": pd.Timedelta(minutes=5)},
    ("S", "40_0"): {"gap": pd.Timedelta(minutes=15), "max_rows": 200, "max_duration": pd.Timedelta(minutes=8)},
}

SATELLITE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#0f766e",
    "#be123c",
    "#4f46e5",
    "#ca8a04",
    "#0891b2",
    "#84cc16",
    "#f43f5e",
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.4,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean_output() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ["S*.png", "S*.pdf", "all_tracks_*.png", "all_tracks_*.pdf", ALL_TRACKS_HTML, "index_manifest.csv", "index.html"]:
        for path in OUT_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def display_op(op_id: object) -> str:
    text = str(op_id)
    if text.endswith("_0") and text[:-2].isdigit():
        text = text[:-2]
    return f"OP{text}"


def robust_range(values: pd.Series, floor: float | None = None, ceil: float | None = None, pad: float = 0.12) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return (0.0, 1.0)
    lo = float(np.nanpercentile(arr, 2.0))
    hi = float(np.nanpercentile(arr, 98.0))
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    lo -= span * pad
    hi += span * pad
    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    return lo, hi


def fixed_panel_scale(y_col: str, df: pd.DataFrame) -> tuple[tuple[float, float], list[float]] | None:
    if y_col == "cn0_dbhz":
        phases = set(df.get("phase", pd.Series(dtype=str)).dropna().astype(str).unique())
        if phases == {"S"}:
            return S_CN0_LIMITS, S_CN0_TICKS
        return CTL_CN0_LIMITS, CTL_CN0_TICKS
    return FIXED_PANEL_SCALES.get(y_col)


def color_map(df: pd.DataFrame) -> dict[str, str]:
    sats = sorted(str(sat) for sat in df["sat"].dropna().unique())
    return {sat: SATELLITE_PALETTE[i % len(SATELLITE_PALETTE)] for i, sat in enumerate(sats)}


def add_independent_minute_roti(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[ROTI_INDEPENDENT_COL] = np.nan
    required = ["phase", "op_id", "sat", "signal_id", "segment_id", "time_utc", "rot_tecu_per_min"]
    if any(col not in out.columns for col in required) or out.empty:
        return out

    key_cols = ["phase", "op_id", "sat", "signal_id", "segment_id"]
    work = out.sort_values(key_cols + ["time_utc"]).copy()
    rot = pd.to_numeric(work["rot_tecu_per_min"], errors="coerce")
    valid_time = work["time_utc"].notna()
    key_change = work[key_cols].astype(str).ne(work[key_cols].astype(str).shift()).any(axis=1)
    time_gap = work["time_utc"].diff().dt.total_seconds().abs().gt(1.5).fillna(True)
    new_chunk = key_change | time_gap | rot.isna() | rot.shift().isna()
    work["_roti_chunk"] = new_chunk.cumsum()
    finite = valid_time & rot.notna()
    if finite.any():
        grouped = (
            work.loc[finite]
            .assign(_utc_minute=work.loc[finite, "time_utc"].dt.floor("min"))
            .groupby(["_roti_chunk", "_utc_minute"], dropna=False)["rot_tecu_per_min"]
            .transform(lambda s: s.std(ddof=1) if s.notna().sum() >= 2 else np.nan)
        )
        out.loc[grouped.index, ROTI_INDEPENDENT_COL] = grouped.to_numpy(dtype=float)
    return out


def legend_column_count(handles: list[mpl.lines.Line2D]) -> int:
    return min(9, max(1, int(np.ceil(len(handles) / 2)) + 1))


def ordered_phase_op_members() -> list[tuple[str, str]]:
    members: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for group in CTL_GROUPS:
        for phase, op_id in group["members"]:
            item = (str(phase), str(op_id))
            if item not in seen:
                seen.add(item)
                members.append(item)
    for group in S_GROUPS:
        for op_id in group["ops"]:
            item = ("S", str(op_id))
            if item not in seen:
                seen.add(item)
                members.append(item)
    return members


def wanted_tangent_members() -> set[tuple[str, str]]:
    return set(ordered_phase_op_members())


def wrap_lon(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return ((arr + 180.0) % 360.0) - 180.0


def recompute_los_tangent(chunk: pd.DataFrame) -> pd.DataFrame:
    vector_cols = [
        "rx_x_j2000_km",
        "rx_y_j2000_km",
        "rx_z_j2000_km",
        "sat_x_j2000_km",
        "sat_y_j2000_km",
        "sat_z_j2000_km",
    ]
    for col in vector_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    valid = chunk[vector_cols].notna().all(axis=1).to_numpy()
    if not valid.any():
        return pd.DataFrame()

    work = chunk.loc[valid].copy()
    rx = work[["rx_x_j2000_km", "rx_y_j2000_km", "rx_z_j2000_km"]].to_numpy(dtype=float)
    sat = work[["sat_x_j2000_km", "sat_y_j2000_km", "sat_z_j2000_km"]].to_numpy(dtype=float)
    los = sat - rx
    denom = np.einsum("ij,ij->i", los, los)
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = -np.einsum("ij,ij->i", rx, los) / denom
    lam_clip = np.clip(lam, 0.0, 1.0)
    closest = rx + lam_clip[:, None] * los
    height = np.linalg.norm(closest, axis=1) - EARTH_RADIUS_KM

    out = work[
        [
            "phase",
            "op_id",
            "sat",
            "signal_id",
            "signal_name",
            "segment_id",
            "time_utc",
            "rx_gps_seconds",
            "lugre_tangent_lat",
            "lugre_tangent_lon",
        ]
    ].copy()
    out["lugre_lat"] = pd.to_numeric(out.pop("lugre_tangent_lat"), errors="coerce")
    out["lugre_lon"] = wrap_lon(pd.to_numeric(out.pop("lugre_tangent_lon"), errors="coerce"))
    out["lugre_h_tan_km"] = height
    out["lugre_tangent_lambda"] = lam
    out["lugre_tangent_lambda_clip"] = lam_clip
    out["coordinate_source"] = "recomputed_J2000_LOS_closest_point"
    finite = (
        np.isfinite(height)
        & np.isfinite(lam)
        & np.isfinite(out["lugre_lat"].to_numpy(dtype=float))
        & np.isfinite(out["lugre_lon"].to_numpy(dtype=float))
        & (denom > 0.0)
        & (lam >= 0.0)
        & (lam <= 1.0)
    )
    return out.loc[finite].copy()


def load_recomputed_tangent_rows() -> pd.DataFrame:
    if RECOMPUTED_TANGENT_CACHE.exists():
        df = pd.read_csv(RECOMPUTED_TANGENT_CACHE, low_memory=False)
    else:
        if not LINE_GEOMETRY_CSV.exists():
            raise FileNotFoundError(f"Missing J2000 line geometry CSV: {LINE_GEOMETRY_CSV}")
        usecols = [
            "phase",
            "op_id",
            "sat",
            "signal_id",
            "signal_name",
            "segment_id",
            "time_utc",
            "rx_gps_seconds",
            "rx_x_j2000_km",
            "rx_y_j2000_km",
            "rx_z_j2000_km",
            "sat_x_j2000_km",
            "sat_y_j2000_km",
            "sat_z_j2000_km",
            "lugre_tangent_lat",
            "lugre_tangent_lon",
        ]
        wanted = wanted_tangent_members()
        frames = []
        for chunk in pd.read_csv(LINE_GEOMETRY_CSV, usecols=usecols, chunksize=250_000, low_memory=False):
            chunk["phase"] = chunk["phase"].astype(str)
            chunk["op_id"] = chunk["op_id"].astype(str)
            mask = pd.MultiIndex.from_arrays([chunk["phase"], chunk["op_id"]]).isin(wanted)
            chunk = chunk.loc[mask].copy()
            if chunk.empty:
                continue
            recomputed = recompute_los_tangent(chunk)
            if not recomputed.empty:
                frames.append(recomputed)
        if not frames:
            raise RuntimeError(f"No recomputed tangent rows found from {LINE_GEOMETRY_CSV}")
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(RECOMPUTED_TANGENT_CACHE, index=False, encoding="utf-8-sig")

    df["phase"] = df["phase"].astype(str)
    df["op_id"] = df["op_id"].astype(str)
    df["sat"] = df["sat"].astype(str)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    numeric_cols = [
        "rx_gps_seconds",
        "signal_id",
        "segment_id",
        "lugre_lat",
        "lugre_lon",
        "lugre_h_tan_km",
        "lugre_tangent_lambda",
        "lugre_tangent_lambda_clip",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["phase", "op_id", "sat", "time_utc", "lugre_lat", "lugre_lon", "lugre_h_tan_km"]).sort_values(
        ["phase", "op_id", "sat", "signal_id", "segment_id", "time_utc"]
    )


def load_ctl_rows() -> pd.DataFrame:
    cols = [
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
    if CTL_CACHE.exists():
        df = pd.read_csv(CTL_CACHE, low_memory=False)
    else:
        df = pd.read_excel(DATA_XLSX, sheet_name="all_heights_master", usecols=cols)
        df = df[df["phase"].astype(str).isin(["C", "T", "L"])].copy()
        df.to_csv(CTL_CACHE, index=False, encoding="utf-8-sig")
    df["phase"] = df["phase"].astype(str)
    df["op_id"] = df["op_id"].astype(str)
    df["sat"] = df["sat"].astype(str)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    numeric_cols = [
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
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["phase", "op_id", "sat", "time_utc"]).sort_values(
        ["phase", "op_id", "sat", "signal_id", "segment_id", "time_utc"]
    ).pipe(add_independent_minute_roti)


def load_s_rows() -> pd.DataFrame:
    cols = [
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
    if S_CACHE.exists():
        df = pd.read_csv(S_CACHE, low_memory=False)
    else:
        manifest = pd.read_csv(S_TRACK_MANIFEST)
        wanted_ops = {op for group in S_GROUPS for op in group["ops"]}
        manifest = manifest[manifest["op_id"].astype(str).isin(wanted_ops)].dropna(subset=["csv"]).copy()
        frames = []
        for rel_csv in manifest["csv"].astype(str):
            path = ROOT / rel_csv
            frames.append(pd.read_csv(path, usecols=cols, low_memory=False))
        if not frames:
            raise RuntimeError(f"No S phase track CSVs found from {S_TRACK_MANIFEST}")
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(S_CACHE, index=False, encoding="utf-8-sig")
    df["phase"] = df["phase"].astype(str)
    df["op_id"] = df["op_id"].astype(str)
    df["sat"] = df["sat"].astype(str)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    numeric_cols = [
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
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["phase", "op_id", "sat", "time_utc"]).sort_values(
        ["phase", "op_id", "sat", "signal_id", "segment_id", "time_utc"]
    ).pipe(add_independent_minute_roti)


def observation_mask(df: pd.DataFrame) -> pd.Series:
    observed = pd.Series(False, index=df.index)
    for col in OBSERVATION_COLUMNS:
        if col in df.columns:
            observed |= pd.to_numeric(df[col], errors="coerce").notna()
    return df["time_utc"].notna() & observed


def observed_time_span(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    mask = observation_mask(df)
    if not mask.any():
        mask = df["time_utc"].notna()
    return df.loc[mask, "time_utc"].min(), df.loc[mask, "time_utc"].max()


def trim_sparse_tail_cluster(phase: str, op_id: str, sub: pd.DataFrame) -> pd.DataFrame:
    rule = SPARSE_TAIL_TRIM_RULES.get((str(phase), str(op_id)))
    if rule is None:
        return sub
    observed = sub.loc[observation_mask(sub), ["time_utc"]].sort_values("time_utc").dropna()
    if observed.empty:
        return sub
    times = observed["time_utc"].drop_duplicates().reset_index(drop=True)
    gaps = times.diff()
    large_gap_positions = list(np.flatnonzero(gaps.gt(rule["gap"]).to_numpy()))
    for pos in reversed(large_gap_positions):
        tail_times = observed.loc[observed["time_utc"].ge(times.iloc[pos]), "time_utc"]
        tail_duration = tail_times.max() - tail_times.min()
        if len(tail_times) <= int(rule["max_rows"]) or tail_duration <= rule["max_duration"]:
            keep_end = times.iloc[pos - 1]
            return sub[sub["time_utc"].le(keep_end)].copy()
    return sub


def display_width_hours(actual_h: float, default_min_h: float = DISPLAY_HOURS_PER_WINDOW) -> float:
    if actual_h < SHORT_WINDOW_THRESHOLD_HOURS:
        return max(SHORT_WINDOW_DISPLAY_HOURS, actual_h)
    return max(default_min_h, actual_h)


def compact_axis(
    df: pd.DataFrame,
    members: list[tuple[str, str]],
    default_min_display_h: float = DISPLAY_HOURS_PER_WINDOW,
    short_display_h: float = SHORT_WINDOW_DISPLAY_HOURS,
) -> tuple[pd.DataFrame, list[dict[str, object]], float]:
    frames = []
    windows: list[dict[str, object]] = []
    offset = 0.0
    for phase, op_id in members:
        sub = df[(df["phase"].eq(str(phase))) & (df["op_id"].eq(str(op_id)))].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("time_utc")
        sub = trim_sparse_tail_cluster(str(phase), str(op_id), sub)
        if sub.empty:
            continue
        t0, t1 = observed_time_span(sub)
        sub = sub[sub["time_utc"].between(t0, t1)].copy()
        actual_h = max((t1 - t0).total_seconds() / 3600.0, 1.0 / 60.0)
        if actual_h < SHORT_WINDOW_THRESHOLD_HOURS:
            display_h = max(short_display_h, actual_h)
        else:
            display_h = max(default_min_display_h, actual_h)
        elapsed_h = (sub["time_utc"] - t0).dt.total_seconds() / 3600.0
        sub["plot_x"] = offset + elapsed_h / actual_h * display_h
        windows.append(
            {
                "phase": phase,
                "op_id": op_id,
                "label": f"{phase} {display_op(op_id)}",
                "t0": t0,
                "t1": t1,
                "x0": offset,
                "x1": offset + display_h,
                "xc": offset + display_h / 2.0,
                "duration_h": actual_h,
            }
        )
        frames.append(sub)
        offset += display_h + DISPLAY_GAP_HOURS
    if not frames:
        return pd.DataFrame(), [], 0.0
    return pd.concat(frames, ignore_index=True), windows, max(0.0, offset - DISPLAY_GAP_HOURS)


def add_time_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    colors: dict[str, str],
    y_limits: tuple[float, float] | None = None,
    y_ticks: list[float] | None = None,
    zero_line: bool = False,
) -> None:
    fixed_scale = fixed_panel_scale(y_col, df)
    if fixed_scale is not None:
        y_limits, y_ticks = fixed_scale
    for sat, sub in df.groupby("sat", sort=False):
        y = pd.to_numeric(sub[y_col], errors="coerce")
        mask = sub["plot_x"].notna() & y.notna()
        if not mask.any():
            continue
        ax.scatter(
            sub.loc[mask, "plot_x"],
            y.loc[mask],
            s=1.2,
            color=colors.get(str(sat), "#64748b"),
            alpha=0.46,
            edgecolors="none",
            rasterized=True,
        )
    if zero_line:
        ax.axhline(0.0, color="#94a3b8", lw=0.65, alpha=0.75)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e7eb", alpha=0.72, lw=0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.62)
        spine.set_color("#222222")
    ax.tick_params(axis="both", which="major", labelsize=6.4, length=2.0, width=0.5, pad=1.0)


def add_altitude_layers(ax: plt.Axes, ymax: float) -> None:
    layers = [
        (50.0, 90.0, "D", "#f8fafc"),
        (90.0, 150.0, "E", "#eef6ff"),
        (150.0, 600.0, "F", "#f7f3ff"),
        (600.0, 1000.0, "Topside", "#f6fff8"),
    ]
    for lo, hi, label, color in layers:
        ax.axhspan(lo, min(hi, ymax), color=color, alpha=0.72, zorder=0)
        if lo < ymax:
            y = np.sqrt(lo * min(hi, ymax))
            ax.text(
                0.006,
                y,
                label,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=6.4,
                color="#64748b",
                zorder=2,
            )


def add_altitude_panel(ax: plt.Axes, df: pd.DataFrame, colors: dict[str, str]) -> None:
    ymin = 50.0
    ymax = 1000.0
    add_altitude_layers(ax, ymax)
    for sat, sub in df.groupby("sat", sort=False):
        y = pd.to_numeric(sub["lugre_h_tan_km"], errors="coerce")
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
    ax.set_ylabel("h$_{tan}$\n(km)")
    ax.set_xlabel("UTC time", labelpad=-2.0)
    ax.grid(True, color="#cbd5e1", alpha=0.32, lw=0.48)
    ax.tick_params(axis="both", which="major", labelsize=6.4, length=2.0, width=0.5, pad=1.0)


def split_map_chunks(sub: pd.DataFrame) -> list[pd.DataFrame]:
    sub = sub.sort_values("time_utc").dropna(subset=["lugre_lon", "lugre_lat"]).copy()
    if len(sub) < 2:
        return []
    lon = sub["lugre_lon"].to_numpy(dtype=float)
    lat = sub["lugre_lat"].to_numpy(dtype=float)
    if "rx_gps_seconds" in sub.columns:
        seconds = pd.to_numeric(sub["rx_gps_seconds"], errors="coerce")
        time_delta = seconds.diff().to_numpy(dtype=float)
        time_threshold = 3.0
    else:
        time_delta = sub["time_utc"].diff().dt.total_seconds().to_numpy(dtype=float)
        positive = time_delta[np.isfinite(time_delta) & (time_delta > 0)]
        median_delta = float(np.nanmedian(positive)) if len(positive) else 10.0
        time_threshold = max(30.0, median_delta * 4.0)
    time_gap = np.nan_to_num(time_delta, nan=0.0, posinf=time_threshold + 1.0) > time_threshold
    lon_delta = np.abs(np.diff(lon))
    lon_gap = np.r_[False, lon_delta > 180.0]
    lat_delta = np.abs(np.diff(lat))
    angular_gap = np.r_[False, np.hypot(np.minimum(lon_delta, 360.0 - lon_delta), lat_delta) > 12.0]
    run_id = np.cumsum(time_gap | lon_gap | angular_gap)
    return [piece for _, piece in sub.groupby(run_id, sort=False) if len(piece) >= 2]


def add_track_map(ax: plt.Axes, df: pd.DataFrame, colors: dict[str, str]) -> None:
    h = pd.to_numeric(df["lugre_h_tan_km"], errors="coerce")
    map_df = df[h.between(50.0, 1000.0)].copy()
    if map_df.empty:
        map_df = df.copy()
    ax.set_global()
    ax.set_aspect("auto")
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.35, edgecolor="#9ca3af", zorder=3)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.grid(True, color="#e5e7eb", alpha=0.72, lw=0.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(axis="both", which="major", labelsize=6.5, length=2.4, width=0.5, pad=1.2)
    for sat, sat_df in map_df.groupby("sat", sort=False):
        color = colors.get(str(sat), "#64748b")
        group_cols = [col for col in ["phase", "op_id", "sat", "signal_id", "segment_id"] if col in sat_df.columns]
        for _, sub in sat_df.groupby(group_cols, sort=False, dropna=False):
            for chunk in split_map_chunks(sub):
                ax.plot(
                    chunk["lugre_lon"].to_numpy(dtype=float),
                    chunk["lugre_lat"].to_numpy(dtype=float),
                    transform=ccrs.PlateCarree(),
                    color=color,
                    lw=0.92,
                    alpha=0.82,
                    zorder=4,
                )


def phase_op_color_map(df: pd.DataFrame) -> dict[tuple[str, str], str]:
    cmap = plt.get_cmap("tab20")
    ordered = [item for item in ordered_phase_op_members() if ((df["phase"].eq(item[0])) & (df["op_id"].eq(item[1]))).any()]
    return {item: mpl.colors.to_hex(cmap(i % cmap.N)) for i, item in enumerate(ordered)}


def add_all_tracks_map(ax: plt.Axes, df: pd.DataFrame, colors: dict[tuple[str, str], str]) -> None:
    h = pd.to_numeric(df["lugre_h_tan_km"], errors="coerce")
    map_df = df[h.between(50.0, 1000.0)].copy()
    ax.set_global()
    ax.set_aspect("auto")
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.38, edgecolor="#9ca3af", zorder=3)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.grid(True, color="#e5e7eb", alpha=0.72, lw=0.42)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(axis="both", which="major", labelsize=7.0, length=2.8, width=0.55, pad=1.8)
    for (phase, op_id, sat, signal_id, segment_id), sub in map_df.groupby(
        ["phase", "op_id", "sat", "signal_id", "segment_id"], sort=False, dropna=False
    ):
        color = colors.get((str(phase), str(op_id)), "#64748b")
        for chunk in split_map_chunks(sub):
            ax.plot(
                chunk["lugre_lon"].to_numpy(dtype=float),
                chunk["lugre_lat"].to_numpy(dtype=float),
                transform=ccrs.PlateCarree(),
                color=color,
                lw=0.82,
                alpha=0.72,
                zorder=4,
            )


def write_all_tracks_page(png: Path, pdf: Path, row_count: int, iono_count: int, html_path: Path) -> None:
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html lang=\"en\">",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<title>All OPs ECEF/geographic 50-1000 km tangent tracks</title>",
                "<style>",
                "body{margin:0;font-family:Arial,sans-serif;background:#f8fafc;color:#111827;}",
                "header{padding:18px 24px;background:#fff;border-bottom:1px solid #e5e7eb;}",
                "main{max-width:1480px;margin:0 auto;padding:18px;}",
                "img{display:block;width:100%;height:auto;border:1px solid #e5e7eb;background:white;}",
                "h1{font-size:20px;margin:0 0 6px;}p{font-size:13px;color:#475569;margin:6px 0 0;}",
                "a{color:#1d4ed8;text-decoration:none;}a:hover{text-decoration:underline;}",
                "</style>",
                "</head>",
                "<body>",
                "<header>",
                "<h1>All OPs ECEF/geographic 50-1000 km tangent tracks</h1>",
                f"<p>Rows in recomputed tangent cache: {row_count:,}; rows inside 50-1000 km: {iono_count:,}. <a href=\"{html.escape(pdf.name)}\">PDF</a></p>",
                "</header>",
                "<main>",
                f"<a href=\"{html.escape(png.name)}\"><img src=\"{html.escape(png.name)}\" alt=\"All CTLS 50-1000 km tangent tracks\"></a>",
                "</main>",
                "</body>",
                "</html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def plot_all_tracks_overview(tangent_df: pd.DataFrame) -> dict[str, str]:
    iono = tangent_df[pd.to_numeric(tangent_df["lugre_h_tan_km"], errors="coerce").between(50.0, 1000.0)].copy()
    if iono.empty:
        raise RuntimeError("No 50-1000 km tangent rows available for all-track overview")
    colors = phase_op_color_map(iono)
    tracks = iono.groupby(["phase", "op_id", "sat", "signal_id", "segment_id"], dropna=False).ngroups

    fig = plt.figure(figsize=(15.4, 7.2), facecolor="white")
    fig.suptitle("All OPs ECEF/geographic tangent tracks, 50-1000 km", fontsize=13.0, fontweight="bold", y=0.985)
    fig.text(
        0.055,
        0.94,
        f"Rows: {len(iono):,} | Tracks: {tracks:,} | Geometry: J2000 LOS closest point; map uses ECEF/geographic tangent latitude/longitude",
        fontsize=8.0,
        color="#475569",
        ha="left",
        va="top",
    )
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    add_all_tracks_map(ax, iono, colors)

    handles = [
        mpl.lines.Line2D([], [], color=color, lw=2.0, label=f"{phase} {display_op(op_id)}")
        for (phase, op_id), color in colors.items()
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            frameon=False,
            ncol=min(9, max(1, int(np.ceil(len(handles) / 2)))),
            bbox_to_anchor=(0.52, 0.91),
            handlelength=1.8,
            handletextpad=0.35,
            columnspacing=0.95,
            borderaxespad=0.0,
            title="Phase / OP",
            title_fontsize=7.2,
        )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.09)

    png = OUT_DIR / f"{ALL_TRACKS_STEM}.png"
    pdf = OUT_DIR / f"{ALL_TRACKS_STEM}.pdf"
    page = OUT_DIR / ALL_TRACKS_HTML
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    write_all_tracks_page(png, pdf, len(tangent_df), len(iono), page)
    return {"png": png.name, "pdf": pdf.name, "html": page.name, "points": str(len(iono))}


def aligned_integer_time_ticks(
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    step_hours: int = 2,
    edge_pad: pd.Timedelta = pd.Timedelta(minutes=30),
) -> list[pd.Timestamp]:
    cursor = pd.Timestamp(t0).ceil("h")
    while cursor.hour % step_hours != 0:
        cursor += pd.Timedelta(hours=1)
    ticks = []
    while cursor < t1:
        if cursor > t0 + edge_pad and cursor < t1 - edge_pad:
            ticks.append(cursor)
        cursor += pd.Timedelta(hours=step_hours)
    return ticks


def compact_tick_items(
    windows: list[dict[str, object]],
    include_final_end: bool = True,
    include_integer_ticks: bool = False,
) -> tuple[list[float], list[str]]:
    items: list[tuple[float, pd.Timestamp, str]] = []
    for w in windows:
        items.append((float(w["x0"]), pd.Timestamp(w["t0"]), "start"))
        if include_integer_ticks:
            t0 = pd.Timestamp(w["t0"])
            t1 = pd.Timestamp(w["t1"])
            duration = max((t1 - t0).total_seconds(), 1.0)
            x0 = float(w["x0"])
            x1 = float(w["x1"])
            for tick in aligned_integer_time_ticks(t0, t1):
                fraction = (tick - t0).total_seconds() / duration
                items.append((x0 + fraction * (x1 - x0), tick, "integer"))
    if include_final_end and windows:
        last = windows[-1]
        items.append((float(last["x1"]), pd.Timestamp(last["t1"]), "end"))
    items = sorted(items, key=lambda item: item[0])
    compacted: list[tuple[float, pd.Timestamp, str]] = []
    for item in items:
        if compacted and abs(item[0] - compacted[-1][0]) < 0.28 and item[2] == "integer":
            continue
        compacted.append(item)
    return [item[0] for item in compacted], [item[1].strftime("%m-%d\n%H:%M") for item in compacted]


def add_compact_window_marks(
    axes: list[plt.Axes],
    windows: list[dict[str, object]],
    xmax: float,
    include_integer_ticks: bool = False,
) -> None:
    ticks, labels = compact_tick_items(windows, include_final_end=True, include_integer_ticks=include_integer_ticks)
    for ax in axes:
        for prev, w in zip(windows[:-1], windows[1:]):
            ax.axvline(float(prev["x1"]), color="#94a3b8", lw=0.62, ls="--", alpha=0.76, zorder=1)
            ax.axvline(float(w["x0"]), color="#94a3b8", lw=0.62, ls="--", alpha=0.76, zorder=1)
        ax.set_xlim(-0.05, xmax + 0.05)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=0)
        ax.xaxis.offsetText.set_visible(False)
    altitude_ax = axes[-1]
    ymin, ymax = altitude_ax.get_ylim()
    y = ymin * (ymax / ymin) ** 0.075
    for w in windows:
        altitude_ax.text(
            float(w["xc"]),
            y,
            str(w["label"]),
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.70},
            zorder=5,
        )


def add_datetime_time_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    colors: dict[str, str],
    y_limits: tuple[float, float] | None = None,
    y_ticks: list[float] | None = None,
    zero_line: bool = False,
) -> None:
    fixed_scale = fixed_panel_scale(y_col, df)
    if fixed_scale is not None:
        y_limits, y_ticks = fixed_scale
    for sat, sub in df.groupby("sat", sort=False):
        y = pd.to_numeric(sub[y_col], errors="coerce")
        mask = sub["time_utc"].notna() & y.notna()
        if not mask.any():
            continue
        ax.scatter(
            sub.loc[mask, "time_utc"],
            y.loc[mask],
            s=1.2,
            color=colors.get(str(sat), "#64748b"),
            alpha=0.46,
            edgecolors="none",
            rasterized=True,
        )
    if zero_line:
        ax.axhline(0.0, color="#94a3b8", lw=0.65, alpha=0.75)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e7eb", alpha=0.72, lw=0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.62)
        spine.set_color("#222222")
    ax.tick_params(axis="both", which="major", labelsize=6.4, length=2.0, width=0.5, pad=1.0)


def add_datetime_altitude_panel(ax: plt.Axes, df: pd.DataFrame, colors: dict[str, str]) -> None:
    ymin = 50.0
    ymax = 1000.0
    add_altitude_layers(ax, ymax)
    for sat, sub in df.groupby("sat", sort=False):
        y = pd.to_numeric(sub["lugre_h_tan_km"], errors="coerce")
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
    ax.set_ylabel("h$_{tan}$\n(km)")
    ax.set_xlabel("UTC time", labelpad=-2.0)
    ax.grid(True, color="#cbd5e1", alpha=0.32, lw=0.48)
    ax.tick_params(axis="both", which="major", labelsize=6.4, length=2.0, width=0.5, pad=1.0)


def exact_datetime_ticks(t0: pd.Timestamp, t1: pd.Timestamp, step_hours: float = 2.0) -> list[pd.Timestamp]:
    ticks = [pd.Timestamp(t0)]
    ticks.extend(aligned_integer_time_ticks(pd.Timestamp(t0), pd.Timestamp(t1), step_hours=int(step_hours)))
    if abs((pd.Timestamp(t1) - ticks[-1]).total_seconds()) > 60:
        ticks.append(pd.Timestamp(t1))
    return ticks


def set_exact_datetime_axis(axes: list[plt.Axes], t0: pd.Timestamp, t1: pd.Timestamp) -> None:
    ticks = exact_datetime_ticks(t0, t1)
    tick_values = [tick.to_pydatetime() for tick in ticks]
    for ax in axes:
        ax.set_xlim(t0.to_pydatetime(), t1.to_pydatetime())
        ax.set_xticks(tick_values)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
        ax.xaxis.offsetText.set_visible(False)
        ax.margins(x=0)


def datetime_op_windows(df: pd.DataFrame) -> list[dict[str, object]]:
    windows = []
    for op_id, sub in df.groupby("op_id", sort=False):
        t0, t1 = observed_time_span(sub)
        windows.append({"op_id": str(op_id), "label": display_op(op_id), "t0": t0, "t1": t1, "tc": t0 + (t1 - t0) / 2})
    return sorted(windows, key=lambda item: item["t0"])


def add_datetime_op_marks(axes: list[plt.Axes], windows: list[dict[str, object]]) -> None:
    if not windows:
        return
    for window in windows[1:]:
        for ax in axes:
            ax.axvline(pd.Timestamp(window["t0"]).to_pydatetime(), color="#cbd5e1", lw=0.55, ls="--", alpha=0.72, zorder=1)
    altitude_ax = axes[-1]
    ymin, ymax = altitude_ax.get_ylim()
    y = ymin * (ymax / ymin) ** 0.075
    for window in windows:
        altitude_ax.text(
            pd.Timestamp(window["tc"]).to_pydatetime(),
            y,
            str(window["label"]),
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.70},
            zorder=5,
        )


def apply_compact_windows(track_df: pd.DataFrame, windows: list[dict[str, object]]) -> pd.DataFrame:
    frames = []
    for window in windows:
        phase = window.get("phase")
        op_id = str(window["op_id"])
        t0 = pd.Timestamp(window["t0"])
        t1 = pd.Timestamp(window["t1"])
        mask = track_df["op_id"].astype(str).eq(op_id) & track_df["time_utc"].between(t0, t1, inclusive="both")
        if phase is not None and "phase" in track_df.columns:
            mask &= track_df["phase"].astype(str).eq(str(phase))
        sub = track_df[mask].copy()
        if sub.empty:
            continue
        duration_s = max((t1 - t0).total_seconds(), 1.0)
        x0 = float(window["x0"])
        x1 = float(window["x1"])
        fraction = (sub["time_utc"] - t0).dt.total_seconds() / duration_s
        sub["plot_x"] = x0 + fraction * (x1 - x0)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=list(track_df.columns) + ["plot_x"])
    return pd.concat(frames, ignore_index=True)


def trim_ops(df: pd.DataFrame, phase: str, ops: list[str]) -> pd.DataFrame:
    frames = []
    for op in ops:
        sub = df[df["op_id"].astype(str).eq(str(op))].copy()
        if sub.empty:
            continue
        frames.append(trim_sparse_tail_cluster(phase, str(op), sub))
    if not frames:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(frames, ignore_index=True)


def utc_range_text(t0: pd.Timestamp, t1: pd.Timestamp) -> str:
    return f"{pd.Timestamp(t0):%Y-%m-%d %H:%M} - {pd.Timestamp(t1):%Y-%m-%d %H:%M} UTC"


def add_figure_context(fig: plt.Figure, t0: pd.Timestamp, t1: pd.Timestamp, axis_note: str = "") -> None:
    axis_part = "x-axis: MM-DD HH:MM"
    if axis_note:
        axis_part = f"{axis_part}, {axis_note}"
    fig.text(
        0.065,
        0.944,
        f"UTC time: {utc_range_text(t0, t1)} | {axis_part} | h$_{{tan}}$/map: 50-1000 km, J2000 LOS",
        fontsize=7.5,
        color="#475569",
        ha="left",
        va="top",
    )


def plot_ctl_group(all_df: pd.DataFrame, tangent_df: pd.DataFrame, group: dict[str, object]) -> dict[str, str]:
    compact_df, windows, xmax = compact_axis(all_df, group["members"])
    if compact_df.empty:
        raise RuntimeError(f"No data for {group['figure']}")
    track_compact_df = apply_compact_windows(tangent_df, windows)
    colors = color_map(compact_df)
    tracks = compact_df.groupby(["phase", "op_id", "sat", "signal_id"], dropna=False).ngroups
    segments = compact_df.groupby(["phase", "op_id", "sat", "signal_id", "segment_id"], dropna=False).ngroups
    t0 = pd.to_datetime(compact_df["time_utc"].min(), utc=True)
    t1 = pd.to_datetime(compact_df["time_utc"].max(), utc=True)
    window_text = ", ".join(str(w["label"]) for w in windows)

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs = fig.add_gridspec(7, 1, height_ratios=GRID_HEIGHT_RATIOS, hspace=0.22)
    fig.suptitle(
        f"{group['figure']}: {group['title']} - C/N$_0$, $\\delta$C/N$_0$, TEC, ROT, ROTI, altitude, and track",
        fontsize=12.0,
        fontweight="bold",
        y=0.985,
    )
    add_figure_context(fig, t0, t1, "gaps omitted")

    axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]
    add_time_panel(axes[0], compact_df, "cn0_dbhz", "Raw C/N$_0$", "C/N$_0$\n(dB-Hz)", colors, robust_range(compact_df["cn0_dbhz"], floor=15.0))
    add_time_panel(
        axes[1],
        compact_df,
        "plot_cn0_detrended_db",
        "$\\delta$C/N$_0$",
        "$\\delta$C/N$_0$\n(dB)",
        colors,
        robust_range(compact_df["plot_cn0_detrended_db"], floor=-8.0, ceil=8.0),
        zero_line=True,
    )
    add_time_panel(axes[2], compact_df, "stec_tecu", "Observed slant TEC", "STEC\n(TECU)", colors, robust_range(compact_df["stec_tecu"], floor=0.0))
    add_time_panel(
        axes[3],
        compact_df,
        "rot_tecu_per_min",
        "ROT rate of TEC",
        "ROT\n(TECU/min)",
        colors,
        robust_range(compact_df["rot_tecu_per_min"], floor=-75.0, ceil=75.0),
        zero_line=True,
    )
    add_time_panel(
        axes[4],
        compact_df,
        ROTI_INDEPENDENT_COL,
        "ROTI 1 min independent",
        "ROTI\n(TECU/min)",
        colors,
        robust_range(compact_df[ROTI_INDEPENDENT_COL], floor=0.0),
    )
    add_altitude_panel(axes[5], track_compact_df, colors)
    add_compact_window_marks(axes, windows, xmax)

    ax_map = fig.add_subplot(gs[6, 0], projection=ccrs.PlateCarree())
    add_track_map(ax_map, track_compact_df, colors)

    handles = [mpl.lines.Line2D([], [], marker=".", ls="", ms=4.2, color=color, label=sat) for sat, color in colors.items()]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            ncol=legend_column_count(handles),
            bbox_to_anchor=(0.965, 0.965),
            handletextpad=0.25,
            columnspacing=0.8,
            borderaxespad=0.0,
            title="Satellite",
            title_fontsize=6.9,
        )
    fig.subplots_adjust(left=0.065, right=0.965, top=0.91, bottom=0.065)
    png = OUT_DIR / f"{group['out_stem']}.png"
    pdf = OUT_DIR / f"{group['out_stem']}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "figure": str(group["figure"]),
        "title": str(group["title"]),
        "png": png.name,
        "pdf": pdf.name,
        "points": str(len(compact_df)),
        "windows": window_text,
    }


def plot_s_group(s_df: pd.DataFrame, tangent_df: pd.DataFrame, group: dict[str, object]) -> dict[str, str]:
    ops = [str(op) for op in group["ops"]]
    phase_df = trim_ops(s_df, "S", ops)
    if phase_df.empty:
        raise RuntimeError(f"No S phase rows for {group['figure']}: {ops}")
    t0, t1 = observed_time_span(phase_df)
    phase_df = phase_df[phase_df["time_utc"].between(t0, t1)].sort_values(
        ["op_id", "sat", "signal_id", "segment_id", "time_utc"]
    )
    track_phase_df = tangent_df[
        tangent_df["phase"].astype(str).eq("S")
        & tangent_df["op_id"].astype(str).isin(ops)
        & tangent_df["time_utc"].between(t0, t1, inclusive="both")
    ].copy()
    colors = color_map(phase_df)
    tracks = phase_df.groupby(["op_id", "sat", "signal_id"], dropna=False).ngroups
    segments = phase_df.groupby(["op_id", "sat", "signal_id", "segment_id"], dropna=False).ngroups
    windows = datetime_op_windows(phase_df)
    window_text = ", ".join(str(window["label"]) for window in windows)

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs = fig.add_gridspec(7, 1, height_ratios=GRID_HEIGHT_RATIOS, hspace=0.22)
    fig.suptitle(
        f"{group['figure']}: {group['title']} - C/N$_0$, $\\delta$C/N$_0$, TEC, ROT, ROTI, altitude, and track",
        fontsize=12.0,
        fontweight="bold",
        y=0.985,
    )
    add_figure_context(fig, t0, t1)

    axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]
    add_datetime_time_panel(
        axes[0],
        phase_df,
        "cn0_dbhz",
        "Raw C/N$_0$",
        "C/N$_0$\n(dB-Hz)",
        colors,
        robust_range(phase_df["cn0_dbhz"], floor=15.0),
    )
    add_datetime_time_panel(
        axes[1],
        phase_df,
        "plot_cn0_detrended_db",
        "$\\delta$C/N$_0$",
        "$\\delta$C/N$_0$\n(dB)",
        colors,
        robust_range(phase_df["plot_cn0_detrended_db"], floor=-8.0, ceil=8.0),
        zero_line=True,
    )
    add_datetime_time_panel(
        axes[2],
        phase_df,
        "stec_tecu",
        "Observed slant TEC",
        "STEC\n(TECU)",
        colors,
        robust_range(phase_df["stec_tecu"], floor=0.0),
    )
    add_datetime_time_panel(
        axes[3],
        phase_df,
        "rot_tecu_per_min",
        "ROT rate of TEC",
        "ROT\n(TECU/min)",
        colors,
        robust_range(phase_df["rot_tecu_per_min"], floor=-75.0, ceil=75.0),
        zero_line=True,
    )
    add_datetime_time_panel(
        axes[4],
        phase_df,
        ROTI_INDEPENDENT_COL,
        "ROTI 1 min independent",
        "ROTI\n(TECU/min)",
        colors,
        robust_range(phase_df[ROTI_INDEPENDENT_COL], floor=0.0),
    )
    add_datetime_altitude_panel(axes[5], track_phase_df, colors)
    set_exact_datetime_axis(axes, t0, t1)
    add_datetime_op_marks(axes, windows)

    ax_map = fig.add_subplot(gs[6, 0], projection=ccrs.PlateCarree())
    add_track_map(ax_map, track_phase_df, colors)

    handles = [mpl.lines.Line2D([], [], marker=".", ls="", ms=4.2, color=color, label=sat) for sat, color in colors.items()]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            ncol=legend_column_count(handles),
            bbox_to_anchor=(0.965, 0.965),
            handletextpad=0.25,
            columnspacing=0.8,
            borderaxespad=0.0,
            title="Satellite",
            title_fontsize=6.9,
        )
    fig.subplots_adjust(left=0.065, right=0.965, top=0.91, bottom=0.065)
    png = OUT_DIR / f"{group['out_stem']}.png"
    pdf = OUT_DIR / f"{group['out_stem']}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "figure": str(group["figure"]),
        "title": str(group["title"]),
        "png": png.name,
        "pdf": pdf.name,
        "points": str(len(phase_df)),
        "windows": f"{window_text}; {t0:%Y-%m-%dT%H:%M:%SZ} to {t1:%Y-%m-%dT%H:%M:%SZ}",
    }


def plot_s_compact_group(s_df: pd.DataFrame, tangent_df: pd.DataFrame, group: dict[str, object]) -> dict[str, str]:
    ops = [str(op) for op in group["ops"]]
    phase_df = trim_ops(s_df, "S", ops)
    if phase_df.empty:
        raise RuntimeError(f"No S phase rows for {group['figure']}: {ops}")
    members = [("S", op) for op in ops]
    compact_df, windows, xmax = compact_axis(
        phase_df,
        members,
        default_min_display_h=0.0,
        short_display_h=S_COMPACT_MIN_DISPLAY_HOURS,
    )
    if compact_df.empty:
        raise RuntimeError(f"No compacted S phase rows for {group['figure']}: {ops}")
    track_compact_df = apply_compact_windows(tangent_df[tangent_df["phase"].astype(str).eq("S")].copy(), windows)
    colors = color_map(compact_df)
    tracks = compact_df.groupby(["op_id", "sat", "signal_id"], dropna=False).ngroups
    segments = compact_df.groupby(["op_id", "sat", "signal_id", "segment_id"], dropna=False).ngroups
    t0 = pd.to_datetime(compact_df["time_utc"].min(), utc=True)
    t1 = pd.to_datetime(compact_df["time_utc"].max(), utc=True)
    window_text = ", ".join(display_op(w["op_id"]) for w in windows)

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs = fig.add_gridspec(7, 1, height_ratios=GRID_HEIGHT_RATIOS, hspace=0.22)
    fig.suptitle(
        f"{group['figure']}: {group['title']} - C/N$_0$, $\\delta$C/N$_0$, TEC, ROT, ROTI, altitude, and track",
        fontsize=12.0,
        fontweight="bold",
        y=0.985,
    )
    add_figure_context(fig, t0, t1, "OP gaps omitted")

    axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]
    add_time_panel(
        axes[0],
        compact_df,
        "cn0_dbhz",
        "Raw C/N$_0$",
        "C/N$_0$\n(dB-Hz)",
        colors,
        robust_range(compact_df["cn0_dbhz"], floor=15.0),
    )
    add_time_panel(
        axes[1],
        compact_df,
        "plot_cn0_detrended_db",
        "$\\delta$C/N$_0$",
        "$\\delta$C/N$_0$\n(dB)",
        colors,
        robust_range(compact_df["plot_cn0_detrended_db"], floor=-8.0, ceil=8.0),
        zero_line=True,
    )
    add_time_panel(
        axes[2],
        compact_df,
        "stec_tecu",
        "Observed slant TEC",
        "STEC\n(TECU)",
        colors,
        robust_range(compact_df["stec_tecu"], floor=0.0),
    )
    add_time_panel(
        axes[3],
        compact_df,
        "rot_tecu_per_min",
        "ROT rate of TEC",
        "ROT\n(TECU/min)",
        colors,
        robust_range(compact_df["rot_tecu_per_min"], floor=-75.0, ceil=75.0),
        zero_line=True,
    )
    add_time_panel(
        axes[4],
        compact_df,
        ROTI_INDEPENDENT_COL,
        "ROTI 1 min independent",
        "ROTI\n(TECU/min)",
        colors,
        robust_range(compact_df[ROTI_INDEPENDENT_COL], floor=0.0),
    )
    add_altitude_panel(axes[5], track_compact_df, colors)
    add_compact_window_marks(axes, windows, xmax, include_integer_ticks=True)

    ax_map = fig.add_subplot(gs[6, 0], projection=ccrs.PlateCarree())
    add_track_map(ax_map, track_compact_df, colors)

    handles = [mpl.lines.Line2D([], [], marker=".", ls="", ms=4.2, color=color, label=sat) for sat, color in colors.items()]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            ncol=legend_column_count(handles),
            bbox_to_anchor=(0.965, 0.965),
            handletextpad=0.25,
            columnspacing=0.8,
            borderaxespad=0.0,
            title="Satellite",
            title_fontsize=6.9,
        )
    fig.subplots_adjust(left=0.065, right=0.965, top=0.91, bottom=0.065)
    png = OUT_DIR / f"{group['out_stem']}.png"
    pdf = OUT_DIR / f"{group['out_stem']}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "figure": str(group["figure"]),
        "title": str(group["title"]),
        "png": png.name,
        "pdf": pdf.name,
        "points": str(len(compact_df)),
        "windows": f"{window_text}; compacted {t0:%Y-%m-%dT%H:%M:%SZ} to {t1:%Y-%m-%dT%H:%M:%SZ}",
    }


def plot_s_groups(s_df: pd.DataFrame, tangent_df: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    for group in S_GROUPS:
        if len(group["ops"]) > 1:
            rows.append(plot_s_compact_group(s_df, tangent_df, group))
        else:
            rows.append(plot_s_group(s_df, tangent_df, group))
    return rows


def write_index(
    rows: list[dict[str, str]],
    target: Path,
    relative_prefix: str = "",
    all_tracks: dict[str, str] | None = None,
) -> None:
    cards = []
    for row in rows:
        img = f"{relative_prefix}{row['png']}"
        pdf = f"{relative_prefix}{row['pdf']}"
        cards.append(
            "\n".join(
                [
                    "<article>",
                    f"  <h2>{html.escape(row['figure'])}: {html.escape(row['title'])}</h2>",
                    f"  <a href=\"{html.escape(img)}\"><img src=\"{html.escape(img)}\" alt=\"{html.escape(row['figure'] + ': ' + row['title'])}\"></a>",
                    f"  <p><a href=\"{html.escape(pdf)}\">PDF</a></p>",
                    "</article>",
                ]
            )
        )
    if all_tracks is not None:
        all_tracks_href = f"{relative_prefix}{all_tracks['html']}"
        all_tracks_text = (
            f"<p class=\"tool-link\"><a href=\"{html.escape(all_tracks_href)}\" target=\"_blank\" rel=\"noopener\">"
            "Open all 50-1000 km tangent tracks in a separate window</a></p>"
        )
    else:
        all_tracks_text = ""
    target.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html lang=\"en\">",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<title>LuGRE phase story supplementary figures S1-S5</title>",
                "<style>",
                "body{margin:0;font-family:Arial,sans-serif;background:#f8fafc;color:#111827;}",
                "header{padding:24px 32px;background:#fff;border-bottom:1px solid #e5e7eb;position:sticky;top:0;z-index:2;}",
                "main{max-width:1440px;margin:0 auto;padding:22px;}",
                "article{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin:0 0 18px;}",
                "h1{font-size:22px;margin:0 0 6px;}h2{font-size:16px;margin:0 0 10px;}",
                "p{color:#475569;font-size:13px;line-height:1.45;margin:10px 0 0;}",
                ".tool-link a{display:inline-block;margin-top:4px;padding:6px 10px;border:1px solid #bfdbfe;border-radius:6px;background:#eff6ff;}",
                "img{display:block;width:100%;height:auto;border:1px solid #e5e7eb;border-radius:6px;}",
                "a{color:#1d4ed8;text-decoration:none;}a:hover{text-decoration:underline;}",
                "</style>",
                "</head>",
                "<body>",
                "<header>",
                "<h1>LuGRE phase story supplementary figures S1-S5</h1>",
                "<p>CTL figures use compact observing-window axes: long gaps are removed, regular CTL windows use about 4 h display width, short windows are narrowed, and tick labels show actual UTC month-day and hour:minute. CTL is compressed into one figure; T OP3 is omitted because it has very few samples, and the sparse L OP37 tail is trimmed. S phase days use observed start/end limits; multi-OP S days are compacted with inter-OP gaps removed. C/N0 scales use 18-48 dB-Hz for S1 and 18-42 dB-Hz for S2-S5; ROTI is recomputed as an independent UTC-minute standard deviation of ROT within each continuous arc; all tangent-height and map panels use J2000 receiver/GNSS vectors with the LOS closest point recomputed from the line-geometry source, restricted to 50-1000 km.</p>",
                all_tracks_text,
                "</header>",
                "<main>",
                *cards,
                "</main>",
                "</body>",
                "</html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(rows: list[dict[str, str]]) -> None:
    with (OUT_DIR / "index_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "title", "png", "pdf", "points", "windows"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    setup_style()
    clean_output()
    all_df = load_ctl_rows()
    s_df = load_s_rows()
    tangent_df = load_recomputed_tangent_rows()
    all_tracks = plot_all_tracks_overview(tangent_df)
    rows: list[dict[str, str]] = []
    for group in CTL_GROUPS:
        rows.append(plot_ctl_group(all_df, tangent_df, group))
    rows.extend(plot_s_groups(s_df, tangent_df))
    write_manifest(rows)
    write_index(rows, OUT_DIR / "index.html", all_tracks=all_tracks)
    write_index(rows, ROOT / "index.html", relative_prefix="index_ctl_compact_1fig_trimmed/", all_tracks=all_tracks)
    print(f"output_dir={OUT_DIR}")
    print(f"figures={len(rows)}")
    print(f"root_index={ROOT / 'index.html'}")


if __name__ == "__main__":
    main()
