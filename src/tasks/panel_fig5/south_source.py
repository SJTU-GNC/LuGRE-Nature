from __future__ import annotations

import importlib.util
import json
import math
import sys
import tarfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "panel_ready" / "Fig5"
OUT_DIR = ROOT / "work" / "panel_fig5"
NORTH_SCRIPT = TASK_DIR / "north_source.py"
SOUTH_ROOT = DATA_ROOT / "south"
SOUTH_DATA = SOUTH_ROOT
LEORO_ROLLING_MEDIAN_POINTS = DATA_ROOT / "common" / "leoro_rolling_points.csv"
LEORO_PYDEPS = ROOT / "runtime" / "python"
PLANETIQ_CONPHS_TAR = DATA_ROOT / "external_not_packaged" / "planetiq_conPhs.tar.gz"
SPIRE_CONPHS_TAR = DATA_ROOT / "external_not_packaged" / "spire_conPhs.tar.gz"
LEORO_SOUTH_ROLLING_L1L2_POINTS = OUT_DIR / "fig4_south_leoro_cn0_rolling_median_l1l2_points_v5.csv"

NORTH_PNG = OUT_DIR / "fig4_north_pair_nature_layout_candidate_v43_v40_layout_fig7_basemap.png"
SOUTH_PNG = OUT_DIR / "fig4_south_pair_match_north_layout_v9_compact_horizontal.png"
SOUTH_PDF = OUT_DIR / "fig4_south_pair_match_north_layout_v9_compact_horizontal.pdf"
COMBINED_PNG = OUT_DIR / "fig4_north_south_side_by_side_v9_compact_horizontal.png"
COMBINED_PDF = OUT_DIR / "fig4_north_south_side_by_side_v9_compact_horizontal.pdf"
QA_OUT = OUT_DIR / "fig4_north_south_side_by_side_v9_compact_horizontal_QA.csv"
MANIFEST_OUT = OUT_DIR / "fig4_north_south_side_by_side_v9_compact_horizontal_source_manifest.csv"
README_OUT = OUT_DIR / "README_fig4_north_south_side_by_side_v9_compact_horizontal.md"


def load_north_module():
    spec = importlib.util.spec_from_file_location("fig4_north_v43_template", NORTH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


north = load_north_module()

plt = north.plt
np = north.np
pd = north.pd
mdates = north.mdates
ScalarMappable = north.ScalarMappable
Normalize = north.Normalize
PowerNorm = north.PowerNorm
Line2D = north.Line2D
Circle = north.Circle
FixedFormatter = north.FixedFormatter
FixedLocator = north.FixedLocator
NullFormatter = north.NullFormatter

INK = north.INK
MUTED = north.MUTED
GRID = north.GRID
AXIS = north.AXIS
SPINE = north.SPINE
LUGRE = north.LUGRE
LEORO = north.LEORO
SITE_RED = north.SITE_RED
ROTI_CMAP = north.ROTI_CMAP
SD_CMAP = north.SD_CMAP
ISMR_CMAP = north.ISMR_CMAP


SOUTH_MAP_TITLES = {
    9: "A  OP38 Galileo E21 | south polar day | 2025-03-03 12:23-12:32 UTC",
    17: "F  OP38 Galileo E23 | south polar day | 2025-03-03 14:06-14:13 UTC",
}

_LEORO_ROLLING_CACHE: pd.DataFrame | None = None
_LEORO_ROLLING_NOTES: list[str] = []


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def polar_xy_south(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    r = (90.0 + lat) / 30.0
    th = np.deg2rad(lon)
    # Match the Extended Data Fig. 7 south-polar convention:
    # 0 deg to the right, 90 deg E at the top, 90 deg W at the bottom.
    return r * np.cos(th), r * np.sin(th)


def scatter_south(ax, lat, lon, **kwargs):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat <= -60.0)
    if ok.sum() == 0:
        return None
    x, y = polar_xy_south(lat[ok], lon[ok])
    return ax.scatter(x, y, **kwargs)


def plot_south_line(ax, lat, lon, **kwargs) -> None:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat <= -60.0)
    if ok.sum() < 2:
        return
    lat = lat[ok]
    lon = lon[ok]
    breaks = np.flatnonzero(np.abs(np.diff(lon)) > 150.0) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(lon)]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        x, y = polar_xy_south(lat[s:e], lon[s:e])
        ax.plot(x, y, **kwargs)


def polar_xy_to_lonlat_south(points, *, plot_xy: bool = False) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
        return np.array([]), np.array([])
    x = arr[:, 0]
    y = arr[:, 1]
    radius = np.hypot(x, y)
    lat = -90.0 + radius
    ref_y = -y if plot_xy else y
    lon = np.rad2deg(np.arctan2(x, ref_y))
    lon = ((lon + 180.0) % 360.0) - 180.0
    ok = np.isfinite(lat) & np.isfinite(lon)
    return lat[ok], lon[ok]


def draw_south_polar_frame(ax) -> None:
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(-1.055, 1.055)
    ax.set_ylim(-1.055, 1.055)
    ax.axis("off")
    lat_label_lon = 67.5
    for lat, lw, alpha in [(-60, 0.95, 0.96), (-70, 0.62, 0.84), (-80, 0.62, 0.84)]:
        r = (90.0 + lat) / 30.0
        ax.add_patch(Circle((0, 0), r, fill=False, ec=GRID, lw=lw, alpha=alpha, zorder=3))
        tx, ty = polar_xy_south([lat], [lat_label_lon])
        ax.text(tx[0] * 1.018, ty[0] * 1.018, f"{abs(lat)}°S", fontsize=6.1, color=MUTED, fontweight="bold", ha="center", va="center", zorder=40)
    ax.text(0.018, -0.018, "90°S", fontsize=5.9, color=MUTED, fontweight="bold", alpha=0.85, zorder=40)
    for lon in range(-180, 181, 45):
        x, y = polar_xy_south([-60], [lon])
        ax.plot([0, x[0]], [0, y[0]], color=GRID, lw=0.45, alpha=0.62, zorder=3)
    labels = [(0, "0°"), (45, "45°E"), (90, "90°E"), (135, "135°E"), (180, "180°"), (-135, "135°W"), (-90, "90°W"), (-45, "45°W")]
    for lon, label in labels:
        x, y = polar_xy_south([-59.3], [lon])
        ax.text(x[0], y[0], label, fontsize=6.1, color=MUTED, fontweight="bold", ha="center", va="center", zorder=40)


def draw_antarctic_basemap(ax) -> int:
    count = 0
    for feature, color, lw, alpha, zorder in [
        (north.cfeature.COASTLINE.with_scale("110m"), "#8C96A3", 0.36, 0.72, 2.1),
        (north.cfeature.BORDERS.with_scale("110m"), "#C9CED6", 0.16, 0.58, 2.0),
    ]:
        for geom in feature.geometries():
            for lat, lon in north._iter_geometry_lonlat_lines(geom):
                ok = np.isfinite(lat) & np.isfinite(lon) & (lat <= -60.0)
                if ok.sum() < 2:
                    continue
                idx = np.flatnonzero(ok)
                breaks = np.where(np.diff(idx) > 1)[0] + 1
                for segment in np.split(idx, breaks):
                    if len(segment) < 2:
                        continue
                    plot_south_line(ax, lat[segment], lon[segment], color=color, lw=lw, alpha=alpha, zorder=zorder)
                    count += 1
    return count


def draw_south_roti_background(ax, roti: pd.DataFrame):
    if roti.empty:
        return None
    r = roti.copy()
    for col in ["lat_center", "lon_center", "roti5_p95", "n", "stations"]:
        if col in r.columns:
            r[col] = pd.to_numeric(r[col], errors="coerce")
    r = r.dropna(subset=["lat_center", "lon_center", "roti5_p95"])
    r = r[r["lat_center"].le(-60.0)]
    total_cells = len(r)
    r = r[np.isfinite(r["roti5_p95"])]
    if r.empty:
        return None
    px, py = polar_xy_south(r["lat_center"], r["lon_center"])
    points = np.column_stack([px, py])
    values = r["roti5_p95"].to_numpy(dtype=float)
    grid_n = 420
    gx = np.linspace(-1.0, 1.0, grid_n)
    gy = np.linspace(-1.0, 1.0, grid_n)
    X, Y = np.meshgrid(gx, gy)
    R = np.hypot(X, Y)
    linear = north.griddata(points, values, (X, Y), method="linear")
    nearest = north.griddata(points, values, (X, Y), method="nearest")
    tree = north.cKDTree(points)
    nearest_dist, _ = tree.query(np.column_stack([X.ravel(), Y.ravel()]), k=1)
    nearest_dist = nearest_dist.reshape(X.shape)
    coverage_radius = 0.044
    coverage = (nearest_dist <= coverage_radius) & (R <= 1.0)
    Z = np.where(np.isfinite(linear), linear, nearest)
    Z = np.where(coverage, Z, np.nan)
    Z = np.clip(Z, 0, 4)
    im = ax.imshow(
        np.ma.masked_invalid(Z),
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap=ROTI_CMAP,
        norm=PowerNorm(gamma=1.25, vmin=0, vmax=4),
        alpha=0.82,
        interpolation="bilinear",
        zorder=0,
    )
    im.set_clip_path(Circle((0, 0), 1.0, transform=ax.transData))
    im.roti_total_cells = int(total_cells)
    im.roti_valid_cells = int(len(r))
    im.roti_colored_grid_fraction = float(np.isfinite(Z).sum() / (R <= 1.0).sum())
    return im


def bearing_distance_from_station(lat_deg, lon_deg, station_lat, station_lon) -> tuple[np.ndarray, np.ndarray]:
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat1 = math.radians(float(station_lat))
    lon1 = math.radians(float(station_lon))
    dlon = lon - lon1
    y = np.sin(dlon) * np.cos(lat)
    x = np.cos(lat1) * np.sin(lat) - np.sin(lat1) * np.cos(lat) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    a = np.sin((lat - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    distance = 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return bearing, distance


def angle_delta_deg(angle, center) -> np.ndarray:
    return ((np.asarray(angle, dtype=float) - float(center) + 180.0) % 360.0) - 180.0


def prepare_fir_fan_bearings(fan: pd.DataFrame) -> pd.DataFrame:
    if fan.empty:
        return pd.DataFrame()
    f = fan.copy()
    for col in ["beam", "bmazm_deg", "range_center_km", "mean_power_db", "mean_velocity_ms"]:
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors="coerce")
    f = f.dropna(subset=["beam", "bmazm_deg", "range_center_km"])
    if f.empty:
        return f
    beams = np.array(sorted(f["beam"].dropna().unique()), dtype=float)
    raw_center = float(np.nanmedian(f["bmazm_deg"]))
    raw_span = float(np.nanpercentile(np.abs(angle_delta_deg(f["bmazm_deg"].to_numpy(dtype=float), raw_center)), 98))
    if len(beams) >= 4 and raw_span < 2.0:
        beam_mid = float(np.nanmedian(beams))
        beam_sep = 3.24
        beam_angles = {beam: raw_center + (float(beam) - beam_mid) * beam_sep for beam in beams}
        f["_bearing_deg"] = f["beam"].map(beam_angles)
        f["_bearing_note"] = "bearing expanded from beam number because FIR bmazm_deg is single-valued"
    else:
        beam_angles = {beam: float(f.loc[f["beam"].eq(beam), "bmazm_deg"].median()) for beam in beams}
        f["_bearing_deg"] = f["beam"].map(beam_angles)
        f["_bearing_note"] = "bearing from bmazm_deg"
    return f


def fir_fan_latlon_for_map(case: dict) -> pd.DataFrame:
    station = case["station"]
    st_lat = float(station.get("lat_use", station.get("lat", -51.8314)))
    st_lon = float(station.get("lon_use", station.get("lon", -58.9793)))
    fan = prepare_fir_fan_bearings(case["fan"])
    if fan.empty:
        return fan
    fan = fan.dropna(subset=["_bearing_deg", "range_center_km", "mean_power_db"])
    fan = fan[fan["mean_power_db"].gt(0)].copy()
    if fan.empty:
        return fan
    latlon = [north.destination_point(st_lat, st_lon, float(b), float(r)) for b, r in zip(fan["_bearing_deg"], fan["range_center_km"])]
    if latlon:
        lat, lon = np.array(latlon).T
        fan["map_lat"] = lat
        fan["map_lon"] = lon
        fan = fan[fan["map_lat"].le(-60.0)].copy()
    return fan


def profile_to_frame(profile: dict, label: str, color: str, marker: str) -> pd.DataFrame:
    n = min(len(profile.get("time", [])), len(profile.get("height", [])), len(profile.get("detrended", [])))
    if n == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "time": pd.to_datetime(profile["time"][:n], utc=True, errors="coerce"),
            "height": pd.to_numeric(pd.Series(profile["height"][:n]), errors="coerce"),
            "dcn0": pd.to_numeric(pd.Series(profile["detrended"][:n]), errors="coerce"),
            "label": label,
            "color": color,
            "marker": marker,
        }
    ).dropna(subset=["time", "height", "dcn0"])


def leoro_short_label(occ: dict) -> str:
    source = str(occ.get("source") or "LEO-RO")
    label = str(occ.get("label") or source)
    event_text = " ".join(str(x) for x in [label, occ.get("id", ""), (occ.get("meta") or {}).get("eventId", "")])
    match = north.re.search(r"(?<![A-Z0-9])([GREJC]\d{2})(?=[_.\-\s]|$)", event_text.upper())
    if match:
        return f"{source} {match.group(1)}"
    return source


def leoro_source_file_from_occ(occ: dict) -> str | None:
    text = " ".join(str(x) for x in [occ.get("label", ""), occ.get("id", ""), (occ.get("meta") or {}).get("eventId", "")])
    match = north.re.search(r"conPhs_[A-Z0-9]+\.\d{4}\.\d{3}\.\d{2}\.\d{2}\.[A-Z]\d{2}_\d{4}\.\d{4}_nc", text)
    return match.group(0) if match else None


GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


def read_nc_scalar(ds, names: list[str]) -> float:
    for name in names:
        if name in ds.variables:
            arr = np.asarray(ds.variables[name][:], dtype=float).ravel()
            if arr.size and np.isfinite(arr[0]):
                return float(arr[0])
    return np.nan


def raw_snr_to_rolling_median_points(
    *,
    mission: str,
    tar_path: Path,
    source_file: str,
    signal: str,
    snr_var: str,
) -> pd.DataFrame:
    if not tar_path.exists():
        _LEORO_ROLLING_NOTES.append(f"{source_file} {signal}: missing tar {tar_path}")
        return pd.DataFrame()
    if LEORO_PYDEPS.exists() and str(LEORO_PYDEPS) not in sys.path:
        sys.path.insert(0, str(LEORO_PYDEPS))
    try:
        from netCDF4 import Dataset
    except Exception as exc:
        _LEORO_ROLLING_NOTES.append(f"{source_file} {signal}: netCDF4 unavailable ({exc})")
        return pd.DataFrame()

    with tarfile.open(tar_path, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if Path(m.name).name == source_file or source_file in m.name), None)
        if member is None:
            _LEORO_ROLLING_NOTES.append(f"{source_file} {signal}: absent from raw conPhs tar")
            return pd.DataFrame()
        payload = tar.extractfile(member).read()

    ds = Dataset("inmemory.nc", memory=payload)
    try:
        if snr_var not in ds.variables or "time" not in ds.variables or "occheight" not in ds.variables:
            _LEORO_ROLLING_NOTES.append(f"{source_file} {signal}: required variables missing")
            return pd.DataFrame()
        time_s = np.asarray(ds.variables["time"][:], dtype=float)
        height = np.asarray(ds.variables["occheight"][:], dtype=float)
        snr = np.asarray(ds.variables[snr_var][:], dtype=float)
        n = min(len(time_s), len(height), len(snr))
        if n == 0:
            return pd.DataFrame()
        time_s = time_s[:n]
        height = height[:n]
        snr = snr[:n]
        start_gps = read_nc_scalar(ds, ["startTime"])
        leapsec = read_nc_scalar(ds, ["leapsec"])
        if not np.isfinite(leapsec):
            leapsec = 18.0
        gps_time = start_gps + time_s
        ok = np.isfinite(gps_time) & np.isfinite(height) & np.isfinite(snr) & (snr > 0) & (height >= 50.0) & (height <= 1000.0)
        if ok.sum() == 0:
            return pd.DataFrame()
        work = pd.DataFrame({"gps_time": gps_time[ok], "height": height[ok], "snr": snr[ok]})
        work["sec"] = np.floor(work["gps_time"]).astype("int64")
        one_sec = work.groupby("sec", sort=True).agg(
            tangent_height_km=("height", "median"),
            snr_median=("snr", "median"),
            n_samples_snr=("snr", "size"),
        ).reset_index()
        if one_sec.empty:
            return pd.DataFrame()
        median_count = float(np.nanmedian(one_sec["n_samples_snr"]))
        min_samples = max(1, int(math.floor(0.8 * median_count)))
        one_sec = one_sec[one_sec["n_samples_snr"].ge(min_samples)].copy()
        if one_sec.empty:
            return pd.DataFrame()
        one_sec["cn0_proxy_dbhz"] = 20.0 * np.log10(one_sec["snr_median"].to_numpy(dtype=float))
        usable_n = len(one_sec)
        window = int(round((2.0 / 9.0) * usable_n))
        window = max(9, min(61, window))
        if window % 2 == 0:
            window += 1
        one_sec["cn0_fit_dbhz"] = one_sec["cn0_proxy_dbhz"].rolling(window=window, center=True, min_periods=1).median()
        one_sec["delta_cn0_db"] = one_sec["cn0_proxy_dbhz"] - one_sec["cn0_fit_dbhz"]
        one_sec["time_utc"] = [
            GPS_EPOCH + timedelta(seconds=float(sec) - float(leapsec))
            for sec in one_sec["sec"].to_numpy(dtype=float)
        ]
        prn_match = north.re.search(r"\.([A-Z]\d{2})_", source_file)
        prn = prn_match.group(1) if prn_match else ""
        profile_id = f"{mission}:{source_file.replace('_nc', '')}:{signal}"
        out = pd.DataFrame(
            {
                "mission": mission,
                "profile_id": profile_id,
                "source_file": source_file,
                "signal": signal,
                "time_utc": pd.to_datetime(one_sec["time_utc"], utc=True),
                "latitude_band_magnetic": "",
                "geomagnetic_dip_lat_deg": np.nan,
                "tangent_height_km": one_sec["tangent_height_km"],
                "delta_cn0_db": one_sec["delta_cn0_db"],
                "cn0_proxy_dbhz": one_sec["cn0_proxy_dbhz"],
                "cn0_fit_dbhz": one_sec["cn0_fit_dbhz"],
                "n_samples_snr": one_sec["n_samples_snr"],
                "source_kind": "fig4_raw_conphs_1hz_dynamic_rolling_median_cn0_proxy",
                "cn0_proxy_method": "raw conPhs high-rate SNR to 1 Hz median-SNR C/N0 proxy minus centered dynamic rolling-median baseline; window=min(61 s,max(9 s,about 2/9 usable segment length)), odd",
                "raw_snr_variable": snr_var,
                "gnss_prn": prn,
                "rolling_window_s": window,
                "min_samples_per_second": min_samples,
            }
        ).dropna(subset=["time_utc", "tangent_height_km", "delta_cn0_db"])
        _LEORO_ROLLING_NOTES.append(f"{source_file} {signal}: recomputed from raw conPhs {snr_var}, n={len(out)}, window={window}s")
        return out
    finally:
        ds.close()


def ensure_south_leoro_l1l2_cache() -> Path:
    if LEORO_SOUTH_ROLLING_L1L2_POINTS.exists():
        return LEORO_SOUTH_ROLLING_L1L2_POINTS
    raw_cases = [
        ("PlanetiQ", PLANETIQ_CONPHS_TAR, "conPhs_GN04.2025.062.12.17.G18_0001.0001_nc", "L1", "pL1Snr"),
        ("PlanetiQ", PLANETIQ_CONPHS_TAR, "conPhs_GN04.2025.062.12.17.G18_0001.0001_nc", "L2", "pL2Snr"),
        ("Spire", SPIRE_CONPHS_TAR, "conPhs_S194.2025.062.14.11.C25_0001.0001_nc", "L1", "pL1Snr"),
        ("Spire", SPIRE_CONPHS_TAR, "conPhs_S194.2025.062.14.11.C25_0001.0001_nc", "L2", "pL2Snr"),
    ]
    frames = [
        raw_snr_to_rolling_median_points(mission=mission, tar_path=tar_path, source_file=source_file, signal=signal, snr_var=snr_var)
        for mission, tar_path, source_file, signal, snr_var in raw_cases
    ]
    df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    df.to_csv(LEORO_SOUTH_ROLLING_L1L2_POINTS, index=False, encoding="utf-8-sig")
    return LEORO_SOUTH_ROLLING_L1L2_POINTS


def load_leoro_rolling_cache() -> pd.DataFrame:
    global _LEORO_ROLLING_CACHE
    if _LEORO_ROLLING_CACHE is not None:
        return _LEORO_ROLLING_CACHE
    source_files = {
        "conPhs_GN04.2025.062.12.17.G18_0001.0001_nc",
        "conPhs_S194.2025.062.14.11.C25_0001.0001_nc",
    }
    chunks = []
    cache_path = ensure_south_leoro_l1l2_cache()
    if cache_path.exists():
        for chunk in pd.read_csv(cache_path, chunksize=250_000):
            if "source_file" not in chunk.columns:
                continue
            sel = chunk["source_file"].astype(str).isin(source_files)
            if sel.any():
                chunks.append(chunk.loc[sel].copy())
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.DataFrame()
    if not df.empty:
        for col in ["time_utc", "tangent_height_km", "delta_cn0_db", "cn0_proxy_dbhz", "cn0_fit_dbhz"]:
            if col == "time_utc":
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            elif col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["time_utc", "tangent_height_km", "delta_cn0_db"])
    _LEORO_ROLLING_CACHE = df
    return _LEORO_ROLLING_CACHE


def rolling_leoro_profiles(occ: dict) -> list[pd.DataFrame]:
    source_file = leoro_source_file_from_occ(occ)
    if not source_file:
        return []
    df = load_leoro_rolling_cache()
    if df.empty:
        _LEORO_ROLLING_NOTES.append(f"{source_file}: no rolling-median dynamic points found")
        return []
    use = df[df["source_file"].astype(str).eq(source_file)].copy()
    if use.empty:
        _LEORO_ROLLING_NOTES.append(f"{source_file}: absent from rolling-median dynamic point source")
        return []
    frames = []
    for signal, sub in use.groupby("signal", sort=True):
        mission = str(sub["mission"].dropna().iloc[0]) if "mission" in sub.columns and sub["mission"].notna().any() else str(occ.get("source") or "LEO-RO")
        prn_match = north.re.search(r"\.([A-Z]\d{2})_", source_file)
        prn = prn_match.group(1) if prn_match else ""
        signal_label = str(signal).replace("_CA", "").replace("_P", "")
        label = f"{mission} {prn} {signal_label}".strip()
        color = "#E45756" if "L1" in signal_label.upper() else "#7B61FF"
        marker = "D" if "L1" in signal_label.upper() else "s"
        out = pd.DataFrame(
            {
                "time": sub["time_utc"],
                "height": sub["tangent_height_km"],
                "dcn0": sub["delta_cn0_db"],
                "label": label,
                "color": color,
                "marker": marker,
            }
        ).dropna(subset=["time", "height", "dcn0"]).sort_values("time")
        if not out.empty:
            frames.append(out)
    signals = sorted(str(x) for x in use["signal"].dropna().unique()) if "signal" in use.columns else []
    if not any("L2" in s.upper() for s in signals):
        _LEORO_ROLLING_NOTES.append(f"{source_file}: rolling-median dynamic source contains {','.join(signals) or 'no signal'} only; L2 point residual not plotted")
    return frames


def south_lugre_profiles(case: dict) -> list[pd.DataFrame]:
    frames = []
    sat = str(case["track"]["sat"].iloc[0])
    seen_e5 = 0
    for item in case["ro"].get("lugres", [])[:4]:
        label = str(item.get("label", "LuGRE"))
        if "E1" in label or "L1" in label:
            frames.append(profile_to_frame(item.get("profile") or {}, f"LuGRE {sat} E1", "#2F80ED", "o"))
        elif "E5" in label or "L5" in label:
            seen_e5 += 1
            suffix = "E5a" if seen_e5 == 1 else "E5a"
            frames.append(profile_to_frame(item.get("profile") or {}, f"LuGRE {sat} {suffix}", "#22A766", "^"))
    return [frame for frame in frames if not frame.empty]


def south_leoro_profiles(case: dict) -> list[pd.DataFrame]:
    occ = case["ro"].get("occultation") or {}
    rolling_frames = rolling_leoro_profiles(occ)
    if rolling_frames:
        return rolling_frames
    frames = []
    channels = occ.get("channelSignalProfiles") or occ.get("_channel_signal") or {}
    short = leoro_short_label(occ)
    if isinstance(channels, dict) and channels:
        for ch, prof in channels.items():
            frames.append(profile_to_frame(prof, f"{short} {ch}", "#E45756" if str(ch).upper() == "L1" else "#7B61FF", "o" if str(ch).upper() == "L1" else "D"))
    else:
        frames.append(profile_to_frame(occ.get("profile") or {}, short, LEORO, "D"))
    return [frame for frame in frames if not frame.empty]


def load_south_case(track_index: int) -> dict:
    track = read_csv(SOUTH_DATA / "track.csv")
    ro = json.loads((SOUTH_DATA / "ro_focus.json").read_text(encoding="utf-8"))
    sd_row = read_csv(SOUTH_DATA / "selected_superdarn_row.csv")
    station = read_csv(SOUTH_DATA / "selected_superdarn_station.csv")
    fan = read_csv(SOUTH_DATA / "fan_cells.csv")
    context = read_csv(SOUTH_DATA / "fan_context.csv")
    return {
        "track_index": track_index,
        "track": track,
        "ro": ro,
        "sd": sd_row.iloc[0].to_dict() if not sd_row.empty else {},
        "station": station.iloc[0].to_dict() if not station.empty else {},
        "fan": fan,
        "superdarn_context": context,
        "roti": read_csv(SOUTH_DATA / "roti_context_field.csv"),
        "ismr": read_csv(SOUTH_DATA / "ismr_context_field.csv"),
        "ionprf": read_csv(SOUTH_DATA / "legacy_ionprf_profile.csv"),
        "iri": read_csv(SOUTH_DATA / "legacy_iri_profile.csv"),
        "los_paths": read_csv(SOUTH_DATA / "lugre_los_paths.csv"),
    }


def draw_south_los_paths_from_csv(ax, los: pd.DataFrame, *, max_paths: int = 54) -> None:
    if los.empty:
        return
    l = los.copy()
    for col in ["los_path_index", "latitude_deg", "longitude_deg", "tangent_latitude_deg", "tangent_longitude_deg"]:
        l[col] = pd.to_numeric(l[col], errors="coerce")
    groups = [(idx, group.sort_values("point_index")) for idx, group in l.groupby("los_path_index")]
    if not groups:
        return
    step = max(1, int(math.ceil(len(groups) / max_paths)))
    for _, group in groups[::step]:
        plot_south_line(ax, group["latitude_deg"], group["longitude_deg"], color="#24364D", lw=max(0.48, 2.45 * 0.30), alpha=0.32, zorder=12)
    # The LuGRE tangent-point track itself is drawn once in draw_south_map().
    # Keeping only the faint LOS rays here avoids the double-thick TP line.


def draw_south_ro_los_paths(ax, paths, *, max_paths: int = 42) -> None:
    paths = paths or []
    if not paths:
        return
    step = max(1, int(math.ceil(len(paths) / max_paths)))
    centers_lat = []
    centers_lon = []
    for path in paths[::step]:
        lat, lon = polar_xy_to_lonlat_south(path, plot_xy=True)
        plot_south_line(ax, lat, lon, color="#D97706", lw=0.58, alpha=0.20, zorder=12)
    for path in paths:
        arr = np.asarray(path, dtype=float)
        if arr.ndim == 2 and len(arr):
            mid = arr[len(arr) // 2, :2]
            lat, lon = polar_xy_to_lonlat_south([mid], plot_xy=True)
            if len(lat):
                centers_lat.append(lat[0])
                centers_lon.append(lon[0])
    if len(centers_lat) >= 2:
        plot_south_line(ax, centers_lat, centers_lon, color=LEORO, lw=1.85, alpha=0.96, zorder=22)


def draw_south_map(ax, fig, case: dict, label: str) -> dict:
    draw_south_polar_frame(ax)
    basemap_segments = draw_antarctic_basemap(ax)
    ax.text(0.0, 1.085, SOUTH_MAP_TITLES.get(case["track_index"], f"{label}  Map + SuperDARN"), fontsize=8.1, weight="bold", color=INK, ha="center", va="bottom")
    roti_im = draw_south_roti_background(ax, case["roti"])

    ismr = case["ismr"].copy()
    ismr_scatter = None
    ismr_points_total = 0
    if not ismr.empty:
        for col in ["ipp_lat", "ipp_lon", "s4"]:
            ismr[col] = pd.to_numeric(ismr[col], errors="coerce")
        ismr = ismr.dropna(subset=["ipp_lat", "ipp_lon", "s4"])
        ismr = ismr[ismr["ipp_lat"].le(-60.0)]
        ismr_points_total = len(ismr)
        ismr_scatter = scatter_south(ax, ismr["ipp_lat"], ismr["ipp_lon"], c=ismr["s4"], cmap=ISMR_CMAP, norm=Normalize(0, 0.5), s=3.4, alpha=0.82, linewidths=0, zorder=8, rasterized=True)

    sd_scatter = None
    fan_plot = fir_fan_latlon_for_map(case)
    if not fan_plot.empty:
        if len(fan_plot) > 1100:
            fan_plot = fan_plot.sort_values(["_bearing_deg", "range_center_km"])
            keep_idx = np.linspace(0, len(fan_plot) - 1, 1100).round().astype(int)
            fan_plot = fan_plot.iloc[np.unique(keep_idx)].copy()
        for _, group in fan_plot.groupby("_bearing_deg"):
            group = group.sort_values("range_center_km")
            if len(group) < 2:
                continue
            plot_south_line(ax, group["map_lat"], group["map_lon"], color="#64748B", lw=0.36, alpha=0.28, zorder=8.5)
        sd_scatter = scatter_south(
            ax,
            fan_plot["map_lat"],
            fan_plot["map_lon"],
            c=fan_plot["mean_power_db"],
            cmap=SD_CMAP,
            norm=Normalize(0, 12),
            marker="s",
            s=8.3,
            alpha=1.0,
            linewidths=0,
            zorder=9,
        )

    draw_south_los_paths_from_csv(ax, case["los_paths"], max_paths=54)
    draw_south_ro_los_paths(ax, case["ro"].get("ro_los_paths_xy") or [], max_paths=42)

    track = case["track"].copy()
    plot_south_line(ax, track["lugre_lat"], track["lugre_lon"], color=LUGRE, lw=2.05, alpha=0.97, zorder=25)
    scatter_south(ax, [track["lugre_lat"].iloc[0]], [track["lugre_lon"].iloc[0]], s=38, color=LUGRE, edgecolors="white", linewidths=0.5, zorder=27)
    scatter_south(ax, [track["lugre_lat"].iloc[-1]], [track["lugre_lon"].iloc[-1]], s=58, marker="X", color=LUGRE, edgecolors="white", linewidths=0.5, zorder=27)

    sd = case["sd"]
    if sd:
        scatter_south(ax, [float(sd["nearest_track_lat"])], [float(sd["nearest_track_lon"])], s=88, marker="o", facecolors="#F8FAFC", edgecolors=LUGRE, linewidths=2.0, zorder=33)
        nx, ny = polar_xy_south([float(sd["nearest_track_lat"])], [float(sd["nearest_track_lon"])])
        ax.text(nx[0] + 0.025, ny[0] + 0.005, "LuGRE", color=LUGRE, fontsize=6.4, weight="bold", zorder=34)

    ro_lat, ro_lon = polar_xy_to_lonlat_south([case["ro"].get("ro_nearest_xy", [])], plot_xy=False)
    if len(ro_lat):
        scatter_south(ax, ro_lat, ro_lon, s=78, marker="o", color=LEORO, edgecolors="white", linewidths=0.9, zorder=32)
        rx, ry = polar_xy_south(ro_lat, ro_lon)
        ax.text(rx[0] + 0.025, ry[0] - 0.025, "LEO-RO", color=LEORO, fontsize=6.4, weight="bold", zorder=34)

    if roti_im is not None:
        cax = ax.inset_axes([-0.062, 0.705, 0.018, 0.190])
        cb = fig.colorbar(ScalarMappable(norm=PowerNorm(gamma=1.25, vmin=0, vmax=4), cmap=ROTI_CMAP), cax=cax, orientation="vertical", ticks=[0, 2, 4])
        cb.ax.tick_params(labelsize=4.6, length=1.4, pad=1)
        cb.set_label("ROTI (TECU/min)", fontsize=5.0, labelpad=1.5, color=AXIS, fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_color(AXIS)
            tick.set_fontweight("bold")
    if sd_scatter is not None:
        cax = ax.inset_axes([-0.062, 0.475, 0.018, 0.190])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 12), cmap=SD_CMAP), cax=cax, orientation="vertical", ticks=[0, 6, 12])
        cb.ax.tick_params(labelsize=4.6, length=1.4, pad=1)
        cb.set_label("SD power (dB)", fontsize=5.0, labelpad=1.5, color=AXIS, fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_color(AXIS)
            tick.set_fontweight("bold")
    if ismr_scatter is not None:
        cax = ax.inset_axes([-0.062, 0.245, 0.018, 0.190])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 0.5), cmap=ISMR_CMAP), cax=cax, orientation="vertical", ticks=[0.0, 0.25, 0.5])
        cb.ax.tick_params(labelsize=4.6, length=1.4, pad=1)
        cb.set_label(r"ISMR $S_4$", fontsize=5.0, labelpad=1.5, color=AXIS, fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_color(AXIS)
            tick.set_fontweight("bold")

    return {
        "track_index": case["track_index"],
        "outer_latitude_deg": -60,
        "roti_valid_cells_plotted": getattr(roti_im, "roti_valid_cells", 0),
        "roti_colored_grid_fraction": getattr(roti_im, "roti_colored_grid_fraction", np.nan),
        "fig7_basemap_line_segments_plotted": basemap_segments,
        "ismr_ipp_points_plotted": int(ismr_points_total),
        "superdarn_fan_power_cells_plotted": int(len(fan_plot)) if not fan_plot.empty else 0,
        "superdarn_fan_bearing_note": str(fan_plot["_bearing_note"].dropna().iloc[0]) if not fan_plot.empty and "_bearing_note" in fan_plot.columns and fan_plot["_bearing_note"].notna().any() else "",
        "fir_site_lat": case["station"].get("lat_use", np.nan),
        "fir_site_lon": case["station"].get("lon_use", np.nan),
        "matched_cell_lat": sd.get("nearest_cell_lat") if sd else np.nan,
        "matched_cell_lon": sd.get("nearest_cell_lon") if sd else np.nan,
    }


def draw_south_height_time(ax, case: dict, label: str) -> None:
    north.draw_height_layers(ax)
    profiles = south_lugre_profiles(case) + south_leoro_profiles(case)
    for df in profiles:
        ax.plot(df["time"], df["height"], color=df["color"].iloc[0], lw=1.15, alpha=0.88)
    north.set_height_axis(ax)
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis="y", labelleft=True, labelright=False)
    ax.set_ylabel("Tangent height [km]", fontsize=7.0, color=AXIS, fontweight="bold", labelpad=2)
    ax.set_title(f"{label}  Height-time", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=3)
    all_times = pd.concat([df["time"] for df in profiles if "time" in df.columns], ignore_index=True).dropna() if profiles else pd.Series(dtype="datetime64[ns, UTC]")
    track_times = pd.to_datetime(case["track"]["time_utc"], utc=True, errors="coerce").dropna()
    combined_times = pd.concat([track_times, all_times], ignore_index=True).dropna()
    start = combined_times.min().floor("min") if not combined_times.empty else pd.NaT
    end = combined_times.max().ceil("min") if not combined_times.empty else pd.NaT
    if pd.isna(start) or pd.isna(end):
        start = pd.Timestamp("2025-03-03 12:00", tz="UTC")
        end = start + pd.Timedelta(minutes=8)
    span_min = max(1.0, (end - start).total_seconds() / 60.0)
    tick_freq = "3min" if span_min > 12 else "2min"
    ticks = list(pd.date_range(start, end, freq=tick_freq))
    if not ticks or ticks[0] != start:
        ticks.insert(0, start)
    ax.set_xlim(start, end)
    ax.set_xticks([mdates.date2num(t.to_pydatetime()) for t in ticks])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
    ax.set_xlabel("UTC", fontsize=7.0, color=AXIS, fontweight="bold", labelpad=2)
    ax.text(0.98, 0.96, f"{start:%Y-%m-%d} | OP38 S", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=AXIS, fontweight="bold")
    north.make_axis_text_bold(ax)


def draw_south_dcn0_panel(ax, case: dict, label: str) -> None:
    north.draw_height_layers(ax)
    handles = []
    seen = set()
    for df in south_lugre_profiles(case) + south_leoro_profiles(case):
        ax.scatter(df["dcn0"], df["height"], s=8.5, marker=df["marker"].iloc[0], color=df["color"].iloc[0], alpha=0.66, linewidths=0)
        lab = df["label"].iloc[0]
        if lab not in seen:
            handles.append(Line2D([0], [0], marker=df["marker"].iloc[0], color="none", markerfacecolor=df["color"].iloc[0], markeredgecolor="none", markersize=4.1, label=lab))
            seen.add(lab)
    ax.axvline(0, color="#334155", lw=0.65, ls=(0, (2.5, 2.2)), alpha=0.65)
    ax.set_xlim(-6, 6)
    ax.set_xticks([-6, -3, 0, 3, 6])
    north.set_height_axis(ax, side="right")
    ax.set_title(f"{label}  $\\delta C/N_0$", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=3)
    ax.set_xlabel(r"$\delta C/N_0$ (dB)", fontsize=7.0, color=AXIS, fontweight="bold", labelpad=2)
    north.make_axis_text_bold(ax)
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=6.2,
        frameon=False,
        ncol=2,
        handletextpad=0.24,
        columnspacing=0.58,
        labelspacing=0.20,
        borderaxespad=0.20,
    )


def draw_south_ne_panel(ax, case: dict, label: str) -> dict:
    north.draw_height_layers(ax)
    iri = case["iri"].copy()
    ion = case["ionprf"].copy()
    for col in ["height_km", "log10_ne"]:
        if col in iri.columns:
            iri[col] = pd.to_numeric(iri[col], errors="coerce")
        if col in ion.columns:
            ion[col] = pd.to_numeric(ion[col], errors="coerce")
    iri = iri.dropna(subset=["height_km", "log10_ne"])
    ion = ion.dropna(subset=["height_km", "log10_ne"])
    if not iri.empty:
        ax.plot(iri["log10_ne"], iri["height_km"], color="#1769AA", lw=1.75, ls=(0, (3.5, 2.0)), label="IRI", zorder=4)
    if not ion.empty:
        ax.plot(ion["log10_ne"], ion["height_km"], color="#B42318", lw=1.65, ls="-", label="YunYao", zorder=5)
    north.set_height_axis(ax, label="Height [km]")
    ax.set_xlim(8.0, 12.0)
    ax.set_xticks([8, 10, 12])
    ax.set_title(f"{label}  $N_e(h)$", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=3)
    ax.set_xlabel(r"$\log_{10} N_e$ [m$^{-3}$]", fontsize=7.0, color=AXIS, fontweight="bold", labelpad=2)
    north.make_axis_text_bold(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=5.8, frameon=False, handlelength=1.6, borderaxespad=0.0)
    return {
        "figure_track_index": case["track_index"],
        "iri_rows": len(iri),
        "yunyao_rows": len(ion),
        "profile_source": "south_pair_restored_package_yunyao_plus_iri",
    }


def draw_south_fan_panel(ax, case: dict, label: str) -> None:
    fan = prepare_fir_fan_bearings(case["fan"])
    ax.set_title(f"{label}  SuperDARN fan", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=3)
    if fan.empty:
        ax.text(0.5, 0.5, "No fan table", transform=ax.transAxes, ha="center", va="center", fontsize=7, color=MUTED)
        return
    for col in ["_bearing_deg", "range_center_km", "mean_power_db"]:
        fan[col] = pd.to_numeric(fan[col], errors="coerce")
    fan = fan.dropna(subset=["_bearing_deg", "range_center_km", "mean_power_db"])
    display_bearing = fan["_bearing_deg"].to_numpy(dtype=float)
    sc = ax.scatter(np.deg2rad(display_bearing), fan["range_center_km"], c=fan["mean_power_db"], cmap=SD_CMAP, norm=Normalize(0, 12), s=10, alpha=0.82, linewidths=0)
    station = case["station"]
    st_lat = float(station.get("lat_use", station.get("lat", -51.8314)))
    st_lon = float(station.get("lon_use", station.get("lon", -58.9793)))
    # South-pole fan azimuths are displayed with the sector opening upward,
    # matching the accepted north-pole E/J visual orientation. The map panel
    # still uses the geodetic FIR fan coordinates from the source table.
    ax.set_theta_zero_location("S", offset=18)
    ax.set_theta_direction(-1)
    theta_min = float(np.nanmin(display_bearing))
    theta_max = float(np.nanmax(display_bearing))
    theta_pad = max(4.0, 0.06 * (theta_max - theta_min))
    ax.set_thetamin(theta_min - theta_pad)
    ax.set_thetamax(theta_max + theta_pad)
    ax.set_ylim(0, min(3500, max(800, float(fan["range_center_km"].max()) + 100)))
    range_ticks = [1000, 2000, 3000]
    range_ticks = [tick for tick in range_ticks if tick <= ax.get_ylim()[1]]
    ax.set_yticks(range_ticks)
    ax.set_yticklabels([])
    track = case["track"].copy()
    b, d = bearing_distance_from_station(track["lugre_lat"], track["lugre_lon"], st_lat, st_lon)
    ok = np.isfinite(b) & np.isfinite(d) & (d <= ax.get_ylim()[1])
    if ok.sum() >= 2:
        ax.plot(np.deg2rad(b[ok]), d[ok], color="#111827", lw=1.35, alpha=0.95, zorder=8, label="LuGRE TP")
    ax.grid(True, color=GRID, alpha=0.68, lw=0.55)
    ax.tick_params(axis="x", labelsize=6.2, colors=AXIS, width=0.65, length=2.5, pad=-2.0)
    ax.tick_params(axis="y", length=0)
    tick_theta_deg = theta_max + theta_pad + 5.0
    label_theta_deg = theta_max + theta_pad + 13.0

    def radial_text_angle(theta_deg: float) -> float:
        theta = np.deg2rad(theta_deg)
        r0 = ax.get_ylim()[1] * 0.35
        r1 = ax.get_ylim()[1] * 0.85
        p0 = ax.transData.transform((theta, r0))
        p1 = ax.transData.transform((theta, r1))
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        return angle

    tick_theta = np.deg2rad(tick_theta_deg)
    range_label_rotation = radial_text_angle(label_theta_deg)
    for tick in range_ticks:
        ax.text(tick_theta, tick, f"{tick}", rotation=range_label_rotation, rotation_mode="anchor", fontsize=6.1, fontweight="bold", color=AXIS, ha="left", va="center", clip_on=False)
    ax.text(np.deg2rad(label_theta_deg), ax.get_ylim()[1] * 0.66, "Range (km)", rotation=range_label_rotation, rotation_mode="anchor", ha="left", va="center", fontsize=5.9, fontweight="bold", color=AXIS, clip_on=False)
    cax = ax.inset_axes([0.09, -0.075, 0.82, 0.045])
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal", ticks=[0, 4, 8, 12])
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.set_label("Power (dB)", fontsize=6.1, color=AXIS, fontweight="bold")
    for tick in cbar.ax.get_xticklabels():
        tick.set_color(AXIS)
        tick.set_fontweight("bold")
    north.make_axis_text_bold(ax)


def draw_south_pair_block() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )
    cases = [load_south_case(9), load_south_case(17)]
    letters = [("a", "b", "c", "d", "e"), ("f", "g", "h", "i", "j")]
    fig = plt.figure(figsize=(12.95, 10.6), dpi=260, facecolor="white")
    outer = fig.add_gridspec(
        2,
        3,
        left=0.030,
        right=0.955,
        bottom=0.062,
        top=0.972,
        width_ratios=[2.05, 0.48, 0.48],
        hspace=0.112,
        wspace=0.045,
    )
    qa_rows = []
    for r, case in enumerate(cases):
        ax_map = fig.add_subplot(outer[r, 0])
        north.shift_axis(ax_map, dx=0.018)
        qa = draw_south_map(ax_map, fig, case, letters[r][0].upper())
        sub = outer[r, 1:].subgridspec(2, 2, hspace=0.30, wspace=0.16)
        ax_b = fig.add_subplot(sub[0, 0])
        ax_c = fig.add_subplot(sub[0, 1])
        ax_d = fig.add_subplot(sub[1, 0])
        ax_e = fig.add_subplot(sub[1, 1], projection="polar")
        north.align_fan_axis(fig, ax_e, ax_c, ax_d)
        draw_south_height_time(ax_b, case, letters[r][1].upper())
        draw_south_dcn0_panel(ax_c, case, letters[r][2].upper())
        qa.update(draw_south_ne_panel(ax_d, case, letters[r][3].upper()))
        draw_south_fan_panel(ax_e, case, letters[r][4].upper())
        north.align_fan_axis(fig, ax_e, ax_c, ax_d)
        qa_rows.append(qa)

    fig.legend(
        handles=north.map_legend_handles(),
        loc="center left",
        bbox_to_anchor=(0.153, 0.550),
        frameon=True,
        facecolor="white",
        edgecolor="#E2E8F0",
        fontsize=5.15,
        ncol=1,
        handlelength=1.25,
        columnspacing=0.50,
        handletextpad=0.35,
    )
    fig.savefig(SOUTH_PNG, dpi=260)
    fig.savefig(SOUTH_PDF)
    plt.close(fig)
    pd.DataFrame(qa_rows).to_csv(QA_OUT, index=False, encoding="utf-8-sig")


def combine_side_by_side() -> None:
    north_img = Image.open(NORTH_PNG).convert("RGB")
    south_img = Image.open(SOUTH_PNG).convert("RGB")
    if south_img.size != north_img.size:
        south_img = south_img.resize(north_img.size, Image.Resampling.LANCZOS)

    def crop_horizontal_white_margin(img: Image.Image, margin_px: int = 10) -> Image.Image:
        """Tighten only left/right white canvas; keep panel heights unchanged."""
        arr = np.asarray(img)
        nonwhite = np.any(arr < 252, axis=2)
        cols = np.where(nonwhite.any(axis=0))[0]
        if cols.size == 0:
            return img
        left = max(0, int(cols[0]) - margin_px)
        right = min(img.width, int(cols[-1]) + margin_px + 1)
        return img.crop((left, 0, right, img.height))

    north_img = crop_horizontal_white_margin(north_img, margin_px=10)
    south_img = crop_horizontal_white_margin(south_img, margin_px=10)
    gap = 12
    canvas = Image.new("RGB", (north_img.width + gap + south_img.width, north_img.height), "white")
    canvas.paste(north_img, (0, 0))
    canvas.paste(south_img, (north_img.width + gap, 0))
    canvas.save(COMBINED_PNG)
    canvas.save(COMBINED_PDF, "PDF", resolution=300.0)


def write_notes() -> None:
    MANIFEST_OUT.write_text(
        "\n".join(
            [
                "role,file_path,notes",
                f"north_pair_png,{NORTH_PNG},Accepted north v43 block used unchanged.",
                f"south_pair_compact_data,{SOUTH_DATA},South track009/017 compact source tables from restored LuGRE LOS package.",
                f"south_leoro_raw_planetiq_conphs,{PLANETIQ_CONPHS_TAR},Raw conPhs source for PlanetiQ G18 L1/L2 rolling-median C/N0-proxy residuals; 0.05 Hz high-pass samples are not used.",
                f"south_leoro_raw_spire_conphs,{SPIRE_CONPHS_TAR},Raw conPhs source for Spire C25 L1/L2 rolling-median C/N0-proxy residuals; 0.05 Hz high-pass samples are not used.",
                f"south_leoro_rolling_median_l1l2_points,{LEORO_SOUTH_ROLLING_L1L2_POINTS},Generated D-drive point table used by south B/G/C/H.",
                f"south_pair_png,{SOUTH_PNG},South block redrawn with the same size/layout/fonts/colorbars as north v43.",
                f"combined_png,{COMBINED_PNG},North and south blocks placed side by side with identical block dimensions.",
                f"combined_pdf,{COMBINED_PDF},PDF export of the side-by-side candidate.",
            ]
        ),
        encoding="utf-8",
    )
    README_OUT.write_text(
        "\n".join(
            [
                "# Fig.4 north/south side-by-side candidate v9 compact horizontal",
                "",
                "This version keeps the accepted north v43 block unchanged and redraws the south OP38 E21/E23 cases from the compact restored LuGRE LOS package with the same v43 layout, font sizes, axis styling, colorbars and panel proportions.",
                "",
                "Compared with v8, v9 keeps the same south LOS styling and upright south SuperDARN fan inset, but trims only the left/right white canvas and reduces the north-south gap in the combined image. No data, panel content, axis ranges, or internal evidence layers are changed.",
                "",
                "Core layout: north pair on the left, south pair on the right; each half has the same two-row A-E/F-J structure.",
                "",
                "South data sources include LuGRE track tables, restored LuGRE LOS paths, FIR SuperDARN segment fan/context footprints, ROTI context grids, ISMR S4 IPP context points, and YunYao/IRI electron-density profiles from the provided south package.",
                "",
                f"South LEO-RO C/N0 residuals use raw conPhs high-rate SNR from `{PLANETIQ_CONPHS_TAR}` and `{SPIRE_CONPHS_TAR}`. The script converts each signal to 1 Hz median-SNR C/N0 proxy and subtracts the same centered dynamic rolling-median baseline selected for Fig.3. The generated point table is `{LEORO_SOUTH_ROLLING_L1L2_POINTS}`.",
                "",
                "For the two south compact cases, v5 plots PlanetiQ G18 L1/L2 and Spire C25 L1/L2 point residuals from the same rolling-median method. The 0.05 Hz high-pass sample tables and 60 s window statistics are intentionally not used for C/H.",
                "",
                "The FIR radar site is outside the 60 deg S polar circle, so the south maps show the FIR power fan cells and matched LuGRE/LEO-RO geometry rather than moving the station marker into the polar circle. Because the FIR fan table stores a single bmazm_deg for all beams, v4 expands the map/fan bearing by beam number using the restored south-package logic.",
                "",
                "Rolling-median notes: " + ("; ".join(sorted(set(_LEORO_ROLLING_NOTES))) if _LEORO_ROLLING_NOTES else "none"),
                "",
                f"South PNG: `{SOUTH_PNG}`",
                f"South PDF: `{SOUTH_PDF}`",
                f"Combined PNG: `{COMBINED_PNG}`",
                f"Combined PDF: `{COMBINED_PDF}`",
                f"QA: `{QA_OUT}`",
                f"Manifest: `{MANIFEST_OUT}`",
            ]
        ),
        encoding="utf-8",
    )


def build() -> None:
    draw_south_pair_block()
    combine_side_by_side()
    write_notes()
    print(SOUTH_PNG)
    print(SOUTH_PDF)
    print(COMBINED_PNG)
    print(COMBINED_PDF)


if __name__ == "__main__":
    build()
