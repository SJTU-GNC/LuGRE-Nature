from __future__ import annotations
TRACK_KEYS = ["phase", "op_id", "sat", "signal_id", "segment_id"]

CN0_RESID_COL = "cn0_detrend_zero_mean_dbhz"

CN0_ABS_RESID_COL = "cn0_detrend_zero_mean_abs_dbhz"
POLAR_LAT_DEG = 60.0

MAP_LAT_MIN = 60.0

CMAX_DBHZ = 4.0

BOUNDARIES = [0.0, 0.01, 0.025, 0.063, 0.109, 0.170, 0.268, 0.410, 0.615, 0.929, 1.374, 2.203, 4.0]

def wrap_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0

def iter_lines(geometry):
    geom_type = geometry.geom_type
    if geom_type == "LineString":
        yield np.asarray(geometry.coords)
    elif geom_type == "MultiLineString":
        for line in geometry.geoms:
            yield np.asarray(line.coords)
    elif geom_type == "Polygon":
        yield np.asarray(geometry.exterior.coords)
        for interior in geometry.interiors:
            yield np.asarray(interior.coords)
    elif geom_type == "MultiPolygon":
        for poly in geometry.geoms:
            yield np.asarray(poly.exterior.coords)
            for interior in poly.interiors:
                yield np.asarray(interior.coords)

def polar_radius(lat: np.ndarray, hemisphere: str) -> np.ndarray:
    if hemisphere == "north":
        return 90.0 - lat
    return 90.0 + lat

def setup_axis(ax: plt.Axes, hemisphere: str, title: str) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.0, 90.0 - MAP_LAT_MIN)
    ax.set_facecolor("white")
    ax.set_title(title, pad=18, fontsize=14)
    ax.grid(True, color="#b7c0ca", linestyle="-", linewidth=0.9, alpha=0.30)
    degs = np.arange(0, 360, 45)
    labels = ["0°", "45°E", "90°E", "135°E", "180°", "135°W", "90°W", "45°W"]
    ax.set_thetagrids(degs, labels=labels)
    ax.set_yticks([10, 20, 30])
    if hemisphere == "north":
        ax.set_yticklabels(["80°N", "70°N", "60°N"])
        ax.text(0, 0, "90°N", ha="center", va="center", fontsize=8.5, color="#26323f", zorder=5)
    else:
        ax.set_yticklabels(["80°S", "70°S", "60°S"])
        ax.text(0, 0, "90°S", ha="center", va="center", fontsize=8.5, color="#26323f", zorder=5)
    for deg in degs:
        theta = np.deg2rad(deg)
        ax.plot([theta, theta], [0, 90 - MAP_LAT_MIN], color="#8fa3b5", lw=0.7, ls="-", alpha=0.24, zorder=0)

def kept_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = []
    ends = []
    in_run = False
    start = 0
    for idx, keep in enumerate(mask):
        if keep and not in_run:
            start = idx
            in_run = True
        elif not keep and in_run:
            starts.append(start)
            ends.append(idx)
            in_run = False
    if in_run:
        starts.append(start)
        ends.append(len(mask))
    return [(start, end) for start, end in zip(starts, ends) if end - start >= 2]

def split_geo_piece(lon: np.ndarray, lat: np.ndarray, hemisphere: str) -> list[tuple[np.ndarray, np.ndarray]]:
    keep = lat >= (MAP_LAT_MIN - 0.6) if hemisphere == "north" else lat <= -(MAP_LAT_MIN - 0.6)
    pieces = []
    for start, end in kept_runs(keep):
        lon_run = wrap_lon(lon[start:end])
        lat_run = lat[start:end]
        jumps = np.where(np.abs(np.diff(lon_run)) > 40.0)[0] + 1
        starts = np.r_[0, jumps]
        ends = np.r_[jumps, len(lon_run)]
        for sub_start, sub_end in zip(starts, ends):
            if sub_end - sub_start >= 2:
                pieces.append((lon_run[sub_start:sub_end], lat_run[sub_start:sub_end]))
    return pieces

def draw_land(ax: plt.Axes, hemisphere: str) -> None:
    land_path = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    reader = shpreader.Reader(land_path)
    for record in reader.records():
        for coords in iter_lines(record.geometry):
            if coords.size == 0:
                continue
            lon = coords[:, 0].astype(float)
            lat = coords[:, 1].astype(float)
            for lon_piece, lat_piece in split_geo_piece(lon, lat, hemisphere):
                if len(lon_piece) < 3:
                    continue
                radius = polar_radius(lat_piece, hemisphere)
                if np.nanmax(np.abs(np.diff(radius))) > 12.0:
                    continue
                ax.fill(
                    np.deg2rad(lon_piece),
                    radius,
                    color="white",
                    alpha=1.0,
                    linewidth=0,
                    zorder=0.18,
                )

def draw_coastlines(ax: plt.Axes, hemisphere: str) -> None:
    draw_land(ax, hemisphere)
    coast_path = shpreader.natural_earth(resolution="50m", category="physical", name="coastline")
    reader = shpreader.Reader(coast_path)
    for record in reader.records():
        for coords in iter_lines(record.geometry):
            if coords.size == 0:
                continue
            lon = coords[:, 0].astype(float)
            lat = coords[:, 1].astype(float)
            for lon_piece, lat_piece in split_geo_piece(lon, lat, hemisphere):
                radius = polar_radius(lat_piece, hemisphere)
                if np.nanmax(np.abs(np.diff(radius))) > 12.0:
                    continue
                ax.plot(
                    np.deg2rad(lon_piece),
                    radius,
                    color="#7e8b96",
                    lw=0.74,
                    alpha=0.88,
                    zorder=0.5,
                )

def split_track(track: pd.DataFrame, hemisphere: str) -> list[pd.DataFrame]:
    lat = track["lugre_lat"].to_numpy(dtype=float)
    keep = lat >= POLAR_LAT_DEG if hemisphere == "north" else lat <= -POLAR_LAT_DEG
    if keep.sum() < 2:
        return []
    sub = track.loc[keep].sort_values("time_utc").copy()
    lon = sub["lugre_lon"].to_numpy(dtype=float)
    time_gap = sub["rx_gps_seconds"].diff().fillna(1.0).to_numpy(dtype=float) > 1.5
    lon_gap = np.r_[False, np.abs(np.diff(lon)) > 180.0]
    run_id = np.cumsum(time_gap | lon_gap)
    return [piece for _, piece in sub.groupby(run_id, sort=False) if len(piece) >= 2]

def add_tracks(ax: plt.Axes, df: pd.DataFrame, hemisphere: str, cmap, norm, linewidth: float) -> int:
    count = 0
    for _, track in df.sort_values("time_utc").groupby(base.TRACK_KEYS, dropna=False):
        pieces = split_track(track, hemisphere)
        if pieces:
            count += 1
        for piece in pieces:
            lon = piece["lugre_lon"].to_numpy(dtype=float)
            lat = piece["lugre_lat"].to_numpy(dtype=float)
            theta = np.deg2rad(lon)
            radius = polar_radius(lat, hemisphere)
            xy = np.column_stack([theta, radius])
            values_raw = np.clip(piece[base.CN0_ABS_RESID_COL].to_numpy(dtype=float), 0.0, CMAX_DBHZ)
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            values = np.clip((values_raw[:-1] + values_raw[1:]) / 2.0, 0.0, CMAX_DBHZ)
            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=linewidth,
                alpha=1.0,
                capstyle="round",
                joinstyle="round",
                rasterized=True,
                zorder=4.2,
            )
            lc.set_array(values)
            ax.add_collection(lc)
    return count
