from __future__ import annotations
def set_panel_header(ax: plt.Axes, label: str, title: str, label_x: float = PANEL_LABEL_X) -> None:
    ax.text(
        label_x,
        PANEL_HEADER_Y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )
    ax.text(
        0.0,
        PANEL_HEADER_Y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=mpl.rcParams.get("axes.titlesize", 7.4),
        color="#111111",
        clip_on=False,
    )

def geometry_ionosphere_spans(
    height: pd.DataFrame,
    low_km: float = 50.0,
    high_km: float = 1000.0,
    max_gap_s: float = 90.0,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if height.empty:
        return spans
    h = height.copy()
    h["time_dt"] = pd.to_datetime(h["time_dt"], utc=True, errors="coerce")
    h["h_tan_km"] = pd.to_numeric(h["h_tan_km"], errors="coerce")
    valid = h[h["h_tan_km"].between(low_km, high_km)].sort_values("time_dt")
    for _, group in valid.groupby("track_label", sort=False):
        times = group["time_dt"].dropna().sort_values().to_list()
        if not times:
            continue
        start = prev = times[0]
        for t in times[1:]:
            if (t - prev).total_seconds() > max_gap_s:
                spans.append((start, prev))
                start = t
            prev = t
        spans.append((start, prev))
    spans.sort(key=lambda pair: pair[0])
    return spans

def merge_time_spans(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    join_gap_s: float = 2.0,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda pair: pair[0])
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + pd.Timedelta(seconds=join_gap_s):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged

def time_segments(rows: pd.DataFrame, max_gap_s: float = 90.0) -> list[pd.DataFrame]:
    if rows.empty:
        return []
    ordered = rows.dropna(subset=["time_dt"]).sort_values("time_dt").reset_index(drop=True)
    if len(ordered) < 2:
        return [ordered]
    diffs = ordered["time_dt"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)
    segment_id = np.cumsum(diffs > max_gap_s)
    return [piece for _, piece in ordered.groupby(segment_id, sort=False) if not piece.empty]

def map_extent(sim_track: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    lon = pd.to_numeric(sim_track["tp_lon_deg"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(sim_track["tp_lat_deg"], errors="coerce").to_numpy(dtype=float)
    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]
    if lon.size == 0 or lat.size == 0:
        return (-180.0, 180.0), (-90.0, 90.0)
    xlim = (max(-180.0, float(np.nanmin(lon)) - 9.0), min(180.0, float(np.nanmax(lon)) + 9.0))
    ylim = (max(-88.0, float(np.nanmin(lat)) - 5.0), min(88.0, float(np.nanmax(lat)) + 7.0))
    return xlim, ylim

def plot_time_pieces_local(
    ax: plt.Axes,
    rows: pd.DataFrame,
    y_col: str,
    color: str,
    *,
    linewidth: float,
    alpha: float = 1.0,
    linestyle: str | tuple[float, tuple[float, ...]] = "-",
    max_gap_s: float = 90.0,
    zorder: float = 3.0,
) -> None:
    rows = rows.dropna(subset=["time_dt", y_col]).sort_values("time_dt")
    for piece in time_segments(rows, max_gap_s=max_gap_s):
        if len(piece) < 2:
            continue
        ax.plot(
            piece["time_dt"],
            piece[y_col],
            color=color,
            lw=linewidth,
            alpha=alpha,
            ls=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )
def style_nature_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.62)
        spine.set_color("#222222")
    ax.tick_params(axis="both", which="major", labelsize=6.3, width=0.62, length=3.0, direction="out", pad=2.0)
    ax.grid(True, axis="y", color="#d5d9df", lw=0.45, alpha=0.65)

def format_lon(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1.0e-9:
        return "0"
    hemi = "E" if value > 0 else "W"
    return f"{abs(value):.0f}{hemi}"

def format_lat(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1.0e-9:
        return "0"
    hemi = "N" if value > 0 else "S"
    return f"{abs(value):.0f}{hemi}"

def draw_nature_map_base(
    ax: plt.Axes,
    land_shapes: list[list[tuple[float, float]]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.set_facecolor("#ffffff")
    for poly in land_shapes:
        if len(poly) < 3:
            continue
        xs = np.asarray([p[0] for p in poly], dtype=float)
        ys = np.asarray([p[1] for p in poly], dtype=float)
        if np.nanmax(xs) < xlim[0] or np.nanmin(xs) > xlim[1] or np.nanmax(ys) < ylim[0] or np.nanmin(ys) > ylim[1]:
            continue
        ax.fill(xs, ys, facecolor="#f0f1f2", edgecolor="#bfc5cc", linewidth=0.35, alpha=1.0, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    lon_span = abs(xlim[1] - xlim[0])
    if lon_span <= 14.0:
        lon_step = 2.0
    elif lon_span <= 50.0:
        lon_step = 10.0
    else:
        lon_step = 20.0
    ax.xaxis.set_major_locator(MultipleLocator(lon_step))
    ax.yaxis.set_major_locator(MultipleLocator(10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.grid(True, color="#d7dce2", lw=0.45, alpha=0.78, zorder=1)
    ax.set_xlabel("Longitude", fontsize=6.6, labelpad=2.0)
    ax.set_ylabel("Latitude", fontsize=6.6, labelpad=2.0)
    style_nature_axis(ax)
