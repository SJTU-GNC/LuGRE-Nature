from __future__ import annotations
def split_track_hemi(track: pd.DataFrame, hemisphere: str) -> list[pd.DataFrame]:
    lat = track["lugre_lat"].to_numpy(dtype=float)
    keep = lat >= polar_map.POLAR_LAT_DEG if hemisphere == "north" else lat <= -polar_map.POLAR_LAT_DEG
    if keep.sum() < 2:
        return []
    sub = track.loc[keep].sort_values("time_utc").copy()
    lon = sub["lugre_lon"].to_numpy(dtype=float)
    time_gap = sub["rx_gps_seconds"].diff().fillna(1.0).to_numpy(dtype=float) > 1.5
    lon_gap = np.r_[False, np.abs(np.diff(lon)) > 180.0]
    run_id = np.cumsum(time_gap | lon_gap)
    return [piece for _, piece in sub.groupby(run_id, sort=False) if len(piece) >= 2]

def draw_selected_track_on_polar(ax, track: pd.DataFrame, hemisphere: str, cmap, norm, vmax: float) -> None:
    value_col = polar_map.base.CN0_ABS_RESID_COL
    if value_col not in track.columns:
        track = web.with_cn0_residual(track)
        track[value_col] = np.abs(track[polar_map.base.CN0_RESID_COL].to_numpy(dtype=float))
    for piece in split_track_hemi(track, hemisphere):
        theta = np.deg2rad(piece["lugre_lon"].to_numpy(dtype=float))
        radius = polar_map.polar_radius(piece["lugre_lat"].to_numpy(dtype=float), hemisphere)
        if len(theta) < 2:
            continue
        xy = np.column_stack([theta, radius])
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        values_raw = piece[value_col].to_numpy(dtype=float)
        values = np.clip((values_raw[:-1] + values_raw[1:]) / 2.0, 0.0, vmax)
        ax.add_collection(
            LineCollection(
                segments,
                colors="white",
                linewidths=3.0,
                alpha=0.96,
                capstyle="round",
                joinstyle="round",
                zorder=5.8,
            )
        )
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=1.68,
            alpha=0.98,
            capstyle="round",
            joinstyle="round",
            zorder=6.2,
        )
        lc.set_array(values)
        ax.add_collection(lc)

def add_native_label(ax, track: pd.DataFrame, hemisphere: str, label: str) -> None:
    pieces = split_track_hemi(track, hemisphere)
    if not pieces:
        return
    all_parts = pd.concat(pieces, ignore_index=True)
    theta = np.deg2rad(all_parts["lugre_lon"].to_numpy(dtype=float))
    radius = polar_map.polar_radius(all_parts["lugre_lat"].to_numpy(dtype=float), hemisphere)
    ax.text(
        float(np.nanmean(theta)),
        float(np.nanmean(radius)),
        str(label),
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#111111",
        bbox={"boxstyle": "circle,pad=0.16", "fc": "white", "ec": "#111111", "lw": 0.9, "alpha": 0.97},
        zorder=8,
    )
