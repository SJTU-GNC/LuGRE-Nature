from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "panel_ready" / "Fig5"
OUT_DIR = ROOT / "work" / "panel_fig5"
NORTH_SCRIPT = TASK_DIR / "north_source.py"
SOUTH_SCRIPT = TASK_DIR / "south_source.py"

PNG_OUT = OUT_DIR / "fig4_north_south_v37_yunyao_ionprf_ne_colors.png"
PDF_OUT = OUT_DIR / "fig4_north_south_v37_yunyao_ionprf_ne_colors.pdf"
QA_OUT = OUT_DIR / "fig4_north_south_v37_yunyao_ionprf_ne_colors_QA.csv"
MANIFEST_OUT = OUT_DIR / "fig4_north_south_v37_yunyao_ionprf_ne_colors_source_manifest.csv"
README_OUT = OUT_DIR / "README_fig4_north_south_v37_yunyao_ionprf_ne_colors.md"
ISR_AUDIT_OUT = OUT_DIR / "fig4_north_south_v35_isr_time_coverage_audit.csv"
PFISR_PROFILE_SOURCE = DATA_ROOT / "common" / "pfisr_profiles.csv"
PFISR_MATCH_SUMMARY = OUT_DIR / "fig4_pfisr_profiles_d_i_match_summary.csv"
PFISR_IRI_PROFILE_SOURCE = OUT_DIR / "fig4_pfisr_site_iri2016_profiles_d_i_50_1000km.csv"
PFISR_SITE_LAT = 65.13
PFISR_SITE_LON = -147.471
ROLLING_POINTS_SOURCE = DATA_ROOT / "common" / "leoro_rolling_points.csv"
ROLLING_POINTS_OUT = OUT_DIR / "fig4_north_south_v37_yunyao_ionprf_ne_colors_leoro_1hz_points.csv"
HIGHRATE_POINTS_SOURCE = DATA_ROOT / "common" / "leoro_highrate_points.csv"
HIGHRATE_L1_POINTS_OUT = OUT_DIR / "fig4_north_south_v37_yunyao_ionprf_ne_colors_highrate_l1_points.csv"
YUNYAO_PROFILE_PACKAGE = DATA_ROOT / "common" / "yunyao_profiles"
YUNYAO_SUMMARY_SOURCE = YUNYAO_PROFILE_PACKAGE / "matched_profiles.csv"

SOUTH_FAN_INSET_RANGE_MAX_KM = 6500.0
YUNYAO_MAP_COLOR = "#7C3AED"
YUNYAO_NE_COLOR = "#B42318"
ISR_NE_COLOR = "#1769AA"
IRI_COLOR = "#111827"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


north = load_module("fig4_north_v43_source", NORTH_SCRIPT)
south = load_module("fig4_south_v9_source", SOUTH_SCRIPT)

plt = north.plt
pd = north.pd
ScalarMappable = north.ScalarMappable
Normalize = north.Normalize
PowerNorm = north.PowerNorm

_draw_south_fan_panel_original = south.draw_south_fan_panel


def draw_south_fan_panel_6500km(ax, case: dict, label: str) -> None:
    """Keep the accepted south fan style but show the long FIR ranges."""
    _draw_south_fan_panel_original(ax, case, label)
    ax.set_ylim(0, SOUTH_FAN_INSET_RANGE_MAX_KM)
    for text in list(ax.texts):
        if text.get_text() in {"1000", "2000", "3000", "Range (km)"}:
            text.remove()
    range_ticks = [2000, 4000, 6000]
    ax.set_yticks(range_ticks)
    ax.set_yticklabels([])

    def radial_text_angle(theta_deg: float) -> float:
        theta = north.np.deg2rad(theta_deg)
        r0 = ax.get_ylim()[1] * 0.35
        r1 = ax.get_ylim()[1] * 0.85
        p0 = ax.transData.transform((theta, r0))
        p1 = ax.transData.transform((theta, r1))
        angle = north.np.degrees(north.np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        return angle

    tick_theta_deg = ax.get_thetamax() + 5.0
    label_theta_deg = ax.get_thetamax() + 13.0
    tick_theta = north.np.deg2rad(tick_theta_deg)
    range_label_rotation = radial_text_angle(label_theta_deg)
    for tick in range_ticks:
        ax.text(
            tick_theta,
            tick,
            f"{tick}",
            rotation=range_label_rotation,
            rotation_mode="anchor",
            fontsize=6.1,
            fontweight="bold",
            color=north.AXIS,
            ha="left",
            va="center",
            clip_on=False,
        )
    ax.text(
        north.np.deg2rad(label_theta_deg),
        ax.get_ylim()[1] * 0.64,
        "Range (km)",
        rotation=range_label_rotation,
        rotation_mode="anchor",
        ha="left",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color=north.AXIS,
        clip_on=False,
    )


south.draw_south_fan_panel = draw_south_fan_panel_6500km

_north_map_legend_handles_original = north.map_legend_handles


def north_map_legend_handles_pfisr_site():
    handles = _north_map_legend_handles_original()
    for handle in handles:
        if handle.get_label() == "ISR context":
            handle.set_label("PFISR site")
    handles.append(
        north.Line2D(
            [0],
            [0],
            color=YUNYAO_MAP_COLOR,
            lw=1.35,
            marker="P",
            markerfacecolor=YUNYAO_MAP_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.0,
            label="YunYao RO track",
        )
    )
    return handles


north.map_legend_handles = north_map_legend_handles_pfisr_site


def with_pfisr_site_context(case: dict) -> dict:
    """Use the ISR station that supplies panels d/i on north map panels."""
    updated = dict(case)
    ctx = dict(updated.get("isr_context") or {})
    ctx.update(
        {
            "station": "PFISR",
            "station_lat": PFISR_SITE_LAT,
            "station_lon": PFISR_SITE_LON,
            "distance_km": "",
            "time_gap_hr": 0,
        }
    )
    updated["isr_context"] = ctx
    return updated

_ROLLING_POINT_CACHE = None
_HIGHRATE_L1_CACHE = None
_YUNYAO_SUMMARY_CACHE = None
_YUNYAO_PROFILE_CACHE = {}


def load_yunyao_summary() -> pd.DataFrame:
    global _YUNYAO_SUMMARY_CACHE
    if _YUNYAO_SUMMARY_CACHE is not None:
        return _YUNYAO_SUMMARY_CACHE
    if not YUNYAO_SUMMARY_SOURCE.exists():
        _YUNYAO_SUMMARY_CACHE = pd.DataFrame()
        return _YUNYAO_SUMMARY_CACHE
    df = pd.read_csv(YUNYAO_SUMMARY_SOURCE)
    df["track_index"] = pd.to_numeric(df["track_index"], errors="coerce").astype("Int64")
    for col in [
        "time_gap_to_lugre_window_min",
        "min_distance_to_lugre_tangent_km",
        "representative_tp_latitude_deg",
        "representative_tp_longitude_deg",
        "representative_tp_altitude_km",
        "profile_point_count",
        "profile_altitude_min_km",
        "profile_altitude_max_km",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _YUNYAO_SUMMARY_CACHE = df
    return _YUNYAO_SUMMARY_CACHE


def load_yunyao_profile(track_index: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    track_index = int(track_index)
    if track_index in _YUNYAO_PROFILE_CACHE:
        return _YUNYAO_PROFILE_CACHE[track_index]
    summary = load_yunyao_summary()
    row_df = summary[summary["track_index"].eq(track_index)].copy()
    if row_df.empty:
        result = (pd.DataFrame(), pd.DataFrame(), {})
        _YUNYAO_PROFILE_CACHE[track_index] = result
        return result
    row = row_df.iloc[0].to_dict()
    obs_rel = str(row.get("observed_profile_csv", "")).replace("\\", "/")
    iri_rel = str(row.get("iri_profile_csv", "")).replace("\\", "/")
    obs_path = YUNYAO_PROFILE_PACKAGE / obs_rel
    iri_path = YUNYAO_PROFILE_PACKAGE / iri_rel
    obs = pd.read_csv(obs_path) if obs_path.exists() else pd.DataFrame()
    iri = pd.read_csv(iri_path) if iri_path.exists() else pd.DataFrame()
    for df in [obs, iri]:
        for col in [
            "altitude_km",
            "latitude_deg",
            "longitude_deg",
            "electron_density_m3",
            "log10_electron_density_m3",
            "iri_electron_density_m3",
            "iri_log10_electron_density_m3",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if not obs.empty:
        obs = obs.dropna(subset=["altitude_km", "log10_electron_density_m3"]).sort_values("altitude_km")
    if not iri.empty:
        iri = iri.dropna(subset=["altitude_km", "iri_log10_electron_density_m3"]).sort_values("altitude_km")
    result = (obs, iri, row)
    _YUNYAO_PROFILE_CACHE[track_index] = result
    return result


def yunyao_metadata_for_track(track_index: int) -> dict:
    obs, iri, meta = load_yunyao_profile(track_index)
    if not meta:
        return {
            "yunyao_track_index": track_index,
            "yunyao_status": "missing",
            "yunyao_profile_package": str(YUNYAO_PROFILE_PACKAGE),
        }
    return {
        "yunyao_track_index": track_index,
        "yunyao_status": "plotted",
        "yunyao_leo": meta.get("leo", ""),
        "yunyao_prn": meta.get("prn", ""),
        "yunyao_event_time_utc": meta.get("event_time_utc", ""),
        "yunyao_time_gap_min": meta.get("time_gap_to_lugre_window_min", ""),
        "yunyao_min_distance_km": meta.get("min_distance_to_lugre_tangent_km", ""),
        "yunyao_profile_rows": int(len(obs)),
        "yunyao_iri_rows": int(len(iri)),
        "yunyao_altitude_min_km": float(obs["altitude_km"].min()) if not obs.empty else "",
        "yunyao_altitude_max_km": float(obs["altitude_km"].max()) if not obs.empty else "",
        "yunyao_profile_package": str(YUNYAO_PROFILE_PACKAGE),
    }


def load_rolling_point_table():
    global _ROLLING_POINT_CACHE
    if _ROLLING_POINT_CACHE is not None:
        return _ROLLING_POINT_CACHE
    df = pd.read_csv(ROLLING_POINTS_SOURCE)
    for col in ["time_utc", "tangent_height_roll_km", "delta_rolling_db", "cn0_median_snr_dbhz", "fit_rolling_dbhz"]:
        if col == "time_utc":
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_utc", "tangent_height_roll_km", "delta_rolling_db"]).copy()
    keep = df["panel"].astype(str).isin(["c", "h", "m", "r"])
    df = df.loc[keep].copy()
    df.to_csv(ROLLING_POINTS_OUT, index=False, encoding="utf-8-sig")
    _ROLLING_POINT_CACHE = df
    return _ROLLING_POINT_CACHE


def rolling_leoro_profiles_for_panel(panel: str):
    df = load_rolling_point_table()
    use = df[df["panel"].astype(str).eq(panel)].copy()
    if use.empty:
        return []
    frames = []
    for signal, sub in use.groupby("signal", sort=True):
        mission = str(sub["mission"].dropna().iloc[0])
        event_label = str(sub["event_label"].dropna().iloc[0])
        # Keep the received GNSS PRN visible in the legend.
        prn = ""
        parts = event_label.split("/")
        if len(parts) > 1:
            tokens = parts[1].strip().split()
            if len(tokens) >= 2:
                prn = tokens[1]
        signal_text = str(signal).upper()
        color = "#E45756" if "L1" in signal_text else "#7B61FF"
        marker = "o" if "L1" in signal_text else "D"
        label = f"{mission} {prn} {signal_text}".strip()
        frame = pd.DataFrame(
            {
                "time": sub["time_utc"],
                "height": sub["tangent_height_roll_km"],
                "dcn0": sub["delta_rolling_db"],
                "label": label,
                "color": color,
                "marker": marker,
            }
        ).dropna(subset=["time", "height", "dcn0"]).sort_values("time")
        if not frame.empty:
            frames.append(frame)
    return frames


def load_highrate_l1_table():
    global _HIGHRATE_L1_CACHE
    if _HIGHRATE_L1_CACHE is not None:
        return _HIGHRATE_L1_CACHE
    df = pd.read_csv(HIGHRATE_POINTS_SOURCE)
    for col in ["time_utc", "height_km", "elapsed_s", "delta_highrate_db"]:
        if col == "time_utc":
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_utc", "height_km", "delta_highrate_db"]).copy()
    df = df[df["panel"].astype(str).isin(["c", "h", "m", "r"]) & df["signal"].astype(str).str.upper().eq("L1")].copy()
    df.to_csv(HIGHRATE_L1_POINTS_OUT, index=False, encoding="utf-8-sig")
    _HIGHRATE_L1_CACHE = df
    return _HIGHRATE_L1_CACHE


def highrate_l1_for_panel(panel: str):
    df = load_highrate_l1_table()
    use = df[df["panel"].astype(str).eq(panel)].copy()
    if use.empty:
        return use
    return use.sort_values("time_utc")


def draw_dcn0_panel_with_highrate_background(ax, case: dict, label: str, *, panel: str, lugre_func, leoro_func) -> None:
    north.draw_height_layers(ax)
    high = highrate_l1_for_panel(panel)
    handles = []
    seen = set()
    if not high.empty:
        ax.scatter(
            high["delta_highrate_db"],
            high["height_km"],
            s=2.9,
            marker="o",
            color="#7B8490",
            alpha=0.165,
            linewidths=0,
            rasterized=True,
            zorder=1.5,
        )
    for df in lugre_func(case) + leoro_func(case):
        ax.scatter(
            df["dcn0"],
            df["height"],
            s=8.5,
            marker=df["marker"].iloc[0],
            color=df["color"].iloc[0],
            alpha=0.66,
            linewidths=0,
            zorder=3.2,
        )
        lab = df["label"].iloc[0]
        if lab not in seen:
            handles.append(
                north.Line2D(
                    [0],
                    [0],
                    marker=df["marker"].iloc[0],
                    color="none",
                    markerfacecolor=df["color"].iloc[0],
                    markeredgecolor="none",
                    markersize=4.1,
                    label=lab,
                )
            )
            seen.add(lab)
    ax.axvline(0, color="#334155", lw=0.65, ls=(0, (2.5, 2.2)), alpha=0.65, zorder=2.2)
    ax.set_xlim(-6, 6)
    ax.set_xticks([-6, -3, 0, 3, 6])
    north.set_height_axis(ax, side="right")
    ax.set_title(f"{label}  $\\delta C/N_0$", loc="left", fontsize=8.6, fontweight="bold", color=north.INK, pad=3)
    ax.set_xlabel(r"$\delta C/N_0$ (dB)", fontsize=7.0, color=north.AXIS, fontweight="bold", labelpad=2)
    north.make_axis_text_bold(ax)
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=5.7,
        frameon=False,
        ncol=2,
        handletextpad=0.18,
        columnspacing=0.45,
        labelspacing=0.16,
        borderaxespad=0.16,
    )


def north_leoro_profiles_1hz_rolling(case: dict):
    panel_by_track = {145: "c", 187: "h"}
    panel = panel_by_track.get(int(case.get("track_index", -1)))
    if not panel:
        return []
    return rolling_leoro_profiles_for_panel(panel)


def south_leoro_profiles_1hz_rolling(case: dict):
    panel_by_track = {9: "m", 17: "r"}
    panel = panel_by_track.get(int(case.get("track_index", -1)))
    if not panel:
        return []
    return rolling_leoro_profiles_for_panel(panel)


north.leoro_profiles = north_leoro_profiles_1hz_rolling
south.south_leoro_profiles = south_leoro_profiles_1hz_rolling


def draw_yunyao_ne_panel(ax, figure_track_index: int, label: str, *, include_pfisr_context: bool = False) -> dict:
    """Plot path-near YunYao ionPrf Ne plus its matched IRI profile."""
    north.draw_height_layers(ax)
    obs, iri, meta = load_yunyao_profile(int(figure_track_index))

    if not iri.empty:
        ax.plot(
            iri["iri_log10_electron_density_m3"],
            iri["altitude_km"],
            color=IRI_COLOR,
            lw=1.85,
            ls=(0, (3.5, 2.0)),
            label="IRI (YunYao)",
            zorder=4,
        )

    if not obs.empty:
        yy_label = "YunYao RO"
        if meta:
            yy_label = f"YunYao {meta.get('leo', '')} {meta.get('prn', '')}".strip()
        ax.plot(
            obs["log10_electron_density_m3"],
            obs["altitude_km"],
            color=YUNYAO_NE_COLOR,
            lw=1.85,
            ls="-",
            label=yy_label,
            zorder=6,
        )

    pfisr = pd.DataFrame()
    if include_pfisr_context and PFISR_PROFILE_SOURCE.exists():
        pfisr_all = pd.read_csv(PFISR_PROFILE_SOURCE)
        pfisr_all["track_index"] = pd.to_numeric(pfisr_all["track_index"], errors="coerce")
        pfisr = pfisr_all[pfisr_all["track_index"].eq(float(figure_track_index))].copy()
        for col in ["height_km", "log10_ne", "relative_dne"]:
            if col in pfisr.columns:
                pfisr[col] = pd.to_numeric(pfisr[col], errors="coerce")
        pfisr = pfisr.dropna(subset=["height_km", "log10_ne"]).sort_values("height_km")
    if not pfisr.empty:
        ax.plot(
            pfisr["log10_ne"],
            pfisr["height_km"],
            color=ISR_NE_COLOR,
            lw=1.35,
            ls="-",
            label="ISR (PFISR)",
            zorder=5,
        )

    north.set_height_axis(ax, label="Height [km]")
    ax.set_xlim(8.0, 12.25)
    ax.set_xticks([8, 10, 12])
    ax.set_title(f"{label}  $N_e(h)$", loc="left", fontsize=8.6, fontweight="bold", color=north.INK, pad=3)
    ax.set_xlabel(r"$\log_{10} N_e$ [m$^{-3}$]", fontsize=7.0, color=north.AXIS, fontweight="bold", labelpad=2)
    north.make_axis_text_bold(ax)
    ax.legend(loc="upper left", fontsize=5.5, frameon=False, handlelength=2.2, bbox_to_anchor=(0.02, 0.98))
    qa = {
        "ne_track": figure_track_index,
        "ne_primary_source": "YunYao ionPrf + matched IRI",
        "ne_yunyao_summary_source": str(YUNYAO_SUMMARY_SOURCE),
        "ne_observed_yunyao_rows": int(len(obs)),
        "ne_yunyao_iri_rows": int(len(iri)),
        "observed_isr_ne_rows": int(len(pfisr)),
        "isr_ne_status": "pfisr_context_line_retained" if include_pfisr_context and not pfisr.empty else "",
    }
    qa.update(yunyao_metadata_for_track(int(figure_track_index)))
    return qa


def draw_north_ne_panel_yunyao(ax, prof_all: pd.DataFrame, matched_all: pd.DataFrame, figure_track_index: int, label: str) -> dict:
    return draw_yunyao_ne_panel(ax, int(figure_track_index), label, include_pfisr_context=True)


def draw_south_ne_panel_yunyao(ax, case: dict, label: str) -> dict:
    return draw_yunyao_ne_panel(ax, int(case["track_index"]), label, include_pfisr_context=False)


north.draw_ne_panel = draw_north_ne_panel_yunyao


south.draw_south_ne_panel = draw_south_ne_panel_yunyao


def overlay_yunyao_track_on_map(ax, track_index: int) -> dict:
    obs, _, meta = load_yunyao_profile(int(track_index))
    if obs.empty:
        return {"map_yunyao_status": "missing", "map_yunyao_track": int(track_index)}
    lat = obs["latitude_deg"]
    lon = obs["longitude_deg"]
    is_north = int(track_index) in {145, 187}
    if is_north:
        north.plot_polar_line(ax, lat, lon, color="white", lw=2.4, alpha=0.90, zorder=36)
        north.plot_polar_line(ax, lat, lon, color=YUNYAO_MAP_COLOR, lw=1.35, alpha=0.98, zorder=37)
        if meta:
            north.scatter_polar(
                ax,
                [float(meta["representative_tp_latitude_deg"])],
                [float(meta["representative_tp_longitude_deg"])],
                s=46,
                marker="P",
                color=YUNYAO_MAP_COLOR,
                edgecolors="white",
                linewidths=0.55,
                zorder=38,
            )
    else:
        south.plot_south_line(ax, lat, lon, color="white", lw=2.4, alpha=0.90, zorder=36)
        south.plot_south_line(ax, lat, lon, color=YUNYAO_MAP_COLOR, lw=1.35, alpha=0.98, zorder=37)
        if meta:
            south.scatter_south(
                ax,
                [float(meta["representative_tp_latitude_deg"])],
                [float(meta["representative_tp_longitude_deg"])],
                s=46,
                marker="P",
                color=YUNYAO_MAP_COLOR,
                edgecolors="white",
                linewidths=0.55,
                zorder=38,
            )
    qa = yunyao_metadata_for_track(int(track_index))
    qa.update(
        {
            "map_yunyao_status": "plotted",
            "map_yunyao_track": int(track_index),
            "map_yunyao_line_points": int(len(obs)),
        }
    )
    return qa


def draw_map_without_duplicate_colorbars(draw_func, ax, fig, case: dict, label: str) -> dict:
    """Draw the accepted map panel but remove its per-panel inset colorbars."""
    before = set(fig.axes)
    qa = draw_func(ax, fig, case, label)
    qa.update(overlay_yunyao_track_on_map(ax, int(case.get("track_index", -1))))
    for text in ax.texts:
        if text.get_text() == "ISR ctx":
            text.set_text("PFISR")
    # Matplotlib inset axes created through ax.inset_axes are often stored as
    # child axes rather than only as top-level figure axes. Remove both forms
    # so the combined figure keeps one shared colorbar stack.
    for child_ax in list(getattr(ax, "child_axes", [])):
        child_ax.remove()
    for extra_ax in list(fig.axes):
        if extra_ax not in before and extra_ax is not ax:
            extra_ax.remove()
    return qa


def add_shared_map_colorbars(fig, x: float = 0.481, y0: float = 0.402, w: float = 0.010) -> None:
    specs = [
        (y0 + 0.195, 0.115, ScalarMappable(norm=PowerNorm(gamma=1.25, vmin=0, vmax=4), cmap=north.ROTI_CMAP), [0, 2, 4], "ROTI (TECU/min)"),
        (y0 + 0.065, 0.115, ScalarMappable(norm=Normalize(0, 12), cmap=north.SD_CMAP), [0, 6, 12], "SD power (dB)"),
        (y0 - 0.065, 0.115, ScalarMappable(norm=Normalize(0, 0.5), cmap=north.ISMR_CMAP), [0.0, 0.25, 0.5], r"ISMR $S_4$"),
    ]
    for y, h, sm, ticks, label in specs:
        cax = fig.add_axes([x, y, w, h])
        cb = fig.colorbar(sm, cax=cax, orientation="vertical", ticks=ticks)
        cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
        cb.set_label(label, fontsize=5.6, labelpad=1.8, color=north.AXIS, fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_color(north.AXIS)
            tick.set_fontweight("bold")


def add_map_legend_and_polar_tags(fig, north_map_center: float, south_map_center: float) -> None:
    fig.legend(
        handles=north.map_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.491, 0.047),
        frameon=True,
        facecolor="white",
        edgecolor="#E2E8F0",
        fontsize=5.3,
        ncol=1,
        handlelength=1.20,
        columnspacing=0.50,
        handletextpad=0.35,
    )
    fig.text(north_map_center, 0.027, "North polar", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=north.INK)
    fig.text(south_map_center, 0.027, "South polar", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=north.INK)


def side_axes(fig, x0: float, y_top: float, y_bottom: float, w: float, h: float, gap: float):
    return (
        fig.add_axes([x0, y_top, w, h]),
        fig.add_axes([x0 + w + gap, y_top, w, h]),
        fig.add_axes([x0, y_bottom, w, h]),
        fig.add_axes([x0 + w + gap, y_bottom, w, h], projection="polar"),
    )


def draw_north_side(fig, case: dict, labels: tuple[str, str, str, str], x0: float, y_top: float, y_bottom: float) -> dict:
    ax_b, ax_c, ax_d, ax_e = side_axes(fig, x0, y_top, y_bottom, 0.086, 0.165, 0.014)
    north.draw_height_time(ax_b, case, labels[0])
    panel_by_track = {145: "c", 187: "h"}
    draw_dcn0_panel_with_highrate_background(
        ax_c,
        case,
        labels[1],
        panel=panel_by_track.get(int(case.get("track_index", -1)), labels[1]),
        lugre_func=north.lugre_profiles,
        leoro_func=north.leoro_profiles,
    )
    prof_all, matched_all = draw_north_side.prof_all
    qa = north.draw_ne_panel(ax_d, prof_all, matched_all, case["track_index"], labels[2])
    north.draw_fan_panel(ax_e, case, labels[3])
    return qa


def draw_south_side(fig, case: dict, labels: tuple[str, str, str, str], x0: float, y_top: float, y_bottom: float) -> dict:
    ax_b, ax_c, ax_d, ax_e = side_axes(fig, x0, y_top, y_bottom, 0.086, 0.165, 0.014)
    south.draw_south_height_time(ax_b, case, labels[0])
    panel_by_track = {9: "m", 17: "r"}
    draw_dcn0_panel_with_highrate_background(
        ax_c,
        case,
        labels[1],
        panel=panel_by_track.get(int(case.get("track_index", -1)), labels[1]),
        lugre_func=south.south_lugre_profiles,
        leoro_func=south.south_leoro_profiles,
    )
    qa = south.draw_south_ne_panel(ax_d, case, labels[2])
    south.draw_south_fan_panel(ax_e, case, labels[3])
    return qa


def build() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )

    # Replace the original uppercase map panel letters in the source titles
    # before drawing; no raster overlay is used.
    north.MAP_TITLES = {
        145: "a  OP76 GPS G01 | north polar night | 2025-03-15 16:24-16:29 UTC",
        187: "f  OP76 GPS G26 | north polar night | 2025-03-16 13:38-13:46 UTC",
    }
    south.SOUTH_MAP_TITLES = {
        9: "k  OP38 Galileo E21 | south polar day | 2025-03-03 12:23-12:32 UTC",
        17: "p  OP38 Galileo E23 | south polar day | 2025-03-03 14:06-14:13 UTC",
    }

    prof_all, matched_all = north.load_ne_profiles()
    draw_north_side.prof_all = (prof_all, matched_all)
    north_cases = [with_pfisr_site_context(north.load_case(145)), with_pfisr_site_context(north.load_case(187))]
    south_cases = [south.load_south_case(9), south.load_south_case(17)]

    fig = plt.figure(figsize=(16.40, 10.25), dpi=260, facecolor="white")

    # Accepted compact structure: north diagnostics left, four maps centered,
    # south diagnostics right. Axis positions are code-level, not image crops.
    left_x = 0.018
    right_x = 0.784
    map_shift_left = 0.012
    north_map_x = 0.244 - map_shift_left
    south_map_x = 0.521 - map_shift_left
    map_w = 0.242
    map_h = 0.395
    map_top_y = 0.535
    map_bottom_y = 0.075

    qa_rows = []
    ax_na = fig.add_axes([north_map_x, map_top_y, map_w, map_h])
    qa_rows.append(draw_map_without_duplicate_colorbars(north.draw_map, ax_na, fig, north_cases[0], "a"))
    ax_nf = fig.add_axes([north_map_x, map_bottom_y, map_w, map_h])
    qa_rows.append(draw_map_without_duplicate_colorbars(north.draw_map, ax_nf, fig, north_cases[1], "f"))
    ax_sk = fig.add_axes([south_map_x, map_top_y, map_w, map_h])
    qa_rows.append(draw_map_without_duplicate_colorbars(south.draw_south_map, ax_sk, fig, south_cases[0], "k"))
    ax_sp = fig.add_axes([south_map_x, map_bottom_y, map_w, map_h])
    qa_rows.append(draw_map_without_duplicate_colorbars(south.draw_south_map, ax_sp, fig, south_cases[1], "p"))

    qa_rows[0].update(draw_north_side(fig, north_cases[0], ("b", "c", "d", "e"), left_x, 0.770, 0.555))
    qa_rows[1].update(draw_north_side(fig, north_cases[1], ("g", "h", "i", "j"), left_x, 0.285, 0.070))
    qa_rows[2].update(draw_south_side(fig, south_cases[0], ("l", "m", "n", "o"), right_x, 0.770, 0.555))
    qa_rows[3].update(draw_south_side(fig, south_cases[1], ("q", "r", "s", "t"), right_x, 0.285, 0.070))

    add_shared_map_colorbars(fig, x=0.496 - map_shift_left, y0=0.402, w=0.010)
    add_map_legend_and_polar_tags(
        fig,
        north_map_center=north_map_x + map_w / 2,
        south_map_center=south_map_x + map_w / 2,
    )

    fig.savefig(PNG_OUT, dpi=260)
    fig.savefig(PDF_OUT)
    plt.close(fig)

    pd.DataFrame(qa_rows).to_csv(QA_OUT, index=False, encoding="utf-8-sig")
    MANIFEST_OUT.write_text(
        "\n".join(
            [
                "role,file_path,notes",
                f"north_source_script,{NORTH_SCRIPT},Imported and redrawn from source functions; no raster panel-label editing.",
                f"south_source_script,{SOUTH_SCRIPT},Imported and redrawn from source functions; no raster panel-label editing. South o/t fan insets are expanded to {SOUTH_FAN_INSET_RANGE_MAX_KM:.0f} km; south k/p maps retain the original v23 FIR fan coverage.",
                f"leoro_rolling_1hz_source,{ROLLING_POINTS_SOURCE},All LEO-RO C/N0 proxy residual panels c/h/m/r use delta_rolling_db from the 1 Hz C/N0 proxy plus centered rolling-median baseline diagnostic table; high-rate std and Chebyshev residuals are not plotted.",
                f"leoro_rolling_1hz_points_plotted,{ROLLING_POINTS_OUT},Filtered point table for the LEO-RO points plotted in panels c/h/m/r.",
                f"leoro_highrate_l1_source,{HIGHRATE_POINTS_SOURCE},High-rate signed L1 C/N0 proxy residuals used only as a gray background layer in panels c/h/m/r.",
                f"leoro_highrate_l1_points_plotted,{HIGHRATE_L1_POINTS_OUT},Filtered high-rate L1 point table plotted as gray background in panels c/h/m/r.",
                f"isr_time_coverage_audit,{ISR_AUDIT_OUT},Companion audit showing the nearest available local ISR profile times for the two north OP76 events.",
                f"yunyao_ionprf_iri_package,{YUNYAO_PROFILE_PACKAGE},Four YunYao ionPrf electron-density profiles and their matched-location IRI profiles for panels d/i/n/s and the YunYao RO map tracks in panels a/f/k/p.",
                f"yunyao_ionprf_iri_summary,{YUNYAO_SUMMARY_SOURCE},Track-level timing, distance, representative point and source-path summary for the four YunYao profiles.",
                f"pfisr_profile_source,{PFISR_PROFILE_SOURCE},QC-filtered PFISR vertical-beam event-window median Ne profiles retained as gray context only in north panels d/i; relative_dne<=1.0.",
                f"pfisr_profile_summary,{PFISR_MATCH_SUMMARY},Summary of PFISR event-window matching and altitude coverage.",
                f"combined_png,{PNG_OUT},Source-redrawn compact Fig.4 montage with lowercase labels, shared central map colorbars, 6500-km south fan insets, unified 1 Hz rolling-median LEO-RO C/N0 proxy residuals, lighter unlabeled high-rate L1 residual background, YunYao ionPrf plus matched IRI Ne profiles in d/i/n/s, and YunYao RO tracks overlaid in a/f/k/p. Ne-panel color convention: YunYao red, ISR blue, IRI black; the plotted IRI curves correspond to YunYao locations and are labelled IRI (YunYao).",
                f"combined_pdf,{PDF_OUT},PDF export of the source-redrawn compact Fig.4 montage.",
                f"qa_csv,{QA_OUT},Combined QA rows from map/electron-density panels.",
            ]
        ),
        encoding="utf-8",
    )
    README_OUT.write_text(
        "\n".join(
            [
                "# Fig.4 v37 with YunYao ionPrf Ne profiles and revised Ne colors",
                "",
                "This version keeps the accepted v25 combined north/south Fig.4 layout, maps, colorbars, and SuperDARN fan panels. Panels c/h/m/r use the same 1 Hz C/N0 proxy plus centered rolling-median baseline detrending method as v27, with an additional pale gray background layer showing high-rate L1 signed C/N0 proxy residuals. The high-rate background is intentionally not included in the legend.",
                "",
                "Compared with v36, the layout, data sources, map overlays, and all non-Ne panels are unchanged. The only visual change is the electron-density color/legend convention: YunYao RO profiles are red, ISR context is blue, and IRI is black. Because the plotted IRI curves are computed at the YunYao matched locations, the legend label is `IRI (YunYao)`. If a future version plots an ISR-site IRI curve, it should be labelled `IRI (ISR)`.",
                "",
                f"Panel identifiers are generated as lowercase text at draw time: north cases a-j and south cases k-t. Map titles are also lowercase in the source title dictionaries. Per-map inset colorbars are removed at draw time and replaced by one shared central map-colorbar stack. The four central map panels keep the accepted v23 positions and original k/p FIR fan coverage. The o/t south SuperDARN fan insets are expanded to {SOUTH_FAN_INSET_RANGE_MAX_KM:.0f} km so their displayed range is comparable to the long-range FIR cells visible on the south-polar maps.",
                "",
                "Main LEO-RO method: high-rate SNR is converted to a 1 Hz C/N0 proxy using the per-second median SNR, then a centered rolling-median baseline is subtracted. The plotted colored LEO-RO points are the signed residual `delta_rolling_db`. The high-rate gray background is `delta_highrate_db` for L1 only and is included as an inspection/context layer to reveal sub-second variability and possible product discontinuities. It should not be quoted as the quantitative event amplitude without additional step/jump screening.",
                "",
                "The high-rate per-second std diagnostic remains rejected for Fig.4 because it responds to step-like level jumps, especially in the PlanetiQ example. The Chebyshev residual is not plotted here.",
                "",
                "Panels d/i/n/s draw the new YunYao ionPrf profiles from `ro_ne_profiles_track009_017_145_187_ionprf_iri_package.zip` and the corresponding IRI profiles computed at each YunYao matched profile location. North panels d/i retain the PFISR vertical-beam profile as a thin gray context line because it is time-matched but spatially farther from the LuGRE/SuperDARN LYR cells. The YunYao profiles are closer to the LuGRE tangent paths and are therefore the primary Ne context in this version.",
                "",
                "YunYao profile matches: track 145 uses Y038/E05 (time gap 27.06 min, minimum distance 3.67 km); track 187 uses Y031/E10 (3.66 min, 0.48 km); track 009 uses Y040/C19 (6.34 min, 36.19 km); track 017 uses Y016/C39 (0.00 min, 3.26 km).",
                "",
                "No ROTI/ISMR/SuperDARN source, time window, C/N0 method, fan layout, or overall panel layout was intentionally changed relative to v35.",
                "",
                f"PNG: `{PNG_OUT}`",
                f"PDF: `{PDF_OUT}`",
                f"QA: `{QA_OUT}`",
                f"Manifest: `{MANIFEST_OUT}`",
                f"LEO-RO plotted points: `{ROLLING_POINTS_OUT}`",
                f"LEO-RO high-rate L1 background points: `{HIGHRATE_L1_POINTS_OUT}`",
                f"ISR time-coverage audit: `{ISR_AUDIT_OUT}`",
                f"YunYao ionPrf/IRI package: `{YUNYAO_PROFILE_PACKAGE}`",
                f"YunYao profile match summary: `{YUNYAO_SUMMARY_SOURCE}`",
                f"PFISR Ne profiles retained as context in d/i: `{PFISR_PROFILE_SOURCE}`",
                f"PFISR match summary: `{PFISR_MATCH_SUMMARY}`",
            ]
        ),
        encoding="utf-8",
    )

    print(PNG_OUT)
    print(PDF_OUT)
    print(QA_OUT)


if __name__ == "__main__":
    build()
