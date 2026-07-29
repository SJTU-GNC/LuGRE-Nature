from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TASK_DIR = Path(__file__).resolve().parent
REPRO_ROOT = TASK_DIR.parents[2]
DATA_DIR = REPRO_ROOT / "data" / "analysis_ready" / "ED1"
OUTDIR = REPRO_ROOT / "outputs" / "ED1"
OUTDIR.mkdir(parents=True, exist_ok=True)

MAIN = DATA_DIR / "Polar_scintillation_support_1min_merged_20250115_20250317.csv"
OMNI2 = DATA_DIR / "OMNI2_Dst_F107_20250115_20250317.csv"
WINDOW_METRICS = DATA_DIR / "fig2_window_metrics_with_baseline.csv"
OPTABLE_CANDIDATES = [DATA_DIR / "OPTABLE.csv"]

OUT_STEM = "Extended_Data_Fig_01_space_weather_lugre_context_surface_phase"

XLIM_START = pd.Timestamp("2025-03-03 00:00:00")
XLIM_END = pd.Timestamp("2025-03-17 00:00:00")
DATE_TICKS = [
    pd.Timestamp("2025-03-03"),
    pd.Timestamp("2025-03-05"),
    pd.Timestamp("2025-03-07"),
    pd.Timestamp("2025-03-09"),
    pd.Timestamp("2025-03-11"),
    pd.Timestamp("2025-03-13"),
    pd.Timestamp("2025-03-15"),
    pd.Timestamp("2025-03-17"),
]
FONT_SCALE = 1.5


COLORS = {
    "ink": (27, 28, 30),
    "muted": (98, 101, 106),
    "grid": (224, 226, 230),
    "panel_bg": (252, 252, 251),
    "span": (176, 58, 66),
    "bz": (39, 91, 154),
    "bz_fill": (202, 222, 244),
    "speed": (55, 126, 82),
    "pressure": (166, 75, 67),
    "f107": (142, 83, 118),
    "kp": (45, 48, 53),
    "pcn": (44, 119, 137),
    "ae": (222, 126, 30),
    "au": (0, 135, 98),
    "al": (106, 61, 154),
    "lugre_bar": (185, 69, 58),
    "lugre_point": (65, 65, 65),
    "all_ref": (158, 162, 168),
    "all_ref_light": (220, 222, 225),
}

PHASE_COLORS = {
    "C": (54, 104, 171),
    "T": (48, 139, 124),
    "L": (105, 83, 164),
    "S": (189, 72, 66),
}


@dataclass(frozen=True)
class Panel:
    key: str
    label: str
    ylabel: str
    ymin: float
    ymax: float
    ticks: tuple[float, ...]


PANELS = [
    Panel("f107", "a", "F10.7\n(sfu)", 130, 250, (150, 200, 250)),
    Panel("bz", "b", "IMF Bz\n(nT)", -15, 15, (-10, 0, 10)),
    Panel("kp", "c", "Kp", 0, 6.5, (0, 3, 5, 6)),
    Panel("pcn", "d", "PCN\n(mV/m)", -0.5, 7.5, (0, 3, 6)),
    Panel("auroral", "e", "AE/AU/AL\n(nT)", -1600, 1800, (-1000, 0, 1000)),
    Panel("lugre", "f", "Polar OP\nP90 of P95\n|δC/N0|\n(dB)", 0, 5.0, (0, 1.5, 3, 4.5)),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidate = Path(
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Required exact-rendering font is missing: {candidate}. "
            "Run verify_package.cmd for the full environment check."
        )
    return ImageFont.truetype(str(candidate), size=size)


def scaled_font(base_size: float, scale: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return font(int(round(base_size * scale * FONT_SCALE)), bold=bold)


def load_context() -> pd.DataFrame:
    cols = [
        "datetime_utc",
        "pcn_mV_m",
        "ae_nT",
        "au_nT",
        "al_nT",
        "bz_gsm_nT",
        "flow_speed_km_s",
        "pressure_nPa",
        "kp_3h",
    ]
    df = pd.read_csv(MAIN, usecols=cols)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert(None)
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("datetime_utc").sort_index()

    hourly = pd.DataFrame(index=df.resample("1h").mean().index)
    hourly["bz"] = df["bz_gsm_nT"].resample("1h").mean()
    hourly["speed"] = df["flow_speed_km_s"].resample("1h").mean()
    hourly["pressure"] = df["pressure_nPa"].resample("1h").mean()
    hourly["pcn"] = df["pcn_mV_m"].resample("1h").mean()
    hourly["ae"] = df["ae_nT"].resample("1h").max()
    hourly["au"] = df["au_nT"].resample("1h").max()
    hourly["al"] = df["al_nT"].resample("1h").min()
    hourly["kp"] = df["kp_3h"].resample("1h").mean()
    omni = pd.read_csv(OMNI2, parse_dates=["datetime_utc"])
    omni["datetime_utc"] = pd.to_datetime(omni["datetime_utc"], utc=True).dt.tz_convert(None)
    omni = omni.set_index("datetime_utc").sort_index()
    hourly = hourly.join(omni["f107_sfu"].rename("f107"), how="left")
    hourly["f107"] = hourly["f107"].ffill().bfill()
    return hourly


def find_optable() -> Path:
    for candidate in OPTABLE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No LuGRE OPTABLE.csv found in the configured candidate paths.")


def summarize_op_metrics(metrics: pd.DataFrame, prefix: str) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["phase", "op_id"], as_index=False)
        .agg(
            n_windows=("a95_db", "size"),
            p90_a95_db=("a95_db", lambda s: float(s.quantile(0.90))),
            max_a95_db=("a95_db", "max"),
            event_count=("is_event", "sum"),
            event_rate=("is_event", "mean"),
        )
    )
    return grouped.rename(
        columns={
            "n_windows": f"{prefix}_n_windows",
            "p90_a95_db": f"{prefix}_p90_a95_db",
            "max_a95_db": f"{prefix}_max_a95_db",
            "event_count": f"{prefix}_event_count",
            "event_rate": f"{prefix}_event_rate",
        }
    )


def load_op_cn0_stats() -> pd.DataFrame:
    metrics = pd.read_csv(WINDOW_METRICS)
    metrics["op_id"] = metrics["op_id"].astype(str).str.strip()
    metrics["phase"] = metrics["phase"].astype(str).str.strip()
    metrics["lat_group"] = metrics["lat_group"].astype(str).str.strip()
    metrics["a95_db"] = pd.to_numeric(metrics["a95_db"], errors="coerce")
    if pd.api.types.is_bool_dtype(metrics["is_event"]):
        metrics["is_event"] = metrics["is_event"].fillna(False)
    else:
        metrics["is_event"] = metrics["is_event"].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})

    all_grouped = summarize_op_metrics(metrics, "all")
    polar_grouped = summarize_op_metrics(metrics[metrics["lat_group"].str.lower().eq("polar")], "polar")
    grouped = all_grouped.merge(polar_grouped, on=["phase", "op_id"], how="left")

    optable = pd.read_csv(find_optable(), dtype={"OP ID": str}, low_memory=False)
    optable["op_id"] = optable["OP ID"].astype(str).str.strip()
    optable["opt_phase"] = optable["Phase"].astype(str).str.strip()
    optable["start"] = pd.to_datetime(optable["RTP_start (yyyy/mm/dd hh:mm:ss UTC)"], errors="coerce")
    optable["end"] = pd.to_datetime(optable["RTP_end (yyyy/mm/dd hh:mm:ss UTC)"], errors="coerce")
    optable["distance_re"] = pd.to_numeric(optable["Initial Altitude (RE)"], errors="coerce")

    keep = ["op_id", "opt_phase", "start", "end", "distance_re", "Mode"]
    out = grouped.merge(optable[keep], on="op_id", how="left")
    out["phase"] = out["phase"].where(out["phase"].notna(), out["opt_phase"])
    out = out.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)
    out["center"] = out["start"] + (out["end"] - out["start"]) / 2
    out["duration_days"] = (out["end"] - out["start"]).dt.total_seconds() / 86400.0
    out["has_polar_windows"] = out["polar_n_windows"].fillna(0).gt(0)
    return out


def clean_series(series: pd.Series, ymin: float, ymax: float) -> pd.Series:
    return series.clip(lower=ymin, upper=ymax)


def make_canvas(scale: int = 2) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    width, height = 1800 * scale, 1810 * scale
    img = Image.new("RGB", (width, height), (255, 255, 255))
    return img, ImageDraw.Draw(img, "RGBA"), scale


def x_map(ts: pd.Timestamp, t0: pd.Timestamp, t1: pd.Timestamp, left: int, right: int) -> float:
    frac = (ts - t0).total_seconds() / (t1 - t0).total_seconds()
    return left + frac * (right - left)


def y_map(value: float, ymin: float, ymax: float, top: int, bottom: int) -> float:
    frac = (value - ymin) / (ymax - ymin)
    return bottom - frac * (bottom - top)


def draw_text_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, font=fnt, fill=fill)


def draw_rotated_text_centered(
    img: Image.Image,
    xy: tuple[float, float],
    text: str,
    fnt,
    fill: tuple[int, int, int],
    angle: float = 60.0,
) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=fnt)
    tw = max(1, bbox[2] - bbox[0] + 10)
    th = max(1, bbox[3] - bbox[1] + 10)
    tile = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
    td = ImageDraw.Draw(tile)
    td.text((5 - bbox[0], 5 - bbox[1]), text, font=fnt, fill=fill + (255,))
    rotated = tile.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    img.paste(rotated, (int(xy[0] - rotated.width / 2), int(xy[1] - rotated.height / 2)), rotated)


def draw_multiline_right(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fnt, fill) -> None:
    lines = text.split("\n")
    line_h = fnt.size + 7
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        draw.text((x - (bbox[2] - bbox[0]), y + i * line_h), line, font=fnt, fill=fill)


def to_points(
    times: pd.Index,
    values: pd.Series,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    rect: tuple[int, int, int, int],
    ymin: float,
    ymax: float,
) -> list[tuple[float, float]]:
    left, top, right, bottom = rect
    pts: list[tuple[float, float]] = []
    for ts, value in zip(times, values):
        if pd.isna(value):
            continue
        pts.append((x_map(pd.Timestamp(ts), t0, t1, left, right), y_map(float(value), ymin, ymax, top, bottom)))
    return pts


def draw_polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color, width: int) -> None:
    if len(pts) < 2:
        return
    # Break visually long gaps caused by missing records.
    segment = [pts[0]]
    for prev, curr in zip(pts, pts[1:]):
        if curr[0] - prev[0] > 35:
            if len(segment) > 1:
                draw.line(segment, fill=color + (255,), width=width, joint="curve")
            segment = [curr]
        else:
            segment.append(curr)
    if len(segment) > 1:
        draw.line(segment, fill=color + (255,), width=width, joint="curve")


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float, float, float],
    color: tuple[int, int, int],
    width: int,
    dash: int,
    gap: int,
) -> None:
    x0, y0, x1, y1 = xy
    if abs(y1 - y0) < 1e-6:
        x = x0
        while x < x1:
            draw.line((x, y0, min(x + dash, x1), y1), fill=color + (255,), width=width)
            x += dash + gap
    elif abs(x1 - x0) < 1e-6:
        y = y0
        while y < y1:
            draw.line((x0, y, x1, min(y + dash, y1)), fill=color + (255,), width=width)
            y += dash + gap


def op_label_offsets(
    op_stats: pd.DataFrame,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    rect: tuple[int, int, int, int],
    scale: int,
) -> dict[int, tuple[float, float]]:
    positioned = []
    for idx, row in op_stats.iterrows():
        positioned.append((idx, x_map(row["center"], t0, t1, rect[0], rect[2])))
    positioned.sort(key=lambda item: item[1])

    clusters: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for item in positioned:
        if current and item[1] - current[-1][1] > 64 * scale:
            clusters.append(current)
            current = []
        current.append(item)
    if current:
        clusters.append(current)

    offsets: dict[int, tuple[float, float]] = {}
    y_pattern = [0, 7, -6, 5, -4, 3]
    for cluster in clusters:
        n = len(cluster)
        if n == 1:
            offsets[cluster[0][0]] = (0.0, 0.0)
            continue
        step = 18 * scale
        for i, (idx, _) in enumerate(cluster):
            dx = (i - (n - 1) / 2.0) * step
            dy = y_pattern[i % len(y_pattern)] * scale
            offsets[idx] = (dx, dy)
    return offsets


def kp_active_spans(kp: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    active = kp.dropna() >= 5.0
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    prev: pd.Timestamp | None = None
    for ts, is_active in active.items():
        ts = pd.Timestamp(ts)
        if is_active and start is None:
            start = ts
        if (not is_active) and start is not None and prev is not None:
            spans.append((start, prev + pd.Timedelta(hours=1)))
            start = None
        prev = ts
    if start is not None and prev is not None:
        spans.append((start, prev + pd.Timedelta(hours=1)))
    return spans


def draw_axes(
    draw: ImageDraw.ImageDraw,
    panel: Panel,
    rect: tuple[int, int, int, int],
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    shaded_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    panel_label_font,
    axis_font,
    small_font,
    scale: int,
) -> None:
    left, top, right, bottom = rect
    draw.rectangle(rect, fill=COLORS["panel_bg"] + (255,))

    # Pale red guides denote Kp >= 5 geomagnetic activity intervals.
    for start, end in shaded_spans:
        x0 = x_map(start, t0, t1, left, right)
        x1 = x_map(end, t0, t1, left, right)
        draw.rectangle((x0, top, x1, bottom), fill=COLORS["span"] + (24,))

    for tick in panel.ticks:
        y = y_map(tick, panel.ymin, panel.ymax, top, bottom)
        draw.line((left, y, right, y), fill=COLORS["grid"] + (255,), width=max(1, scale))
        draw.line((left - 8 * scale, y, left, y), fill=COLORS["ink"] + (255,), width=max(1, scale))
        draw.text((left - 16 * scale, y - 10 * scale), f"{tick:g}", font=axis_font, fill=COLORS["muted"], anchor="ra")

    # Date reference ticks.
    for dt in DATE_TICKS:
        x = x_map(dt, t0, t1, left, right)
        draw.line((x, top, x, bottom), fill=(238, 238, 238, 255), width=max(1, scale))
        draw.line((x, bottom, x, bottom + 8 * scale), fill=COLORS["ink"] + (255,), width=max(1, scale))
        if panel.key == "lugre":
            draw_text_centered(draw, (x, bottom + 82 * scale), dt.strftime("%b %d"), small_font, COLORS["muted"])

    draw.line((left, top, left, bottom), fill=COLORS["ink"] + (255,), width=max(2, scale))
    draw.line((left, bottom, right, bottom), fill=COLORS["ink"] + (255,), width=max(2, scale))
    draw.text((left - 122 * scale, top - 6 * scale), panel.label, font=panel_label_font, fill=COLORS["ink"])
    draw_multiline_right(draw, left - 72 * scale, top + 0.5 * (bottom - top) - 34 * scale, panel.ylabel, axis_font, COLORS["ink"])


def draw_legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, tuple[int, int, int]]], x: int, y: int, fnt, scale: int) -> None:
    cursor = x
    for label, color in items:
        draw.line((cursor, y + 8 * scale, cursor + 42 * scale, y + 8 * scale), fill=color + (255,), width=4 * scale)
        draw.text((cursor + 52 * scale, y - 1 * scale), label, font=fnt, fill=COLORS["muted"])
        cursor += (52 + 11 * len(label)) * scale


def draw_figure() -> Path:
    context = load_context()
    op_stats = load_op_cn0_stats()

    t0 = XLIM_START
    t1 = XLIM_END
    context = context.loc[(context.index >= t0) & (context.index <= t1)]
    op_stats = op_stats[(op_stats["center"] >= t0) & (op_stats["center"] <= t1)].copy()
    shaded_spans = kp_active_spans(context["kp"])

    img, draw, scale = make_canvas(scale=2)
    W, H = img.size

    axis_font = scaled_font(18, scale)
    small_font = scaled_font(16, scale)
    panel_label_font = scaled_font(23, scale, bold=True)

    left = 310 * scale
    right = W - 255 * scale
    panel_h = 205 * scale
    gap = 72 * scale
    top0 = 58 * scale
    rects = []
    for i in range(len(PANELS)):
        top = top0 + i * (panel_h + gap)
        rects.append((left, top, right, top + panel_h))

    for panel, rect in zip(PANELS, rects):
        draw_axes(draw, panel, rect, t0, t1, shaded_spans, panel_label_font, axis_font, small_font, scale)

    # Panel a: F10.7 solar radio flux, a proxy for solar EUV background ionization.
    rect = rects[0]
    f107 = clean_series(context["f107"], PANELS[0].ymin, PANELS[0].ymax)
    draw_polyline(draw, to_points(context.index, f107, t0, t1, rect, PANELS[0].ymin, PANELS[0].ymax), COLORS["f107"], 3 * scale)
    draw.text((rect[0] + 18 * scale, rect[1] + 18 * scale), "solar radio flux / EUV background proxy", font=small_font, fill=COLORS["f107"])

    # Panel b: IMF Bz. Fill southward values to show input favorable for reconnection.
    rect = rects[1]
    bz = clean_series(context["bz"], PANELS[1].ymin, PANELS[1].ymax)
    pts = to_points(context.index, bz, t0, t1, rect, PANELS[1].ymin, PANELS[1].ymax)
    zero_y = y_map(0, PANELS[1].ymin, PANELS[1].ymax, rect[1], rect[3])
    neg_poly = [(rect[0], zero_y)]
    for ts, val in zip(context.index, bz):
        if pd.isna(val) or val >= 0:
            continue
        neg_poly.append((x_map(pd.Timestamp(ts), t0, t1, rect[0], rect[2]), y_map(float(val), PANELS[1].ymin, PANELS[1].ymax, rect[1], rect[3])))
    neg_poly.append((rect[2], zero_y))
    if len(neg_poly) > 3:
        draw.polygon(neg_poly, fill=COLORS["bz_fill"] + (115,))
    draw.line((rect[0], zero_y, rect[2], zero_y), fill=(150, 150, 150, 255), width=scale)
    draw_polyline(draw, pts, COLORS["bz"], 3 * scale)
    draw.text((rect[2] - 246 * scale, rect[1] + 14 * scale), "southward IMF favors coupling", font=small_font, fill=COLORS["bz"])

    # Panel c: Kp global geomagnetic activity.
    rect = rects[2]
    kp = clean_series(context["kp"], PANELS[2].ymin, PANELS[2].ymax)
    draw_polyline(draw, to_points(context.index, kp, t0, t1, rect, PANELS[2].ymin, PANELS[2].ymax), COLORS["kp"], 3 * scale)
    kp5_y = y_map(5, PANELS[2].ymin, PANELS[2].ymax, rect[1], rect[3])
    draw_dashed_line(draw, (rect[0], kp5_y, rect[2], kp5_y), COLORS["span"], 2 * scale, 15 * scale, 10 * scale)
    draw.text((rect[2] - 120 * scale, kp5_y - 24 * scale), "Kp = 5", font=small_font, fill=COLORS["span"])

    # Panel d: PCN polar-cap convection proxy.
    rect = rects[3]
    pcn = clean_series(context["pcn"], PANELS[3].ymin, PANELS[3].ymax)
    pts = to_points(context.index, pcn, t0, t1, rect, PANELS[3].ymin, PANELS[3].ymax)
    draw_polyline(draw, pts, COLORS["pcn"], 3 * scale)
    draw.text((rect[0] + 18 * scale, rect[1] + 18 * scale), "polar-cap convection response", font=small_font, fill=COLORS["pcn"])

    # Panel e: auroral electrojet activity, retaining the familiar AE/AU/AL family.
    rect = rects[4]
    zero_y = y_map(0, PANELS[4].ymin, PANELS[4].ymax, rect[1], rect[3])
    draw.line((rect[0], zero_y, rect[2], zero_y), fill=(145, 145, 145, 255), width=scale)
    ae = clean_series(context["ae"], PANELS[4].ymin, PANELS[4].ymax)
    au = clean_series(context["au"], PANELS[4].ymin, PANELS[4].ymax)
    al = clean_series(context["al"], PANELS[4].ymin, PANELS[4].ymax)
    draw_polyline(draw, to_points(context.index, ae, t0, t1, rect, PANELS[4].ymin, PANELS[4].ymax), COLORS["ae"], 3 * scale)
    draw_polyline(draw, to_points(context.index, au, t0, t1, rect, PANELS[4].ymin, PANELS[4].ymax), COLORS["au"], 3 * scale)
    draw_polyline(draw, to_points(context.index, al, t0, t1, rect, PANELS[4].ymin, PANELS[4].ymax), COLORS["al"], 3 * scale)
    draw_legend(draw, [("AE", COLORS["ae"]), ("AU", COLORS["au"]), ("AL", COLORS["al"])], rect[0] + 18 * scale, rect[1] - 44 * scale, small_font, scale)

    # Panel f: OP-level LuGRE C/N0 fluctuation statistics, emphasizing polar-only windows.
    rect = rects[5]
    thr_y = y_map(1.5, PANELS[5].ymin, PANELS[5].ymax, rect[1], rect[3])
    draw_dashed_line(draw, (rect[0], thr_y, rect[2], thr_y), COLORS["span"], 2 * scale, 15 * scale, 10 * scale)
    ref_label_x = x_map(pd.Timestamp("2025-03-07 12:00:00"), t0, t1, rect[0], rect[2])
    draw.text((int(ref_label_x), thr_y - 24 * scale), "1.5 dB reference level", font=small_font, fill=COLORS["span"])

    draw_legend(
        draw,
        [("all-window reference", COLORS["all_ref"]), ("max P95 |δC/N0|", COLORS["lugre_point"])],
        rect[0] + 18 * scale,
        rect[1] - 48 * scale,
        small_font,
        scale,
    )

    label_offsets = op_label_offsets(op_stats, t0, t1, rect, scale)
    for idx, (_, row) in enumerate(op_stats.iterrows()):
        x_center = x_map(row["center"], t0, t1, rect[0], rect[2])
        phase_color = PHASE_COLORS.get(str(row["phase"]), COLORS["lugre_bar"])
        y0 = y_map(0, PANELS[5].ymin, PANELS[5].ymax, rect[1], rect[3])
        y_all = y_map(row["all_p90_a95_db"], PANELS[5].ymin, PANELS[5].ymax, rect[1], rect[3])
        all_radius = (4.2 + np.sqrt(min(float(row["all_n_windows"]), 180.0)) * 0.26) * scale
        draw.line((x_center, y0, x_center, y_all), fill=COLORS["all_ref"] + (155,), width=6 * scale)
        draw.ellipse(
            (x_center - all_radius, y_all - all_radius, x_center + all_radius, y_all + all_radius),
            fill=COLORS["all_ref_light"] + (170,),
            outline=COLORS["all_ref"] + (210,),
            width=max(1, scale),
        )
        y_all_max = y_map(row["all_max_a95_db"], 0, 10, rect[1], rect[3])
        r_all = 4 * scale
        draw.line((x_center - r_all, y_all_max - r_all, x_center + r_all, y_all_max + r_all), fill=COLORS["all_ref"] + (185,), width=2 * scale)
        draw.line((x_center - r_all, y_all_max + r_all, x_center + r_all, y_all_max - r_all), fill=COLORS["all_ref"] + (185,), width=2 * scale)

        has_polar = bool(row["has_polar_windows"])
        if has_polar:
            y_p90 = y_map(row["polar_p90_a95_db"], PANELS[5].ymin, PANELS[5].ymax, rect[1], rect[3])
            draw.line((x_center, y0, x_center, y_p90), fill=phase_color + (255,), width=4 * scale)
            polar_n = float(row["polar_n_windows"])
            event_rate = float(row["polar_event_rate"])
            radius = (5.2 + np.sqrt(min(polar_n, 180.0)) * 0.42) * scale
            outline_w = max(2 * scale, int((2.0 + 3.5 * event_rate) * scale))
            draw.ellipse(
                (x_center - radius, y_p90 - radius, x_center + radius, y_p90 + radius),
                fill=phase_color + (235,),
                outline=phase_color + (255,),
                width=outline_w,
            )
            y_max = y_map(row["polar_max_a95_db"], 0, 10, rect[1], rect[3])
            r = 5 * scale
            draw.line((x_center - r, y_max - r, x_center + r, y_max + r), fill=(255, 255, 255, 230), width=5 * scale)
            draw.line((x_center - r, y_max + r, x_center + r, y_max - r), fill=(255, 255, 255, 230), width=5 * scale)
            draw.line((x_center - r, y_max - r, x_center + r, y_max + r), fill=phase_color + (255,), width=2 * scale)
            draw.line((x_center - r, y_max + r, x_center + r, y_max - r), fill=phase_color + (255,), width=2 * scale)

        op_label = f"OP{str(row['op_id']).split('_')[0]}"
        dx, dy = label_offsets.get(row.name, (0.0, 0.0))
        label_color = phase_color if has_polar else COLORS["muted"]
        draw_rotated_text_centered(img, (x_center + dx, rect[3] + 34 * scale + dy), op_label, small_font, label_color, angle=60)

    for tick in (0, 5, 10):
        y = y_map(tick, 0, 10, rect[1], rect[3])
        draw.line((rect[2], y, rect[2] + 8 * scale, y), fill=COLORS["lugre_point"] + (255,), width=max(1, scale))
        draw.text((rect[2] + 16 * scale, y - 10 * scale), f"{tick:g}", font=axis_font, fill=COLORS["lugre_point"])
    draw_multiline_right(draw, rect[2] + 166 * scale, rect[1] + 50 * scale, "max\nP95\n|δC/N0|\n(dB)", axis_font, COLORS["lugre_point"])

    img_small = img.resize((W // scale, H // scale), Image.Resampling.LANCZOS)
    out_png = OUTDIR / f"{OUT_STEM}.png"
    img_small.save(out_png, dpi=(300, 300))
    return out_png


if __name__ == "__main__":
    output = draw_figure()
    print(output)
