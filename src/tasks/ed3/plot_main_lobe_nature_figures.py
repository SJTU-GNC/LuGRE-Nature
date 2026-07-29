from __future__ import annotations

from pathlib import Path
import csv
import math
import urllib.request

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageFilter
except Exception:  # pragma: no cover - fallback handled at draw time.
    Image = None
    ImageFilter = None


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[2]
OUT_DIR = PACKAGE_ROOT / "work" / "incomplete" / "ED3"
FIG_DIR = OUT_DIR / "main_lobe_figures"
POINT_CSV = OUT_DIR / "lugre_gnss_main_lobe_point_check.csv"
GPS_III_IMAGE_URL = "https://archive.gps.gov/multimedia/images/GPS-III-A.jpg"
GPS_III_RAW = FIG_DIR / "gps_iii_public_domain_source.jpg"
GPS_III_CUTOUT = FIG_DIR / "gps_iii_satellite_cutout.png"
USER_EARTH_IMAGE = PACKAGE_ROOT / "assets" / "ED3" / "earth_user_reference_globe.png"
EARTH_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Blue_Marble_2002.png/1280px-Blue_Marble_2002.png"
EARTH_RAW = FIG_DIR / "earth_blue_marble_2002_map.png"
EARTH_CUTOUT = FIG_DIR / "earth_user_reference_globe.png"

COLORS = {
    "GPS": "#2E86DE",
    "Galileo": "#E69F00",
    "GPS L1": "#2E86DE",
    "GPS L5": "#8CC5FF",
    "Galileo E1": "#E69F00",
    "Galileo E5a": "#F2C46D",
    "main": "#5DA5DA",
    "side": "#8E6BBE",
    "observed": "#2CA25F",
    "ink": "#1B2632",
    "muted": "#6B7280",
    "grid": "#E6ECF2",
}

SIGNAL_ORDER = ["GPS L1", "GPS L5", "Galileo E1", "Galileo E5a"]
SIGNAL_LIMITS = {"GPS L1": 23.5, "GPS L5": 26.0, "Galileo E1": 20.5, "Galileo E5a": 23.5}
SIGNAL_SHORT_LABELS = {
    "GPS L1": "GPS L1 C/A",
    "GPS L5": "GPS L5Q",
    "Galileo E1": "Gal E1C",
    "Galileo E5a": "Gal E5a-Q",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_points() -> pd.DataFrame:
    rows = []
    with POINT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signal = row.get("signal_name", "")
            if not signal.startswith(("GPS", "Galileo")):
                continue
            in_main_lobe = str(row.get("in_main_lobe", "True")).strip().lower() == "true"
            if not in_main_lobe:
                continue
            try:
                angle = float(row["tx_off_boresight_deg"])
                half_angle = float(row["main_lobe_half_angle_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((signal, angle, half_angle, in_main_lobe))
    df = pd.DataFrame(rows, columns=["signal_name", "tx_off_boresight_deg", "main_lobe_half_angle_deg", "in_main_lobe"])
    df["constellation"] = df["signal_name"].str.extract(r"^(GPS|Galileo)")
    df["signal_name"] = pd.Categorical(df["signal_name"], categories=SIGNAL_ORDER, ordered=True)
    return df.dropna(subset=["constellation", "signal_name", "tx_off_boresight_deg"])


def prepare_satellite_cutout() -> Path | None:
    if GPS_III_CUTOUT.exists():
        try:
            existing = Image.open(GPS_III_CUTOUT).convert("RGBA") if Image is not None else None
            if existing is not None and existing.getbbox() is not None:
                return GPS_III_CUTOUT
        except Exception:
            pass
    if Image is None:
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not GPS_III_RAW.exists():
        urllib.request.urlretrieve(GPS_III_IMAGE_URL, GPS_III_RAW)
    img = Image.open(GPS_III_RAW).convert("RGBA")
    # Crop away the Earth limb and keep the GPS III spacecraft with solar panels.
    crop = img.crop((250, 390, 3070, 2260))
    arr = np.array(crop)
    rgb = arr[:, :, :3].astype(float)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    alpha = np.where(luminance > 24, 255, 0).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=0.65))
    crop.putalpha(alpha_img)
    bbox = crop.getbbox()
    if bbox:
        crop = crop.crop(bbox)
    crop.thumbnail((420, 300), Image.Resampling.LANCZOS)
    crop.save(GPS_III_CUTOUT)
    return GPS_III_CUTOUT


def prepare_earth_cutout() -> Path | None:
    if Image is None:
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if USER_EARTH_IMAGE.exists():
        img = Image.open(USER_EARTH_IMAGE).convert("RGBA")
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        arr = np.array(img)
        alpha = arr[:, :, 3]
        edge = alpha < 80
        arr[edge, :3] = np.array([55, 145, 220], dtype=np.uint8)
        arr[alpha < 24, 3] = 0
        img = Image.fromarray(arr, mode="RGBA")
        img.thumbnail((620, 620), Image.Resampling.LANCZOS)
        img.save(EARTH_CUTOUT)
        return EARTH_CUTOUT

    if EARTH_CUTOUT.exists():
        try:
            existing = Image.open(EARTH_CUTOUT).convert("RGBA")
            if existing is not None and existing.getbbox() is not None:
                return EARTH_CUTOUT
        except Exception:
            pass
    if not EARTH_RAW.exists():
        request = urllib.request.Request(
            EARTH_IMAGE_URL,
            headers={"User-Agent": "Mozilla/5.0 Codex figure generation"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            EARTH_RAW.write_bytes(response.read())

    src = Image.open(EARTH_RAW).convert("RGB")
    src_arr = np.asarray(src).astype(float)
    src_h, src_w = src_arr.shape[:2]
    size = 520
    radius = size * 0.465
    center = (size - 1) / 2.0
    lon0 = math.radians(25.0)
    lat0 = math.radians(-10.0)
    sin_lat0, cos_lat0 = math.sin(lat0), math.cos(lat0)

    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx - center) / radius
    y = -(yy - center) / radius
    rho2 = x * x + y * y
    visible = rho2 <= 1.0
    z = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))

    lat = np.arcsin(np.clip(y * cos_lat0 + z * sin_lat0, -1.0, 1.0))
    lon = lon0 + np.arctan2(x, z * cos_lat0 - y * sin_lat0)
    u = ((lon + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi) * (src_w - 1)
    v = (math.pi / 2.0 - lat) / math.pi * (src_h - 1)

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = (u0 + 1) % src_w
    v1 = np.clip(v0 + 1, 0, src_h - 1)
    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]
    top = src_arr[v0, u0] * (1.0 - du) + src_arr[v0, u1] * du
    bottom = src_arr[v1, u0] * (1.0 - du) + src_arr[v1, u1] * du
    globe = top * (1.0 - dv) + bottom * dv

    limb = np.sqrt(np.clip(rho2, 0.0, 1.0))
    shade = 0.70 + 0.30 * z
    blue_rim = np.clip((limb - 0.82) / 0.18, 0.0, 1.0)[..., None]
    globe = globe * shade[..., None]
    globe = globe * (1.0 - 0.22 * blue_rim) + np.array([118.0, 185.0, 230.0]) * 0.22 * blue_rim

    alpha = np.zeros((size, size), dtype=np.uint8)
    alpha[visible] = 255
    out = np.dstack([np.clip(globe, 0, 255).astype(np.uint8), alpha])
    out[~visible, :3] = 0
    Image.fromarray(out, mode="RGBA").save(EARTH_CUTOUT)
    return EARTH_CUTOUT


def smoothed_histogram(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sigma = 1.45
    radius = int(math.ceil(sigma * 4))
    idx = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (idx / sigma) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    return centers, smooth


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        FIG_DIR / f"{stem}.png",
        FIG_DIR / f"{stem}.pdf",
        FIG_DIR / f"{stem}.svg",
    ]
    for path in paths:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if path.suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
    return paths


def draw_satellite(ax: plt.Axes, xy: tuple[float, float], compact: bool = False) -> None:
    cutout = prepare_satellite_cutout()
    if cutout and cutout.exists():
        arr = plt.imread(cutout)
        image = OffsetImage(arr, zoom=0.135 if compact else 0.24)
        box = AnnotationBbox(image, xy, frameon=False, pad=0.0, zorder=8)
        ax.add_artist(box)
        return
    # Fallback: simple GNSS-like bus and solar panels.
    x, y = xy
    panel_color = "#5B83C7"
    ax.add_patch(patches.Rectangle((x - 0.42, y - 0.09), 0.30, 0.18, angle=16, facecolor=panel_color, edgecolor="white", linewidth=0.4, zorder=8))
    ax.add_patch(patches.Rectangle((x + 0.12, y - 0.09), 0.30, 0.18, angle=16, facecolor=panel_color, edgecolor="white", linewidth=0.4, zorder=8))
    ax.add_patch(patches.Rectangle((x - 0.08, y - 0.10), 0.16, 0.20, angle=16, facecolor="#B7A78B", edgecolor=COLORS["ink"], linewidth=0.4, zorder=9))


def draw_earth(ax: plt.Axes, xy: tuple[float, float], compact: bool = False) -> None:
    cutout = prepare_earth_cutout()
    if cutout and cutout.exists():
        arr = plt.imread(cutout)
        image = OffsetImage(arr, zoom=0.175 if compact else 0.16)
        box = AnnotationBbox(image, xy, frameon=False, pad=0.0, zorder=7)
        ax.add_artist(box)
        return
    ax.add_patch(patches.Circle(xy, 0.55 if compact else 0.62, facecolor="#D8ECFF", edgecolor="#356AA0", linewidth=0.8, zorder=7))


def draw_earth_scaled(ax: plt.Axes, xy: tuple[float, float], radius: float) -> None:
    cutout = prepare_earth_cutout()
    if cutout and cutout.exists():
        arr = plt.imread(cutout)
        x, y = xy
        ax.imshow(
            arr,
            extent=(x - radius, x + radius, y - radius, y + radius),
            interpolation="lanczos",
            zorder=7,
        )
        return
    ax.add_patch(patches.Circle(xy, radius, facecolor="#D8ECFF", edgecolor="#356AA0", linewidth=0.8, zorder=7))


def make_angle_statistics_figure(df: pd.DataFrame) -> list[Path]:
    groups = SIGNAL_ORDER
    x_min, x_max = 11.2, 16.9
    bins = np.linspace(x_min, x_max, 70)

    fig = plt.figure(figsize=(7.2, 4.55))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.85, 1.45], hspace=0.07)
    ax = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[1, 0], sharex=ax)

    max_y = 0.0
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group in groups:
        values = df.loc[df["signal_name"] == group, "tx_off_boresight_deg"].to_numpy()
        x, y = smoothed_histogram(values, bins)
        curves[group] = (values, x, y)
        max_y = max(max_y, float(y.max()))

    for group in groups:
        values, x, y = curves[group]
        color = COLORS[group]
        ax.fill_between(x, y, 0, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2.1, label=f"{SIGNAL_SHORT_LABELS[group]} (n={len(values):,})")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_y * 1.22)
    ax.set_ylabel("Probability density")
    ax.tick_params(labelbottom=False)
    despine(ax)
    ax.legend(loc="upper left", ncol=2, handlelength=2.8, columnspacing=1.4)
    ax.text(
        0.01,
        1.03,
        "a",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0.99,
        0.94,
        "Side-lobe receptions excluded before statistics\n"
        "5-95% intervals describe main-lobe points only",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=COLORS["muted"],
    )

    y_positions = {group: float(len(groups) - idx - 1) for idx, group in enumerate(groups)}
    for group in groups:
        values = curves[group][0]
        q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        y = y_positions[group]
        color = COLORS[group]
        ax_box.plot([q[0], q[6]], [y, y], color=color, linewidth=1.1, alpha=0.75)
        ax_box.plot([q[1], q[5]], [y, y], color=color, linewidth=5.8, alpha=0.24, solid_capstyle="round")
        rect = patches.Rectangle(
            (q[2], y - 0.15),
            q[4] - q[2],
            0.30,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.20,
        )
        ax_box.add_patch(rect)
        ax_box.plot([q[3], q[3]], [y - 0.20, y + 0.20], color=color, linewidth=2.0)
        ax_box.scatter([q[0], q[6]], [y, y], s=12, color=color, zorder=3)
        ax_box.text(
            x_max - 0.02,
            y + 0.23,
            f"median {q[3]:.2f}; range {q[0]:.2f}-{q[6]:.2f} deg",
            ha="right",
            va="center",
            fontsize=7.3,
            color=COLORS["muted"],
        )

    ax_box.set_ylim(-0.55, len(groups) - 0.45)
    ax_box.set_yticks([y_positions[group] for group in groups])
    ax_box.set_yticklabels([f"{SIGNAL_SHORT_LABELS[group]} ({SIGNAL_LIMITS[group]:g} deg)" for group in groups])
    ax_box.set_xlabel(r"Transmitter off-boresight angle, $\alpha$ (deg)")
    ax_box.grid(True, axis="x", color=COLORS["grid"], linewidth=0.6)
    ax_box.grid(False, axis="y")
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)
    ax_box.text(
        0.01,
        1.06,
        "b",
        transform=ax_box.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )
    fig.text(
        0.015,
        0.985,
        "LuGRE GNSS transmitter off-boresight angle statistics",
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=COLORS["ink"],
    )
    paths = save_figure(fig, "figure1_gps_galileo_off_boresight_statistics_nature")
    plt.close(fig)
    return paths


def polar_xy(origin: tuple[float, float], length: float, angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    return origin[0] + length * math.cos(angle), origin[1] + length * math.sin(angle)


def make_lobe_schematic_figure(df: pd.DataFrame) -> list[Path]:
    observed_min = float(df["tx_off_boresight_deg"].min())
    observed_max = float(df["tx_off_boresight_deg"].max())

    fig, (ax_geo, ax_gain) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.0),
        gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.22},
    )

    # Geometry schematic
    ax_geo.set_aspect("equal")
    ax_geo.set_xlim(-0.45, 6.9)
    ax_geo.set_ylim(-2.75, 2.75)
    ax_geo.axis("off")
    sat = (0.0, 0.0)
    earth = (5.95, 0.0)
    radius = 5.2

    main = patches.Wedge(sat, radius, -26, 26, facecolor=COLORS["main"], alpha=0.16, edgecolor="none")
    obs_upper = patches.Wedge(sat, radius * 0.98, observed_min, observed_max, facecolor=COLORS["observed"], alpha=0.26, edgecolor="none")
    obs_lower = patches.Wedge(sat, radius * 0.98, -observed_max, -observed_min, facecolor=COLORS["observed"], alpha=0.20, edgecolor="none")
    ax_geo.add_patch(main)
    ax_geo.add_patch(obs_upper)
    ax_geo.add_patch(obs_lower)

    for angle, color, lw, ls in [
        (26, COLORS["main"], 1.2, "-"),
        (-26, COLORS["main"], 1.2, "-"),
        (20.5, "#4B6CB7", 0.9, (0, (3.5, 3.0))),
        (-20.5, "#4B6CB7", 0.9, (0, (3.5, 3.0))),
        (observed_min, COLORS["observed"], 1.1, "-"),
        (observed_max, COLORS["observed"], 1.1, "-"),
    ]:
        end = polar_xy(sat, radius, angle)
        ax_geo.plot([sat[0], end[0]], [sat[1], end[1]], color=color, linewidth=lw, linestyle=ls)

    for angle, length, width, alpha in [
        (39, 2.9, 0.50, 0.25),
        (-39, 2.9, 0.50, 0.25),
        (58, 2.35, 0.36, 0.18),
        (-58, 2.35, 0.36, 0.18),
        (75, 1.75, 0.25, 0.14),
        (-75, 1.75, 0.25, 0.14),
    ]:
        cx, cy = polar_xy(sat, length, angle)
        ax_geo.add_patch(
            patches.Ellipse(
                (cx, cy),
                width=width,
                height=0.22,
                angle=angle,
                facecolor=COLORS["side"],
                edgecolor="none",
                alpha=alpha,
            )
        )

    ax_geo.annotate("", xy=(earth[0] - 0.48, earth[1]), xytext=(sat[0], sat[1]), arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": COLORS["ink"]})
    ax_geo.add_patch(patches.Circle(earth, 0.43, facecolor="#D8ECFF", edgecolor="#356AA0", linewidth=0.8))
    ax_geo.add_patch(patches.Circle(sat, 0.15, facecolor=COLORS["ink"], edgecolor="none"))
    lugre_point = polar_xy(sat, 4.85, 14.2)
    ax_geo.scatter([lugre_point[0]], [lugre_point[1]], s=24, facecolor=COLORS["observed"], edgecolor="white", linewidth=0.6, zorder=5)
    ax_geo.text(sat[0], sat[1] - 0.38, "GNSS SV", ha="center", va="top", fontsize=8, fontweight="bold")
    ax_geo.text(earth[0], earth[1], "Earth", ha="center", va="center", fontsize=7.5, color="#1E3A5F", fontweight="bold")
    ax_geo.text(2.65, -0.24, "boresight / nadir", ha="center", va="top", fontsize=7, color=COLORS["muted"])
    ax_geo.text(2.35, 1.05, "main lobe", ha="center", va="center", fontsize=8, color="#285B8F", fontweight="bold")
    ax_geo.text(2.18, 2.02, "E1 limit 20.5 deg", fontsize=6.8, color="#4B6CB7")
    ax_geo.text(2.65, 2.22, "widest limit 26 deg", fontsize=6.8, color="#285B8F")
    ax_geo.text(3.72, 1.36, f"LuGRE observed\n{observed_min:.1f}-{observed_max:.1f} deg", fontsize=6.9, color=COLORS["observed"], fontweight="bold", ha="left")
    ax_geo.text(0.76, -2.08, "side lobes beyond the main-lobe cut-off", fontsize=7.2, color="#6D4FA1")
    ax_geo.text(-0.10, 2.55, "a", fontsize=11, fontweight="bold")

    # Idealized antenna pattern
    angles = np.linspace(0, 80, 700)
    gain = np.where(
        angles <= 26,
        0.93 * np.exp(-(angles / 17.2) ** 2) + 0.055,
        0.19 * (np.sin((angles - 26) / 6.2) ** 2) * np.exp(-(angles - 26) / 45) + 0.035,
    )
    ax_gain.axvspan(0, 26, color=COLORS["main"], alpha=0.13, linewidth=0)
    ax_gain.axvspan(26, 80, color=COLORS["side"], alpha=0.10, linewidth=0)
    ax_gain.axvspan(observed_min, observed_max, color=COLORS["observed"], alpha=0.20, linewidth=0)
    ax_gain.plot(angles, gain, color=COLORS["ink"], linewidth=1.7)
    for angle, color in [
        (20.5, "#4B6CB7"),
        (23.5, "#B91C1C"),
        (26.0, "#B91C1C"),
    ]:
        ax_gain.axvline(angle, color=color, linewidth=0.9, linestyle=(0, (3, 3)))
    ax_gain.text(5, 0.94, "main lobe", fontsize=8, color="#285B8F", fontweight="bold")
    ax_gain.text(43, 0.24, "side lobes", fontsize=8, color="#6D4FA1", fontweight="bold")
    ax_gain.annotate(
        f"LuGRE observed\n{observed_min:.1f}-{observed_max:.1f} deg",
        xy=((observed_min + observed_max) / 2, 0.84),
        xytext=(34, 0.86),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": COLORS["observed"]},
        fontsize=7.2,
        color=COLORS["observed"],
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax_gain.plot([49, 55], [0.99, 0.99], color="#4B6CB7", linewidth=0.9, linestyle=(0, (3, 3)), clip_on=False)
    ax_gain.text(56.5, 0.99, "20.5 deg limit", fontsize=6.7, color="#4B6CB7", va="center")
    ax_gain.plot([49, 55], [0.93, 0.93], color="#B91C1C", linewidth=0.9, linestyle=(0, (3, 3)), clip_on=False)
    ax_gain.text(56.5, 0.93, "23.5 / 26 deg limits", fontsize=6.7, color="#B91C1C", va="center")
    ax_gain.set_xlim(0, 80)
    ax_gain.set_ylim(0, 1.08)
    ax_gain.set_xlabel(r"Off-boresight angle, $\alpha$ (deg)")
    ax_gain.set_ylabel("Relative transmit gain (a.u.)")
    ax_gain.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    despine(ax_gain)
    ax_gain.grid(True, axis="both", color=COLORS["grid"], linewidth=0.6)
    ax_gain.text(-0.16, 1.03, "b", transform=ax_gain.transAxes, fontsize=11, fontweight="bold")

    fig.text(
        0.015,
        0.985,
        "GNSS transmitter main-lobe and side-lobe geometry for LuGRE",
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=COLORS["ink"],
    )
    paths = save_figure(fig, "figure2_gnss_main_sidelobe_schematic_nature")
    plt.close(fig)
    return paths


def draw_gain_curve(ax: plt.Axes, observed_min: float, observed_max: float, compact: bool = False) -> None:
    angles = np.linspace(0, 80, 700)
    main_gain = 0.98 * np.exp(-(angles / 15.6) ** 2)
    side_gain = (
        0.16 * np.exp(-((angles - 35.5) / 5.8) ** 2)
        + 0.115 * np.exp(-((angles - 55.0) / 6.8) ** 2)
        + 0.085 * np.exp(-((angles - 74.0) / 7.2) ** 2)
        + 0.030 * np.exp(-(angles - 26).clip(min=0) / 42.0)
    )
    side_weight = 1.0 / (1.0 + np.exp(-(angles - 25.0) / 2.2))
    gain = main_gain + side_weight * side_gain
    gain /= gain.max()
    ax.axvspan(0, 26, color=COLORS["main"], alpha=0.13, linewidth=0)
    ax.axvspan(26, 80, color=COLORS["side"], alpha=0.10, linewidth=0)
    ax.axvspan(observed_min, observed_max, color=COLORS["observed"], alpha=0.20, linewidth=0)
    ax.plot(angles, gain, color=COLORS["ink"], linewidth=1.45 if compact else 1.7)
    for angle, color in [(20.5, "#4B6CB7"), (23.5, "#B91C1C"), (26.0, "#B91C1C")]:
        ax.axvline(angle, color=color, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1.12 if compact else 1.08)
    ax.set_xlabel(r"$\alpha$ (deg)", labelpad=1 if compact else 3)
    ax.set_ylabel("gain (a.u.)" if compact else "Relative transmit gain (a.u.)", labelpad=1 if compact else 3)
    if compact:
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=6.4, pad=1.5)
        ax.text(4, 0.94, "main lobe", fontsize=6.5, color="#285B8F", fontweight="bold")
        ax.text(43, 0.23, "side lobe", fontsize=6.5, color="#6D4FA1", fontweight="bold")
        lugre_center = 0.5 * (observed_min + observed_max)
        label_specs = [
            (lugre_center, 1.02, f"LuGRE ({observed_min:.1f}-{observed_max:.1f}$^\\circ$)", COLORS["observed"]),
            (20.5, 0.91, "Gal E1C (20.5$^\\circ$)", "#4B6CB7"),
            (23.5, 0.80, "GPS L1 C/A / Gal E5a-Q (23.5$^\\circ$)", "#B91C1C"),
            (26.0, 0.69, "GPS L5Q (26$^\\circ$)", "#B91C1C"),
        ]
        for x0, y0, text, color in label_specs:
            ax.annotate(
                text,
                xy=(x0, y0),
                xytext=(38.0, y0),
                color=color,
                fontsize=5.9,
                va="center",
                arrowprops={"arrowstyle": "->", "lw": 0.7, "color": color, "shrinkA": 2, "shrinkB": 1},
            )
    else:
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.text(5, 0.94, "main lobe", fontsize=8, color="#285B8F", fontweight="bold")
        ax.text(43, 0.24, "side lobes", fontsize=8, color="#6D4FA1", fontweight="bold")
    despine(ax)
    ax.grid(True, axis="both", color=COLORS["grid"], linewidth=0.55)


def draw_signal_selection_panel(ax: plt.Axes, df: pd.DataFrame, observed_min: float, observed_max: float) -> None:
    x_max = 30.0
    row_height = 0.62
    y_positions = {group: float(len(SIGNAL_ORDER) - idx - 1) for idx, group in enumerate(SIGNAL_ORDER)}

    ax.set_xlim(0, x_max)
    ax.set_ylim(-1.02, len(SIGNAL_ORDER) - 0.02)

    for row_idx, group in enumerate(SIGNAL_ORDER):
        y = y_positions[group]
        limit = SIGNAL_LIMITS[group]
        color = COLORS[group]
        values = np.sort(df.loc[df["signal_name"] == group, "tx_off_boresight_deg"].to_numpy())
        if len(values) == 0:
            continue
        q = np.quantile(values, [0.0, 0.05, 0.50, 0.95, 1.0])

        # Classification domains: statistics are calculated only from the retained domain.
        ax.add_patch(
            patches.Rectangle((0, y - row_height / 2), limit, row_height, facecolor="#DCEEFF", edgecolor="none", alpha=0.70, zorder=0)
        )
        ax.add_patch(
            patches.Rectangle((limit, y - row_height / 2), x_max - limit, row_height, facecolor="#F1E8FB", edgecolor="#B99AD9", linewidth=0.18, hatch="///", alpha=0.44, zorder=0)
        )
        ax.plot([limit, limit], [y - row_height / 2, y + row_height / 2], color="#B91C1C" if limit >= 23.5 else "#4B6CB7", linewidth=1.0, linestyle=(0, (3.0, 2.4)), zorder=2)

        # A deterministic sparse rug makes the retained LuGRE population visible without plotting every point.
        if len(values) > 180:
            sample_idx = np.linspace(0, len(values) - 1, 180).astype(int)
            sample = values[sample_idx]
        else:
            sample = values
        jitter = ((np.arange(len(sample)) % 9) - 4) * 0.010
        ax.scatter(sample, np.full_like(sample, y - 0.16) + jitter, s=3.5, facecolor=color, edgecolor="none", alpha=0.20, zorder=3)

        ax.plot([q[0], q[4]], [y + 0.06, y + 0.06], color=color, linewidth=1.15, alpha=0.74, solid_capstyle="round", zorder=4)
        ax.plot([q[1], q[3]], [y + 0.06, y + 0.06], color=color, linewidth=5.4, alpha=0.70, solid_capstyle="round", zorder=5)
        ax.scatter([q[2]], [y + 0.06], s=24, facecolor="white", edgecolor=color, linewidth=1.15, zorder=6)

    ax.axvline(13.8, color="#7D8793", linewidth=0.80, linestyle=(0, (2.6, 2.4)), zorder=2)
    ax.text(13.8 + 0.25, len(SIGNAL_ORDER) - 0.18, "Earth-limb 13.8 deg", fontsize=5.3, color="#7D8793", va="top")

    legend_y = -0.68
    ax.add_patch(patches.Rectangle((0.9, legend_y - 0.08), 0.58, 0.16, facecolor="#DCEEFF", edgecolor="none", alpha=0.85, zorder=7))
    ax.text(1.65, legend_y, "main lobe retained", fontsize=5.35, color="#285B8F", va="center", ha="left", zorder=8)
    ax.add_patch(patches.Rectangle((10.2, legend_y - 0.08), 0.58, 0.16, facecolor="#F1E8FB", edgecolor="#B99AD9", linewidth=0.2, hatch="///", alpha=0.62, zorder=7))
    ax.text(10.95, legend_y, "side lobe excluded", fontsize=5.35, color="#6D4FA1", va="center", ha="left", zorder=8)
    x0 = 20.2
    ax.plot([x0, x0 + 1.15], [legend_y, legend_y], color=COLORS["GPS L1"], linewidth=5.0, alpha=0.70, solid_capstyle="round", zorder=8)
    ax.scatter([x0 + 0.58], [legend_y], s=22, facecolor="white", edgecolor=COLORS["GPS L1"], linewidth=1.0, zorder=9)
    ax.text(x0 + 1.36, legend_y, "5-95% + median", fontsize=5.35, color=COLORS["ink"], va="center", ha="left", zorder=8)

    ax.set_xlabel(r"Transmitter off-boresight angle, $\alpha$ (deg)", labelpad=2)
    ax.set_yticks([y_positions[group] for group in SIGNAL_ORDER])
    ax.set_yticklabels([f"{SIGNAL_SHORT_LABELS[group]}\nlimit {SIGNAL_LIMITS[group]:g} deg" for group in SIGNAL_ORDER], fontsize=5.45)
    ax.tick_params(axis="y", pad=3, length=0)
    ax.set_xticks([0, 10, 13.8, 20, 26, 30])
    ax.set_xticklabels(["0", "10", "13.8", "20", "26", "30"], fontsize=5.85)
    ax.tick_params(axis="x", pad=1.5, length=2.5)
    ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.grid(False, axis="y")
    despine(ax)


def draw_lobe_geometry(ax_geo: plt.Axes, observed_min: float, observed_max: float, compact: bool = False) -> None:
    ax_geo.set_aspect("equal")
    ax_geo.set_xlim(-0.45, 6.9)
    ax_geo.set_ylim(-2.75, 2.75)
    ax_geo.axis("off")
    sat = (0.0, 0.0)
    earth = (5.95, 0.0)
    radius = 5.2

    ax_geo.add_patch(patches.Wedge(sat, radius, -26, 26, facecolor=COLORS["main"], alpha=0.16, edgecolor="none"))
    ax_geo.add_patch(patches.Wedge(sat, radius * 0.98, observed_min, observed_max, facecolor=COLORS["observed"], alpha=0.26, edgecolor="none"))
    ax_geo.add_patch(patches.Wedge(sat, radius * 0.98, -observed_max, -observed_min, facecolor=COLORS["observed"], alpha=0.20, edgecolor="none"))
    if compact:
        for angle1, angle2, length, alpha in [(31, 45, radius * 0.68, 0.16), (-45, -31, radius * 0.68, 0.14)]:
            ax_geo.add_patch(
                patches.Wedge(
                    sat,
                    length,
                    angle1,
                    angle2,
                    facecolor=COLORS["side"],
                    alpha=alpha,
                    edgecolor="none",
                )
            )

    threshold_lines = [
        (26, COLORS["main"], 1.0, "-"),
        (-26, COLORS["main"], 1.0, "-"),
    ]
    if not compact:
        threshold_lines.extend(
            [
                (20.5, "#4B6CB7", 0.8, (0, (3.5, 3.0))),
                (-20.5, "#4B6CB7", 0.8, (0, (3.5, 3.0))),
            ]
        )
    threshold_lines.extend(
        [
            (observed_min, COLORS["observed"], 1.0, "-"),
            (observed_max, COLORS["observed"], 1.0, "-"),
        ]
    )
    for angle, color, lw, ls in threshold_lines:
        end = polar_xy(sat, radius, angle)
        ax_geo.plot([sat[0], end[0]], [sat[1], end[1]], color=color, linewidth=lw, linestyle=ls)

    side_lobes = []
    if not compact:
        side_lobes.extend(
            [
                (39, 2.9, 0.50, 0.24),
                (-39, 2.9, 0.50, 0.24),
                (58, 2.35, 0.36, 0.17),
                (-58, 2.35, 0.36, 0.17),
                (75, 1.75, 0.25, 0.13),
                (-75, 1.75, 0.25, 0.13),
            ]
        )
    for angle, length, width, alpha in side_lobes:
        cx, cy = polar_xy(sat, length, angle)
        ax_geo.add_patch(
            patches.Ellipse(
                (cx, cy),
                width=width,
                height=0.22,
                angle=angle,
                facecolor=COLORS["side"],
                edgecolor="none",
                alpha=alpha,
            )
        )

    ax_geo.annotate("", xy=(earth[0] - 0.48, earth[1]), xytext=(sat[0], sat[1]), arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": COLORS["ink"]})
    draw_earth(ax_geo, earth, compact=compact)
    if compact:
        draw_satellite(ax_geo, (sat[0] - 0.05, sat[1] + 0.02), compact=True)
    else:
        ax_geo.add_patch(patches.Circle(sat, 0.15, facecolor=COLORS["ink"], edgecolor="none"))
    lugre_point = polar_xy(sat, 4.85, 14.2)
    ax_geo.scatter([lugre_point[0]], [lugre_point[1]], s=22, facecolor=COLORS["observed"], edgecolor="white", linewidth=0.6, zorder=5)
    ax_geo.text(sat[0] - 0.12, sat[1] - 0.60, "GNSS satellite", ha="center", va="top", fontsize=6.8, fontweight="bold")
    ax_geo.text(earth[0], earth[1] - (0.62 if compact else 0.68), "Earth", ha="center", va="top", fontsize=7.0, color="#1E3A5F", fontweight="bold")
    if compact:
        ax_geo.text(2.42, 0.78, "main lobe", ha="center", va="center", fontsize=7.0, color="#285B8F", fontweight="bold")
        ax_geo.text(3.36, 1.42, "LuGRE range", fontsize=6.8, color=COLORS["observed"], fontweight="bold", ha="left")
        ax_geo.text(1.25, 1.65, "side lobe", fontsize=6.0, color="#6D4FA1", ha="center")
        ax_geo.text(1.25, -1.65, "side lobe", fontsize=6.0, color="#6D4FA1", ha="center")
    else:
        ax_geo.text(2.55, -0.22, "boresight / nadir", ha="center", va="top", fontsize=6.6, color=COLORS["muted"])
        ax_geo.text(2.10, 0.91, "main lobe", ha="center", va="center", fontsize=7.4, color="#285B8F", fontweight="bold")
        ax_geo.text(2.12, 1.86, "20.5 deg", fontsize=6.2, color="#4B6CB7")
        ax_geo.text(2.80, 2.28, "26 deg", fontsize=6.2, color="#285B8F")
        ax_geo.text(3.72, 1.36, f"LuGRE observed\n{observed_min:.1f}-{observed_max:.1f} deg", fontsize=6.9, color=COLORS["observed"], fontweight="bold", ha="left")
        ax_geo.text(0.76, -2.08, "side lobes beyond the main-lobe cut-off", fontsize=7.2, color="#6D4FA1")


def draw_lobe_geometry_vertical(ax_geo: plt.Axes, observed_min: float, observed_max: float) -> None:
    ax_geo.set_aspect("equal")
    ax_geo.set_xlim(-6.35, 6.35)
    ax_geo.set_ylim(-13.85, 1.58)
    ax_geo.axis("off")
    ax_geo.set_xticks([])
    ax_geo.set_yticks([])
    ax_geo.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    earth_limb_deg = 13.8
    vertical_shift = -0.54
    sat_image = (0.0, 0.58 + vertical_shift)
    antenna = (0.0, 0.16 + vertical_shift)
    earth = (0.0, -10.48 + vertical_shift)
    center_distance = antenna[1] - earth[1]
    main_edge_radius = center_distance * math.sin(math.radians(26.0))
    tangent_length = center_distance * math.cos(math.radians(26.0))
    earth_radius = center_distance * math.sin(math.radians(earth_limb_deg))
    beam_length = center_distance + earth_radius * 1.25

    def beam_angle(offset_deg: float) -> float:
        return -90.0 + offset_deg

    def surface_length(offset_deg: float) -> float:
        theta = math.radians(abs(offset_deg))
        disc = earth_radius * earth_radius - (center_distance * math.sin(theta)) ** 2
        if disc <= 0:
            return tangent_length
        return center_distance * math.cos(theta) - math.sqrt(disc)

    ax_geo.add_patch(
        patches.Wedge(
            antenna,
            beam_length,
            beam_angle(-26.0),
            beam_angle(26.0),
            facecolor=COLORS["main"],
            alpha=0.16,
            edgecolor="none",
        )
    )
    for a0, a1, length, alpha in [
        (-66, -28, beam_length * 0.50, 0.14),
        (28, 66, beam_length * 0.50, 0.14),
        (-82, -66, beam_length * 0.37, 0.09),
        (66, 82, beam_length * 0.37, 0.09),
    ]:
        ax_geo.add_patch(
            patches.Wedge(
                antenna,
                length,
                beam_angle(a0),
                beam_angle(a1),
                facecolor=COLORS["side"],
                alpha=alpha,
                edgecolor="none",
            )
        )
    for a0, a1, alpha in [
        (observed_min, observed_max, 0.28),
        (-observed_max, -observed_min, 0.21),
    ]:
        ax_geo.add_patch(
            patches.Wedge(
                antenna,
                beam_length * 0.99,
                beam_angle(a0),
                beam_angle(a1),
                facecolor=COLORS["observed"],
                alpha=alpha,
                edgecolor="none",
            )
        )

    for offset, color, lw, ls in [
        (-26.0, COLORS["main"], 1.0, "-"),
        (26.0, COLORS["main"], 1.0, "-"),
        (-23.5, "#B91C1C", 0.62, (0, (3.0, 2.2))),
        (23.5, "#B91C1C", 0.62, (0, (3.0, 2.2))),
        (-20.5, "#4B6CB7", 0.62, (0, (2.2, 2.0))),
        (20.5, "#4B6CB7", 0.62, (0, (2.2, 2.0))),
        (-earth_limb_deg, "#7D8793", 0.70, (0, (3.0, 2.6))),
        (earth_limb_deg, "#7D8793", 0.70, (0, (3.0, 2.6))),
        (-observed_max, COLORS["observed"], 0.95, "-"),
        (-observed_min, COLORS["observed"], 0.95, "-"),
        (observed_min, COLORS["observed"], 0.95, "-"),
        (observed_max, COLORS["observed"], 0.95, "-"),
    ]:
        length = beam_length * (0.98 if abs(offset) >= observed_min else 0.88)
        end = polar_xy(antenna, length, beam_angle(offset))
        ax_geo.plot([antenna[0], end[0]], [antenna[1], end[1]], color=color, linewidth=lw, linestyle=ls, zorder=3)

    legend_x = -6.04
    legend_y = -0.62
    ax_geo.text(legend_x, legend_y + 0.38, "public SSV reference limits", fontsize=5.45, color=COLORS["muted"], ha="left", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35})
    for idx, (label, color, ls) in enumerate([
        ("Gal E1C 20.5$^\\circ$", "#4B6CB7", (0, (2.2, 2.0))),
        ("GPS L1 C/A, Gal E5a-Q 23.5$^\\circ$", "#B91C1C", (0, (3.0, 2.2))),
        ("GPS L5Q 26$^\\circ$", "#285B8F", (0, (3.6, 2.2))),
    ]):
        yy = legend_y - idx * 0.72
        ax_geo.plot([legend_x, legend_x + 0.58], [yy, yy], color=color, linewidth=0.85, linestyle=ls, zorder=10)
        ax_geo.text(legend_x + 0.72, yy, label, fontsize=5.05, color=color, ha="left", va="center", zorder=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.2})

    earth_top = (earth[0], earth[1] + earth_radius)
    ax_geo.plot(
        [antenna[0], earth_top[0]],
        [antenna[1], earth_top[1]],
        color="#7D8793",
        linewidth=0.70,
        linestyle=(0, (3.0, 2.6)),
        zorder=4,
    )

    draw_satellite(ax_geo, sat_image, compact=True)
    draw_earth_scaled(ax_geo, earth, earth_radius)

    lugre_r = beam_length * 0.56
    lugre_point = polar_xy(antenna, lugre_r, beam_angle(-14.2))
    ax_geo.scatter([lugre_point[0]], [lugre_point[1]], s=20, facecolor=COLORS["observed"], edgecolor="white", linewidth=0.55, zorder=12)

    ax_geo.text(
        3.12,
        sat_image[1] + 0.18,
        "GNSS satellite",
        fontsize=6.3,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
        zorder=14,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
    )
    ax_geo.text(0.5, -0.075, "Earth", transform=ax_geo.transAxes, ha="center", va="top", fontsize=6.8, fontweight="bold", color="#1E3A5F", clip_on=False)
    ax_geo.text(0.0, -3.05, "main lobe", ha="center", va="center", fontsize=6.9, color="#285B8F", fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.25})
    ax_geo.text(-4.02, -3.80, "side lobe", ha="center", va="center", fontsize=5.9, color="#6D4FA1", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.2})
    ax_geo.text(4.02, -3.80, "side lobe", ha="center", va="center", fontsize=5.9, color="#6D4FA1", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.2})
    limb_label_y = earth[1] + earth_radius + 0.20 + earth_radius * 0.22
    limb_arc_y = earth[1] + earth_radius + 0.06
    limb_arc_r = (limb_arc_y - antenna[1]) / math.sin(math.radians(beam_angle(earth_limb_deg)))
    limb_arc_start = (earth[0], limb_arc_y)
    limb_arc_end = polar_xy(antenna, limb_arc_r, beam_angle(earth_limb_deg))
    limb_arc = patches.FancyArrowPatch(
        limb_arc_start,
        limb_arc_end,
        connectionstyle="arc3,rad=-0.28",
        arrowstyle="<->",
        mutation_scale=10.0,
        linewidth=1.0,
        color="#7D8793",
        zorder=18,
    )
    ax_geo.add_patch(limb_arc)
    ax_geo.text(
        0.5 * (limb_arc_start[0] + limb_arc_end[0]),
        limb_label_y,
        "rep. Earth-limb\n13.8$^\\circ$",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#28313D",
        zorder=19,
    )
    ax_geo.annotate(
        "LuGRE",
        xy=lugre_point,
        xytext=(lugre_point[0] - 1.76, lugre_point[1] + 0.44),
        ha="right",
        va="center",
        fontsize=6.5,
        color=COLORS["observed"],
        fontweight="bold",
        zorder=13,
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": COLORS["observed"], "shrinkA": 0.5, "shrinkB": 1},
    )


def make_combined_ab_figure(df: pd.DataFrame) -> list[Path]:
    observed_min = float(df["tx_off_boresight_deg"].min())
    observed_max = float(df["tx_off_boresight_deg"].max())

    fig = plt.figure(figsize=(8.15, 6.95))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.03, 1.34], hspace=0.36)

    # Panel a: signal-band angle distributions with compact summary strips.
    gs_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[1.95, 1.08], hspace=0.06)
    ax = fig.add_subplot(gs_a[0, 0])
    ax_box = fig.add_subplot(gs_a[1, 0], sharex=ax)
    x_min, x_max = 11.2, 16.9
    bins = np.linspace(x_min, x_max, 68)
    curves = {}
    max_y = 0.0
    for group in SIGNAL_ORDER:
        values = df.loc[df["signal_name"] == group, "tx_off_boresight_deg"].to_numpy()
        x, y = smoothed_histogram(values, bins)
        curves[group] = (values, x, y)
        max_y = max(max_y, float(y.max()))
    for group in SIGNAL_ORDER:
        values, x, y = curves[group]
        color = COLORS[group]
        ax.fill_between(x, y, 0, color=color, alpha=0.13, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.55, label=f"{SIGNAL_SHORT_LABELS[group]} (n={len(values):,})")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_y * 1.18)
    ax.set_ylabel("Probability density")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=2.0, borderpad=0.1, ncol=2, columnspacing=1.0)
    ax.text(
        0.98,
        0.92,
        "side-lobes excluded before statistics",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    despine(ax)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COLORS["ink"])
        ax.spines[side].set_linewidth(0.75)
    ax.text(
        0.0,
        1.13,
        "a  Main-lobe off-boresight angle statistics by signal",
        transform=ax.transAxes,
        fontsize=10.4,
        fontweight="bold",
        va="bottom",
    )

    y_positions = {group: float(len(SIGNAL_ORDER) - idx - 1) for idx, group in enumerate(SIGNAL_ORDER)}
    for group in SIGNAL_ORDER:
        y = y_positions[group]
        values = curves[group][0]
        q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        color = COLORS[group]
        ax_box.plot([q[0], q[6]], [y, y], color=color, linewidth=0.9, alpha=0.75)
        ax_box.plot([q[1], q[5]], [y, y], color=color, linewidth=4.5, alpha=0.58, solid_capstyle="round")
        ax_box.add_patch(
            patches.Rectangle((q[2], y - 0.13), q[4] - q[2], 0.26, facecolor="white", edgecolor=color, linewidth=0.85, alpha=0.72)
        )
        ax_box.plot([q[3], q[3]], [y - 0.17, y + 0.17], color=color, linewidth=1.7)
        ax_box.scatter([q[0], q[6]], [y, y], s=10, color=color, zorder=3)
        ax_box.text(x_max - 0.03, y + 0.20, f"median {q[3]:.2f}", ha="right", va="center", fontsize=5.6, color=COLORS["muted"])
    ax_box.set_ylim(-0.52, len(SIGNAL_ORDER) - 0.48)
    ax_box.set_yticks([y_positions[group] for group in SIGNAL_ORDER])
    ax_box.set_yticklabels([f"{SIGNAL_SHORT_LABELS[group]} ({SIGNAL_LIMITS[group]:g} deg)" for group in SIGNAL_ORDER])
    ax_box.set_xlabel(r"Transmitter off-boresight angle, $\alpha$ (deg)")
    ax_box.grid(True, axis="x", color=COLORS["grid"], linewidth=0.55)
    ax_box.grid(False, axis="y")
    for side in ["top", "right", "bottom", "left"]:
        ax_box.spines[side].set_visible(True)
        ax_box.spines[side].set_color(COLORS["ink"])
        ax_box.spines[side].set_linewidth(0.75)

    gs_bc = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[0.92, 1.18], wspace=0.42)

    # Panel b: vertical lobe geometry with the transmitting satellite above Earth.
    ax_b = fig.add_subplot(gs_bc[0, 0])
    draw_lobe_geometry_vertical(ax_b, observed_min, observed_max)
    ax_b.text(
        -0.17,
        1.03,
        "b  Geometry and SSV reference limits",
        transform=ax_b.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="bottom",
    )

    # Panel c: signal-by-signal main-lobe selection and side-lobe exclusion.
    gain_ax = fig.add_subplot(gs_bc[0, 1])
    draw_signal_selection_panel(gain_ax, df, observed_min, observed_max)
    gain_ax.text(
        -0.10,
        1.03,
        "c  Main-lobe selection by signal",
        transform=gain_ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="bottom",
    )

    lower_left = ax_b.get_position().x0
    lower_right = gain_ax.get_position().x1
    for top_ax in [ax, ax_box]:
        pos = top_ax.get_position()
        top_ax.set_position([lower_left, pos.y0, lower_right - lower_left, pos.height])

    paths = save_figure(fig, "lugre_main_lobe_ab_figure_nature")
    plt.close(fig)
    return paths


def main() -> None:
    configure_style()
    df = load_points()
    outputs = []
    outputs.extend(make_angle_statistics_figure(df))
    outputs.extend(make_lobe_schematic_figure(df))
    outputs.extend(make_combined_ab_figure(df))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()


