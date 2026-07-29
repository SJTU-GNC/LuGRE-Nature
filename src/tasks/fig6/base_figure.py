from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
ROOT = PACKAGE_ROOT / "outputs" / "Fig6"
ZIP_ASSET_DIR = PACKAGE_ROOT / "assets" / "Fig6"
PREV = PACKAGE_ROOT / "data" / "analysis_ready" / "Fig6"

PANEL_A = ZIP_ASSET_DIR / "polar_schematic.png"
PANEL_B = ZIP_ASSET_DIR / "scale_schematic.png"
RA_SOURCE = PREV / "response_ratio_source.csv"
LOS_SOURCE = PREV / "projected_scale_source.csv"
LOS_SUMMARY = PREV / "projected_scale_summary.csv"
NORM_SUMMARY = PREV / "normalized_scale_summary.csv"

OUT_PNG = ROOT / "fig5_six_panel_nature_v23_panel_a_height_aligned.png"
OUT_PDF = ROOT / "fig5_six_panel_nature_v23_panel_a_height_aligned.pdf"
OUT_SUMMARY = ROOT / "fig5_six_panel_nature_v23_source_summary.csv"
OUT_README = ROOT / "README_fig5_six_panel_nature_v23.md"


COL_GROUND = "#6e7b88"
COL_LEO = "#c48224"
COL_LUGRE = "#1676b7"
COL_EQ = "#dfb766"
COL_POLAR = "#2b84b6"
COL_GRID = "#dfe5eb"
COL_TEXT = "#202a33"
SHADE_GROUND_LEO = "#dce4ea"
SHADE_LEO = "#ead4ad"
SHADE_LUGRE = "#d8edf8"
SHADE_LUGRE_SCALE = "#e5f2fb"


def crop_white(img: Image.Image, margin: int = 20, threshold: int = 14) -> Image.Image:
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        rgb = white.convert("RGB")
    else:
        rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.point(lambda p: 255 if p > threshold else 0).getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    return rgb.crop(
        (
            max(0, left - margin),
            max(0, top - margin),
            min(img.width, right + margin),
            min(img.height, bottom + margin),
        )
    )


def logfmt(x, _pos):
    if x >= 1:
        return f"{x:g}"
    return f"{x:.2g}"


def panel_label(ax, label: str):
    ax.text(
        -0.035,
        1.022,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def outside_title(ax, title: str):
    ax.text(
        0.50,
        1.022,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color=COL_TEXT,
        clip_on=False,
    )


def style_axes(ax):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#8b98a5")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=7.5, width=0.8, length=3)


def image_panel(ax, path: Path, label: str, margin: int = 18):
    img = crop_white(Image.open(path), margin=margin)
    ax.imshow(img, aspect="auto")
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec="#8b98a5", lw=0.8, zorder=10))
    panel_label(ax, label)
    outside_title(ax, "Lunar-distance GNSS geometry")


def draw_scale_map(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e2, 1.2e4)
    ax.set_ylim(1, 1e6)
    ax.axvspan(260, 360, color=SHADE_GROUND_LEO, alpha=0.62, zorder=0)
    ax.axvspan(400, 850, color=SHADE_LEO, alpha=0.52, zorder=0)
    ax.axvspan(1900, 2500, color=SHADE_LUGRE, alpha=0.72, zorder=0)

    boxes = [
        ("Ground\nreceiver", 260, 360, 1e2, 1e3, COL_GROUND, "260-360 m"),
        ("LEO-RO", 400, 850, 1e3, 4.5e3, COL_LEO, "400-850 m"),
        ("LuGRE", 1900, 2500, 7e4, 3.5e5, COL_LUGRE, "1.9-2.5 km"),
    ]
    for name, x0, x1, y0, y1, color, txt in boxes:
        ax.fill_between([x0, x1], y0, y1, color=color, alpha=0.55, lw=0, zorder=2)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, lw=0.9, zorder=3)
        ax.vlines([x0, x1], 1, y0, colors=color, linestyles=(0, (3, 2)), lw=0.75, alpha=0.85, zorder=1)
        ax.hlines([y0, y1], 1e2, x0, colors=color, linestyles=(0, (3, 2)), lw=0.75, alpha=0.75, zorder=1)
        if name.startswith("Ground"):
            ax.text(x0 * 0.9, np.sqrt(y0 * y1), f"{name}\n{txt}", ha="right", va="center", fontsize=6.4, color=COL_TEXT)
        else:
            ax.text(x1 * 1.08, np.sqrt(y0 * y1), f"{name}\n{txt}", ha="left", va="center", fontsize=6.4, color=COL_TEXT)

    ax.set_xlabel(r"Fresnel scale, $R_F$ (m)", fontsize=8)
    ax.set_ylabel(r"Receiver distance, $D_R$ (km)", fontsize=8)
    ax.grid(color=COL_GRID, lw=0.55, alpha=0.65)
    ax.xaxis.set_major_locator(FixedLocator([1e2, 3e2, 1e3, 3e3, 1e4]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axes(ax)
    panel_label(ax, "b")
    outside_title(ax, "Link distance and Fresnel scale")


def draw_ra(ax):
    df = pd.read_csv(RA_SOURCE)
    regimes = df[(df["kind"] == "regime") & (df["include_in_visual_guide"].astype(bool))].copy()
    missions = df[(df["kind"] == "leo_mission") & (df["include_in_visual_guide"].astype(bool)) & (df["label"] != "TSX")].copy()

    ax.axhspan(0, 1, color="#f7f7f7", zorder=0)
    ax.axhline(1.0, color="#9ca3af", lw=0.9, ls="--", zorder=1)
    ax.axvspan(0.25, 0.36, color=SHADE_GROUND_LEO, alpha=0.62, zorder=0)
    ax.axvspan(0.40, 0.75, color=SHADE_LEO, alpha=0.52, zorder=0)
    ax.axvspan(1.91, 2.21, color=SHADE_LUGRE, alpha=0.65, zorder=0)

    for _, r in missions.iterrows():
        ax.errorbar(
            r["rF_ref_km"],
            r["R_A_polar_over_equatorial"],
            xerr=[[r["rF_ref_km"] - r["rF_min_km"]], [r["rF_max_km"] - r["rF_ref_km"]]],
            fmt="o",
            ms=3.6,
            color=COL_LEO,
            ecolor=COL_LEO,
            alpha=0.55,
            capsize=2,
            lw=0.9,
            zorder=3,
        )

    colors = {"ground": COL_GROUND, "leo": COL_LEO, "moon": COL_LUGRE}
    markers = {"ground": "s", "leo": "D", "moon": "*"}
    for _, r in regimes.iterrows():
        ax.errorbar(
            r["rF_ref_km"],
            r["R_A_polar_over_equatorial"],
            xerr=[[max(r["rF_ref_km"] - r["rF_min_km"], 0)], [max(r["rF_max_km"] - r["rF_ref_km"], 0)]],
            fmt=markers[r["regime"]],
            ms=7.2 if r["regime"] != "moon" else 11,
            color=colors[r["regime"]],
            ecolor=colors[r["regime"]],
            mec="white",
            mew=0.8,
            capsize=3,
            lw=1.2,
            zorder=4,
        )

    ax.annotate(
        "",
        xy=(1.86, 2.47),
        xytext=(0.48, 1.18),
        arrowprops=dict(arrowstyle="->", lw=1.25, color="#9ba6af"),
        zorder=2,
    )
    ax.text(
        0.57,
        1.34,
        "response changes from equatorial/comparable\n"
        "to polar-dominant in the lunar-distance regime",
        color="#5d6874",
        fontsize=6.9,
        ha="left",
        va="center",
    )
    ax.text(0.275, 0.62, "Ground\nISMR $S_4$", color=COL_GROUND, fontsize=7, ha="left", va="top")
    ax.text(0.58, 0.94, "LEO-RO", color=COL_LEO, fontsize=7, ha="left", va="bottom")
    ax.text(1.72, 2.64, "LuGRE", color=COL_LUGRE, fontsize=8, ha="right", va="bottom")
    ax.text(0.26, 3.04, "ground/LEO\n$R_F$", color="#607080", fontsize=6.8, ha="left", va="top")
    ax.text(1.88, 3.04, "LuGRE\n$R_F$", color=COL_LUGRE, fontsize=6.8, ha="left", va="top")
    ax.text(2.62, 1.03, "$R_A=1$", fontsize=6.8, color="#6b7280", va="bottom", ha="right")

    ax.set_xscale("log")
    ax.set_xlim(0.18, 3.0)
    ax.set_ylim(0.35, 3.15)
    ax.set_xlabel(r"L1 Fresnel scale, $R_F$ (km)", fontsize=8)
    ax.set_ylabel(r"$R_A={\rm P95}_{polar}/{\rm P95}_{equatorial}$", fontsize=8)
    ax.grid(color=COL_GRID, lw=0.6, alpha=0.7)
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 0.7, 1, 2, 3]))
    ax.xaxis.set_major_formatter(FuncFormatter(logfmt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    style_axes(ax)
    panel_label(ax, "c")
    outside_title(ax, "Polar/equatorial amplitude response versus Fresnel scale")


def draw_ratio(ax):
    df = pd.read_csv(RA_SOURCE)
    regimes = df[df["kind"] == "regime"].copy()
    order = ["Ground ISMR S4", "LEO-RO aggregate", "LuGRE lunar link"]
    labels = ["Ground\nISMR $S_4$", "LEO-RO\n$\\delta C/N_0$ proxy", "LuGRE\n$\\delta C/N_0$"]
    vals = [float(regimes.loc[regimes["label"] == o, "R_A_polar_over_equatorial"].iloc[0]) for o in order]
    x = np.arange(len(order))
    width = 0.34
    ax.bar(x - width / 2, np.ones(len(order)), width=width, color=COL_EQ, edgecolor="none", label="Equatorial")
    polar_colors = [COL_GROUND, "#b37a23", COL_LUGRE]
    ax.bar(x + width / 2, vals, width=width, color=polar_colors, edgecolor="none", label="Polar")
    for xi, v in zip(x, vals):
        ax.text(xi + width / 2, v + 0.08, f"{v:.2f}x", ha="center", va="bottom", fontsize=7, color=COL_TEXT)
    ax.axhline(1, color="#8c8c8c", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.3)
    ax.set_ylim(0, 3.1)
    ax.set_ylabel("Amplitude response\nnormalized to equatorial", fontsize=8)
    ax.text(0.02, 0.94, "Equatorial", transform=ax.transAxes, color=COL_EQ, fontsize=7.5, va="top")
    ax.text(0.43, 0.94, "Polar", transform=ax.transAxes, color=COL_GROUND, fontsize=7.5, va="top")
    ax.grid(axis="y", color=COL_GRID, lw=0.6, alpha=0.75)
    style_axes(ax)
    panel_label(ax, "d")
    outside_title(ax, "Amplitude-response contrast")


def draw_los_hist(ax):
    src = pd.read_csv(LOS_SOURCE)
    src = src[np.isfinite(src["L_LOS_km"]) & (src["L_LOS_km"] > 0)].copy()
    compact = src[src["match_class"].astype(str).str.contains("strict", case=False, na=False)]
    if compact.empty:
        compact = src[src["match_class"].astype(str).str.contains("compact", case=False, na=False)]
    summary = pd.read_csv(LOS_SUMMARY).set_index("metric")["value"].to_dict()
    med_all = float(summary["median_all_km"])
    med_compact = float(summary["median_compact_km"])
    lugre_l1 = float(summary["lugre_l1_RF_km"])
    lugre_l5 = float(summary["lugre_l5_RF_km"])
    n_all = int(float(summary["n_all"]))
    n_compact = int(float(summary["n_compact"]))

    ax.axvspan(lugre_l1, lugre_l5, color=SHADE_LUGRE, alpha=0.72, zorder=0)
    bins = np.logspace(np.log10(0.06), np.log10(300), 42)
    ax.hist(src["L_LOS_km"], bins=bins, color="#b5bec5", alpha=0.88, edgecolor="white", lw=0.25, label=f"all matches (n={n_all})", zorder=2)
    ax.hist(compact["L_LOS_km"], bins=bins, histtype="step", color="#17212b", lw=1.4, label=f"compact subset (n={n_compact})", zorder=4)
    ax.axvline(med_all, color="#ffffff", lw=1.2, zorder=5)
    ax.axvline(med_compact, color="#17212b", lw=1.0, ls="--", zorder=5)

    ax.text(np.sqrt(lugre_l1 * lugre_l5), 21.75, "LuGRE $R_F$", fontsize=6.8, color=COL_LUGRE, ha="center", va="top", zorder=7)
    ax.text(
        210.0,
        6.1,
        f"median all = {med_all:.2f} km\nmedian compact = {med_compact:.2f} km",
        fontsize=6.5,
        ha="right",
        va="bottom",
        zorder=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.3),
    )
    ax.set_xscale("log")
    ax.set_xlim(0.06, 300)
    ax.set_ylim(0, 22.5)
    ax.set_xlabel(r"Projected structure scale, $L_{\rm LOS}=|V_{\rm LOS}|/f$ (km)", fontsize=8)
    ax.set_ylabel("Matched windows (count)", fontsize=8)
    ax.grid(axis="x", color=COL_GRID, lw=0.6, alpha=0.75)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100, 300]))
    ax.xaxis.set_major_formatter(FuncFormatter(logfmt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper right", fontsize=6.3, frameon=False)
    style_axes(ax)
    panel_label(ax, "e")
    outside_title(ax, "LuGRE-SuperDARN projected structure scale")


def draw_norm(ax):
    norm = pd.read_csv(NORM_SUMMARY)
    y = {"ground": 2.6, "LEO-RO": 1.55, "LuGRE": 0.5}
    labels = {"ground": "ground\n$R_F$ ref.", "LEO-RO": "LEO-RO\n$R_F$ ref.", "LuGRE": "LuGRE\n$R_F$ ref."}
    all_cols = {"ground": COL_GROUND, "LEO-RO": COL_LEO, "LuGRE": COL_LUGRE}
    comp_col = "#17212b"
    ax.axvspan(0.5, 2, color=SHADE_LUGRE_SCALE, alpha=0.65, zorder=0)
    ax.axvline(1, color="#222", lw=0.9)
    for _, r in norm.iterrows():
        ref = r["reference"]
        yy = y[ref]
        ax.hlines(yy + 0.16, r["all_iqr_low"], r["all_iqr_high"], color=all_cols[ref], lw=1.5)
        ax.hlines(yy + 0.16, r["all_iqr_low"], r["all_iqr_high"], color=all_cols[ref], lw=6, alpha=0.68)
        ax.plot(r["all_median_L_over_RF"], yy + 0.16, "o", color="white", mec=all_cols[ref], mew=1.2, ms=4.6)
        ax.hlines(yy - 0.16, r["compact_iqr_low"], r["compact_iqr_high"], color=comp_col, lw=1.5)
        ax.hlines(yy - 0.16, r["compact_iqr_low"], r["compact_iqr_high"], color=comp_col, lw=4.8, alpha=0.98)
        ax.plot(r["compact_median_L_over_RF"], yy - 0.16, "o", color="white", mec=comp_col, mew=1.2, ms=4.6)
        ax.text(0.065, yy, labels[ref], ha="left", va="center", fontsize=8, color=COL_TEXT)
    ax.text(1.06, 3.05, r"$L_{\rm LOS}=R_F$", fontsize=7.5, ha="left", va="top", color=COL_TEXT)
    ax.plot([], [], color="#8aa0ad", lw=5, label="all IQR")
    ax.plot([], [], color=comp_col, lw=4, label="compact IQR")
    ax.legend(loc="lower right", fontsize=6.3, frameon=False)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 100)
    ax.set_ylim(0, 3.25)
    ax.set_yticks([])
    ax.set_xlabel(r"Projected scale / reference $R_F$ (L1/E1)", fontsize=8)
    ax.grid(axis="x", color=COL_GRID, lw=0.6, alpha=0.75)
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.3, 1, 3, 10, 30, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(logfmt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    style_axes(ax)
    panel_label(ax, "f")
    outside_title(ax, r"Same scale normalized by reference $R_F$")


def write_outputs_summary():
    rows = [
        {"metric": "panel_a_source", "value": str(PANEL_A)},
        {"metric": "panel_b_source", "value": str(PANEL_B)},
        {"metric": "RA_source", "value": str(RA_SOURCE)},
        {"metric": "LOS_source", "value": str(LOS_SOURCE)},
        {"metric": "normalized_summary", "value": str(NORM_SUMMARY)},
    ]
    for _, r in pd.read_csv(RA_SOURCE).iterrows():
        if str(r["include_in_visual_guide"]).lower() == "true":
            rows.append({"metric": f"RA_{r['label']}", "value": f"{r['R_A_polar_over_equatorial']:.3f}"})
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    OUT_README.write_text(
        "# Fig.5 six-panel Nature-style candidate v23\n\n"
        "Panel A uses the updated user-provided polar schematic. Panel B is redrawn "
        "from the user-provided scale schematic so that the ground/LEO/LuGRE Fresnel "
        "backgrounds match panels C and E. C-F are redrawn from the original source data "
        "to remove nested panel letters and large embedded titles.\n\n"
        "Panel logic: A mechanism; B theoretical Fresnel-scale map; C polar/equatorial "
        "response versus Fresnel scale; D ratio interpretation; E SuperDARN-derived projected "
        "structure scale; F Fresnel-normalized projected scale.\n\n"
        "Caution: E/F use LOS-projected SuperDARN velocity and LuGRE temporal frequency. "
        "`L_LOS` is a projected lower-bound scale, not a full two-dimensional irregularity size.\n",
        encoding="utf-8",
    )


def main():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
        }
    )
    fig = plt.figure(figsize=(7.25, 8.15), facecolor="white")
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.88, 0.94, 0.94],
        width_ratios=[1.55, 1.0],
        left=0.065,
        right=0.985,
        top=0.965,
        bottom=0.065,
        wspace=0.27,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    image_panel(ax_a, PANEL_A, "a", margin=12)
    ax_b = fig.add_subplot(gs[0, 1])
    draw_scale_map(ax_b)
    ax_c = fig.add_subplot(gs[1, 0])
    draw_ra(ax_c)
    ax_d = fig.add_subplot(gs[1, 1])
    draw_ratio(ax_d)
    ax_e = fig.add_subplot(gs[2, 0])
    draw_los_hist(ax_e)
    ax_f = fig.add_subplot(gs[2, 1])
    draw_norm(ax_f)
    fig.savefig(OUT_PNG, dpi=450)
    fig.savefig(OUT_PDF)
    write_outputs_summary()
    print(json.dumps({"png": str(OUT_PNG), "pdf": str(OUT_PDF), "summary": str(OUT_SUMMARY)}, indent=2))


if __name__ == "__main__":
    main()
