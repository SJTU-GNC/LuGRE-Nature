from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import plot_main_lobe_gps_galileo as core
import plot_main_lobe_previous_ab_gps_galileo as figure_code


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DERIVED_DIR = ROOT / "derived_data"


def build_inputs():
    core.configure_style()
    plt.rcParams.update({"font.size": 7.2, "axes.titlesize": 7.6})

    points = figure_code.load_previous_points()
    raw_windows, _ = core.load_windows()
    prn_svn = core.load_prn_svn_map()

    summaries: dict[str, core.PatternSummary] = {}
    gps_valleys: dict[str, dict[str, object]] = {}
    missing: dict[str, list[int]] = {}
    for signal in ("GPS L1", "GPS L5"):
        patterns, _, missing_svns = figure_code.load_gps_pattern_set(
            signal, raw_windows, prn_svn
        )
        summaries[signal] = core.summarize_satellite_patterns(
            patterns,
            "NAVCEN measured SV patterns; equal weight per patterned SVN",
        )
        gps_valleys[signal] = figure_code.summarize_gps_first_valleys(patterns)
        missing[signal] = missing_svns

    summaries["Galileo E1"] = core.load_galileo_summary("Galileo E1")
    summaries["Galileo E5a"] = core.load_galileo_summary("Galileo E5a")

    e1 = figure_code.first_valley_summary(
        core.load_grap_band(core.GRAP_DIR / "GRAP_File_E1_.xlsx")
    )
    e5a = figure_code.first_valley_summary(
        core.load_grap_band(core.GRAP_DIR / "GRAP_File_E5a_.xlsx")
    )
    return points, summaries, gps_valleys, e1, e5a, missing


def save_panel_b(
    points: pd.DataFrame,
    gps_valleys: dict[str, dict[str, object]],
    e1: dict[str, float],
    e5a: dict[str, float],
) -> list[Path]:
    observed_min = float(points["tx_off_boresight_deg"].min())
    observed_max = float(points["tx_off_boresight_deg"].max())

    fig = plt.figure(figsize=(4.25, 5.25))
    grid = fig.add_gridspec(
        1, 1, left=0.03, right=0.97, bottom=0.10, top=0.90
    )
    ax = fig.add_subplot(grid[0, 0])
    figure_code.draw_previous_panel_b(
        ax,
        observed_min,
        observed_max,
        gps_valleys["GPS L1"],
        gps_valleys["GPS L5"],
        e1,
        e5a,
    )
    position = ax.get_position()
    fig.text(
        position.x0,
        position.y1 + 0.022,
        "b  Earth-limb and pattern-derived lobe diagnostics",
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=figure_code.OLD_COLORS["ink"],
    )
    paths = figure_code.save_panel_figure(fig, OUTPUT_DIR / "panel_b")
    plt.close(fig)
    return paths


def save_panel_c(
    points: pd.DataFrame,
    summaries: dict[str, core.PatternSummary],
) -> list[Path]:
    fig = plt.figure(figsize=(6.35, 4.85))
    grid = fig.add_gridspec(
        1, 1, left=0.10, right=0.985, bottom=0.18, top=0.86
    )
    axes, legend = core.draw_panel_c(fig, grid[0, 0], summaries, points)
    fig.text(
        axes[0].get_position().x0,
        axes[0].get_position().y1 + 0.038,
        "c  Transmit-pattern evidence by signal",
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=figure_code.OLD_COLORS["ink"],
    )
    fig.legend(
        handles=legend,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.025),
        ncol=3,
        columnspacing=1.15,
        handlelength=1.8,
        handletextpad=0.42,
        labelspacing=0.45,
        fontsize=5.8,
    )
    paths = figure_code.save_panel_figure(
        fig, OUTPUT_DIR / "panel_c", tight=False
    )
    plt.close(fig)
    return paths


def write_first_valley_summary(
    gps_valleys: dict[str, dict[str, object]],
    e1: dict[str, float],
    e5a: dict[str, float],
) -> Path:
    rows = []
    for signal, values, source in (
        ("GPS L1", gps_valleys["GPS L1"], "NAVCEN measured SV patterns"),
        ("GPS L5", gps_valleys["GPS L5"], "NAVCEN measured SV patterns"),
        ("Galileo E1", e1, "JRC GRAP FOC population model"),
        ("Galileo E5a", e5a, "JRC GRAP FOC population model"),
    ):
        rows.append(
            {
                "signal": signal,
                "first_valley_min_deg": values["min"],
                "first_valley_p05_deg": values["p05"],
                "first_valley_median_deg": values["median"],
                "first_valley_p95_deg": values["p95"],
                "first_valley_max_deg": values["max"],
                "source": source,
                "interpretation": "diagnostic first valley; not a fixed lobe edge",
            }
        )
    path = DERIVED_DIR / "panel_b_first_valley_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    DERIVED_DIR.mkdir(exist_ok=True)
    points, summaries, gps_valleys, e1, e5a, missing = build_inputs()
    paths = save_panel_b(points, gps_valleys, e1, e5a)
    paths.extend(save_panel_c(points, summaries))
    paths.append(write_first_valley_summary(gps_valleys, e1, e5a))

    for path in paths:
        print(path)
    print(f"GPS missing public patterns: {missing}")


if __name__ == "__main__":
    main()