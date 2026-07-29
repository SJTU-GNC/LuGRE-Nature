#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data' / 'analysis_ready' / 'Fig3'
OUTDIR = ROOT / 'outputs' / 'Fig3'
OUTBASE = OUTDIR / 'Fig3'

# Softened version of the latitude palette used in the paper.
COL = {
    'Equatorial': '#2B9CB5',
    'Mid-lat': '#8163D8',
    'Polar': '#E34E4E',
    'Equatorial_fill': '#E1F4F7',
    'Mid-lat_fill': '#EEE8FA',
    'Polar_fill': '#FBE4E2',
    'Day': '#C9932E',
    'Night': '#2F6F9F',
    'Day_fill': '#F7E7C3',
    'Night_fill': '#DCEBF6',
    'Sunlit': '#C9932E',
    'Dark': '#2F6F9F',
    'Sunlit_fill': '#F7E7C3',
    'Dark_fill': '#DCEBF6',
    'Text': '#111827',
    'Axis': '#384154',
    'Grid': '#D8E0EA',
    'LightGrid': '#EEF2F6',
    'Dark': '#111827',
    'TableHeader': '#EDF2F7',
    'TableAlt': '#F8FAFC',
    'TableEdge': '#D5DEE8',
    'Highlight': '#FFF1E8'
}
ORDER = ['Equatorial', 'Mid-lat', 'Polar']
LABELS = {'Equatorial': 'Equat.', 'Mid-lat': 'Mid-lat', 'Polar': 'Polar'}
FULL_LABELS = {'Equatorial': 'Equatorial', 'Mid-lat': 'Mid-latitude', 'Polar': 'Polar'}

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8.4,
    'xtick.labelsize': 6.4,
    'ytick.labelsize': 6.4,
    'legend.fontsize': 6.7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.45,
    'ytick.minor.width': 0.45,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'figure.dpi': 170,
    'savefig.dpi': 600,
})

def clean(ax, grid='y'):
    for s in ['left', 'bottom', 'top', 'right']:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(COL['Axis'])
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=COL['Text'], width=0.6)
    if grid:
        ax.grid(axis=grid, color=COL['Grid'], lw=0.42, alpha=0.70)
    ax.set_axisbelow(True)

def panel_label(ax, label, x=-0.11, y=1.10):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='left', va='bottom', color=COL['Text'])

def wilson_ci(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k/n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n))/denom
    half = z*np.sqrt((p*(1-p) + z**2/(4*n))/n)/denom
    return 100*(center-half), 100*(center+half)

def draw_violin(ax, df, metric, ylabel, log=False):
    data = [df.loc[df['lat_group'].eq(g), metric].dropna().values for g in ORDER]
    positions = np.arange(len(ORDER))
    parts = ax.violinplot(data, positions=positions, widths=0.68, showmeans=False,
                          showextrema=False, showmedians=False)
    rng = np.random.default_rng(2)
    for i, (body, g, vals) in enumerate(zip(parts['bodies'], ORDER, data)):
        body.set_facecolor(COL[g+'_fill'])
        body.set_edgecolor(COL[g])
        body.set_linewidth(0.75)
        body.set_alpha(0.58)
        if len(vals):
            x = i + rng.normal(0, 0.040, size=len(vals))
            ax.scatter(x, vals, s=4.5, color=COL[g], alpha=0.18, lw=0, zorder=2)
            q1, med, q3 = np.nanpercentile(vals, [25, 50, 75])
            ax.plot([i, i], [q1, q3], color=COL['Dark'], lw=1.0, zorder=4)
            ax.plot([i-0.13, i+0.13], [med, med], color=COL['Dark'], lw=1.15, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[g] for g in ORDER])
    ax.set_ylabel(ylabel, labelpad=0)
    if log:
        ax.set_yscale('log')
    clean(ax)

def draw_signal_table(ax, sig):
    ax.axis('off')
    sig_order = ['GPS L1', 'GPS L5', 'Galileo E1', 'Galileo E5a']
    sig = sig.set_index('signal_name').loc[sig_order].reset_index()
    rows = []
    for _, r in sig.iterrows():
        rows.append([
            ({'Galileo E1': 'Gal. E1', 'Galileo E5a': 'Gal. E5a'}.get(r['signal_name'], r['signal_name'])),
            f"{int(r['events'])}/{int(r['total_windows'])}",
            f"{r['event_fraction_percent']:.1f}",
            f"{r['median_width_db']:.2f}",
            f"{r['median_dominant_frequency_hz']:.3f}",
        ])
    col_labels = ['Signal', 'events/n', 'event\n(%)', 'width\n(dB)', 'dom. f\n(Hz)']
    col_widths = [0.25, 0.20, 0.16, 0.17, 0.20]
    table = ax.table(cellText=rows, colLabels=col_labels, colLoc='center', cellLoc='center',
                     loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(6.25)
    table.scale(1.02, 1.43)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COL['TableEdge'])
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(COL['TableHeader'])
            cell.set_text_props(weight='bold', color=COL['Text'])
        else:
            cell.set_facecolor('#FFFFFF' if row % 2 else COL['TableAlt'])
            cell.set_text_props(color=COL['Text'])
            if col == 0:
                cell.set_text_props(ha='left')
    ax.set_title('L5/E5a events are broader', pad=3)

def make_figure():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    asd = pd.read_csv(DATA/'fig3_panel_a_asd_paired_long.csv')
    rate = pd.read_csv(DATA/'fig3_panel_b1_event_fraction.csv')
    events = pd.read_csv(DATA/'fig3_panel_b_event_level_points.csv')
    daynight = pd.read_csv(DATA/'fig3_panel_c_sza_summary.csv')
    alt = pd.read_csv(DATA/'fig3_panel_d_altitude_layer_heatmap.csv')
    sig = pd.read_csv(DATA/'fig3_panel_e_signal_robustness.csv')

    fig = plt.figure(figsize=(8.35, 6.38), constrained_layout=False)
    gs = GridSpec(3, 24, figure=fig, height_ratios=[1.05, 1.16, 1.28],
                  hspace=0.50, wspace=0.42, left=0.056, right=0.992, bottom=0.076, top=0.965)

    # Panel a: ASD, grouped into signal pairs.
    ax_a1 = fig.add_subplot(gs[0, 0:11])
    ax_a2 = fig.add_subplot(gs[0, 12:23], sharey=ax_a1)
    axes_a = [
        (ax_a1, 'L1 / E1', 'Median ASD: GPS L1 / Galileo E1'),
        (ax_a2, 'L5 / E5a', 'Median ASD: GPS L5 / Galileo E5a'),
    ]
    linestyle_map = {'GPS': '-', 'Galileo': (0, (3.2, 2.0))}
    lw_map = {'Polar': 1.45, 'Mid-lat': 1.20, 'Equatorial': 1.20}
    for ax, pair, title in axes_a:
        sub = asd[asd['asd_pair'].eq(pair)].copy()
        for (signal, lat), d in sub.groupby(['signal_name', 'lat_group']):
            fam = 'GPS' if 'GPS' in signal else 'Galileo'
            d = d.sort_values('frequency_hz')
            ax.plot(d['frequency_hz'], d['median_asd_dbhz_per_sqrt_hz'],
                    color=COL[lat], lw=lw_map.get(lat, 1.2), ls=linestyle_map[fam],
                    solid_capstyle='round')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.004, 0.35)
        ax.set_ylim(0.055, 1.4)
        ax.set_title(title, pad=3)
        ax.set_xlabel('Frequency (Hz)')
        clean(ax, grid='both')
        ax.grid(axis='both', which='major', color=COL['Grid'], lw=0.42, alpha=0.70)
        ax.grid(axis='both', which='minor', color=COL['LightGrid'], lw=0.28, alpha=0.65)
    ax_a1.set_ylabel('Median ASD')
    plt.setp(ax_a2.get_yticklabels(), visible=False)
    panel_label(ax_a1, 'a', x=-0.10, y=1.02)

    handles_lat = [Line2D([0],[0], color=COL[g], lw=1.75, label=FULL_LABELS[g]) for g in ['Polar','Mid-lat','Equatorial']]
    handles_style = [Line2D([0],[0], color=COL['Dark'], lw=1.2, ls='-', label='GPS'),
                     Line2D([0],[0], color=COL['Dark'], lw=1.2, ls=(0,(3.2,2.0)), label='Galileo')]
    # Keep legend inside panel a, in the lower part of the left ASD panel.
    leg = ax_a1.legend(handles=handles_lat + handles_style, ncol=5, loc='lower left',
                       bbox_to_anchor=(0.01, 0.035), frameon=True, handlelength=1.55,
                       columnspacing=0.86, borderaxespad=0.0, borderpad=0.22,
                       labelspacing=0.25)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('none')
    leg.get_frame().set_alpha(0.82)

    # Panel b: event summary, with conclusion-forward subpanel titles.
    ax_b1 = fig.add_subplot(gs[1, 0:7])
    ax_b2 = fig.add_subplot(gs[1, 8:15])
    ax_b3 = fig.add_subplot(gs[1, 16:23])

    rate = rate.set_index('lat_group').loc[ORDER].reset_index()
    x = np.arange(len(ORDER))
    y = rate['event_fraction_percent'].to_numpy()
    lo, hi = wilson_ci(rate['events'], rate['total_windows'])
    yerr = np.vstack([y-lo, hi-y])
    ax_b1.bar(x, y, color=[COL[g+'_fill'] for g in ORDER], edgecolor=[COL[g] for g in ORDER],
              linewidth=0.75, alpha=0.97, width=0.56, zorder=3)
    ax_b1.errorbar(x, y, yerr=yerr, fmt='none', ecolor=COL['Dark'], elinewidth=0.8,
                   capsize=2.5, capthick=0.8, zorder=4)
    for xi, yi, hi_val, ev, n in zip(x, y, hi, rate['events'], rate['total_windows']):
        ax_b1.text(xi, hi_val + 1.3, f'{int(ev)}/{int(n)}', ha='center', va='bottom', fontsize=6.4, color=COL['Text'])
    ax_b1.set_xticks(x)
    ax_b1.set_xticklabels([LABELS[g] for g in ORDER])
    ax_b1.set_ylabel('Event fraction (%)', labelpad=1.5)
    ax_b1.set_ylim(0, 48)
    ax_b1.set_title('Polar events are more common', pad=3)
    clean(ax_b1)
    panel_label(ax_b1, 'b', x=-0.13, y=1.02)

    draw_violin(ax_b2, events, 'signed_p95_p05_width_db', 'P95-P05 width (dB)')
    ax_b2.set_ylim(0, 11.0)
    ax_b2.set_title('Polar events are broader', pad=3)

    draw_violin(ax_b3, events, 'dominant_frequency_hz', 'Dominant frequency (Hz)', log=True)
    ax_b3.set_ylim(0.008, 0.60)
    ax_b3.set_yticks([0.01, 0.03, 0.1, 0.3])
    ax_b3.get_yaxis().set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
    ax_b3.set_title('Polar variations are faster', pad=3)

    # Panel c: SZA-defined day-night event fraction.
    ax_c = fig.add_subplot(gs[2, 0:7])
    sub_order = ['All windows', 'Polar only']
    dn_order = ['Day', 'Night']
    x0 = np.arange(len(sub_order))
    bw = 0.34
    for k, dn in enumerate(dn_order):
        vals = []
        ylo = []
        yhi = []
        labels = []
        for sub in sub_order:
            r = daynight[(daynight['subset'].eq(sub)) & (daynight['daynight'].eq(dn))].iloc[0]
            vals.append(r['event_rate_percent'])
            ylo.append(r['event_rate_percent'] - r['event_rate_ci_low_percent'])
            yhi.append(r['event_rate_ci_high_percent'] - r['event_rate_percent'])
            labels.append(f"{int(r['event_windows'])}/{int(r['total_windows'])}")
        xpos = x0 + (k-0.5)*bw
        ax_c.bar(xpos, vals, width=bw*0.88, color=COL[dn+'_fill'], edgecolor=COL[dn],
                 linewidth=0.75, alpha=0.98, zorder=3, label=dn)
        ax_c.errorbar(xpos, vals, yerr=np.vstack([ylo, yhi]), fmt='none',
                      ecolor=COL['Dark'], elinewidth=0.75, capsize=2.0, capthick=0.75, zorder=4)
        for xi, yi, hi_extra, lab in zip(xpos, vals, yhi, labels):
            ax_c.text(xi, yi + hi_extra + 1.4, lab, ha='center', va='bottom', fontsize=5.8, color=COL['Text'])
    ax_c.set_xticks(x0)
    ax_c.set_xticklabels(['All windows', 'Polar only'])
    ax_c.set_ylabel('Event fraction (%)', labelpad=1.5)
    ax_c.set_ylim(0, 58)
    ax_c.set_title('Night events are more common', pad=3)
    ax_c.legend(frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0.02, 0.985),
                handlelength=1.0, columnspacing=0.8, borderaxespad=0)
    clean(ax_c)
    panel_label(ax_c, 'c', x=-0.13, y=1.02)

    # Panel d: heatmap without a colour bar; values are printed in each cell.
    ax_d = fig.add_subplot(gs[2, 8:15])
    row_order = ['Topside', 'F2', 'F1', 'E', 'D']
    col_order = ['Equatorial', 'Mid-lat', 'Polar']
    mat = np.full((len(row_order), len(col_order)), np.nan)
    labels = [['' for _ in col_order] for __ in row_order]
    for _, r in alt.iterrows():
        if r['height_layer'] in row_order and r['lat_group'] in col_order:
            i = row_order.index(r['height_layer'])
            j = col_order.index(r['lat_group'])
            if r['total_windows'] > 0 and np.isfinite(r['event_fraction_percent']):
                mat[i, j] = r['event_fraction_percent']
                labels[i][j] = f"{int(round(r['event_fraction_percent']))}%\n{int(r['events'])}/{int(r['total_windows'])}"
            else:
                labels[i][j] = '–'
    cmap = LinearSegmentedColormap.from_list(
        'soft_yorred_refined', ['#FFF8DC', '#F7E7B6', '#F3C977', '#EE8C62', '#D85252']
    )
    ax_d.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax_d.set_xticks(np.arange(len(col_order)))
    ax_d.set_xticklabels(['Equatorial', 'Mid-lat', 'Polar'])
    ax_d.set_yticks(np.arange(len(row_order)))
    ax_d.set_yticklabels(row_order)
    ax_d.set_title('F-region/topside occurrence', pad=3)
    panel_label(ax_d, 'd', x=-0.13, y=1.02)
    ax_d.set_xticks(np.arange(-.5, len(col_order), 1), minor=True)
    ax_d.set_yticks(np.arange(-.5, len(row_order), 1), minor=True)
    ax_d.grid(which='minor', color='white', linestyle='-', linewidth=1.0)
    ax_d.tick_params(which='minor', bottom=False, left=False)
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            val = mat[i, j]
            ax_d.text(j, i, labels[i][j], ha='center', va='center', fontsize=6.2, color='#111827', linespacing=1.05)
    for spine in ax_d.spines.values():
        spine.set_visible(True)
        spine.set_color(COL['TableEdge'])
        spine.set_linewidth(0.55)

    # Panel e: table.
    ax_e = fig.add_subplot(gs[2, 16:23])
    draw_signal_table(ax_e, sig)
    panel_label(ax_e, 'e', x=-0.13, y=1.02)

    fig.savefig(OUTBASE.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    make_figure()
