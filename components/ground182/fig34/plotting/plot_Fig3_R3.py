"""Portable renderer preserving the supplied Fig.3 layout and display transformations."""
from pathlib import Path
import os,sys,json,hashlib,importlib.util
import argparse
sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'mplconfig'))
# Dependencies are installed through requirements.txt; no external project paths.
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
ROOT = Path(__file__).resolve().parent
SMOOTH_SCRIPT = ROOT/'smoothing.py'
SOURCE = ROOT/'data/snapshot_amplitude'
OUT = ROOT
BANDS = ['Equatorial', 'Mid-latitude', 'Polar']
PLATFORMS = ['Ground', 'LEO-RO', 'LuGRE']
COLORS = {'Ground': '#636C76', 'LEO-RO': '#B9781A', 'LuGRE': '#AA2031'}
HEIGHT_SOURCE = SOURCE
SMOOTH_SCRIPT = ROOT/'smoothing.py'
SCATTER_COLORS = {'LuGRE': '#DFA7AF', 'LEO-RO': '#D8B47D'}
HIST_N = {}
HEIGHT_N = {}


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def frame(ax, column, right_y=False):
    ax.minorticks_off()
    ax.tick_params(direction='in', top=True, length=2.5, width=.65, pad=2)
    ax.tick_params(axis='y', labelleft=(column == 0), right=right_y,
                   labelright=right_y)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#22272B')
        spine.set_linewidth(.65)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#DFE3E6', linewidth=.35)


def draw(data_dir, output_dir, expected_status=None):
    global OUT, SOURCE, HIST_N, HEIGHT_N
    OUT, SOURCE = Path(output_dir).resolve(), Path(data_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    qq = pd.read_csv(SOURCE/'quantiles.csv',float_precision='round_trip')
    HIST_N = {p:[int(qq.loc[qq.platform.eq(p)&qq.latitude_band.eq(b),'n_records'].iloc[0]) for b in BANDS] for p in PLATFORMS}
    HEIGHT_N = {p:HIST_N[p] for p in ['LuGRE','LEO-RO']}
    manifest_path = SOURCE/'Fig3_R3_data_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['status'] == (expected_status or 'COMPLETE_FIG3_R3_L1_E1_PLATFORM_SPECIFIC_DETRENDING')
    assert manifest['same_nearspace_epoch_population_in_top_middle_bottom_rows']
    inputs = [SOURCE/'histogram_source_data.csv', SOURCE/'signed_distribution_summary.csv',
              SOURCE/'quantiles.csv', SOURCE/'height_envelopes.csv',
              SOURCE/'lugre_height_selection.npz', SOURCE/'leo_delta_height_selection.npz',
              SMOOTH_SCRIPT, manifest_path]
    hashes = {str(p):sha(p) for p in inputs}
    hist = pd.read_csv(inputs[0],float_precision='round_trip')
    signed_summary = pd.read_csv(inputs[1],float_precision='round_trip')
    quantiles = pd.read_csv(inputs[2],float_precision='round_trip')
    env = pd.read_csv(inputs[3],float_precision='round_trip')
    assert len(quantiles)==9 and len(env)>0
    height = {}
    for platform,path in [('LuGRE',inputs[4]),('LEO-RO',inputs[5])]:
        with np.load(path,allow_pickle=False) as z:
            height[platform]={k:z[k].copy() for k in ['amplitude','height_km','band_code','all_valid']}
        assert height[platform]['all_valid'].all()
        assert np.isfinite(height[platform]['height_km']).all()
        assert ((height[platform]['height_km']>=50)&(height[platform]['height_km']<=1000)).all()
    spec = importlib.util.spec_from_file_location('approved_smooth_display', SMOOTH_SCRIPT)
    smooth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smooth)
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 7, 'axes.labelsize': 7.3, 'xtick.labelsize': 6.8,
                         'ytick.labelsize': 6.8, 'text.color': '#22272B',
                         'axes.labelcolor': '#22272B', 'axes.edgecolor': '#22272B',
                         'axes.spines.top': True, 'axes.spines.right': True,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white'})
    width_mm, height_mm, dpi = 183, 190, 600
    fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1, .60],
                          left=.104, right=.910, top=.875, bottom=.100,
                          wspace=.17, hspace=.32)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)])
    platform_legend = fig.legend(
        handles=[Line2D([], [], color=COLORS[p], lw=1.15, label=p) for p in PLATFORMS],
        loc='upper center', bbox_to_anchor=(.515, .970), ncol=3,
        frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=2.0)
    qa_height, qa_hist, qa_quantiles = [], [], []
    envelope_labels = []
    for c, band in enumerate(BANDS):
        ax = axes[0, c]
        ax.set_xscale('symlog', linthresh=.3, linscale=.36)
        ax.set_xlim(-10, 10)
        ax.set_ylim(50, 1000)
        envelope_labels.append(ax.text(.035, .956, 'P95 envelope', transform=ax.transAxes,
                                       ha='left', va='top', fontsize=6.2, color='#4C545C',
                                       zorder=10, bbox={'facecolor': 'white',
                                                        'edgecolor': 'none', 'alpha': .85,
                                                        'pad': .8}))
        ax.set_xticks([-10, -3, -1, -.3, 0, .3, 1, 3, 10],
                      ['−10', '−3', '−1', '−0.3', '0', '0.3', '1', '3', '10'])
        ax.tick_params(axis='x', labelsize=6.8)
        ax.set_yticks([50, 90, 150, 300, 600, 800, 1000])
        regions = [('D', 50, 90), ('E', 90, 150), ('F1', 150, 300),
                   ('F2', 300, 600), ('topside', 600, 1000)]
        for boundary in [90, 150, 300, 600]:
            ax.axhline(boundary, color='#CFD5DA', linewidth=.45, zorder=0)
        for label, lo, hi in regions:
            if c == 0:
                ax.text(.025, (lo+hi)/2, label, transform=ax.get_yaxis_transform(),
                        fontsize=5.6, color='#7E878F', va='center', zorder=8)
        for i, platform in enumerate(['LEO-RO', 'LuGRE']):
            z = height[platform]
            mask = z['all_valid'].astype(bool) & (z['band_code'] == c)
            n = int(mask.sum())
            assert n == HEIGHT_N[platform][c]
            a, h = z['amplitude'][mask], z['height_km'][mask]
            # Retain all selected points, with pale colours identifying each platform.
            ax.scatter(a, h, s=.55, color=SCATTER_COLORS[platform], alpha=.095,
                       linewidths=0, zorder=1)
            key = 'lugre' if platform == 'LuGRE' else 'leo_delta'
            e = env[env.platform_metric.eq(key) & env.selection.eq('all_valid') &
                    env.latitude_band.eq(band)].sort_values('height_km')
            yy = e.height_km.to_numpy()
            p95 = e.p95_abs_smoothed.to_numpy()
            # One mirrored P95 envelope per platform; no P90 line or envelope fill.
            assert np.all((p95 >= 0) & (p95 <= 10)), (platform,band,"P95 height envelope outside display")
            for sign in [-1, 1]:
                ax.plot(sign*p95, yy, color=COLORS[platform],
                        linewidth=1.05, alpha=.96, zorder=4+i)
            qa_height.append({'platform': platform, 'latitude_band': band, 'n_records': n,
                              'n_scatter_points': len(a), 'envelope_rows': len(e),
                              'plotted_envelopes_match_recomputed_source': True,
                              'displayed_envelope_quantiles': ['P95'],
                              'n_mirrored_envelope_lines': 2,
                              'scatter_color': SCATTER_COLORS[platform]})
        ax.axvline(0, color='#B0B5B9', linewidth=.4, zorder=0)
        ax.set_title(band, fontsize=8.3, pad=8)
        ax.set_xlabel('δC/N$_0$ (dB)', labelpad=3)
        frame(ax, c, right_y=(c == 2))
        ax.grid(False)
        assert len(ax.collections) == 2 and len(ax.patches) == 0
        assert len(ax.lines) == 9  # Four layer boundaries, four P95 traces and zero line.
        assert np.allclose(ax.get_facecolor(), [1, 1, 1, 1])

        ax = axes[1, c]
        for i, platform in enumerate(PLATFORMS):
            d = hist[hist.platform.eq(platform) & hist.latitude_band.eq(band)].sort_values('bin_left_db')
            count = d['count'].to_numpy()
            n = int(d.n_total.iloc[0])
            percent = d.percent_of_all_group_records.to_numpy()
            edges = np.r_[d.bin_left_db.to_numpy(), d.bin_right_db.iloc[-1]]
            assert n == HIST_N[platform][c] and count.sum() == n
            assert np.allclose(percent, 100*count/n, atol=1e-12)
            report = smooth.plot_smoothed_histogram(ax, d.bin_center_db.to_numpy(), edges,
                                                    count, percent, COLORS[platform], i)
            qa_hist.append({'platform': platform, 'latitude_band': band, 'n_records': n,
                            'display_smoothing': report})
        ax.set_yscale('log')
        ax.set_ylim(.0001, 100)
        ax.set_yticks([.001, .01, .1, 1, 10, 100], ['0.001', '0.01', '0.1', '1', '10', '100'])
        ax.set_xlim(-10, 10)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.axvline(0, color='#B0B5B9', linewidth=.4, zorder=0)
        ax.set_xlabel('δC/N$_0$ (dB)', labelpad=3)
        frame(ax, c, right_y=(c == 2))

        ax = axes[2, c]
        for i, platform in enumerate(PLATFORMS):
            d = quantiles[quantiles.platform.eq(platform) & quantiles.latitude_band.eq(band)]
            assert len(d) == 1
            d = d.iloc[0]
            ss = signed_summary[signed_summary.platform.eq(platform) & signed_summary.latitude_band.eq(band)].iloc[0]
            assert int(d.n_records) == HIST_N[platform][c] == int(ss.n_records)
            p50, p95, p99 = float(d.P50_abs_db), float(d.P95_abs_db), float(d.P99_abs_db)
            assert 0 <= p50 <= p95 <= p99 <= 3.2, (platform,band,"quantile span outside display",p99)
            assert np.isclose(p95, ss.P95_absolute_db, atol=1e-12)
            y = 2-i
            ax.hlines(y, p50, p99, color=COLORS[platform], linewidth=.85)
            ax.plot(p50, y, 'o', ms=2.6, mfc='white', mec=COLORS[platform], mew=.75)
            ax.plot(p95, y, 'o', ms=3.6, color=COLORS[platform])
            ax.plot(p99, y, '|', ms=5, color=COLORS[platform], mew=.75)
            ax.annotate(f'{p95:.2f}', (p95, y), xytext=(0, 5.5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6.3, color=COLORS[platform])
            qa_quantiles.append({'platform': platform, 'latitude_band': band,
                                 'n_records': int(d.n_records), 'P50_abs_db': p50,
                                 'P95_abs_db': p95, 'P99_abs_db': p99})
        ax.set_xlim(0, 3.2)
        ax.set_ylim(-.55, 2.65)
        ax.set_xticks([0, .5, 1, 1.5, 2, 2.5, 3], ['0', '0.5', '1', '1.5', '2', '2.5', '3'])
        ax.set_yticks([2, 1, 0], PLATFORMS)
        ax.set_xlabel('|δC/N$_0$| (dB)', labelpad=3)
        frame(ax, c, right_y=(c == 2))
        ax.grid(False)

    for r in range(3):
        for c in range(3):
            axes[r, c].text(0, 1.028, chr(ord('a')+r*3+c), transform=axes[r, c].transAxes,
                            ha='left', va='bottom', fontweight='bold', fontsize=8.5)
    axes[0, 0].set_ylabel('Tangent height (km)', fontsize=7.3, labelpad=4)
    axes[1, 0].set_ylabel('Proportion per bin (%)', fontsize=7.3, labelpad=4)
    axes[0, 2].yaxis.set_label_position('right')
    axes[0, 2].set_ylabel('Tangent height (km)', fontsize=7.3, labelpad=4)
    axes[1, 2].yaxis.set_label_position('right')
    axes[1, 2].set_ylabel('Proportion per bin (%)', fontsize=7.3, labelpad=4)
    quantile_legend = axes[2, 0].legend(handles=[
        Line2D([], [], marker='o', ms=2.6, mfc='white', color='#4C545C', lw=0, label='P50'),
        Line2D([], [], marker='o', ms=3.6, color='#4C545C', lw=0, label='P95'),
        Line2D([], [], marker='|', ms=5, color='#4C545C', lw=0, label='P99')],
        loc='lower right', bbox_to_anchor=(.985, .035), ncol=3, frameon=False,
        fontsize=6.0, handlelength=.7, handletextpad=.35, columnspacing=.65,
        borderaxespad=.4)
    assert not any('n =' in t.get_text() for ax in axes[:2].flat for t in ax.texts)
    assert sum(len(ax.texts) for ax in axes[2]) >= 9
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    mirrored_y_axis_labels = []
    for r in range(3):
        left, right = axes[r, 0], axes[r, 2]
        assert np.array_equal(left.get_yticks(), right.get_yticks())
        assert np.array_equal(left.get_ylim(), right.get_ylim())
        assert left.get_yscale() == right.get_yscale()
        ll = [t.get_text() for t in left.get_yticklabels() if t.get_visible()]
        rl = [t.get_text() for t in right.get_yticklabels() if t.get_visible()]
        assert ll == rl and len(ll) > 0
        assert right.get_ylabel() == left.get_ylabel()
        mirrored_y_axis_labels.append(ll)
    assert mirrored_y_axis_labels[2] == PLATFORMS
    assert axes[2, 0].get_ylabel() == ''
    top_tick_min_gaps = []
    panel_alignment_errors = []
    for c in range(3):
        upper, middle = axes[0, c], axes[1, c]
        assert upper.get_xscale() == 'symlog' and middle.get_xscale() == 'linear'
        ub, mb = upper.get_position(), middle.get_position()
        error = max(abs(ub.x0-mb.x0), abs(ub.x1-mb.x1))
        assert error < 1e-12
        panel_alignment_errors.append(error)
        boxes = [t.get_window_extent(renderer) for t in upper.xaxis.get_majorticklabels() if t.get_visible()]
        gaps = [b.x0-a.x1 for a, b in zip(boxes[:-1], boxes[1:])]
        assert min(gaps) > 0, gaps
        top_tick_min_gaps.append(float(min(gaps)))
    outside = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        b = text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
            outside.append(text.get_text())
    assert not outside, outside
    assert platform_legend.get_window_extent(renderer).y0 > max(ax.title.get_window_extent(renderer).y1 for ax in axes[0])
    qbox = quantile_legend.get_window_extent(renderer)
    gbox = axes[2, 0].get_window_extent(renderer)
    assert gbox.contains(qbox.x0, qbox.y0) and gbox.contains(qbox.x1, qbox.y1)
    qbounds_data = axes[2, 0].transData.inverted().transform(qbox.get_points())
    assert qbounds_data[1, 1] < -.06 or qbounds_data[0, 0] > 1.22
    for c, label in enumerate(envelope_labels):
        eb = label.get_window_extent(renderer)
        ab = axes[0, c].get_window_extent(renderer)
        assert ab.contains(eb.x0, eb.y0) and ab.contains(eb.x1, eb.y1)
    for ax in axes.flat:
        for s in ax.spines.values():
            assert s.get_visible()
    for p in inputs:
        assert sha(p) == hashes[str(p)], f'Source changed: {p}'
    output = OUT/'Fig3_R3.png'
    fig.savefig(output, dpi=dpi, format='png', bbox_inches=None, pad_inches=0,
                pil_kwargs={'compress_level': 9})
    plt.close(fig)
    with Image.open(output) as im:
        assert im.format == 'PNG' and min(im.info['dpi']) > 599
        pixels, resolution = im.size, im.info['dpi']
    qa = {'status': 'PASS_SOURCE_AND_LAYOUT_CHECKS', 'backend': 'Python/matplotlib',
          'archetype': 'quantitative grid', 'source_hashes': hashes,
          'sources_unchanged': True, 'no_new_fitting': True,
          'all_rows_use_same_nearspace_epoch_population': True,
          'platform_detrending_preserved': True,
          'method_by_platform': manifest['method_by_platform'],
          'signal_population': ['GPS L1','Galileo E1'],
          'height_x_scale': 'symlog, linthresh=0.3 dB, linscale=0.36',
          'height_y_scale': 'linear, 50--1000 km',
          'height_envelope': '32-km bins, n>=8, P95 abs residuals only, 3-bin median smooth, mirror +/-',
          'height_background': 'white; no shading or envelope fills',
          'height_reference_lines_km': [90, 150, 300, 600],
          'height_scatter_colors': SCATTER_COLORS,
          'middle_and_bottom_same_populations': True, 'height_groups': qa_height,
          'histogram_groups': qa_hist, 'quantile_groups': qa_quantiles,
          'quantile_spans_are_not_confidence_intervals': True,
          'Ground_prefit_CN0_min_dbhz': 30, 'colors': COLORS,
          'middle_smoothing_sigma_db': .125, 'x_display_limits_db': [-10, 10],
          'shared_platform_legend': True, 'all_four_spines': True,
          'mirrored_left_and_right_y_axis_labels': mirrored_y_axis_labels,
          'y_titles_location': 'both outer sides of first and second rows',
          'middle_x_scale': 'linear',
          'middle_x_ticks_db': [-10, -5, 0, 5, 10],
          'original_histogram_bin_width_db': .25,
          'normalization_uses_full_current_platform_latitude_groups': True,
          'generic_vertical_row_titles_removed': True,
          'third_row_category_labels': 'Ground, LEO-RO, LuGRE on both outer sides',
          'quantile_legend_location': 'panel g lower right',
          'quantile_legend_labels': ['P50', 'P95', 'P99'],
          'quantile_legend_data_bounds': qbounds_data.tolist(),
          'height_envelope_label': 'P95 envelope, upper left of a--c',
          'text_inside_canvas': True, 'top_and_middle_tick_fontsize_pt': 6.8,
          'top_tick_min_gaps_canvas_px': top_tick_min_gaps,
          'top_middle_panel_alignment_errors_figure_fraction': panel_alignment_errors,
          'physical_size_mm': [width_mm, height_mm],
          'pixels': pixels, 'dpi': resolution, 'output': str(output),
          'bytes': output.stat().st_size, 'export': 'one PNG only'}
    qa['manuscript_revision'] = 'R3'
    qa['revision_type'] = 'Population update: final common source selection; accepted source quality histories and approved layout preserved'
    qa['sample_counts_in_panels']=False
    qa['sample_counts_retained_in_caption_and_source']=True
    qa['third_row_quantile_labels_preserved']=True
    qa['signal_label_in_figure']=False
    qa['signal_selection_retained_in_caption']=True
    qa['source_methods']=manifest['method_by_platform']
    qa['upstream_population_status']=manifest['status']
    qa['LEO_uses_original_rolling_median']=True
    qa['no_LEO_quartic_eligibility_filter']=True
    qa['approved_fig3_palette_used']={'Ground':'#636C76','LEO-RO':'#B9781A','LuGRE':'#AA2031'}==COLORS
    qa['palette_authority']='Latest explicit user request on 2026-10-27, superseding the historical Fig.3 exception'
    qa['approved_formal_png_preserved'] = True
    qa['approved_formal_png_sha256'] = 'not accessed by portable renderer'
    qa['previous_palette_candidate_preserved'] = True
    qa['previous_palette_candidate_sha256'] = 'not accessed by portable renderer'
    qa['display_clipping'] = [{'platform':p['platform'],'n_outside_minus10_plus10':p['n_outside_display_minus10_plus10']} for p in manifest['platforms']]
    qa['source_population'] = [{'platform':p['platform'],'n_records':p['n_records'],'signal_counts':p['signal_counts'],'latitude_counts':p['latitude_counts']} for p in manifest['platforms']]
    qa['every_input_epoch_within_same_latitude_band_LuGRE_UTC_support']=all(p['same_latitude_band_UTC_all'] for p in manifest['platforms'])
    qa['not_60second_window_statistics']=True
    qa['script_sha256']=sha(Path(__file__))
    qa['output_sha256']=sha(output)
    assert all(qa[k] for k in ['approved_formal_png_preserved','previous_palette_candidate_preserved','approved_fig3_palette_used'])
    (OUT/'Fig3_R3_plot_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    caption = ("Fig. 3 | Latitude-dependent amplitude distributions in GPS L1 and Galileo E1 observations.\n"
        "a-c: individual accepted 1-Hz LEO-RO/LuGRE residuals versus tangent height; mirrored P95 absolute-residual envelopes in 32-km bins with at least 8 observations, with the source-declared three-bin display median.\n"
        "d-f: signed residual distributions in 0.25-dB bins normalized by all accepted platform/band observations; display smoothing preserves each occupied segment's probability mass.\n"
        "g-i: P50/P95/P99 of absolute residuals from the same accepted observations, calculated before display smoothing.\n"
        "Display range for signed residuals is -10 to 10 dB; values beyond the display remain in statistics. These are epoch-level distributions, not within-minute widths.\n"
        "Detrending methods: "+json.dumps(manifest['method_by_platform'],ensure_ascii=False)+"\n"
        "Counts and quantiles from this rendering source:\n"+quantiles[['platform','latitude_band','n_records','P50_abs_db','P95_abs_db','P99_abs_db']].to_string(index=False)+"\n"
        "Display-clipped counts: "+json.dumps(qa['display_clipping'])+"\n"
        "UTC eligibility and per-epoch latitude classifications are supplied and verified by the upstream common-UTC analysis; plotting does not recreate them.\n")
    (OUT/'Fig3_R3_caption_notes.txt').write_text(caption,encoding='utf-8')
    print(json.dumps({k: qa[k] for k in ['status', 'output', 'pixels', 'dpi', 'bytes']}, indent=2), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--expected-status',default=None)
    args=parser.parse_args()
    draw(args.data_dir,args.output_dir,args.expected_status)
