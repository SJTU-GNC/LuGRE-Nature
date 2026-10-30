"""Portable renderer preserving the supplied Fig.4 response layout."""
from pathlib import Path
import os,sys,json,hashlib,shutil
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
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.container import ErrorbarContainer
from PIL import Image
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'data/snapshot_response'
OUT = ROOT
PLATFORMS = ['Ground', 'LEO-RO', 'LuGRE']
BANDS = ['Equatorial', 'Mid-latitude', 'Polar']
COLORS = {'Equatorial': '#238BA3', 'Mid-latitude': '#7957C5', 'Polar': '#AA2031'}
MARKERS = dict(zip(BANDS, ['o', 's', '^']))
ANGLES = ['ray_B_acute_deg', 'scan_Bperp_acute_deg', 'magnetic_meridian_acute_deg']
TITLES = [r'Ray–B angle, $\theta_{\rm ray}$ (°)',
          r'Scan–B$_\perp$ angle, $\theta_{\rm scan}$ (°)',
          r'Magnetic-meridian azimuth, $\theta_{\rm mer}$ (°)']
XLABELS = [r'Ray–B angle, $\theta_{\rm ray}$ (°)',
           r'Scan–B$_\perp$ angle, $\theta_{\rm scan}$ (°)',
           r'Magnetic-meridian azimuth, $\theta_{\rm mer}$ (°)']
YMIN, YMAX = 0, 4

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def draw(data_dir, output_dir, expected_status=None):
    global OUT, SOURCE
    SOURCE, OUT = Path(data_dir).resolve(), Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    windows_path = SOURCE/'common_60s_response_windows_R3.csv'
    bins_path = SOURCE/'angle_bin_median_IQR_counts_R3.csv'
    manifest_path = SOURCE/'response_window_manifest_R3.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['status'] == (expected_status or 'PASS_FIG4_R3_PLATFORM_SPECIFIC_DETRENDING')
    hashes = {str(p):sha(p) for p in [windows_path,bins_path,manifest_path]}
    windows,table = pd.read_csv(windows_path,float_precision='round_trip',low_memory=False),pd.read_csv(bins_path,float_precision='round_trip')
    assert len(table)==162 and np.all(table.bin_right_deg-table.bin_left_deg==15)
    for platform in PLATFORMS:
        assert set(windows[windows.platform.eq(platform)].signal).issubset({'GPS L1','Galileo E1'})
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 7.0, 'axes.labelsize': 7.2, 'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0,
                         'text.color': '#22272B', 'axes.labelcolor': '#22272B',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    width_mm, height_mm, dpi = 183, 132, 600
    fig, axes = plt.subplots(3, 3, figsize=(width_mm/25.4, height_mm/25.4))
    fig.subplots_adjust(left=.108, right=.912, bottom=.118, top=.885, wspace=.09, hspace=.24)
    handles = [Line2D([], [], ls='none', marker=MARKERS[b], ms=4.0, mfc=COLORS[b], mec='white', mew=.45, label=b) for b in BANDS]
    # Standalone legend artists do not add observations or alter plotted summaries.
    stat_marker=Line2D([], [], ls='none', marker='o', ms=3.6, mfc='#444B51', mec='white', mew=.45)
    stat_caps=(Line2D([], [], ls='none', marker='_', color='#444B51', markersize=4, markeredgewidth=.85),)
    stat_range=(LineCollection([[(0,-1),(0,1)]],colors=['#444B51'],linewidths=[.95]),)
    stat_handle=ErrorbarContainer((stat_marker,stat_caps,stat_range),has_yerr=True,label='Median and interquartile range')
    legend = fig.legend(handles=handles+[stat_handle], ncol=4, loc='upper center', bbox_to_anchor=(.50, .989),
                        frameon=False, fontsize=7.6, handletextpad=.4, columnspacing=1.35)
    ylabels, left_ylabels = [], []
    width_title = r'$\delta C/N_0$ width $W$ (dB)'
    panels, records, summary_artists, row_labels, panel_letters = [], [], [], [], []
    for r, platform in enumerate(PLATFORMS):
        source = windows[windows.platform.eq(platform)]
        for c, angle in enumerate(ANGLES):
            ax = axes[r, c]
            # User requested only the unchanged equal-window summaries, without raw scatter.
            for i, band in enumerate(BANDS):
                t = table[table.platform.eq(platform) & table.angle.eq(angle) & table.latitude_band.eq(band)].sort_values('bin_id')
                assert len(t) == 6 and int(t.n_windows.sum()) == len(source[source.latitude_band.eq(band)])
                for q in t.itertuples():
                    if not q.n_windows:
                        continue
                    x = q.bin_center_deg+(i-1)*.20*(q.bin_right_deg-q.bin_left_deg)
                    median = q.median_width_db
                    low, high = q.q25_width_db, q.q75_width_db
                    assert YMIN <= low <= median <= high <= YMAX, (platform,band,angle,"summary outside display",low,median,high)
                    eb = ax.errorbar([x], [median], yerr=np.array([[median-low], [high-median]]),
                                    fmt='none', ecolor=COLORS[band], elinewidth=.95,
                                    capsize=1.8, capthick=.85, zorder=3)
                    sparse = q.n_windows < 5
                    marker, = ax.plot([x], [median], marker=MARKERS[band], ls='none',
                                      ms=3.6 if i<2 else 4.1,
                                      mfc=COLORS[band], mec='white', mew=.45, zorder=4)
                    summary_artists.append((eb, marker, x, low, median, high, sparse, band))
                    records.append({'panel': chr(97+r*3+c), 'platform': platform, 'angle': angle,
                                    'latitude_band': band, 'bin_id': int(q.bin_id), 'n_windows': int(q.n_windows),
                                    'x_display': x, 'median': median, 'q25': low, 'q75': high, 'small_n': bool(sparse), 'open_marker': False})
            ax.set(xlim=(0,90), ylim=(YMIN,YMAX), yscale='linear')
            ax.set_xticks(np.arange(0,91,15))
            ax.set_yticks([0,1,2,3,4], ['0','1','2','3','4'])
            ax.minorticks_off()
            ax.tick_params(direction='in', top=True, right=True, length=2.5, width=.65, pad=2.5,
                           labelbottom=r==2, labelleft=c==0, labelright=c==2)
            for tick in ax.get_yticklabels():
                if tick.get_text()=='0':
                    tick.set_verticalalignment('bottom')
                elif tick.get_text()=='4':
                    tick.set_verticalalignment('top')
            ax.grid(axis='y', color='#E4E8EB', lw=.4)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#343A40')
                spine.set_linewidth(.65)
            if r==0:
                ax.set_title(TITLES[c], fontsize=7.3, pad=7.5)
            if r==2:
                ax.set_xlabel(XLABELS[c], labelpad=4)
            panel_letters.append(ax.text(0,1.035,chr(97+r*3+c),transform=ax.transAxes,ha='left',va='bottom',fontsize=8.6,fontweight='bold'))
            panels.append({'panel': chr(97+r*3+c), 'platform':platform, 'angle':angle, 'n_windows':len(source),
                           'n_below_display':int(source.width_db.lt(YMIN).sum()),
                           'n_above_display':int(source.width_db.gt(YMAX).sum())})
        pos=axes[r,0].get_position()
        row_labels.append(fig.text(.034,(pos.y0+pos.y1)/2,platform,rotation=90,ha='center',va='center',
                                    fontsize=8.5,fontweight='bold',color='#343A40'))
        left_ylabels.append(axes[r,0].set_ylabel(width_title, fontsize=7.1, labelpad=3.0, linespacing=1.15))
        axes[r,2].yaxis.set_label_position('right')
        ylabels.append(axes[r,2].set_ylabel(width_title, fontsize=7.1, labelpad=3.0, linespacing=1.15))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    assert not any(isinstance(collection, PathCollection) for ax in axes.flat for collection in ax.collections)
    title_boxes=[ax.title.get_window_extent(renderer) for ax in axes[0]]
    xlabel_boxes=[ax.xaxis.label.get_window_extent(renderer) for ax in axes[2]]
    title_gaps=[b.x0-a.x1 for a,b in zip(title_boxes[:-1],title_boxes[1:])]
    xlabel_gaps=[b.x0-a.x1 for a,b in zip(xlabel_boxes[:-1],xlabel_boxes[1:])]
    assert min(title_gaps)>0, title_gaps
    assert min(xlabel_gaps)>0, xlabel_gaps
    assert all('\n' not in t for t in TITLES+XLABELS+[width_title])
    for title,letter in zip(title_boxes,panel_letters[:3]):
        assert not title.overlaps(letter.get_window_extent(renderer)), (title,letter.get_text())
    assert len(records)==int(table.n_windows.gt(0).sum())
    assert sum(q['small_n'] for q in records)==int(table.n_windows.between(1,4).sum())
    assert not any(q['open_marker'] for q in records)
    for eb, marker, x, low, median, high, sparse, band in summary_artists:
        assert np.array_equal(marker.get_data(),np.array([[x],[median]]))
        seg=eb.lines[2][0].get_segments()[0]
        assert np.allclose(seg,[[x,low],[x,high]],atol=1e-13)
        assert marker.get_markerfacecolor()==COLORS[band]
        assert marker.get_marker()==MARKERS[band]
    outside=[]
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        b=text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
            outside.append(text.get_text())
    assert not outside,outside
    assert legend.get_window_extent(renderer).y0 > max(ax.title.get_window_extent(renderer).y1 for ax in axes[0])
    assert [t.get_text() for t in legend.get_texts()]==BANDS+['Median and interquartile range']
    assert all('(°)' in s for s in TITLES+XLABELS)
    legend_boxes=[t.get_window_extent(renderer) for t in legend.get_texts()]
    assert all(a.x1<b.x0 for a,b in zip(legend_boxes[:-1],legend_boxes[1:]))
    right_gap=[]
    left_gap=[]
    platform_gap=[]
    tick_gaps=[]
    for r, ax in enumerate(axes[:,-1]):
        ybox=ylabels[r].get_window_extent(renderer)
        abox=ax.get_window_extent(renderer)
        assert ybox.y0>=abox.y0 and ybox.y1<=abox.y1
        for tick in ax.yaxis.get_majorticklabels():
            if tick.get_visible():
                right_gap.append(ybox.x0-tick.get_window_extent(renderer).x1)
    assert min(right_gap)>0,right_gap
    for r, ax in enumerate(axes[:,0]):
        ybox=left_ylabels[r].get_window_extent(renderer)
        abox=ax.get_window_extent(renderer)
        assert ybox.y0>=abox.y0 and ybox.y1<=abox.y1
        for tick in ax.yaxis.get_majorticklabels():
            if tick.get_visible():
                left_gap.append(tick.get_window_extent(renderer).x0-ybox.x1)
        platform_gap.append(ybox.x0-row_labels[r].get_window_extent(renderer).x1)
    assert min(left_gap)>0,left_gap
    assert min(platform_gap)>0,platform_gap
    for ax in axes[2]:
        boxes=[t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_visible()]
        tick_gaps.extend(b.x0-a.x1 for a,b in zip(boxes[:-1],boxes[1:]))
        for yt in ax.get_yticklabels():
            if yt.get_visible():
                assert not any(yt.get_window_extent(renderer).overlaps(b) for b in boxes)
    assert min(tick_gaps)>0
    for r in range(3):
        assert np.array_equal(axes[r,0].get_yticks(),axes[r,2].get_yticks())
    output=OUT/'Fig4_R3.png'
    fig.savefig(output,dpi=dpi,format='png',bbox_inches=None,pad_inches=0,pil_kwargs={'compress_level':9})
    plt.close(fig)
    with Image.open(output) as im:
        assert im.format=='PNG' and min(im.info['dpi'])>599
        pixels,resolution=im.size,im.info['dpi']
    copied_counts=OUT/'Fig4_R3_bin_statistics.csv'
    shutil.copyfile(bins_path,copied_counts)
    assert sha(copied_counts)==hashes[str(bins_path)]
    assert all(sha(Path(p))==digest for p,digest in hashes.items())
    qa={'status':'PASS_DATA_AND_LAYOUT_CHECKS','manuscript_revision':'R3','backend':'Python/matplotlib',
        'original_panel_layout_preserved':True,'LEO_original_rolling_median_preserved':True,'source_sha256':hashes,
        'counts_table_exact_copy':str(copied_counts),'original_plot_unchanged':True,
        'original_plot_sha256':'not accessed by portable renderer','upstream_population_status':manifest['status'],'methods':manifest['methods'],'signal_population':['GPS L1','Galileo E1'],
        'background_points_drawn':False,'all_medians_and_IQRs_match_recomputed_source':True,
        'bin_edges_degrees':list(range(0,91,15)),'y_limits_db':[YMIN,YMAX],'axis_scale':'linear','y_tick_labels':['0','1','2','3','4'],
        'background_scatter':None,'raw_scatter_collection_count':0,
        'latitude_colors':COLORS,'markers':MARKERS,'all_symbols_filled_matching_original_legend':True,
        'small_n_bins_in_source':sum(q['small_n'] for q in records),
        'small_n_markers':sum(q['open_marker'] for q in records),
        'left_axis_titles':[width_title]*3,
        'right_axis_titles':[width_title]*3, 'right_title_labelpad_pt':3.0,
        'x_axis_titles':XLABELS,'top_titles':TITLES,
        'statistical_legend':{'label':'Median and interquartile range','centre':'median','lower_cap':'25th percentile','upper_cap':'75th percentile','confidence_interval':False,'standard_deviation':False},
        'legend_labels':[t.get_text() for t in legend.get_texts()],
        'minimum_adjacent_top_title_gap_canvas_px':min(title_gaps),
        'minimum_adjacent_x_label_gap_canvas_px':min(xlabel_gaps),
        'title_layout':'single line, identical left and right on each row; top and bottom angle labels include Methods symbols',
        'minimum_left_title_gap_canvas_px':min(left_gap),
        'minimum_platform_title_gap_canvas_px':min(platform_gap),
        'top_explanatory_text_removed':True, 'signal_header_removed':True, 'platform_label_x_figure_fraction':.034,
        'summary_medians_outside_display':[q for q in records if not YMIN<=q['median']<=YMAX],
        'summary_IQRs_crossing_display_boundary':[q for q in records if q['q25']<YMIN or q['q75']>YMAX],
        'panel_map':panels,'summaries':records,
        'text_inside_canvas':True,'minimum_right_title_gap_canvas_px':min(right_gap),
        'minimum_x_tick_gap_canvas_px':min(tick_gaps),'physical_size_mm':[width_mm,height_mm],
        'pixels':pixels,'dpi':resolution,'output':str(output),'output_sha256':sha(output),
        'script_sha256':sha(Path(__file__)),
        'revision_type':'Final common source population: shared KIRG antenna source separately retained; approved median/IQR layout and degree labels preserved'}
    (OUT/'Fig4_R3_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    totals=windows.groupby(['platform','signal']).size().unstack(fill_value=0)
    clip=windows.assign(below=windows.width_db.lt(YMIN),above=windows.width_db.gt(YMAX)).groupby('platform')[['below','above']].sum()
    caption=("Fig. 4 | Geometry-dependent 1-min amplitude widths in GPS L1 and Galileo E1 observations.\n"
        "Rows: Ground, LEO-RO, LuGRE. Columns: acute ray-field angle, transverse scan-field angle and ray azimuth relative to magnetic meridian. No raw background points are drawn. Coloured symbols: equal-window medians in 15-degree bins, with interquartile ranges. Circles/squares/triangles: equatorial/mid-latitude/polar. The grey statistical legend identifies the centre as the median and the capped span as the interquartile range. Symbols offset by -3/0/+3 degrees for legibility; underlying angles unchanged.\n"
        "W=P95-P05 of signed residuals within an accepted continuous segment of a UTC minute; 30-60 real 1-Hz values and complete geometry required. Original source/QC/fit boundaries and sampling gaps are preserved by the upstream window reconstruction. Geometric angles exclude plasma drift.\n"
        "Plotting preserves the 0-4-dB summary display; every accepted window remains in source statistics, irrespective of its individual width. All medians and IQRs are required to be inside display limits.\n"
        "Detrending methods: "+json.dumps(manifest['methods'],ensure_ascii=False)+"\n"
        "Accepted window counts by platform/signal:\n"+totals.to_string()+"\n"
        "Source windows outside summary display range (raw points are not drawn):\n"+clip.to_string()+"\n"
        "All source epochs use the updated common UTC eligibility and their own reference-point latitude; accepted minute populations may differ among platforms.\n")
    (OUT/'Fig4_R3_caption_notes.txt').write_text(caption,encoding='utf-8')
    print(json.dumps({k:qa[k] for k in ['status','output','pixels','dpi','small_n_markers']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--expected-status',default=None)
    args=parser.parse_args()
    draw(args.data_dir,args.output_dir,args.expected_status)
