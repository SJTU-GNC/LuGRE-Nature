"""Reproduce E23 first, then fit E21 with the same measured-only algorithm.

Candidate only: the accepted manuscript residual population is never modified.
Scientific contract: a shows measured C/N0 and one quartic per physical pass
and signal; b shows all candidate residuals; c/d establish occultation geometry.
The original E23 layout is retained, with the user's 20--40 and +/-2 dB zoom.
Only a 600 dpi PNG is exported. Missing C/N0 never enters fit or statistics.
"""
from pathlib import Path
from datetime import timezone
import ast
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
for folder in ['data', 'qa', 'figures', 'captions', 'mplconfig']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(ROOT / 'mplconfig')
sys.path.insert(0, 'D:/Reference/Paper/2026/Nature_LuGRE/Figure/combined_station_global_plus_polar_zooms_package_20260615_1730/_runtime_deps_mpl')
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter

PAPER = Path('D:/Reference/Paper/2026/Nature_LuGRE')
E23_PACKAGE = PAPER / 'Figure/op38_e23_front_arc_example_package_20260616_145440'
FIT = E23_PACKAGE / 'code/GPS_simulator/Build_QuadraticDetrendExample.py'
TEMPLATE = PAPER / 'Figure/extended_data_fig4_wgs84_tangent_height_update_20260701/code/build_op38_e23_nature_style_extract_wgs84_candidate_plot.py'
HELPER = PAPER / 'Figure/op40_e02_package_20260615_whole_arc_cheb4/code/Build_OP40E02Extract.py'
SOURCE = ROOT / 'data/E21_plotted_observations_R3.csv'

def extract(path, names, scope):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if (isinstance(n, ast.ImportFrom) and n.module == '__future__')
             or (isinstance(n, ast.FunctionDef) and n.name in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)

scope = dict(np=np, pd=pd, mpl=mpl, plt=plt, MultipleLocator=MultipleLocator,
             FuncFormatter=FuncFormatter, PANEL_LABEL_X=-.112, PANEL_HEADER_Y=1.026,
             MAP_PANEL_LABEL_X=-.180)
extract(FIT, {'median_abs_deviation', 'binned_median_fit_points', 'fit_poly_trend'}, scope)
extract(TEMPLATE, {'set_panel_header', 'time_segments', 'map_extent',
                   'plot_time_pieces_local', 'geometry_ionosphere_spans', 'merge_time_spans'}, scope)
extract(HELPER, {'style_nature_axis', 'format_lon', 'format_lat', 'draw_nature_map_base'}, scope)
fit = scope['fit_poly_trend']
pieces = scope['time_segments']
header = scope['set_panel_header']

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def evaluate(gps, observed_gps, meta):
    center = float(np.median(observed_gps))
    scale = max(float(np.max(np.abs(observed_gps - center))), 1.)
    return np.polyval(meta['coefficients'], (np.asarray(gps)-center)/scale)

qa = {'status': 'h50_user_requested_correction_not_yet_promoted',
      'failure_in_previous_figure': ['A straight display guide connected independent fits.',
                                    'Whole-arc legend concealed short-segment constant fallbacks.'],
      'algorithm': 'Exact archived E23 robust_poly4_median60s; GPS-anchored bins; sqrt(count) weights; fit only measured C/N0 at tangent height >=50 km; the sole fitting-rule change from the archived E23 method is the user-requested height threshold.',
      'sources': {str(p): digest(p) for p in [FIT, TEMPLATE, HELPER, SOURCE]},
      'E23_reproduction': [], 'E21_segment_reproduction': [], 'E21_joint_fits': [],
      'Fig2f_impact': [], 'no_synthetic_CNo_or_residual_samples': True,
      'accepted_main_figures_and_residuals_modified': False}

# Establish that the implementation really reproduces the approved E23 example.
e23 = pd.read_csv(E23_PACKAGE / 'data/op38_e23_anomaly_focused_cn0_data.csv')
for track, group in e23.groupby('track_label'):
    is_fill = group.cn0_is_filled.astype(str).str.lower().eq('true')
    obs = group[~is_fill]
    t = obs.rx_gps_seconds.to_numpy(float)
    _, meta = fit(t, obs.cn0_dbhz.to_numpy(float), 4)
    pred = evaluate(group.rx_gps_seconds.to_numpy(float), t, meta)
    error = float(np.max(np.abs(pred-group.cn0_trend_dbhz.to_numpy(float))))
    assert error < 1e-10, (track, error)
    qa['E23_reproduction'].append({'track': track, 'observed_n': len(obs),
        'gap_display_n': int(is_fill.sum()), 'fit_bins': meta['fit_points'],
        'baseline_max_error_db': error,
        'true_residual_min_max_db': [float(obs.cn0_detrended_db.min()), float(obs.cn0_detrended_db.max())],
        'outside_original_E23_plus_minus_2dB_axes': int(obs.cn0_detrended_db.abs().gt(2).sum())})

raw = pd.read_csv(SOURCE)
raw['time_dt'] = pd.to_datetime(raw.plot_time, utc=True)
raw['h_tan_km'] = raw.lugre_h_tan_wgs84_km
raw['tp_lat_deg'] = raw.lugre_lat_wgs84
raw['tp_lon_deg'] = raw.lugre_lon_wgs84
raw['track_label'] = raw.signal_id.map({2: 'E21 E1C', 3: 'E21 E5Q'})
assert len(raw) == 6586 and raw.fig2f_sample.sum() == 930
raw['joint_fit_eligible'] = raw.h_tan_km.ge(50)
raw['joint_background_dbhz'] = np.nan
raw['joint_residual_db'] = np.nan
curves = []
reference_coeff = {
  2: [-11.59770801093913, -14.041189635527088, -18.42691782300192, -7.455885302943966, 32.79216194259808],
  3: [-18.148050336557564, -29.974982619741695, -25.55031447167146, -10.399559690301675, 34.35497709081023]}

for (signal, segment), q in raw.groupby(['signal_id', 'segment_id']):
    trend, meta = fit(q.rx_gps_seconds.to_numpy(float), q.cn0_dbhz.to_numpy(float), 4)
    error = float(np.max(abs(trend-q.baseline_dbhz.to_numpy(float))))
    assert error < 1e-10
    qa['E21_segment_reproduction'].append({'signal_id': int(signal), 'segment_id': int(segment),
        'observed_n': len(q), 'actual_degree': meta['degree'], 'method': meta['method'],
        'baseline_max_error_db': error})

for signal, q in raw.groupby('signal_id'):
    observed = q[q.joint_fit_eligible].sort_values('time_dt')
    t = observed.rx_gps_seconds.to_numpy(float)
    assert np.max(np.diff(t)) < 7200, 'Different E23 local arcs may not be merged.'
    _, meta = fit(t, observed.cn0_dbhz.to_numpy(float), 4)
    assert meta['degree'] == 4
    model = evaluate(q.rx_gps_seconds.to_numpy(float), t, meta)
    raw.loc[q.index, 'joint_background_dbhz'] = model
    raw.loc[q.index, 'joint_residual_db'] = q.cn0_dbhz.to_numpy(float)-model
    historical = q[q.h_tan_km.ge(80)].sort_values('time_dt')
    _, historical_meta = fit(historical.rx_gps_seconds.to_numpy(float), historical.cn0_dbhz.to_numpy(float), 4)
    np.testing.assert_allclose(historical_meta['coefficients'], reference_coeff[signal], atol=1e-10, rtol=0)
    old_model = evaluate(q.rx_gps_seconds.to_numpy(float), historical.rx_gps_seconds.to_numpy(float), historical_meta)
    sensitivity = np.max(abs(model-old_model))
    added = observed[observed.h_tan_km.lt(80)]
    assert len(added) == 23, (signal, len(added))
    assert observed.h_tan_km.min() >= 50
    assert np.isfinite(observed.cn0_dbhz).all()
    # Re-run the archived function on identical actual observations to verify determinism.
    _, repeat_meta = fit(t.copy(), observed.cn0_dbhz.to_numpy(float).copy(), 4)
    np.testing.assert_allclose(meta['coefficients'], repeat_meta['coefficients'], atol=0, rtol=0)
    center = float(np.median(t)); scale = float(np.max(abs(t-center)))
    dense_gps = np.arange(t.min(), t.max()+.01, 1.)
    nearest = np.searchsorted(t, dense_gps, side='right')
    left = t[np.maximum(nearest-1, 0)]; right = t[np.minimum(nearest, len(t)-1)]
    hidden = (right-left) > 600
    dense = pd.DataFrame({'signal_id': signal, 'rx_gps_seconds': dense_gps,
        'time_dt': observed.time_dt.iloc[0]+pd.to_timedelta(dense_gps-t.min(), unit='s'),
        'background_dbhz': evaluate(dense_gps, t, meta), 'in_CNo_gap': hidden})
    curves.append(dense)
    qa['E21_joint_fits'].append({'signal_id': int(signal), 'observed_n': len(observed),
        'start_utc': str(observed.time_dt.min()), 'end_utc': str(observed.time_dt.max()),
        'center_gps_seconds': center, 'normalization_scale_s': scale, **meta,
        'h50_vs_archived_h80_max_background_difference_db': float(sensitivity),
        'h80_archive_reproduction_passed': True,
        'h50_fit_min_height_km': float(observed.h_tan_km.min()),
        'added_50_to_80_km_measured_observations_n': len(added),
        'archived_h80_coefficients': historical_meta['coefficients'],
        'actual_input_rows_and_raw_values_preserved': True,
        'observed_max_gap_s': float(np.max(np.diff(t))),
        'gap_baseline_min_max_dbhz': [float(dense.loc[hidden, 'background_dbhz'].min()), float(dense.loc[hidden, 'background_dbhz'].max())]})
    chosen = q[q.fig2f_sample]
    old = chosen.plot_cn0_detrended_db.to_numpy(float)
    new = chosen.cn0_dbhz.to_numpy(float)-evaluate(chosen.rx_gps_seconds.to_numpy(float), t, meta)
    qa['Fig2f_impact'].append({'signal_id': int(signal), 'n': len(chosen),
        'raw_CNo_and_epoch_identities_unchanged': True,
        'old_constant_background_n': int(chosen.constant_fallback.sum()),
        'residual_change_rms_db': float(np.sqrt(np.mean((new-old)**2))),
        'residual_change_max_absolute_db': float(np.max(abs(new-old))),
        'old_and_new_absolute_residual_P95_db': [float(np.quantile(abs(old), .95)), float(np.quantile(abs(new), .95))]})

qa['height50_validation'] = {
    'raw_rows': len(raw),
    'fitted_rows': int(raw.joint_fit_eligible.sum()),
    'added_measured_50_to_80_km_rows': int((raw.joint_fit_eligible & raw.h_tan_km.lt(80)).sum()),
    'below50_not_fitted_rows': int(raw.h_tan_km.lt(50).sum()),
    'threshold_change_only_no_raw_quality_filter_change': True,
    'independent_geometry_input_sha256': digest(ROOT/'data/E21_independent_WGS84_geometry.csv'),
    'producer_sha256': digest(Path(__file__)),
    'archived_producer_sha256': digest(ROOT/'code/archived_check_and_build_E21_whole_arc.py'),
}
qa['high_height_background'] = []
for signal, q in raw[raw.joint_fit_eligible & raw.h_tan_km.gt(1000)].groupby('signal_id'):
    absres = q.joint_residual_db.abs()
    qa['high_height_background'].append({
        'signal_id': int(signal), 'n': len(q),
        'definition': 'mean absolute residual among all measured samples with WGS84 tangent height >1000 km',
        'mean_absolute_residual_db': float(absres.mean()),
        'mean_signed_residual_db': float(q.joint_residual_db.mean()),
        'p95_absolute_residual_db': float(absres.quantile(.95)),
        'mean_absolute_below_half_dB': bool(absres.mean() < .5),
        'no_residual_amplitude_exclusion': True,
    })
raw.to_csv(ROOT/'data/E21_measured_CNo_joint_quartic_candidate.csv', index=False)
curves = pd.concat(curves, ignore_index=True)
curves.to_csv(ROOT/'data/E21_quartic_model_evaluation_not_measurements.csv', index=False)
qa['numerical_checks_passed'] = True
(ROOT/'qa/E21_arc_reconstruction_QA.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')

if '--check-only' in sys.argv:
    print(json.dumps({'QA': str(ROOT/'qa/E21_arc_reconstruction_QA.json'), 'fits': qa['E21_joint_fits'], 'impact': qa['Fig2f_impact']}))
    sys.exit(0)

# Geometry supplied by the independent orbit/receiver-state reconstruction,
# after comparison with the selected WGS84 source (no Hermite/linear guides).
GEOMETRY = ROOT/'data/E21_independent_WGS84_geometry.csv'
geometry_qa = ROOT/'qa/E21_independent_geometry_QA.json'
if not GEOMETRY.exists() or not geometry_qa.exists():
    raise RuntimeError('Independent geometry verification is required before rendering.')
gqa = json.loads(geometry_qa.read_text(encoding='utf-8'))
assert gqa['comparison_to_selected_geometry_passed']
geometry = pd.read_csv(GEOMETRY)
geometry['time_dt'] = pd.to_datetime(geometry.time_utc, utc=True)
observed = raw[raw.joint_fit_eligible].sort_values('time_dt')
start, end = observed.time_dt.min(), observed.time_dt.max()
geometry = geometry[geometry.time_dt.between(start, end)].copy()
geometry['track_label'] = 'E21 E1C'
spans = scope['merge_time_spans'](scope['geometry_ionosphere_spans'](geometry))

mpl.rcParams.update({'font.family': 'Arial', 'font.size': 6.8, 'axes.titlesize': 7.4,
                    'axes.labelsize': 6.9, 'legend.fontsize': 6., 'savefig.dpi': 600})
fig = plt.figure(figsize=(7.35, 5.45), facecolor='white')
gs = fig.add_gridspec(3, 2, width_ratios=[3.15,1.18], height_ratios=[1.08,1.,1.10],
    left=.078, right=.985, bottom=.095, top=.865, hspace=.25, wspace=.24)
aa=fig.add_subplot(gs[0,0]); ab=fig.add_subplot(gs[1,0], sharex=aa)
ac=fig.add_subplot(gs[2,0], sharex=aa); ad=fig.add_subplot(gs[:,1])
palette={2:'#2b5fb3',3:'#c93f3b'}; markers={2:'s',3:'D'}; bridge='#7b818c'
for ax in [aa,ab]:
    for a,b in spans:
        ax.axvspan(a,b,facecolor='#f7dce5',edgecolor='none',alpha=.46,zorder=-.5)
handles=[]
for signal in [2,3]:
    q=observed[observed.signal_id.eq(signal)].sort_values('time_dt')
    color=palette[signal]; marker=markers[signal]
    for ax,col,alpha in [(aa,'cn0_dbhz',.58),(ab,'joint_residual_db',.64)]:
        ax.scatter(q.time_dt,q[col],s=5.2,marker=marker,facecolor=color,
                   edgecolor='none',alpha=alpha,rasterized=True,zorder=3)
    for p in pieces(q,90):
        aa.plot(p.time_dt,p.joint_background_dbhz,color='#2f3440',lw=1.,
                alpha=.72,ls=(0,(4.,2.2)),zorder=4)
    curve=curves[curves.signal_id.eq(signal) & curves.in_CNo_gap]
    for p in pieces(curve,90):
        aa.plot(p.time_dt,p.background_dbhz,color=bridge,lw=.95,alpha=.76,
                ls=(0,(2.2,2.)),zorder=4)
    handles.append(Line2D([0],[0],marker=marker,ls='none',color=color,
        markeredgecolor='none',markersize=4.,label='E21 Galileo '+('E1' if signal==2 else 'E5a')))
    # Retain selected observed-epoch WGS84 points for coloured shell segments.
    geometry_at_observed_epochs=raw[raw.signal_id.eq(signal)].sort_values('time_dt')
    shell=geometry_at_observed_epochs[geometry_at_observed_epochs.h_tan_km.between(50,1000)]
    for p in pieces(shell,90):
        ac.plot(p.time_dt,p.h_tan_km,color=color,lw=1.65,alpha=.9,zorder=5)
        ad.plot(p.tp_lon_deg,p.tp_lat_deg,color=color,lw=2.35 if signal==2 else 1.35,
                alpha=.72 if signal==2 else .88,zorder=6)
    if len(shell):
        first,last=shell.iloc[0],shell.iloc[-1]
        ad.scatter(first.tp_lon_deg,first.tp_lat_deg,s=15,marker='o',facecolor='white',edgecolor=color,lw=.8,zorder=12)
        ad.scatter(last.tp_lon_deg,last.tp_lat_deg,s=17,marker='>',facecolor=color,edgecolor='white',lw=.45,zorder=12)

# Actual geometry at all missing epochs: plotted as a calculated dashed path.
in_gap = np.zeros(len(geometry),bool)
for _,q in observed.groupby('signal_id'):
    for left,right in zip(pieces(q,600)[:-1],pieces(q,600)[1:]):
        in_gap |= geometry.time_dt.between(left.time_dt.max(),right.time_dt.min()).to_numpy()
gap_geo=geometry[in_gap | geometry.h_tan_km.lt(50)]
for p in pieces(gap_geo,90):
    ac.plot(p.time_dt,p.h_tan_km,color=bridge,lw=.95,alpha=.84,ls=(0,(3.2,2.2)),zorder=3)
    ad.plot(p.tp_lon_deg,p.tp_lat_deg,color=bridge,lw=1.55,alpha=.96,ls=(0,(5.,3.)),zorder=10)
header(aa,'a',r'Raw receiver-reported $C/N_0^k$ and whole-arc baseline')
aa.set_ylabel(r'$C/N_0^k$ (dB-Hz)'); aa.set_ylim(20.,40.); aa.set_yticks([20,25,30,35,40])
header(ab,'b',r'Residual $\delta C/N_0^k=C/N_0^k-\widehat{C/N_0^k}$')
ab.set_ylabel(r'$\delta C/N_0^k$ (dB)'); ab.axhline(0,color='#1f2328',lw=.72,alpha=.92,zorder=2)
ab.set_ylim(-2,2); ab.set_yticks([-2,-1,0,1,2])
ac.axhspan(50,1000,facecolor='#f3dce4',edgecolor='none',alpha=.5,zorder=0)
header(ac,'c','Tangent height in the 50-1000 km shell')
ac.set_ylabel('Tangent height (km)'); ac.set_xlabel('UTC on 2025-03-03')
ac.set_ylim(-650,1040); ac.set_yticks([-500,0,250,500,750,1000])
ac.text(.985,.90,'50-1000 km ionosphere',transform=ac.transAxes,ha='right',va='center',
        fontsize=5.6,color='#7f3f52',bbox={'facecolor':'white','edgecolor':'none','alpha':.68,'pad':1.})
for ax in [aa,ab,ac]:
    scope['style_nature_axis'](ax); ax.set_xlim(start,end)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30],tz=timezone.utc))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=timezone.utc))
    ax.grid(True,axis='x',color='#e0e4e9',lw=.42,alpha=.62)
plt.setp(aa.get_xticklabels(),visible=False); plt.setp(ab.get_xticklabels(),visible=False)
scope['draw_nature_map_base'](ad,[],*scope['map_extent'](geometry))
ad.xaxis.set_major_locator(MultipleLocator(40.))
header(ad,'d','Tangent-point footprint',label_x=-.180)
handles.extend([Line2D([0],[0],color='#2f3440',lw=1.,ls=(0,(4.,2.2)),label='fourth-order polynomial fit'),
                Line2D([0],[0],color=bridge,lw=.95,ls=(0,(2.2,2.)),label='model across data gap')])
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.078,.915),ncol=4,frameon=False,
    handlelength=1.8,columnspacing=.95,borderaxespad=0.)
fig.align_ylabels([aa,ab,ac])
out=ROOT/'figures/Extended_Data_CN0_detrending_OP38_E21.png'
fig.savefig(out,dpi=600,bbox_inches='tight',pad_inches=.025); plt.close(fig)
qa.update(output=str(out), output_sha256=digest(out), geometry=gqa,
          visual_review='pending', user_requested_axes={'a_dbhz':[20,40],'b_db':[-2,2]},
          display_zoom_only_no_quality_exclusions=True,
          raw_CNo_outside_a_axes_n=int((~observed.cn0_dbhz.between(20,40)).sum()),
          residuals_outside_b_axes_n=int((~observed.joint_residual_db.between(-2,2)).sum()),
          residual_axis_contains_all_plotted_observations=False)
(ROOT/'qa/E21_arc_reconstruction_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps({'output':str(out),'plotted_observations':len(observed),'QA':str(ROOT/'qa/E21_arc_reconstruction_QA.json')}))
