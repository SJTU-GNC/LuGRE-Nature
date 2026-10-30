"""Recompute Fig.2 from new independent E23-method residuals on fixed cohorts.

The selected panel layout, original observations, geometries and previous QC
identities stay frozen. No previous residual or baseline-transfer value is used
in an updated statistic. Fourth-order eligibility is supplied by the full refit.
"""
from pathlib import Path
import sys, json, hashlib, ast, argparse
sys.dont_write_bytecode = True
PROJECT = Path('H:/LUGRE')
HERE = Path(__file__).resolve().parent
RUN = HERE.parent
OUT = HERE/'derived_source_data'
sys.path[:0] = [str(PROJECT/'python_deps/pyarrow_py312')]
import numpy as np
import pandas as pd

OLD_PROJECT = Path('D:/Reference/Paper/2026/Nature_LuGRE')
OLD_FIGURE = OLD_PROJECT/'Figure'
OLD_SRC = PROJECT/'R3_main_figures_20260920/source_data'
OLD_SELECTED = PROJECT/'figures_R3/Fig2_R3_reorganized_20261013/revision21_L1_E1_polar_tracks_R3'
OLD_MAP = OLD_FIGURE/'centered_dipole_rebuild_fig1_fig3_fig4_20260730/Fig1_centered_dipole_1min_5deg_cn0QC_artifactMask/data'
OLD_PSD_ROOT = OLD_FIGURE/'Fig3_R2_min60_20260915'
OLD_PAIR_ROOT = Path('C:/Users/YR/Documents/Nature_LuGRE/outputs/Fig6_joint_response_R2_INGV_20260920/observation_test')
KEYS = ['phase','op_id','sat','signal_id','segment_id']
LATS = ['Equatorial','Mid-lat','Polar']
SIGNAL_IDS = {'GPS L1':0,'GPS L5':1,'Galileo E1':2,'Galileo E5a':3}
COLORS = {'Equatorial':'#238BA3','Mid-lat':'#7957C5','Polar':'#AA2031'}
SOURCES = {}
QA = {}

def read(path, **kwargs):
    path=Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_parquet(path, **kwargs) if path.suffix=='.parquet' else pd.read_csv(path, **kwargs)

def save(frame, name):
    frame.to_csv(OUT/name,index=False)

def load_functions(path,names,scope):
    path=Path(path);SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
           if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)

def band(mlat,short=False):
    return np.where(np.abs(mlat)<20,'Equatorial',np.where(np.abs(mlat)<60,'Mid-lat' if short else 'Mid-latitude','Polar'))

def latitude(lat,lon):
    la=np.deg2rad(np.asarray(lat,float));lo=np.deg2rad(np.asarray(lon,float))
    return np.rad2deg(np.arcsin(np.clip(np.sin(la)*np.sin(np.deg2rad(80.65))+
        np.cos(la)*np.cos(np.deg2rad(80.65))*np.cos(lo+np.deg2rad(72.68)),-1,1)))

def add_keys(d):
    d=d.copy()
    for k in ['phase','op_id','sat']:d[k]=d[k].astype(str)
    if 'signal_id' not in d:d['signal_id']=d.signal_name.map(SIGNAL_IDS)
    d['signal_id']=pd.to_numeric(d.signal_id).astype(int)
    if 'rx_gps_seconds' in d:
        d['_epoch_ms']=np.rint(pd.to_numeric(d.rx_gps_seconds)*1000).astype('int64')
    elif 'time_utc' in d:
        # These saved map rows retain the original UTC label, not GPS time.
        d['_utc_label_s']=pd.to_datetime(d.time_utc,utc=True,format='mixed').dt.as_unit('ns').astype('int64')//10**9
    return d

def replace_residuals(frame,full,label,keep_all=False):
    d=add_keys(frame)
    ident=['phase','op_id','sat','signal_id']+(['_epoch_ms'] if '_epoch_ms' in d else ['_utc_label_s'])
    assert not d.duplicated(ident).any(),(label,'duplicate source epoch')
    cols=ident+['quartic_background_dbhz','e23_residual_db','e23_abs_residual_db','e23_local_arc_id','fit_degree','fit_eligible']
    source=full[cols+(['cn0_dbhz'] if 'cn0_dbhz' in d else [])].copy()
    assert not source.duplicated(ident).any(),(label,'ambiguous new epoch')
    ren={c:'_new_'+c for c in cols if c not in ident}
    if 'cn0_dbhz' in d:ren['cn0_dbhz']='_new_raw_cn0'
    d=d.merge(source.rename(columns=ren),on=ident,how='left',validate='one_to_one',indicator=True)
    assert d['_merge'].eq('both').all(),(label,d['_merge'].value_counts().to_dict())
    d=d.drop(columns='_merge')
    raw_error=0.
    if '_new_raw_cn0' in d:
        raw_error=float(np.nanmax(np.abs(d.cn0_dbhz-d._new_raw_cn0)))
        assert raw_error<1e-8,(label,raw_error)
        d=d.drop(columns='_new_raw_cn0')
    for c in cols:
        if c not in ident:d[c]=d.pop('_new_'+c)
    eligible=d.fit_eligible.fillna(False).astype(bool)&d.fit_degree.eq(4)&np.isfinite(d.e23_residual_db)
    if 'lugre_h_tan_km' in d:
        eligible &= pd.to_numeric(d.lugre_h_tan_km,errors='coerce').ge(50)
    for c in ['plot_cn0_detrended_db','cn0_detrend_zero_mean_dbhz','cn0_delta_dbhz',
              'plot_cn0_detrended_db_paired','plot_cn0_detrended_db_original_4fit']:
        if c in d:d[c]=d.e23_residual_db
    for c in ['abs_delta_cn0_db','cn0_detrend_zero_mean_abs_dbhz']:
        if c in d:d[c]=d.e23_abs_residual_db
    if 'secondary_detrend_method' in d:d['secondary_detrend_method']='independent_E23_method_poly4'
    QA[label]={'original_rows':len(d),'eligible_rows':int(eligible.sum()),'excluded_no_quartic_or_height_rows':int((~eligible).sum()),
               'raw_cn0_max_abs_change_db':raw_error,'old_QC_and_geometry_identities_frozen':True}
    if 'lugre_h_tan_km' in d:
        QA[label]['minimum_retained_geometry_height_km']=float(d.loc[eligible,'lugre_h_tan_km'].min())
    return d if keep_all else d[eligible].copy()

def map_cells(full):
    old=read(OLD_MAP/'fig1c_l1e1_cn0_1hz_samples_with_5deg_bins_after_artifactMask.csv')
    d=replace_residuals(old,full,'panel_a_map')
    cells=d.groupby(['lat_bin','lon_bin','lat_center','lon_center'],as_index=False).agg(
        metric=('e23_abs_residual_db',lambda s:float(s.quantile(.95))),
        n_samples_1hz=('e23_abs_residual_db','size'),n_minutes=('utc_minute','nunique'),
        n_track_signal_segments=('track_signal_segment','nunique'),n_signals=('signal_name','nunique'))
    cells['mlat_center']=latitude(cells.lat_center,cells.lon_center)
    cells['lat_group']=band(cells.mlat_center)
    cells['metric_name']='cell P95 of all eligible 1-Hz |delta C/N0| samples'
    cells['min_cell_minutes']=1
    save(cells,'spatial_amplitude_cells.csv');save(d,'spatial_amplitude_epoch_source.csv')
    QA['panel_a_map']['occupied_cells']=len(cells)

def strict_peak(time_s,y):
    t=np.asarray(time_s,float);y=np.asarray(y,float);good=np.isfinite(t)&np.isfinite(y)
    t=t[good];y=y[good];order=np.argsort(t);t=t[order];y=y[order]
    if len(y)<30 or not np.all(np.isclose(np.diff(t),1,atol=1e-8,rtol=0)) or np.std(y)<=0:return np.nan
    p=abs(np.fft.rfft((y-y.mean())*np.hanning(len(y))))**2
    f=np.fft.rfftfreq(len(y));valid=(f>=1/120)&(f<=.45)
    return float(f[valid][np.argmax(p[valid])]) if valid.sum()>=2 else np.nan

def events_and_psd(full):
    old=read(OLD_PSD_ROOT/'data/inputs/fig3_points_source.csv')
    # The old event cohort has an additional fixed window-level bias mask.
    # Freeze both its identities and its original time origins; do not select
    # windows anew from the new residual amplitudes.
    origin=old.groupby(KEYS,dropna=False).rx_gps_seconds.transform('min')
    old['_old_window_idx']=np.floor((old.rx_gps_seconds-origin)/60).astype(int)
    old['_old_track_uid']=old[KEYS].astype(str).agg('|'.join,axis=1)
    old_windows=read(Path('C:/Users/YR/Documents/Nature_LuGRE/outputs/Fig6_geometry_sensitivity_R2_INGV_20260918/model/data/window_geometry.csv'))
    allowed=pd.MultiIndex.from_frame(old_windows[['track_uid','window_idx']])
    assert not allowed.duplicated().any() and len(old_windows)==2244
    pts=replace_residuals(old,full,'spectral_event_points')
    save(pts,'fig3_points_source.csv')
    membership=pd.MultiIndex.from_arrays([pts._old_track_uid,pts._old_window_idx]).isin(allowed)
    ep=pts[membership].copy();window_rows=[]
    for (uid,wi),q in ep.groupby(['_old_track_uid','_old_window_idx'],sort=True):
        q=q.sort_values('rx_gps_seconds');y=q.cn0_delta_dbhz.to_numpy(float)
        if len(y)<30:continue
        p05,p50,p95=np.quantile(y,[.05,.5,.95]);pk=strict_peak(q.rx_gps_seconds.to_numpy(),y)
        row={k:q[k].iloc[0] for k in KEYS}
        row.update(track_uid=uid,window_idx=int(wi),signal_name=q.signal_name.iloc[0],
            freq_group='L5/E5a' if q.signal_name.iloc[0] in ['GPS L5','Galileo E5a'] else 'L1/E1',
            daynight=q.daynight.mode().iloc[0],lat_group=q.lat_group.mode().iloc[0],
            mlat_igrf_dip=float(q.mlat_igrf_dip.median()),height_layer=q.height_layer.mode().iloc[0],
            h_tan_km=float(q.lugre_h_tan_km.median()),a95_db=float(np.quantile(abs(y),.95)),
            signed_p05_db=float(p05),signed_median_db=float(p50),signed_p95_db=float(p95),signed_p95_p05_width_db=float(p95-p05),
            rms_db=float(np.sqrt(np.mean(y*y))),dominant_frequency_hz=pk,n_points=len(y),
            duration_s=float(q.rx_gps_seconds.iloc[-1]-q.rx_gps_seconds.iloc[0]),is_event=bool(p95-p05>=1.5))
        window_rows.append(row)
    win=pd.DataFrame(window_rows)
    save(win,'all_60s_windows.csv')
    event=win[win.is_event].copy();event['dominant_frequency_log10_hz']=np.log10(event.dominant_frequency_hz)
    event['lat_order']=event.lat_group.map(dict(zip(LATS,range(3))))
    save(event.sort_values(['lat_order','track_uid','window_idx']),'fig3_panel_b_event_level_points.csv')
    rows=[]
    for layer in ['Topside','F2','F1','E','D']:
        for g in LATS:
            q=win[win.height_layer.eq(layer)&win.lat_group.eq(g)];n=len(q);e=int(q.is_event.sum())
            rows.append(dict(height_layer=layer,lat_group=g,events=e,total_windows=n,event_fraction_percent=100*e/n if n else np.nan))
    save(pd.DataFrame(rows),'fig3_panel_d_altitude_layer_heatmap.csv')
    rates=[]
    for g,q in win.groupby('lat_group'):
        rates.append(dict(lat_group=g,events=int(q.is_event.sum()),total_windows=len(q),event_fraction_percent=100*q.is_event.mean()))
    save(pd.DataFrame(rates),'fig3_panel_b1_event_fraction.csv')
    QA['events']={'valid_windows':len(win),'event_windows':len(event),'width_definition':'P95(residual)-P05(residual)',
        'criterion_db':1.5,'operator':'>=','window_origin':'frozen original signal/segment first point, before new eligibility filtering',
        'window_seconds':60,'minimum_actual_points':30,'window_height_and_latitude':'mode of per-observation categories, original rule',
        'peak_frequency_no_interpolation':True,'counts_by_band':pd.DataFrame(rates).to_dict('records'),
        'frozen_old_bias_screen_window_identities':2244,'new_residual_bias_screen_not_reapplied':True,
        'old_window_identities_lost_for_new_eligibility':2244-len(win),'maximum_event_width_db':float(event.signed_p95_p05_width_db.max()),
        'height_band_rates':rows}
    # Load the accepted strict native-bin Welch function, not its old run body.
    psd_scope=dict(np=np,pd=pd)
    load_functions(OLD_PSD_ROOT/'code/build_psd_min60_native_bins.py',['welch'],psd_scope)
    grid=np.arange(1,129,dtype=float)/256;lo=grid-1/512;hi=grid+1/512;hi[-1]=.5
    spectra=[];metas=[];excluded=[]
    for key,g in pts.groupby(KEYS+['lat_group'],sort=False,dropna=False):
        g=g.sort_values('rx_gps_seconds',kind='stable').drop_duplicates('rx_gps_seconds',keep='first')
        breaks=~np.isclose(g.rx_gps_seconds.diff().to_numpy(),1,atol=1e-8,rtol=0);breaks[0]=False
        for rid,q in g.groupby(np.cumsum(breaks),sort=False):
            ident=dict(zip(KEYS+['lat_group'],key));ident.update(run_id=int(rid),signal_name=q.signal_name.iloc[0])
            if len(q)<60:excluded.append(dict(**ident,n_observed=len(q)));continue
            f,p,m,starts,err=psd_scope['welch'](q.cn0_delta_dbhz.to_numpy(float));fp=f[1:];pp=p[1:]
            vals=np.full(128,np.nan);counts=np.zeros(128,int);supported=(grid>=fp[0]-1e-12)&(grid<=fp[-1]+1e-12)
            for j,(a,b) in enumerate(zip(lo,hi)):
                if not supported[j]:continue
                take=(fp>=a)&((fp<b) if j<127 else (fp<=b));counts[j]=take.sum()
                if counts[j]:vals[j]=np.mean(pp[take])
            uid=len(metas);metas.append(dict(**ident,run_uid=uid,n_observed=len(q),start_gps=q.rx_gps_seconds.iloc[0],end_gps=q.rx_gps_seconds.iloc[-1],welch_nperseg=m,welch_subwindows=len(starts),parseval_error=err))
            spectra.append(pd.DataFrame(dict(run_uid=uid,signal_name=ident['signal_name'],lat_group=ident['lat_group'],frequency_hz=grid,psd=vals,n_native=counts)))
    spec=pd.concat(spectra,ignore_index=True)
    panel=spec.groupby(['signal_name','lat_group','frequency_hz'],as_index=False).agg(median_psd_dbhz2_per_hz=('psd','median'),n_runs=('psd','count'))
    panel['asd_pair']=panel.signal_name.map(lambda s:'L1 / E1' if s in ['GPS L1','Galileo E1'] else 'L5 / E5a')
    save(panel,'fig3_panel_a_psd_no_time_interpolation.csv');save(pd.DataFrame(metas),'psd_retained_runs.csv');save(pd.DataFrame(excluded),'psd_excluded_runs.csv');save(spec,'psd_per_run_frequency_bins.csv')
    QA['PSD']={'retained_runs':len(metas),'retained_actual_points':sum(m['n_observed'] for m in metas),'minimum_run_samples':60,
        'strict_1Hz_run_split':True,'time_interpolation':False,'frequency_interpolation':False,
        'window':'numpy Hanning','nperseg':'min(256,N)','overlap':'min(128,floor(nperseg/2))',
        'subwindow_mean_removal':True,'aggregation':'native positive ordinates into fixed k/256 +/-1/512 Hz bins; median across runs',
        'max_parseval_error':max(m['parseval_error'] for m in metas)}
    comparisons=[]
    for signal,z in panel[panel.frequency_hz.gt(.1)&panel.frequency_hz.le(.45)].groupby('signal_name'):
        piv=z.pivot(index='frequency_hz',columns='lat_group',values='median_psd_dbhz2_per_hz')
        for ref in ['Equatorial','Mid-lat']:
            rat=(piv.Polar/piv[ref]).dropna()
            comparisons.append(dict(signal_name=signal,reference=ref,n_common_frequencies=len(rat),
                polar_over_reference_min=float(rat.min()),polar_over_reference_median=float(rat.median()),
                polar_over_reference_max=float(rat.max()),frequencies_polar_not_larger=rat[rat.le(1)].index.tolist()))
    QA['PSD']['above_0p1_to_0p45_Hz_polar_ratios']=comparisons

def paired(full):
    p=read(OLD_PAIR_ROOT/'all_104_segments_exact_measured_pairs.csv.gz')
    original=read(OLD_PAIR_ROOT/'all_104_natural_segments.csv')
    ident=['phase','op_id','sat','signal_id','_epoch_ms']
    new=full[ident+['e23_residual_db','quartic_background_dbhz','fit_eligible','fit_degree']]
    for side in ['h','l']:
        key=p[['phase','op_id','sat','joint_run_id']].copy()
        key['signal_id']=p['signal_name_'+side].map(SIGNAL_IDS)
        key['_epoch_ms']=np.rint(p['rx_gps_seconds_'+side]*1000).astype('int64')
        q=key.merge(new,on=ident,how='left',validate='many_to_one',indicator=True)
        assert q['_merge'].eq('both').all()
        p['cn0_delta_dbhz_'+side]=q.e23_residual_db.to_numpy()
        p['baseline_accepted_'+side]=q.quartic_background_dbhz.to_numpy()
        p['eligible_'+side]=(q.fit_eligible.fillna(False).astype(bool)&q.fit_degree.eq(4)&np.isfinite(q.e23_residual_db)).to_numpy()
    keep=p.eligible_h&p.eligible_l&pd.to_numeric(p.lugre_h_tan_km,errors='coerce').ge(50)
    QA['paired_points']={'original_fixed_pairs':len(p),'eligible_pairs':int(keep.sum()),'excluded_pairs':int((~keep).sum()),'original_natural_runs':len(original)}
    p=p[keep].copy();p['paired_primary_transfer_applied_l']=False
    for c in [c for c in p if c.startswith('plot_cn0_detrended_db_original_4fit')]:
        p[c]=p['cn0_delta_dbhz_'+('h' if c.endswith('_h') else 'l')]
    rows=[];blocks=[]
    width=lambda z:float(np.quantile(z,.95)-np.quantile(z,.05))
    for run,g in p.groupby('joint_run_id',sort=True):
        g=g.sort_values('rx_gps_seconds_h')
        bad=~np.isclose(g.rx_gps_seconds_h.diff().to_numpy(),1,atol=1e-8,rtol=0);bad[0]=False
        for piece,q in g.groupby(np.cumsum(bad),sort=False):
            if len(q)<180:continue
            block_count=len(q)//60;row=original[original.joint_run_id.eq(run)].iloc[0].to_dict()
            row['original_joint_run_id']=run;row['joint_run_id']=f'{int(run)}_eligible{int(piece)}'
            row['n_measured_pairs']=len(q);row['n_exact_60sample_blocks']=block_count;row['unused_tail_pairs']=len(q)-60*block_count
            row['mlat_median']=q.mlat_centered_dipole.median();row['median_latitude_band']=band([row['mlat_median']])[0]
            labels=q.native_latitude_band.unique();row['latitude_band']=labels[0] if len(labels)==1 else 'Mixed';row['latitude_bands_present']=';'.join(sorted(labels))
            row['high_width_db']=width(q.cn0_delta_dbhz_h);row['low_width_db']=width(q.cn0_delta_dbhz_l)
            hs=[];ls=[]
            for i in range(block_count):
                z=q.iloc[60*i:60*(i+1)];h=width(z.cn0_delta_dbhz_h);l=width(z.cn0_delta_dbhz_l);hs.append(h);ls.append(l)
                blocks.append(dict(joint_run_id=row['joint_run_id'],block_index=i,n_measured_pairs=60,high_width_db=h,low_width_db=l))
            row['high_width_60sample_block_median_db']=float(np.median(hs));row['low_width_60sample_block_median_db']=float(np.median(ls))
            row['width_ratio_60sample_block_median']=float(np.median(np.asarray(ls)/np.asarray(hs)))
            # Old spectrum/correlation summaries are not used by Fig.2 and must
            # not survive as if they had been recomputed with the new residual.
            for c in list(row):
                if c.startswith(('high_spectral','low_spectral','spectral_power','coherence','residual_r','contains_accepted_event','accepted_event','n_accepted_event')):row.pop(c)
            rows.append(row)
    out=pd.DataFrame(rows)
    save(out,'panel_c_paired_frequency_segments.csv');save(out,'paired_geometry.csv');save(pd.DataFrame(blocks),'paired_exact60_blocks.csv')
    save(p,'paired_eligible_measured_epoch_source.csv')
    QA['paired_points'].update(updated_eligible_runs=len(out),complete60_blocks=len(blocks),
        coordinate_definition='per-frequency median of exact contiguous non-overlapping 60-sample block P95-P05 widths within fixed old independent-pair cohorts',
        minimum_pairs=180,colour_definition='absolute centered-dipole latitude median of each retained contiguous paired run',
        old_shared_baseline_excluded_population_not_automatically_added=True)
    paired_summary=[]
    for label,z in [('All',out)]+list(out.groupby('median_latitude_band')):
        broader=z.low_width_60sample_block_median_db.gt(z.high_width_60sample_block_median_db)
        paired_summary.append(dict(latitude_band=label,n_runs=len(z),n_lower_carrier_broader=int(broader.sum()),
            percent_lower_carrier_broader=float(100*broader.mean()),
            median_low_over_high_width=float((z.low_width_60sample_block_median_db/z.high_width_60sample_block_median_db).median()),
            max_lower_carrier_width_db=float(z.low_width_60sample_block_median_db.max())))
    QA['paired_points']['summary_by_display_latitude']=paired_summary

def raw_cases(full):
    for name in ['OP76_G01','OP38_E21']:
        d=read(OLD_SRC/f'Fig2_R3_{name}_measured_samples.csv')
        d=replace_residuals(d,full,'rawcase_'+name,keep_all=True)
        save(d,f'{name}_measured_samples.csv')
    save(read(Path('C:/Users/YR/Documents/Nature_LuGRE/outputs/Fig1_R2_INGV_20260917/data/spatial_roti_cells.csv')),'spatial_roti_cells.csv')

def main():
    a=argparse.ArgumentParser();a.add_argument('--full',type=Path,default=RUN/'data/all_LuGRE_E23_method_residuals.parquet');args=a.parse_args()
    assert PROJECT.is_dir() and HERE.resolve().is_relative_to(Path('E:/LUGRE').resolve())
    OUT.mkdir(parents=True,exist_ok=True)
    full=add_keys(read(args.full))
    full['_utc_label_s']=pd.to_datetime(full.time_utc,utc=True,format='mixed').dt.as_unit('ns').astype('int64')//10**9
    map_cells(full);events_and_psd(full);paired(full);raw_cases(full)
    payload={'figure_contract':{'conclusion':'Reassess the geographic, spectral and paired-frequency LuGRE amplitude characterization with one declared independent E23-method quartic procedure.',
        'archetype':'asymmetric mixed-modality figure','backend':'existing Python/matplotlib workflow','dimensions_mm':[183,170],'export':'600 dpi PNG only',
        'evidence_chain':'a-c amplitude geography; d unchanged phase-derived ROTI; e-f original C/N0; g,j PSD; h paired widths; i occurrence by geometric height; k event widths',
        'review_risks':'old masks frozen; h>=50 fit/statistical domain; h<50 raw retained; degree0 excluded; count changes and no-qualification rows explicit'},
        'scope':'new residuals on frozen prior panel cohorts/QC/artifact masks; no old residual values used',
        'source_hashes':SOURCES,'QA':QA,'all_derived_statistics_recomputed':True,'formal_files_modified':False}
    (HERE/'Fig2_source_rebuild_QA.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(QA,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
