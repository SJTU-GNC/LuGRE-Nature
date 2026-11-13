"""Recompute R4 Fig.2 QC, maps, event widths and PSD from frozen measured inputs."""
from pathlib import Path
import sys,json,hashlib,ast,shutil,os
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
COMPONENT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'fig2'
DATA=ROOT/'data';QA=ROOT/'qa'
for p in [DATA,QA]:p.mkdir(parents=True,exist_ok=True)
OLD=COMPONENT/'data/r4/inputs'
FULL=COMPONENT/'data/full_lugre/native_cn0_verified_geometry_and_expected_results.parquet'
UPDATED=COMPONENT/'data/fig2/updated_points.csv'
WELCH=COMPONENT/'code/r4_welch_native.py'
SEG=['phase','op_id','sat','signal_id','segment_id']
ID=['phase','op_id','sat','signal_id','_epoch_ms']
LATS=['Equatorial','Mid-lat','Polar']
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
sourcefiles=[OLD/'fig3_points_source.parquet',OLD/'all_60s_windows.csv',OLD/'spatial_amplitude_epoch_source.parquet',OLD/'spatial_amplitude_cells.csv',FULL,UPDATED,WELCH]
sources={str(p):sha(p) for p in sourcefiles}
p=pd.read_parquet(OLD/'fig3_points_source.parquet')
ow=pd.read_csv(OLD/'all_60s_windows.csv',dtype={'op_id':str})
assert len(p)==137686 and len(ow)==2244 and not p.duplicated(ID).any()
origin=p.groupby(SEG,dropna=False).rx_gps_seconds.transform('min')
wi=np.floor((p.rx_gps_seconds-origin)/60).astype(int)
assert np.array_equal(wi,p._old_window_idx)
p['qc_window_origin_gps_s']=origin
p['qc_window_idx']=p._old_window_idx
p['qc_track_uid']=p._old_track_uid
p['point_three_condition_eligible']=p.cn0_dbhz.ge(25)&p.fit_eligible&p.fit_degree.eq(4)&np.isfinite(p.e23_residual_db)

def build_ledger(q,uid='qc_track_uid',idx='qc_window_idx'):
    rows=[]
    for (u,w),g in q.groupby([uid,idx],sort=False):
        z=g[g.point_three_condition_eligible].sort_values('rx_gps_seconds')
        assert not z.rx_gps_seconds.duplicated().any()
        n=len(z);p05,med,p95=np.quantile(z.e23_residual_db,[.05,.5,.95]) if n else [np.nan]*3
        enough=n>=30;bias=bool(enough and abs(med)>1)
        r={k:g[k].iloc[0] for k in SEG if k in g}
        r.update(track_uid=u,window_idx=int(w),window_origin_gps_s=float(g.qc_window_origin_gps_s.iloc[0]),
                 window_start_gps_s=float(g.qc_window_origin_gps_s.iloc[0]+60*w),n_original_points=len(g),n_valid=n,
                 point_cn0_lt25_n=int(g.cn0_dbhz.lt(25).sum()),signed_p05_db=float(p05),signed_median_db=float(med),
                 signed_p95_db=float(p95),signed_p95_p05_width_db=float(p95-p05),
                 reject_N_valid_lt30=not enough,reject_abs_median_gt1=bias,qc_keep=bool(enough and not bias),
                 qc_reason='N_valid_below30' if not enough else 'absolute_signed_median_above1dB' if bias else 'retained')
        rows.append(r)
    return pd.DataFrame(rows)
allw=build_ledger(p)
oldids=pd.MultiIndex.from_frame(ow[['track_uid','window_idx']])
allw['in_original_2244_event_domain']=pd.MultiIndex.from_frame(allw[['track_uid','window_idx']]).isin(oldids)
assert len(allw)==2635 and allw.in_original_2244_event_domain.sum()==2244
allw.to_csv(DATA/'all_source_candidate_window_QC_ledger.csv',index=False)
p=p.merge(allw[['track_uid','window_idx','n_valid','signed_median_db','reject_N_valid_lt30','reject_abs_median_gt1','qc_keep','qc_reason','in_original_2244_event_domain']].rename(columns={'track_uid':'qc_track_uid','window_idx':'qc_window_idx'}),on=['qc_track_uid','qc_window_idx'],validate='many_to_one')
p['analysis_qc_keep']=p.point_three_condition_eligible&p.qc_keep
p[['phase','op_id','sat','signal_id','segment_id','_epoch_ms','rx_gps_seconds','cn0_dbhz','e23_residual_db','qc_track_uid','qc_window_idx','qc_window_origin_gps_s','n_valid','signed_median_db','point_three_condition_eligible','reject_N_valid_lt30','reject_abs_median_gt1','analysis_qc_keep','qc_reason','in_original_2244_event_domain']].to_csv(DATA/'spectral_event_all_original_point_QC_ledger.csv',index=False)
ledger=ow.merge(allw,on=['track_uid','window_idx'],suffixes=('_old',''),validate='one_to_one')
assert ledger.n_points.eq(ledger.n_valid).all()
assert np.max(abs(ledger.signed_p95_p05_width_db_old-ledger.signed_p95_p05_width_db))<1e-10
ledger.to_csv(DATA/'original_2244_event_window_QC_ledger.csv',index=False)
# Keep the old event-domain metadata/geometry/latitude assignment; change QC only.
new=ow.copy()
for name in ['signed_p05_db','signed_median_db','signed_p95_db','signed_p95_p05_width_db']:
    new[name]=ledger[name].to_numpy()
new['n_points']=ledger.n_valid.to_numpy();new['current_QC_keep']=ledger.qc_keep.to_numpy()
new=new[new.current_QC_keep].drop(columns='current_QC_keep').copy()
new['is_event']=new.signed_p95_p05_width_db.ge(1.5)
new.to_csv(DATA/'all_60s_windows.csv',index=False)
event=new[new.is_event].copy();event['dominant_frequency_log10_hz']=np.log10(event.dominant_frequency_hz)
event['lat_order']=event.lat_group.map(dict(zip(LATS,range(3))))
event.sort_values(['lat_order','track_uid','window_idx']).to_csv(DATA/'fig3_panel_b_event_level_points.csv',index=False)
rates=[];alt=[];beforeafter=[]
for lat in LATS:
    for stage,t in [('before',ow),('after',new)]:
        z=t[t.lat_group.eq(lat)];e=int(z.signed_p95_p05_width_db.ge(1.5).sum())
        beforeafter.append(dict(group_type='latitude',group=lat,stage=stage,n_windows=len(z),n_events=e,event_fraction_percent=100*e/len(z),median_W_db=float(z.signed_p95_p05_width_db.median()),P95_W_db=float(z.signed_p95_p05_width_db.quantile(.95))))
    z=new[new.lat_group.eq(lat)];rates.append(dict(lat_group=lat,events=int(z.is_event.sum()),total_windows=len(z),event_fraction_percent=float(100*z.is_event.mean())))
pd.DataFrame(rates).to_csv(DATA/'fig3_panel_b1_event_fraction.csv',index=False)
for layer in ['Topside','F2','F1','E','D']:
    for lat in LATS:
        z=new[new.height_layer.eq(layer)&new.lat_group.eq(lat)]
        alt.append(dict(height_layer=layer,lat_group=lat,events=int(z.is_event.sum()),total_windows=len(z),event_fraction_percent=float(100*z.is_event.mean()) if len(z) else np.nan))
pd.DataFrame(alt).to_csv(DATA/'fig3_panel_d_altitude_layer_heatmap.csv',index=False)
for field in ['signal_name','height_layer']:
    for group in ow[field].unique():
        for stage,t in [('before',ow),('after',new)]:
            z=t[t[field].eq(group)]
            beforeafter.append(dict(group_type=field,group=group,stage=stage,n_windows=len(z),n_events=int(z.is_event.sum()),event_fraction_percent=float(100*z.is_event.mean()) if len(z) else np.nan,median_W_db=float(z.signed_p95_p05_width_db.median()) if len(z) else np.nan,P95_W_db=float(z.signed_p95_p05_width_db.quantile(.95)) if len(z) else np.nan))
pd.DataFrame(beforeafter).to_csv(DATA/'event_before_after_latitude_signal_height_summary.csv',index=False)
kept=p[p.analysis_qc_keep].copy()
kept.to_csv(DATA/'fig3_points_source.csv',index=False)
eventreport={'original_window_n':len(ow),'retained_window_n':len(new),'excluded_N_lt30':int(ledger.reject_N_valid_lt30.sum()),'excluded_abs_median_gt1':int(ledger.reject_abs_median_gt1.sum()),'original_event_n':int(ow.is_event.sum()),'retained_event_n':len(event),'rates':rates,'original_source_point_n':len(p),'all_source_candidate_window_n':len(allw),'all_source_retained_window_n':int(allw.qc_keep.sum()),'all_source_retained_point_n':len(kept)}
(QA/'event_strict_QC_R4.json').write_text(json.dumps(eventreport,indent=2),encoding='utf-8')
print('EVENT_READY',json.dumps(eventreport),flush=True)

# Map has an independently frozen spatial-artifact cohort, including167 map-only
# epochs. Preserve all its original identities, and evaluate its own segment bins.
m=pd.read_parquet(OLD/'spatial_amplitude_epoch_source.parquet')
fullcols=['phase','op_id','sat','signal_id','epoch_key','rx_gps_seconds','time_utc','cn0_dbhz']
f=pd.read_parquet(FULL,columns=fullcols);f=f[f.phase.eq('S')].copy()
f['_utc_label_s']=pd.to_datetime(f.time_utc,utc=True,format='mixed').dt.as_unit('ns').astype('int64')//10**9
mid=['phase','op_id','sat','signal_id','_utc_label_s']
assert not f.duplicated(mid).any()
m=m.merge(f[mid+['epoch_key','rx_gps_seconds','cn0_dbhz']],on=mid,how='left',validate='one_to_one',indicator=True)
assert m._merge.eq('both').all();m=m.drop(columns='_merge')
m['_epoch_ms']=m.epoch_key
assert len(m)==83313 and not m.duplicated(ID).any()
m['qc_window_origin_gps_s']=m.groupby(SEG,dropna=False).rx_gps_seconds.transform('min')
m['qc_window_idx']=np.floor((m.rx_gps_seconds-m.qc_window_origin_gps_s)/60).astype(int)
m['qc_track_uid']=m[SEG].astype(str).agg('|'.join,axis=1)
m['point_three_condition_eligible']=m.cn0_dbhz.ge(25)&m.fit_eligible&m.fit_degree.eq(4)&np.isfinite(m.e23_residual_db)
mw=build_ledger(m);mw.to_csv(DATA/'map_original_cohort_window_QC_ledger.csv',index=False)
m=m.merge(mw[['track_uid','window_idx','n_valid','signed_median_db','reject_N_valid_lt30','reject_abs_median_gt1','qc_keep','qc_reason']].rename(columns={'track_uid':'qc_track_uid','window_idx':'qc_window_idx'}),on=['qc_track_uid','qc_window_idx'],validate='many_to_one')
m['analysis_qc_keep']=m.point_three_condition_eligible&m.qc_keep
maporigids=pd.MultiIndex.from_frame(m[ID]);psdorigids=pd.MultiIndex.from_frame(p[ID])
m['map_only_not_in_original_spectral_points']=~maporigids.isin(psdorigids)
assert m.map_only_not_in_original_spectral_points.sum()==167
m.to_csv(DATA/'map_all_original_point_QC_ledger.csv',index=False)
mk=m[m.analysis_qc_keep].copy();mk.to_csv(DATA/'spatial_amplitude_epoch_source.csv',index=False)
cells=mk.groupby(['lat_bin','lon_bin','lat_center','lon_center'],as_index=False).agg(metric=('e23_abs_residual_db',lambda s:float(s.quantile(.95))),n_samples_1hz=('e23_abs_residual_db','size'),n_minutes=('utc_minute','nunique'),n_track_signal_segments=('track_signal_segment','nunique'),n_signals=('signal_name','nunique'))
oc=pd.read_csv(OLD/'spatial_amplitude_cells.csv')
meta=oc[['lat_bin','lon_bin','mlat_center','lat_group']]
cells=cells.merge(meta,on=['lat_bin','lon_bin'],how='left',validate='one_to_one')
assert cells.lat_group.notna().all()
cells['metric_name']='cellP95 of accepted1Hz absdeltaCN0 after three-condition windowQC';cells['min_cell_minutes']=1
cells.to_csv(DATA/'spatial_amplitude_cells.csv',index=False)
up=pd.read_csv(UPDATED,dtype={'op_id':str});up['_epoch_ms']=up.epoch_key
polar=up.merge(mk[ID],on=ID,how='inner',validate='one_to_one')
assert len(polar)==len(mk)
polar.to_csv(DATA/'polar_strict_accepted_points.csv',index=False)
mapreport={'original_points':len(m),'retained_points':len(mk),'original_window_n':len(mw),'retained_window_n':int(mw.qc_keep.sum()),'rejected_N_lt30':int(mw.reject_N_valid_lt30.sum()),'rejected_median_gt1':int(mw.reject_abs_median_gt1.sum()),'original_cells':len(oc),'retained_cells':len(cells),'map_only_original_n':int(m.map_only_not_in_original_spectral_points.sum()),'map_only_retained_n':int(mk.map_only_not_in_original_spectral_points.sum()),'all_retained_point_CN0_ge25':bool(mk.cn0_dbhz.ge(25).all()),'same_a_b_c_accepted_epoch_set':True,'window_origin':'first epoch in each original spatial-artifact-screened source segment; QC count/median evaluated only within this original map cohort, independently of event/PSD scientific point scope'}
(QA/'map_strict_QC_R4.json').write_text(json.dumps(mapreport,indent=2),encoding='utf-8')
print('MAP_READY',json.dumps(mapreport),flush=True)

# Reuse the same strict-native-bin Welch function and aggregation as selected Fig2.
scope={'np':np,'pd':pd}
node=next(n for n in ast.parse(WELCH.read_text('utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='welch')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(WELCH),'exec'),scope)
grid=np.arange(1,129,dtype=float)/256;lo=grid-1/512;hi=grid+1/512;hi[-1]=.5
spectra=[];metas=[];excluded=[]
for key,g in kept.groupby(SEG+['lat_group'],sort=False,dropna=False):
    g=g.sort_values('rx_gps_seconds',kind='stable').drop_duplicates('rx_gps_seconds',keep='first')
    bad=~np.isclose(g.rx_gps_seconds.diff().to_numpy(),1,atol=1e-8,rtol=0);bad[0]=False
    for rid,q in g.groupby(np.cumsum(bad),sort=False):
        ident=dict(zip(SEG+['lat_group'],key));ident.update(run_id=int(rid),signal_name=q.signal_name.iloc[0])
        if len(q)<60:excluded.append(dict(**ident,n_observed=len(q)));continue
        ff,pp,mn,starts,err=scope['welch'](q.cn0_delta_dbhz.to_numpy(float));fp=ff[1:];power=pp[1:]
        vals=np.full(128,np.nan);counts=np.zeros(128,int);supported=(grid>=fp[0]-1e-12)&(grid<=fp[-1]+1e-12)
        for j,(a,b) in enumerate(zip(lo,hi)):
            if not supported[j]:continue
            take=(fp>=a)&((fp<b) if j<127 else (fp<=b));counts[j]=take.sum()
            if counts[j]:vals[j]=np.mean(power[take])
        uid=len(metas);metas.append(dict(**ident,run_uid=uid,n_observed=len(q),start_gps=q.rx_gps_seconds.iloc[0],end_gps=q.rx_gps_seconds.iloc[-1],welch_nperseg=mn,welch_subwindows=len(starts),parseval_error=err))
        spectra.append(pd.DataFrame(dict(run_uid=uid,signal_name=ident['signal_name'],lat_group=ident['lat_group'],frequency_hz=grid,psd=vals,n_native=counts)))
spec=pd.concat(spectra,ignore_index=True)
panel=spec.groupby(['signal_name','lat_group','frequency_hz'],as_index=False).agg(median_psd_dbhz2_per_hz=('psd','median'),n_runs=('psd','count'))
panel['asd_pair']=panel.signal_name.map(lambda s:'L1 / E1' if s in ['GPS L1','Galileo E1'] else 'L5 / E5a')
panel.to_csv(DATA/'fig3_panel_a_psd_no_time_interpolation.csv',index=False)
pd.DataFrame(metas).to_csv(DATA/'psd_retained_runs.csv',index=False);pd.DataFrame(excluded).to_csv(DATA/'psd_excluded_runs.csv',index=False)
spec.to_csv(DATA/'psd_per_run_frequency_bins.csv',index=False)
unchanged={}
for name in ['OP38_E21_measured_samples.csv','OP76_G01_measured_samples.csv','spatial_roti_cells.csv']:
    shutil.copy2(OLD/name,DATA/name);assert sha(OLD/name)==sha(DATA/name);unchanged[name]=sha(DATA/name)
psdreport={'retained_runs':len(metas),'retained_native_points':int(sum(x['n_observed'] for x in metas)),'excluded_short_runs':len(excluded),'continuous_run_minimum':60,'strict_1Hz_gaps_split':True,'method':'unchanged native-bin Hann Welch, no time/frequency interpolation, median across qualified continuous runs','maximum_parseval_error':float(max(x['parseval_error'] for x in metas))}
report={'status':'QC_ONLY_DATA_READY','source_sha256':sources,'three_conditions':'pointCN0>=25dB-Hz;actualvalidseconds>=30 in fixed60s window;abs(signedmedian residual)<=1dB','no_full_native_segment_or_local_CN0_gates_added':True,'no_refit_or_residual_recentering':True,'event':eventreport,'map':mapreport,'PSD':psdreport,'raw_cases_and_ROTI_byte_identical':unchanged,'pre_QC_cohort_identities_preserved':True,'scientific_scope':'Each original panel point range remains an upper bound. Event domain remains original2244window identities. Map preserves its83313point spatial artifact cohort and its map-only167epochs subject to documented same-rule QC. PSD preserves137686point upper bound. No previously excluded point identity is restored.'}
(QA/'event_map_PSD_strict_QC_R4.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('ALL_DATA_READY',json.dumps(psdreport),flush=True)
