"""Recompute E21/high-altitude R4 quality screening from native samples and E23 fits.

The default compact table contains immutable expected_* E23 results already
verified by the optional full native refit. Set MAIN12_REFIT_RESULTS to the
optional full-refit output to use those newly recomputed results instead.
"""
from pathlib import Path
import os,sys,json,hashlib,ast
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
import pyarrow.parquet as pq
C=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MAIN12_OUTPUT',str(C/'outputs')))/'qc_background'
DATA=ROOT/'data';QA=ROOT/'qa'
for p in [DATA,QA]:p.mkdir(parents=True,exist_ok=True)
KEY=['phase','op_id','sat','signal_id','epoch_key']
SEG=['phase','op_id','sat','signal_id','segment_id']
frozen=C/'data/full_lugre/native_cn0_verified_geometry_and_expected_results.parquet'
source=Path(os.environ.get('MAIN12_REFIT_RESULTS',str(frozen)))
schema=set(pq.ParquetFile(source).schema.names)
needed=['phase','op_id','sat','signal_id','signal_name','time_utc','rx_gps_seconds','cn0_dbhz','epoch_key','lugre_h_tan_wgs84_km']
if 'expected_fit_eligible' in schema:
    needed += ['expected_fit_eligible','expected_fit_status','expected_e23_residual_db']
else:
    needed += ['fit_eligible','fit_status','fit_degree','e23_residual_db']
d=pd.read_parquet(source,columns=needed,filters=[('phase','==','S')])
print('QC_BACKGROUND_LUNAR_NATIVE_ROWS',len(d),flush=True)
if 'expected_fit_eligible' in d:
    d=d.rename(columns={k:k.removeprefix('expected_') for k in d if k.startswith('expected_')})
if 'fit_degree' not in d:
    d['fit_degree']=np.where(d.fit_status.eq('quartic'),4,0)
d=d[d.phase.eq('S')].copy()
context=pd.read_parquet(C/'data/r4/qc_background/native_segment_context.parquet',filters=[('phase','==','S')])
context=context[context.phase.eq('S')]
for q in [d,context]:
    for k in ['phase','op_id','sat']:q[k]=q[k].astype(str)
if 'segment_id' not in d:d=d.merge(context,on=KEY,how='left',validate='one_to_one')
assert len(d)==880644 and d.segment_id.notna().all() and not d.duplicated(KEY).any()
del context
d=d.sort_values(SEG+['rx_gps_seconds']).reset_index(drop=True)
d['source_segment_uid']=d[SEG[0]].astype(str)
for k in SEG[1:]:d['source_segment_uid']=d.source_segment_uid+'|'+d[k].astype(str)
d['window_origin_gps_s']=d.groupby(SEG,dropna=False).rx_gps_seconds.transform('min')
d['window_idx']=np.floor((d.rx_gps_seconds-d.window_origin_gps_s+1e-7)/60).astype('int32')
d['window_start_gps_s']=d.window_origin_gps_s+60*d.window_idx
d['window_start_utc']=pd.to_datetime(d.window_start_gps_s-18,unit='s',origin='1980-01-06',utc=True)
d['window_uid']=d.source_segment_uid+'|w'+d.window_idx.astype(str)
d['qualified_original_quartic']=d.fit_eligible&d.fit_degree.eq(4)&np.isfinite(d.e23_residual_db)
d['reject_no_qualified_quartic']=~d.qualified_original_quartic
d['reject_point_cn0_lt25']=d.cn0_dbhz.lt(25)
# No extra segment-C/N0 or moving-local-C/N0 gate is applied in R4.
d['reject_segment_median_cn0_lt25']=False
d['reject_local_60s_median_cn0_lt25']=False
d['point_quality_eligible']=d.qualified_original_quartic & d.cn0_dbhz.ge(25)
d['utc_day']=d.time_utc.str.slice(0,10)
helper=C/'code/r4_background_population.py'
code=helper.read_text('utf-8')
code=code.replace("z.to_csv(DATA / f'{name}_all_points_with_QC_flags.csv', index=False)","\n    if name == 'E21_first_pass':\n        z.to_csv(DATA / f'{name}_all_points_with_QC_flags.csv', index=False)")
scope={'np':np,'pd':pd,'KEY':KEY,'SEG':SEG,'DATA':DATA}
exec(compile(code,str(helper),'exec'),scope)
make=scope['make_population'];qstats=scope['qstats']
high=d[d.lugre_h_tan_wgs84_km.gt(1000)].copy();assert len(high)==722730
ids=pd.read_csv(C/'data/r4/e21/E21_OP38_all_native_observations_with_geometry.csv',usecols=KEY+['time_utc'],dtype={'op_id':str})
ids=ids[pd.to_datetime(ids.time_utc,utc=True)<pd.Timestamp('2025-03-03 15:00',tz='UTC')]
for k in ['phase','op_id','sat']:ids[k]=ids[k].astype(str)
e=d.merge(ids[KEY],on=KEY,how='inner',validate='one_to_one');assert len(e)==6586
print('QC_BACKGROUND_WINDOW_CONTEXT_READY',int(d.window_uid.nunique()),flush=True)
_,ew,ekeep,ereport=make(e,'E21_first_pass')
print('QC_BACKGROUND_E21_RETAINED',len(ekeep),flush=True)
_,hw,hkeep,hreport=make(high,'highalt')
print('QC_BACKGROUND_HIGHALT_RETAINED',len(hkeep),flush=True)
assert len(hkeep)==9178 and len(ekeep)==95
rows=[];sums=[]
for (day,sig),q in hkeep.groupby(['utc_day','signal_name'],sort=True):
    p05,med,p95=np.quantile(q.W_db,[.05,.5,.95])
    rows.append(dict(utc_day=day,signal_name=sig,n_windows=len(q),p05=float(p05),median=float(med),p95=float(p95)))
for sig,q in hkeep.groupby('signal_name',sort=False):sums.append(dict(signal_name=sig,n_windows=len(q),**qstats(q.W_db)))
daily=pd.DataFrame(rows);summary=pd.DataFrame(sums)
daily.to_csv(DATA/'highalt_retained_daily_widths.csv',index=False)
summary.to_csv(DATA/'highalt_retained_width_summary.csv',index=False)
expected=pd.read_csv(C/'data/r4/expected/highalt_retained_width_summary.csv').set_index('signal_name').sort_index()
actual=summary.set_index('signal_name').sort_index()
np.testing.assert_array_equal(actual.n_windows,expected.n_windows)
np.testing.assert_allclose(actual[['p05','median','p95']],expected[['p05','median','p95']],rtol=0,atol=1e-10)
qa=dict(status='PASS_R4_QC_BACKGROUND_RECOMPUTED',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    rules=['Existing qualified fourth-order fit','Point C/N0 >=25 dB-Hz','At least30 actual valid1Hz samples per anchored60s window','Absolute signed residual median <=1dB; failing windows excluded in full'],
    highalt_retained_windows=len(hkeep),highalt_median_W_db=float(hkeep.W_db.median()),reports={'E21_first_pass':ereport,'highalt':hreport},
    signal_summary_matches_selected_R4_with_tolerance_db=1e-10,point_quantile_tail_deletion=False)
(QA/'background_QC_R4.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:qa[k] for k in ['status','highalt_retained_windows','highalt_median_W_db']}),flush=True)
