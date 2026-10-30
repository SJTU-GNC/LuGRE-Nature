"""Recompute all LuGRE C/N0 using the archived E23 algorithm.

Physical-pass candidates are phase/satellite/signal tracks split only where
successive >=50-km measurements are more than 7200 s apart, including across
OP files. This is the global extension of the archived stage-local definition.
Native measurements only; no gap samples, constant fallbacks or interpolation
are admitted as fourth-order residuals. Old results remain untouched.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for folder in ['data', 'qa', 'geometry', 'code']:
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,'H:/LUGRE/python_deps/pyarrow_py312')
import numpy as np
import pandas as pd

PAPER=Path('D:/Reference/Paper/2026/Nature_LuGRE')
SOURCE=Path('H:/LUGRE/lugre_occultation_match/output/phase_story_all_heights_source/lugre_all_visible_master_tangent_cn0_tec_rot_old_logic_all_heights_with_s_phase_detrended_cn0.csv')
GEOMETRY=Path('H:/LUGRE/repro_20260720/shared/WGS84/phase_story_new_reference_tangent_geometry_wgs84_compact.csv')
PATCH_GEOMETRY=ROOT/'geometry/geometry_missing_source_epochs.csv'
FIT=PAPER/'Figure/op38_e23_front_arc_example_package_20260616_145440/code/GPS_simulator/Build_QuadraticDetrendExample.py'
E23=PAPER/'Figure/op38_e23_front_arc_example_package_20260616_145440/data/op38_e23_anomaly_focused_cn0_data.csv'
POINTS=PAPER/'Figure/wgs84_tangent_height_update_20260701/data/Fig1_50_1000_points_cache_4fitting_wgs84.csv'
KEYS=['phase','op_id','sat','signal_id','epoch_key']

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def serial(obj):
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,Path):return str(obj)
    raise TypeError(type(obj))

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=serial),encoding='utf-8')

def keyify(df):
    df['epoch_key']=np.rint(df.rx_gps_seconds.to_numpy(float)*1000).astype('int64')
    assert not df.duplicated(KEYS).any(), 'Duplicate source identity.'
    return df

scope={'np':np,'pd':pd}
tree=ast.parse(FIT.read_text(encoding='utf-8-sig'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
       {'median_abs_deviation','binned_median_fit_points','fit_poly_trend'}]
exec(compile(ast.Module(nodes,type_ignores=[]),str(FIT),'exec'),scope)
fit=scope['fit_poly_trend']

started=time.monotonic()
raw_cache=ROOT/'data/all_LuGRE_positive_native_context.parquet'
if raw_cache.exists():
    raw=pd.read_parquet(raw_cache)
else:
    cols=['phase','op_id','sat','signal_id','signal_name','segment_id','time_utc',
          'rx_gps_seconds','cn0_dbhz','tangent_lat_deg_old_logic',
          'tangent_lon_deg_old_logic','h_tan_km_old_logic','stec_tecu',
          'rot_tecu_per_min','roti_1min','roti_5min',
          's_phase_cn0_trend_dbhz','s_phase_cn0_detrended_db']
    raw=keyify(pd.read_csv(SOURCE,usecols=cols,low_memory=False))
    assert len(raw)==993093 and np.isfinite(raw.cn0_dbhz).all() and raw.cn0_dbhz.gt(0).all()
    raw.to_parquet(raw_cache,index=False)
    write_json(ROOT/'qa/raw_context_cache.json',{'source':SOURCE,'source_sha256':sha(SOURCE),
        'rows':len(raw),'all_positive':True,'no_duplicate_identity':True,
        'counts':raw.groupby(['phase','signal_name']).size().reset_index(name='n').to_dict('records')})
print(json.dumps({'stage':'raw_context_ready','rows':len(raw),'elapsed_s':time.monotonic()-started}),flush=True)
if '--cache-only' in sys.argv:sys.exit(0)

gcols=KEYS[:-1]+['rx_gps_seconds','lugre_lat_wgs84','lugre_lon_wgs84','lugre_h_tan_wgs84_km']
geo=keyify(pd.read_csv(GEOMETRY,usecols=gcols,low_memory=False))
merged=raw.merge(geo[KEYS+gcols[-3:]],on=KEYS,how='left',validate='1:1')
missing_before=int(merged.lugre_h_tan_wgs84_km.isna().sum())
if missing_before:
    if not PATCH_GEOMETRY.exists():
        raise RuntimeError(f'{missing_before} source epochs require independently calculated WGS84 geometry.')
    patch=keyify(pd.read_csv(PATCH_GEOMETRY))
    patch=patch.set_index(KEYS)
    idx=pd.MultiIndex.from_frame(merged[KEYS])
    for col in gcols[-3:]:
        missing=merged[col].isna()
        merged.loc[missing,col]=patch[col].reindex(idx[missing]).to_numpy()
missing_after=int(merged.lugre_h_tan_wgs84_km.isna().sum())
# Keep unresolvable source epochs explicitly; never substitute the older
# receiver frame or claim they supplied qualified fourth-order residuals.
raw=merged
raw['lugre_lat']=raw.lugre_lat_wgs84
raw['lugre_lon']=raw.lugre_lon_wgs84
raw['lugre_h_tan_km']=raw.lugre_h_tan_wgs84_km
raw['fit_input_eligible']=raw.lugre_h_tan_km.ge(50)
raw['fit_eligible']=False
raw['fit_degree']=pd.Series(pd.NA,index=raw.index,dtype='Int8')
raw['e23_local_arc_id']=''
raw['fit_status']=np.where(raw.fit_input_eligible,'pending',
    np.where(raw.lugre_h_tan_km.isna(),'missing_new_reference_geometry','below_50km_fit_domain'))
raw['quartic_background_dbhz']=np.nan
raw['e23_residual_db']=np.nan
raw['e23_abs_residual_db']=np.nan
buffers={col:raw[col].to_numpy(copy=True) for col in
    ['fit_eligible','fit_degree','e23_local_arc_id','fit_status',
     'quartic_background_dbhz','e23_residual_db','e23_abs_residual_db']}

# Original example must still reproduce exactly from its own saved inputs.
e23=pd.read_csv(E23)
example_proof=[]
for track,q in e23.groupby('track_label'):
    obs=q[~q.cn0_is_filled.astype(str).str.lower().eq('true')]
    t=obs.rx_gps_seconds.to_numpy(float)
    _,meta=fit(t,obs.cn0_dbhz.to_numpy(float),4)
    t0=float(np.median(t)); scale=float(np.max(abs(t-t0)))
    pred=np.polyval(meta['coefficients'],(q.rx_gps_seconds.to_numpy(float)-t0)/scale)
    err=float(np.max(abs(pred-q.cn0_trend_dbhz.to_numpy(float))))
    assert err<1e-10
    example_proof.append({'track':track,'baseline_max_abs_error_db':err})

ledger=[]
for (phase,sat,signal),allq in raw.groupby(['phase','sat','signal_id'],sort=True):
    q=allq[allq.fit_input_eligible].sort_values('rx_gps_seconds').copy()
    if q.empty:continue
    times=q.rx_gps_seconds.to_numpy(float)
    q['_arc']=np.cumsum(np.r_[False,np.diff(times)>7200])+1
    for arc,part in q.groupby('_arc',sort=True):
        t=part.rx_gps_seconds.to_numpy(float); y=part.cn0_dbhz.to_numpy(float)
        trend,meta=fit(t,y,4)
        arc_id=f'{phase}_{sat}_sig{int(signal)}_local{int(arc):03d}'
        qualified=meta['degree']==4 and meta['fit_points']>5
        indices=part.index.to_numpy()
        buffers['fit_degree'][indices]=meta['degree']
        buffers['e23_local_arc_id'][indices]=arc_id
        buffers['fit_status'][indices]='quartic' if qualified else meta['method']
        if qualified:
            assert np.isfinite(trend).all()
            buffers['quartic_background_dbhz'][indices]=trend
            buffers['e23_residual_db'][indices]=y-trend
            buffers['e23_abs_residual_db'][indices]=abs(y-trend)
            buffers['fit_eligible'][indices]=True
        center=float(np.median(t)); scale=max(float(np.max(abs(t-center))),1.)
        changes=part.op_id.ne(part.op_id.shift()).to_numpy(copy=True)
        changes[0]=False
        cross=[]
        for pos in np.flatnonzero(changes):
            cross.append({'left_op':str(part.op_id.iloc[pos-1]),'right_op':str(part.op_id.iloc[pos]),
                          'gap_s':float(t[pos]-t[pos-1])})
        xfit=(np.asarray(meta['fit_bin_time_s'])-center)/scale
        entry={'arc_id':arc_id,'phase':phase,'sat':sat,'signal_id':int(signal),
            'signal_name':part.signal_name.iloc[0],'observed_n':len(part),
            'start_utc':part.time_utc.iloc[0],'end_utc':part.time_utc.iloc[-1],
            'gps_start':float(t.min()),'gps_end':float(t.max()),
            'center_gps_seconds':center,'normalization_scale_s':scale,
            'op_ids':list(part.op_id.drop_duplicates()),'cross_op_boundaries':cross,
            'maximum_observed_gap_s':float(np.diff(t).max()) if len(t)>1 else 0.,
            'gaps_over_600s':int((np.diff(t)>600).sum()),
            'admitted_as_fourth_order':qualified,
            'design_condition_number':float(np.linalg.cond(np.vander(xfit,5))) if len(xfit)>=5 else None,
            'residual_min_max_db':[float(np.min(y-trend)),float(np.max(y-trend))],**meta}
        ledger.append(entry)
        if len(ledger)%100==0:
            print(json.dumps({'stage':'fitting','arcs_completed':len(ledger),'elapsed_s':time.monotonic()-started}),flush=True)

for col,values in buffers.items():raw[col]=values
raw['fit_degree']=pd.array(raw.fit_degree,dtype='Int8')
assert not raw.fit_status.eq('pending').any()
raw['plot_cn0_detrended_db']=raw.e23_residual_db
raw['cn0_detrend_zero_mean_dbhz']=raw.e23_residual_db
raw['cn0_detrend_zero_mean_abs_dbhz']=raw.e23_abs_residual_db
raw['freq_group']=raw.signal_name
out=ROOT/'data/all_LuGRE_E23_method_residuals.parquet'
raw.to_parquet(out,index=False)
write_json(ROOT/'data/all_LuGRE_E23_fit_ledger.json',ledger)
pd.DataFrame([{k:v for k,v in row.items() if k not in ['coefficients','fit_bin_time_s','fit_bin_cn0_dbhz','cross_op_boundaries']} for row in ledger]).to_csv(ROOT/'data/all_LuGRE_E23_fit_ledger.csv',index=False)

# Replace residual values only within the existing figure-point identities.
# Geometry, raw measurements and old source identities stay byte-equivalent
# as parsed; old residual columns are retained under explicit legacy names.
points=keyify(pd.read_csv(POINTS,low_memory=False))
identity_cols=KEYS+['cn0_dbhz','fit_eligible','fit_status','fit_degree','e23_local_arc_id',
    'quartic_background_dbhz','e23_residual_db','e23_abs_residual_db']
new=points.merge(raw[identity_cols],on=KEYS,how='left',validate='1:1',suffixes=('','_refit'))
assert len(new)==len(points) and new.cn0_dbhz_refit.notna().all()
raw_error=float((new.cn0_dbhz-new.cn0_dbhz_refit).abs().max())
assert raw_error<1e-10
new=new.drop(columns='cn0_dbhz_refit')
for col in ['plot_cn0_detrended_db','cn0_detrend_zero_mean_dbhz','cn0_detrend_zero_mean_abs_dbhz']:
    new['legacy_'+col]=new[col]
    new[col]=new.e23_abs_residual_db if 'abs' in col else new.e23_residual_db
new.to_csv(ROOT/'data/updated_points.csv',index=False)
new[new.fit_eligible].to_csv(ROOT/'data/updated_points_quartic_eligible.csv',index=False)

qa={'status':'complete_all_source_epochs_accounted_for','revision':'R3',
    'algorithm_source':str(FIT),'algorithm_sha256':sha(FIT),
    'source':str(SOURCE),'source_sha256':sha(SOURCE),
    'WGS84_source':str(GEOMETRY),'WGS84_source_sha256':sha(GEOMETRY),
    'geometry_missing_before_patch':missing_before,'geometry_missing_after_patch':missing_after,
    'scope':'All four LuGRE mission phases and all four GNSS signals in positive-native source; not only existing manuscript subset.',
    'raw_source_rows':len(raw),'invalid_raw_CNo_excluded_upstream':3,
    'raw_phase_signal_counts':raw.groupby(['phase','signal_name']).size().reset_index(name='n').to_dict('records'),
    'global_arc_keys':['phase','sat','signal_id'],'op_file_boundary_does_not_split_arc':True,
    'local_arc_split_rule':'After h>=50 km selection, successive real GPS seconds gap >7200 s.',
    'fit_height_min_km':50,'fit_bins':'absolute GPS 60 s bins; median C/N0; sqrt(count) weight',
    'robust_iteration':'Exact archived E23 function; up to 4 MAD rejection rounds, max(1 dB,2.8 scale).',
    'gap_samples_in_fit_or_statistics':0,
    'degree0_fallbacks_admitted_as_quartic':False,
    'height_domain_extension_below_80km':True,
    'fit_status_counts':raw.fit_status.value_counts().to_dict(),
    'arcs':len(ledger),'qualified_quartic_arcs':sum(x['admitted_as_fourth_order'] for x in ledger),
    'cross_op_arcs':[x['arc_id'] for x in ledger if x['cross_op_boundaries']],
    'E23_independent_reproduction':example_proof,
    'old_figure_point_rows':len(points),'new_figure_point_fit_status':new.fit_status.value_counts().to_dict(),
    'figure_point_raw_CNo_max_abs_difference_db':raw_error,
    'output':str(out),'output_sha256':sha(out),'elapsed_s':time.monotonic()-started,
    'legacy_files_modified':False}
write_json(ROOT/'qa/all_LuGRE_E23_method_QA.json',qa)
print(json.dumps(qa,default=serial),flush=True)
