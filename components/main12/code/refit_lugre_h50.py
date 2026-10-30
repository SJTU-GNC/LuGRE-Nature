"""Optional full-population native C/N0 to E23 h50 residual verification.

Uses only the bundled compact native observations and independently calculated
WGS84 geometry. Expected frozen columns are validation targets, never fit inputs.
This does not rerun spacecraft/GNSS orbit determination or reacquire mission data.
"""
from pathlib import Path
import argparse, ast, json, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'outputs/full_lugre_refit')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
scope={'np':np,'pd':pd}
fit_source=ROOT/'code/e21_fit.py'
tree=ast.parse(fit_source.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
exec(compile(ast.Module(nodes,type_ignores=[]),str(fit_source),'exec'),scope)
fit=scope['fit_poly_trend']
raw=pd.read_parquet(ROOT/'data/full_lugre/native_cn0_verified_geometry_and_expected_results.parquet')
assert len(raw)==993093
assert not raw.duplicated(['phase','op_id','sat','signal_id','epoch_key']).any()
assert np.isfinite(raw.cn0_dbhz).all() and raw.cn0_dbhz.gt(0).all()
eligible=raw.lugre_h_tan_wgs84_km.ge(50)
background=np.full(len(raw),np.nan);residual=np.full(len(raw),np.nan)
status=np.where(eligible,'pending',np.where(raw.lugre_h_tan_wgs84_km.isna(),'missing_new_reference_geometry','below_50km_fit_domain')).astype(object)
arcids=np.full(len(raw),'',dtype=object)
degrees=np.full(len(raw),np.nan);fit_ok=np.zeros(len(raw),dtype=bool)
ledger=[];started=time.monotonic()
for (phase,sat,signal),allq in raw.groupby(['phase','sat','signal_id'],sort=True):
    q=allq[eligible.loc[allq.index]].sort_values('rx_gps_seconds').copy()
    if not len(q):continue
    q['_arc']=np.cumsum(np.r_[False,np.diff(q.rx_gps_seconds.to_numpy(float))>7200])+1
    for arc,p in q.groupby('_arc',sort=True):
        t=p.rx_gps_seconds.to_numpy(float);y=p.cn0_dbhz.to_numpy(float)
        trend,meta=fit(t,y,4)
        qualified=meta['degree']==4 and meta['fit_points']>5
        ix=p.index.to_numpy();name=f'{phase}_{sat}_sig{int(signal)}_local{int(arc):03d}'
        status[ix]='quartic' if qualified else meta['method'];arcids[ix]=name;degrees[ix]=meta['degree']
        if qualified:
            background[ix]=trend;residual[ix]=y-trend;fit_ok[ix]=True
        ledger.append({'arc_id':name,'phase':phase,'sat':sat,'signal_id':int(signal),'observed_n':len(p),
            'op_ids':list(p.op_id.drop_duplicates()),'maximum_observed_gap_s':float(np.diff(t).max()) if len(t)>1 else 0.,
            'admitted_as_fourth_order':qualified,**meta})
        if len(ledger)%100==0:print(json.dumps({'arcs_completed':len(ledger),'elapsed_seconds':time.monotonic()-started}),flush=True)
assert not np.any(status=='pending')
np.testing.assert_array_equal(fit_ok,raw.expected_fit_eligible.to_numpy(bool))
np.testing.assert_array_equal(status,raw.expected_fit_status.to_numpy())
np.testing.assert_array_equal(arcids,raw.expected_e23_local_arc_id.fillna('').to_numpy())
np.testing.assert_allclose(background,raw.expected_quartic_background_dbhz.to_numpy(float),atol=1e-10,rtol=0,equal_nan=True)
np.testing.assert_allclose(residual,raw.expected_e23_residual_db.to_numpy(float),atol=1e-10,rtol=0,equal_nan=True)
out=raw.drop(columns=[c for c in raw if c.startswith('expected_')]).copy()
out['fit_eligible']=fit_ok;out['fit_status']=status;out['e23_local_arc_id']=arcids
out['fit_degree']=pd.array(degrees,dtype='Int8');out['quartic_background_dbhz']=background
out['e23_residual_db']=residual;out['e23_abs_residual_db']=np.abs(residual)
out.to_parquet(args.output/'all_LuGRE_E23_h50_recomputed.parquet',index=False)
(args.output/'fit_ledger.json').write_text(json.dumps(ledger,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
qa={'status':'PASS','native_positive_epochs':len(raw),'arcs':len(ledger),'qualified_quartic_arcs':sum(x['admitted_as_fourth_order'] for x in ledger),
    'fit_height_min_km':50,'statistical_shell_km':[50,1000],'high_altitude_fit_context_preserved':True,
    'global_group_keys':['phase','sat','signal_id'],'split_only_after_height_selection_at_gap_gt_s':7200,
    'synthetic_samples':0,'background_and_residual_maximum_tolerance_db':1e-10,
    'frozen_eligibility_status_and_arc_ids_identical':True,'expected_results_never_enter_fit':True,
    'fit_status_counts':pd.Series(status).value_counts().to_dict(),'elapsed_seconds':time.monotonic()-started}
qa['numpy_version']=np.__version__;qa['pandas_version']=pd.__version__
(args.output/'verification.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps(qa),flush=True)
