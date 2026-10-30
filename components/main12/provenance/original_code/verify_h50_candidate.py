"""Verify the controlled LuGRE height-domain extension against frozen h80 outputs."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
sys.path.insert(0,'H:/LUGRE/python_deps/pyarrow_py312')
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('H:/LUGRE/figures_R3/LuGRE_E23_method_full_refit_R3_20261023')
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def describe(x):
    x=np.asarray(x,float)
    return dict(n=len(x),min=float(np.min(x)),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(np.max(x))) if len(x) else dict(n=0)
old=pd.read_parquet(OLD/'data/all_LuGRE_E23_method_residuals.parquet')
new=pd.read_parquet(ROOT/'data/all_LuGRE_E23_method_residuals.parquet')
keys=['phase','op_id','sat','signal_id','epoch_key']
assert len(new)==len(old)==993093
for cols in [keys,['time_utc','rx_gps_seconds','cn0_dbhz'],['lugre_lat','lugre_lon','lugre_h_tan_km']]:
    pd.testing.assert_frame_equal(new[cols],old[cols],check_exact=True)
eligible=new.fit_eligible.astype(bool)
assert (new.fit_input_eligible==new.lugre_h_tan_km.ge(50)).all()
assert new.loc[eligible,'fit_degree'].eq(4).all()
assert new.loc[eligible,'lugre_h_tan_km'].ge(50).all()
assert new.loc[~eligible,'e23_residual_db'].isna().all()
np.testing.assert_allclose(new.loc[eligible,'e23_residual_db'],new.loc[eligible,'cn0_dbhz']-new.loc[eligible,'quartic_background_dbhz'],rtol=0,atol=0)
np.testing.assert_allclose(new.loc[eligible,'e23_abs_residual_db'],np.abs(new.loc[eligible,'e23_residual_db']),rtol=0,atol=0)
common=eligible & old.fit_eligible.astype(bool)
added=eligible & ~old.fit_eligible.astype(bool)
removed=old.fit_eligible.astype(bool)&~eligible
ledger=json.loads((ROOT/'data/all_LuGRE_E23_fit_ledger.json').read_text(encoding='utf-8'))
qa=json.loads((ROOT/'qa/all_LuGRE_E23_method_QA.json').read_text(encoding='utf-8'))
prior=json.loads((OLD/'qa/all_LuGRE_E23_method_QA.json').read_text(encoding='utf-8'))
for field in ['algorithm_sha256','source_sha256','WGS84_source_sha256','geometry_missing_after_patch']:
    assert qa[field]==prior[field],field
assert qa['fit_height_min_km']==50 and qa['gap_samples_in_fit_or_statistics']==0
assert all(x['maximum_observed_gap_s']<=7200 for x in ledger)
added_rows=new.loc[added,keys+['time_utc','cn0_dbhz','lugre_h_tan_km','fit_degree','e23_local_arc_id','e23_residual_db']]
added_rows.to_csv(ROOT/'qa/newly_eligible_h50_epochs.csv',index=False)
report=dict(status='PASS',controlled_change='LuGRE measured-input and residual-output height threshold 80 to 50 km; exact archived E23 estimator retained',raw_source_rows=len(new),raw_values_geometry_and_identity_unchanged=True,
 old_qualified=int(old.fit_eligible.sum()),new_qualified=int(eligible.sum()),added_qualified=int(added.sum()),removed_qualified=int(removed.sum()),
 added_height_km=describe(new.loc[added,'lugre_h_tan_km']),eligible_h50_to_h80=int((eligible & new.lugre_h_tan_km.ge(50)&new.lugre_h_tan_km.lt(80)).sum()),
 common_qualified_residual_abs_difference_db=describe(np.abs(new.loc[common,'e23_residual_db']-old.loc[common,'e23_residual_db'])),
 old_fit_status_counts=old.fit_status.value_counts().to_dict(),new_fit_status_counts=new.fit_status.value_counts().to_dict(),
 sources_unchanged=['native C/N0 source','WGS84 geometry source','geometry recovery patch','archived E23 fit implementation'],
 old_full_sha256=digest(OLD/'data/all_LuGRE_E23_method_residuals.parquet'),new_full_sha256=digest(ROOT/'data/all_LuGRE_E23_method_residuals.parquet'),
 new_fit_QA=qa,Ground_LEO_unchanged=True,formal_manuscript_promoted=False)
sourceqa=ROOT/'fig2_adapter/Fig2_source_rebuild_QA.json'
if sourceqa.exists():
    a=json.loads((OLD/'fig2_adapter/Fig2_source_rebuild_QA.json').read_text(encoding='utf-8'))
    b=json.loads(sourceqa.read_text(encoding='utf-8'))
    unchanged=[]
    for path,value in a['source_hashes'].items():
        if path.endswith('all_LuGRE_E23_method_residuals.parquet'):continue
        assert path in b['source_hashes'] and value==b['source_hashes'][path],path
        unchanged.append(dict(path=path,sha256=value))
    report['frozen_Fig2_cohorts_QC_and_artifact_mask_source_hashes']=unchanged
    report['Fig2_before']=a['QA'];report['Fig2_after']=b['QA']
(ROOT/'qa/h50_independent_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','raw_source_rows','old_qualified','new_qualified','added_qualified','removed_qualified','added_height_km','common_qualified_residual_abs_difference_db']},indent=2))
