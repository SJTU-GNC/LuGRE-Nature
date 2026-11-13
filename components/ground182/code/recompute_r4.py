"""Recompute R4 LuGRE window QC and figure summaries from package-local inputs.

Ground182 and LEO retain their published processed inputs. This does not rerun
raw multi-network acquisition or any platform's detrending.
"""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from numerical_helpers import _amplitude_tables,build_angle_bins
ROOT=Path(__file__).resolve().parents[1]
BANDS=['Equatorial','Mid-latitude','Polar']
WK=['platform','source_unit','satellite','signal','fit_arc_id','qc_segment_id','latitude_band','continuous_run_id','utc_minute_start_s']

def readcsv(p):return pd.read_csv(p,float_precision='round_trip',low_memory=False)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def key(row):
 result=[]
 for k in WK:
  value=row[k]
  if k in ['continuous_run_id','utc_minute_start_s']:value=str(int(float(value)))
  elif k=='qc_segment_id':
   value=str(value)
   try:
    number=float(value)
    if number.is_integer():value=str(int(number))
   except ValueError:pass
  result.append(str(value))
 return tuple(result)

def verify_tables(out,kept):
 checks={}
 for folder,names in [('amplitude_source',['histogram_source_data.csv','signed_distribution_summary.csv','quantiles.csv','height_envelopes.csv']),('response_source',['common_60s_response_windows_R3.csv','angle_bin_median_IQR_counts_R3.csv'])]:
  for name in names:
   expected=ROOT/'fig34'/folder/name;actual=out/folder/name
   if sha(actual)==sha(expected):checks[folder+'/'+name]='byte identical'
   else:
    a,b=readcsv(actual),readcsv(expected)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=0,atol=5e-14)
    checks[folder+'/'+name]='all numerical values equal within5e-14'
 for name in ['lugre_height_selection.npz','leo_delta_height_selection.npz']:
  with np.load(out/'amplitude_source'/name,allow_pickle=False) as a,np.load(ROOT/'fig34/amplitude_source'/name,allow_pickle=False) as b:
   assert set(a.files)==set(b.files)
   for field in a.files:np.testing.assert_array_equal(a[field],b[field])
  checks[name]='all arrays exactly equal'
 a=pd.read_parquet(out/'illumination/nearspace_solar_windows_R3.parquet')
 b=pd.read_parquet(ROOT/'illumination/nearspace/nearspace_solar_windows_R3.parquet')
 pd.testing.assert_frame_equal(a,b,check_exact=True)
 checks['nearspace_solar_windows_R3.parquet']='all values exactly equal; no solar recomputation'
 return checks

def rebuild(output,verify=True):
 out=Path(output).resolve()
 print('Reconstructing R4 LuGRE window QC from package-local epochs',flush=True)
 for name in ['data','amplitude_source','response_source','illumination']: (out/name).mkdir(parents=True,exist_ok=True)
 x=pd.read_parquet(ROOT/'inputs_r4/LuGRE_comparison_epochs_before_QC.parquet')
 ledger=pd.read_parquet(ROOT/'inputs_r4/LuGRE_comparison_candidate_windows_before_QC.parquet')
 assert len(x)==82761 and len(ledger)==1654
 assert not x.duplicated(['op_id','sat','signal_id','epoch_key']).any()
 point=x.cn0_dbhz.ge(25)&x.fit_eligible.astype(bool)&x.fit_degree.eq(4)&np.isfinite(x.delta_cn0_db)
 assert point.all(),'This release freezes the previously qualified82761-point input.'
 assert np.array_equal(x.delta_cn0_db,x.e23_residual_db)
 assert np.array_equal(x.utc_minute_start_s,np.floor(x.utc_s/60).astype('int64')*60)
 g=x[point].groupby(WK,sort=False,dropna=False,observed=True)
 n=g.size().rename('N_valid');q=g.delta_cn0_db.quantile([.05,.5,.95]).unstack()
 stat=pd.concat([n,q.rename(columns={.05:'current_P05_db',.5:'current_median_db',.95:'current_P95_db'})],axis=1).reset_index()
 ledger=ledger.merge(stat,on=WK,how='left',validate='one_to_one')
 ledger.N_valid=ledger.N_valid.fillna(0).astype(int)
 ledger['reject_N_valid_lt30']=ledger.N_valid.lt(30)
 ledger['reject_abs_median_gt1']=ledger.N_valid.ge(30)&ledger.current_median_db.abs().gt(1)
 ledger['three_conditions_keep']=~ledger.reject_N_valid_lt30&~ledger.reject_abs_median_gt1&ledger.current_median_db.notna()
 ledger['Fig4_LT_keep']=ledger.three_conditions_keep&ledger.complete_geometry_ok
 assert ledger.three_conditions_keep.sum()==1354 and ledger.Fig4_LT_keep.sum()==1342
 assert np.max(abs(ledger.width_db-(ledger.current_P95_db-ledger.current_P05_db)))<1e-12
 indexed=ledger.set_index(WK).three_conditions_keep
 keep=point&pd.MultiIndex.from_frame(x[WK]).map(indexed).to_numpy(dtype=bool)
 kept=x[keep].copy();assert len(kept)==77945
 kept.to_parquet(out/'data/LuGRE_retained_epochs_R4.parquet',index=False,compression='zstd')
 ledger.to_parquet(out/'data/LuGRE_window_QC_ledger_R4.parquet',index=False,compression='zstd')
 groups={('LuGRE',b):kept.loc[kept.latitude_band.eq(b),'delta_cn0_db'].to_numpy() for b in BANDS}
 A=out/'amplitude_source';R=out/'response_source'
 for table,name in zip(_amplitude_tables(groups),['histogram_source_data.csv','signed_distribution_summary.csv','quantiles.csv']):
  if name=='quantiles.csv':
   for band in BANDS:
    z=kept[kept.latitude_band.eq(band)];ix=table.latitude_band.eq(band)
    table.loc[ix,'n_source_units']=z.source_unit.nunique()
    table.loc[ix,'n_fitting_arcs']=z.fit_arc_id.nunique()
  base=readcsv(ROOT/'fig34/amplitude_source'/name)
  pd.concat([base[base.platform.ne('LuGRE')],table],ignore_index=True).to_csv(A/name,index=False,float_format='%.17g')
 height=[]
 for band,z in kept.groupby('latitude_band',sort=False):
  for low in np.arange(50,1000,32):
   t=z[z.height_reference_km.ge(low)&z.height_reference_km.lt(low+32)]
   if len(t)>=8:
    a,b,c=np.quantile(abs(t.delta_cn0_db),[.5,.9,.95])
    height.append(dict(height_low_km=low,height_high_km=low+32,height_km=low+16,n=len(t),p50_abs=a,p90_abs=b,p95_abs=c,platform_metric='lugre',selection='all_valid',latitude_band=band))
 height=pd.DataFrame(height)
 for band in BANDS:
  ix=height.latitude_band.eq(band)
  for field in ['p50','p90','p95']:height.loc[ix,field+'_abs_smoothed']=height.loc[ix,field+'_abs'].rolling(3,center=True,min_periods=1).median()
 base=readcsv(ROOT/'fig34/amplitude_source/height_envelopes.csv')
 pd.concat([base[base.platform_metric.ne('lugre')],height],ignore_index=True).to_csv(A/'height_envelopes.csv',index=False,float_format='%.17g')
 np.savez_compressed(A/'lugre_height_selection.npz',amplitude=kept.delta_cn0_db.to_numpy(),height_km=kept.height_reference_km.to_numpy(),band_code=kept.latitude_band.map(dict(zip(BANDS,range(3)))).to_numpy(),all_valid=np.ones(len(kept),bool))
 shutil.copy2(ROOT/'fig34/amplitude_source/leo_delta_height_selection.npz',A/'leo_delta_height_selection.npz')
 geom=ledger[ledger.Fig4_LT_keep].copy();accepted={key(row) for row in geom.to_dict('records')}
 # Recompute the mask and quantiles above, then retain exact published metadata rows.
 original_rows=ROOT/'inputs_r4/LuGRE_original_response_rows.csv'
 with original_rows.open('rb') as f:
  header=f.readline();columns=next(csv.reader([header.decode('utf-8-sig')]))
  selected=[]
  computed={key(row):row for row in geom.to_dict('records')}
  for line in f:
   row=dict(zip(columns,next(csv.reader([line.decode('utf-8')]))));uid=key(row)
   if uid in accepted:
    r=computed[uid]
    for name,field in [('signed_p05_db','current_P05_db'),('signed_median_db','current_median_db'),('signed_p95_db','current_P95_db')]:assert abs(float(row[name])-r[field])<1e-12
    assert int(float(row['n_points']))==r['N_valid']
    selected.append(line)
 assert len(selected)==1342
 with (ROOT/'fig34/response_source/common_60s_response_windows_R3.csv').open('rb') as src,(R/'common_60s_response_windows_R3.csv').open('wb') as dst:
  assert src.readline()==header;dst.write(header)
  for line in src:
   if not line.startswith(b'LuGRE,'):dst.write(line)
  for line in selected:dst.write(line)
 bins=build_angle_bins(geom);bins=bins[bins.platform.eq('LuGRE')]
 base=readcsv(ROOT/'fig34/response_source/angle_bin_median_IQR_counts_R3.csv')
 pd.concat([base[base.platform.ne('LuGRE')],bins],ignore_index=True).to_csv(R/'angle_bin_median_IQR_counts_R3.csv',index=False,float_format='%.17g')
 near=pd.read_parquet(ROOT/'inputs_r4/nearspace_solar_windows_before_QC.parquet')
 select=near.platform.ne('LuGRE')|near.apply(key,axis=1).isin(accepted)
 near=near[select].reset_index(drop=True)
 assert near.groupby('platform').size().to_dict()=={'LEO-RO':1161,'LuGRE':1342}
 near.to_parquet(out/'illumination/nearspace_solar_windows_R3.parquet',index=False)
 for folder,name in [('amplitude_source','Fig3_R3_data_manifest.json'),('response_source','response_window_manifest_R3.json')]:shutil.copy2(ROOT/'fig34'/folder/name,out/folder/name)
 print('Comparing rebuilt R4 tables with the selected references',flush=True)
 checks=verify_tables(out,kept) if verify else {}
 report={'status':'PASS_R4_NUMERICAL_RECONSTRUCTION' if verify else 'R4_NUMERICAL_RECONSTRUCTION_NOT_COMPARED',
   'original_LuGRE_epochs':len(x),'original_UTC_candidate_windows':len(ledger),'Fig3_quality_windows':int(ledger.three_conditions_keep.sum()),'Fig3_LuGRE_epochs':len(kept),'Fig4_LT_LuGRE_windows':len(geom),'Ground_stations':182,'Ground_epochs':29803114,'Ground_windows':498430,'LEO_windows':1161,
   'QC':{'point_CN0_min_dbHz':25,'minimum_actual_points':30,'maximum_absolute_signed_median_db':1,'entire_failed_window_removed':True},
   'window_partition':'Original half-open UTC minutes split at source,satellite,signal,fit/QC segment,epoch magnetic band and gap>1.5s; not Fig2 event-source anchoring.',
   'methods_unchanged':True,'detrending_rerun':False,'raw_Ground_reprocessing_included':False,'Ground_LEO_processed_tables_unchanged':True,
   'Fig3_includes_quality_windows_without_complete_scan_geometry':True,'Fig4_LT_preserve_original_additional_geometry_eligibility':True,
   'comparison_with_packaged_R4_tables':checks,'all_runtime_inputs_component_relative':True,
   'input_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (ROOT/'inputs_r4').iterdir() if p.is_file()}}
 (out/'numeric_reconstruction_R4.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/numerical_r4');p.add_argument('--no-verify',action='store_true')
 a=p.parse_args();r=rebuild(a.output_dir,verify=not a.no_verify);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
