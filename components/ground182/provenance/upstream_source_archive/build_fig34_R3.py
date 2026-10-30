"""Prepare Fig.3/4 from the common frozen selection without refitting observations."""
from pathlib import Path
import sys,os,json,hashlib,importlib.util,copy,shutil,time,gc
from collections import Counter
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/LUGRE/python_deps/pyarrow_py312')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
BASE=Path('E:/LUGRE/figures_R3/Ground_F_expansion_R3_20261028')
BANDS=['Equatorial','Mid-latitude','Polar']

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def csv(p):return pd.read_csv(p,float_precision='round_trip',low_memory=False)
def parquet_checked(p,digest,cols=None):
 raw=Path(p).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest,p
 d=pq.ParquetFile(pa.BufferReader(raw)).read(columns=cols,use_threads=False).to_pandas()
 del raw
 return d
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 started=time.time();selection=PARENT/'Ground_selection_R3.json';sel=read(selection)
 assert sel['station_count']==len(sel['stations']) and 'KIRG01KIR' not in sel['stations']
 assert sel['support_sha256']=='f560e73f26c6a6726c10920f716449bf8e6ffcfeb0f83be8ed6b5e4b5d553a9e'
 for p in ['amplitude_source','response_source','figures','plotting','captions','comparison']:(ROOT/p).mkdir(parents=True,exist_ok=True)
 completion=ROOT/'Fig34_data_completion_R3.json'
 assert not completion.exists(),'Preserve existing completed figure data'
 ref=load('final_selection_reference','E:/LUGRE/figures_R3/Fig3_Fig4_commonUTC_epochBands_R3_20261027/ground/reference_portable.py')
 chunks={b:[] for b in BANDS};stations={b:set() for b in BANDS};arcs={b:set() for b in BANDS};all_arcs=set();signals=Counter();source_counts=[]
 cols=['source_unit','satellite','signal','fit_arc_id','utc_s','latitude_band','delta_poly4_db','cn0_dbhz']
 for i,p in enumerate(sel['parts'],1):
  d=parquet_checked(p['source'],p['source_sha256'],cols)
  assert len(d)==p['source_rows']
  mp=Path(p['selection_mask']);assert sha(mp)==p['selection_mask_sha256']
  with np.load(mp,allow_pickle=False) as z:
   assert str(z['source_sha256'])==p['source_sha256'] and int(z['source_rows'])==len(d)
   keep=np.unpackbits(z['packed'],bitorder='little')[:len(d)].astype(bool)
  d=d.loc[keep]
  assert len(d)==p['retained_epochs'] and not d.source_unit.eq('KIRG01KIR').any()
  assert set(d.source_unit.unique())<=set(sel['stations'])
  assert np.isfinite(d.delta_poly4_db).all()
  for b,g in d.groupby('latitude_band',observed=True):
   chunks[b].append(g.delta_poly4_db.to_numpy(copy=True));stations[b].update(g.source_unit.unique())
   keys=list(map(tuple,g[['source_unit','satellite','signal','fit_arc_id']].drop_duplicates().to_numpy()));arcs[b].update(keys);all_arcs.update(keys)
  signals.update(d.signal.value_counts().to_dict())
  for (st,sg,b),g in d.groupby(['source_unit','signal','latitude_band'],observed=True):
   source_counts.append(dict(station=st,signal=sg,latitude_band=b,source_part=str(p['source_part']),n_epochs=len(g)))
  if i%10==0 or i==len(sel['parts']):
   write(ROOT/'processing_progress_R3.json',dict(stage='READING_FROZEN_PARTS',parts_completed=i,total_parts=len(sel['parts']),epochs=sum(sum(map(len,x)) for x in chunks.values())))
   print(f'Loaded {i}/{len(sel["parts"])} frozen parts',flush=True)
  del d;gc.collect()
 groups={('Ground',b):np.concatenate(chunks[b]) for b in BANDS};chunks.clear();gc.collect()
 counts={b:len(groups['Ground',b]) for b in BANDS}
 assert counts==sel['latitude_counts'] and sum(counts.values())==sel['Ground_epochs']
 assert set().union(*stations.values())==set(sel['stations'])
 hist,signed,quant=ref._amplitude_tables(groups)
 del groups;gc.collect()
 for b in BANDS:
  mask=quant.latitude_band.eq(b);quant.loc[mask,'n_source_units']=len(stations[b]);quant.loc[mask,'n_fitting_arcs']=len(arcs[b])
 A=ROOT/'amplitude_source';R=ROOT/'response_source'
 nearspace_qa={}
 for f,name in [(hist,'histogram_source_data.csv'),(signed,'signed_distribution_summary.csv'),(quant,'quantiles.csv')]:
  ns=csv(BASE/'fig34/amplitude_source'/name);ns=ns[~ns.platform.eq('Ground')]
  output=pd.concat([f,ns],ignore_index=True);output.to_csv(A/name,index=False,float_format='%.17g')
  actual=csv(A/name);actual=actual[~actual.platform.eq('Ground')].reset_index(drop=True)
  pd.testing.assert_frame_equal(actual,ns.reset_index(drop=True),check_dtype=False,check_exact=True)
  nearspace_qa[name]='all non-Ground numeric values unchanged'
 for name in ['height_envelopes.csv','lugre_height_selection.npz','leo_delta_height_selection.npz']:
  shutil.copy2(BASE/'fig34/amplitude_source'/name,A/name)
  assert sha(A/name)==sha(BASE/'fig34/amplitude_source'/name)
  nearspace_qa[name]='byte-identical'
 w=parquet_checked(sel['windows_source'],sel['windows_sha256'])
 assert len(w)==sel['Ground_windows'] and set(w.source_unit.astype(str))<=set(sel['stations'])
 assert not w.source_unit.eq('KIRG01KIR').any() and w.window_id.is_unique
 assert w.groupby('latitude_band',observed=True).size().to_dict()==sel['window_counts']
 bins=ref.build_angle_bins(w);bins=bins[bins.platform.eq('Ground')].reset_index(drop=True)
 nswin=csv(BASE/'fig34/response_source/common_60s_response_windows_R3.csv');nswin=nswin[~nswin.platform.eq('Ground')]
 nsbins=csv(BASE/'fig34/response_source/angle_bin_median_IQR_counts_R3.csv');nsbins=nsbins[~nsbins.platform.eq('Ground')]
 allwin=pd.concat([w,nswin],ignore_index=True);allbins=pd.concat([bins,nsbins],ignore_index=True)
 allwin.to_csv(R/'common_60s_response_windows_R3.csv',index=False,float_format='%.17g');allbins.to_csv(R/'angle_bin_median_IQR_counts_R3.csv',index=False,float_format='%.17g')
 shutil.copy2(sel['windows_source'],R/'Ground_response_windows_R3.parquet');assert sha(R/'Ground_response_windows_R3.parquet')==sel['windows_sha256']
 for (platform,angle,b),table in allbins.groupby(['platform','angle','latitude_band']):
  g=allwin[allwin.platform.eq(platform)&allwin.latitude_band.eq(b)];assert int(table.n_windows.sum())==len(g)
  for row in table.itertuples():
   v=g[g[angle].ge(row.bin_left_deg)&(g[angle].le(90) if row.bin_right_deg==90 else g[angle].lt(row.bin_right_deg))]
   assert len(v)==row.n_windows
   if len(v):np.testing.assert_allclose(v.width_db.quantile([.5,.25,.75]),[row.median_width_db,row.q25_width_db,row.q75_width_db],rtol=0,atol=4e-15)
 for name,old in [('common_60s_response_windows_R3.csv',nswin),('angle_bin_median_IQR_counts_R3.csv',nsbins)]:
  actual=csv(R/name);actual=actual[~actual.platform.eq('Ground')].reset_index(drop=True)
  # Ground additions can add metadata columns; compare every original near-space column.
  pd.testing.assert_frame_equal(actual[old.columns],old.reset_index(drop=True),check_dtype=False,check_exact=True)
  nearspace_qa[name]='all non-Ground numeric values unchanged'
 common=read(BASE/'fig34/amplitude_source/Fig3_R3_data_manifest.json')
 gp=dict(platform='Ground',n_records=sel['Ground_epochs'],signal_counts=dict(signals),latitude_counts=counts,n_source_units=sel['station_count'],n_fitting_arcs=len(all_arcs),same_latitude_band_UTC_all=True,n_outside_display_minus10_plus10=int(signed.n_records_outside_display.sum()))
 common['platforms']=[gp]+[p for p in common['platforms'] if p['platform']!='Ground']
 method='Unchanged native1Hz independent quartic arc residuals; verified finer-than1dB products; frozen full-native-background quality masks and original common UTC/IPP selection; no latitude-based rejection'
 common['methods']['Ground']=method;common['method_by_platform']['Ground']=method
 common.update(Ground_epochs=sel['Ground_epochs'],window_counts={'Ground':len(w),**{str(k):int(v) for k,v in nswin.groupby('platform').size().items()}},Ground_scope=f"{sel['station_count']} actual Ground epoch contributors in the new screened selection; original receiver identities and source residuals preserved",shared_selection_manifest=str(selection),shared_selection_sha256=sha(selection),selection_variant=sel['variant'])
 cm=copy.deepcopy(common);cm['status']='COMPLETE_FIG3_R3_FINAL_SELECTION';write(A/'Fig3_R3_data_manifest.json',cm)
 cm=copy.deepcopy(common);cm['status']='PASS_FIG4_R3_FINAL_SELECTION';write(R/'response_window_manifest_R3.json',cm)
 pd.DataFrame(source_counts).to_csv(ROOT/'comparison/Station_signal_band_counts_R3.csv',index=False)
 qold=csv(BASE/'fig34/amplitude_source/quantiles.csv');qold=qold[qold.platform.eq('Ground')];qold['version']='previous182_selection'
 qnew=quant.copy();qnew['version']='screened_selection'
 pd.concat([qold,qnew],ignore_index=True).to_csv(ROOT/'comparison/Ground_absolute_residual_quantiles_R3.csv',index=False,float_format='%.17g')
 oldw=pq.read_table(BASE/'fig34/response_source/Ground_response_windows_R3.parquet').to_pandas()
 stats=[]
 for version,windows in [('previous182_selection',oldw),('screened_selection',w)]:
  for b,g in windows.groupby('latitude_band',observed=True):
   v=g.width_db;stats.append(dict(version=version,latitude_band=b,n_windows=len(g),n_stations=g.source_unit.nunique(),P25_W_db=v.quantile(.25),median_W_db=v.median(),P75_W_db=v.quantile(.75),P95_W_db=v.quantile(.95),fraction_W_ge_1p5=float(v.ge(1.5).mean())))
 pd.DataFrame(stats).to_csv(ROOT/'comparison/Ground_window_width_comparison_R3.csv',index=False,float_format='%.17g')
 write(ROOT/'comparison/Population_change_R3.json',dict(old_station_count=oldw.source_unit.nunique(),new_station_count=sel['station_count'],removed_sources=sorted(set(oldw.source_unit)-set(w.source_unit)),added_sources=sorted(set(w.source_unit)-set(oldw.source_unit)),old_windows=len(oldw),new_windows=len(w),residuals_refitted_by_figure_builder=False))
 result=dict(status='PASS_FINAL_SELECTION_FIG34_DATA_READY',station_count=sel['station_count'],window_station_count=int(w.source_unit.nunique()),Ground_epochs=sel['Ground_epochs'],Ground_windows=len(w),latitude_counts=counts,window_counts=sel['window_counts'],band_station_counts={b:len(stations[b]) for b in BANDS},Ground_quantiles=quant.to_dict('records'),shared_selection_sha256=sha(selection),nearspace_checks=nearspace_qa,no_refit=True,elapsed_seconds=time.time()-started)
 write(completion,result);write(ROOT/'processing_progress_R3.json',dict(stage='DATA_COMPLETE',Ground_epochs=sel['Ground_epochs'],Ground_windows=len(w)))
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
