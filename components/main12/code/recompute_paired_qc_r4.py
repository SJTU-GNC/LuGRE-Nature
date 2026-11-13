import os
import json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
COMPONENT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'fig2'
SRC=COMPONENT/'data/r4/inputs'
for pth in [ROOT/'data',ROOT/'qa']:pth.mkdir(parents=True,exist_ok=True)
p=pd.read_parquet(SRC/'paired_eligible_measured_epoch_source.parquet')
runs=pd.read_csv(SRC/'panel_c_paired_frequency_segments.csv')
oldblocks=pd.read_csv(SRC/'paired_exact60_blocks.csv')
original=p.copy(); p['_original_pair_row']=np.arange(len(p)); p['new_joint_run_id']='';p['new_block_index']=-1;p['paired_block_QC_keep']=False;p['paired_block_QC_reason']='old_unused_tail_or_nonqualifying_run'
width=lambda a:float(np.quantile(a,.95)-np.quantile(a,.05))
rowlist=[];blocklist=[];retained=[];max_width_error=0.
# Reconstruct exactly the original 104 qualified contiguous run contexts and their 627 frozen 60-sample blocks.
for run,g in p.groupby('joint_run_id',sort=True):
 g=g.sort_values('rx_gps_seconds_h',kind='stable')
 bad=~np.isclose(g.rx_gps_seconds_h.diff().to_numpy(),1,atol=1e-8,rtol=0);bad[0]=False
 for piece,q in g.groupby(np.cumsum(bad),sort=False):
  rid=f'{int(run)}_eligible{int(piece)}'
  oldrun=runs[runs.joint_run_id.eq(rid)]
  if oldrun.empty:continue
  assert len(oldrun)==1
  row=oldrun.iloc[0].to_dict(); original_blocks=oldblocks[oldblocks.joint_run_id.eq(rid)]
  nblocks=len(q)//60
  assert nblocks==len(original_blocks)==int(row['n_exact_60sample_blocks'])
  accepted_blocks=[];accepted_points=[]
  for bi in range(nblocks):
   z=q.iloc[60*bi:60*(bi+1)].copy()
   wh=width(z.cn0_delta_dbhz_h);wl=width(z.cn0_delta_dbhz_l)
   ob=original_blocks[original_blocks.block_index.eq(bi)].iloc[0]
   max_width_error=max(max_width_error,abs(wh-ob.high_width_db),abs(wl-ob.low_width_db))
   point_good=np.isfinite(z.cn0_delta_dbhz_h)&np.isfinite(z.cn0_delta_dbhz_l)&z.cn0_dbhz_h.ge(25)&z.cn0_dbhz_l.ge(25)
   valid=z[point_good];nv=len(valid)
   # Both frequency widths use identical actual seconds; never compact/repartition after an exclusion.
   mh=float(valid.cn0_delta_dbhz_h.median()) if nv else np.nan
   ml=float(valid.cn0_delta_dbhz_l.median()) if nv else np.nan
   enough=nv>=30; biasok=enough and abs(mh)<=1 and abs(ml)<=1
   keep=bool(enough and biasok)
   reasons=[]
   if not enough:reasons.append('fewer_than30_valid_paired_seconds')
   elif not biasok:reasons.append('absolute_signed_median_above1dB')
   bhr=width(valid.cn0_delta_dbhz_h) if enough else np.nan
   blr=width(valid.cn0_delta_dbhz_l) if enough else np.nan
   vr=float(np.var(valid.cn0_delta_dbhz_l,ddof=1)/np.var(valid.cn0_delta_dbhz_h,ddof=1)) if enough else np.nan
   br=dict(joint_run_id=rid,block_index=bi,n_measured_pairs=len(z),n_valid_paired_seconds=nv,
       n_raw_CN0_below25_high=int(z.cn0_dbhz_h.lt(25).sum()),n_raw_CN0_below25_low=int(z.cn0_dbhz_l.lt(25).sum()),
       high_signed_median_db=mh,low_signed_median_db=ml,high_width_db=bhr,low_width_db=blr,
       original_high_width_db=wh,original_low_width_db=wl,low_over_high_variance_ratio=vr,
       QC_keep=keep,QC_reason=';'.join(reasons) or 'retained',
       start_gps=float(z.rx_gps_seconds_h.iloc[0]),end_gps=float(z.rx_gps_seconds_h.iloc[-1]))
   blocklist.append(br)
   p.loc[z.index,'new_joint_run_id']=rid;p.loc[z.index,'new_block_index']=bi
   p.loc[z.index,'paired_block_QC_reason']=br['QC_reason']
   if keep:
    p.loc[valid.index,'paired_block_QC_keep']=True
    p.loc[z.index[~point_good],'paired_block_QC_reason']='point_CN0_below25_or_nonfinite'
    accepted_blocks.append(br);accepted_points.append(valid)
  # The >=180-run qualification is frozen. No second >=180 gate is introduced after block QC.
  if not accepted_blocks:continue
  v=pd.concat(accepted_points)
  row['original_n_measured_pairs']=row['n_measured_pairs'];row['original_n_exact_60sample_blocks']=nblocks
  row['n_retained_measured_pairs']=len(v);row['n_exact_60sample_blocks']=len(accepted_blocks)
  row['n_rejected_60sample_blocks']=nblocks-len(accepted_blocks)
  row['high_width_db']=width(v.cn0_delta_dbhz_h);row['low_width_db']=width(v.cn0_delta_dbhz_l)
  row['width_low_over_high']=row['low_width_db']/row['high_width_db']
  hs=np.array([b['high_width_db'] for b in accepted_blocks]);ls=np.array([b['low_width_db'] for b in accepted_blocks])
  row['high_width_60sample_block_median_db']=float(np.median(hs));row['low_width_60sample_block_median_db']=float(np.median(ls))
  row['width_ratio_60sample_block_median']=float(np.median(ls/hs))
  row['sample_variance_ratio_60sample_block_median']=float(np.median([b['low_over_high_variance_ratio'] for b in accepted_blocks]))
  row['QC_basis']='Original qualified paired runs; frozen complete60sample blocks; CN0>=25 then Nvalid>=30 then both abs(signed median)<=1dB; unchanged run geometry labels'
  rowlist.append(row)
assert max_width_error<1e-10
ledger=pd.DataFrame(blocklist);out=pd.DataFrame(rowlist)
assert len(ledger)==len(oldblocks)==627
assert ledger[['joint_run_id','block_index']].equals(oldblocks[['joint_run_id','block_index']])
assert ledger.QC_keep.sum()==613 and (~ledger.QC_keep).sum()==14
accepted=ledger[ledger.QC_keep].copy()
ledger.to_csv(ROOT/'data/paired_exact60_block_QC_ledger.csv',index=False)
accepted.to_csv(ROOT/'data/paired_exact60_blocks.csv',index=False)
out.to_csv(ROOT/'data/panel_c_paired_frequency_segments.csv',index=False)
out.to_csv(ROOT/'data/paired_geometry.csv',index=False)
p.to_csv(ROOT/'data/paired_all_original_points_QC_ledger.csv',index=False)
p[p.paired_block_QC_keep].to_csv(ROOT/'data/paired_eligible_measured_epoch_source.csv',index=False)
summary=[]
for label,z in [('All',out)]+list(out.groupby('median_latitude_band')):
 summary.append(dict(latitude_band=label,n_runs=len(z),n_lower_carrier_broader=int(z.low_width_60sample_block_median_db.gt(z.high_width_60sample_block_median_db).sum()),percent_lower_carrier_broader=float(100*z.low_width_60sample_block_median_db.gt(z.high_width_60sample_block_median_db).mean()),median_low_over_high_width=float((z.low_width_60sample_block_median_db/z.high_width_60sample_block_median_db).median()),max_lower_carrier_width_db=float(z.low_width_60sample_block_median_db.max())))
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
qa=dict(status='PAIRED_RECALCULATED_PENDING_INDEPENDENT_QA',original_pairs=len(p),original_qualified_runs=len(runs),retained_runs=len(out),original_exact60_blocks=len(ledger),retained_blocks=len(accepted),rejected_blocks=int((~ledger.QC_keep).sum()),block_count_N_lt30=int(ledger.n_valid_paired_seconds.lt(30).sum()),high_median_fail=int(ledger.high_signed_median_db.abs().gt(1).sum()),low_median_fail=int(ledger.low_signed_median_db.abs().gt(1).sum()),raw_CN0_lt25_high=int(p.cn0_dbhz_h.lt(25).sum()),raw_CN0_lt25_low=int(p.cn0_dbhz_l.lt(25).sum()),all_pairs_retained_in_ledger=True,retained_pairs_in_exact_blocks=int(p.paired_block_QC_keep.sum()),unused_tail_and_nonqualifying_pairs=int(p.new_block_index.lt(0).sum()),max_original_block_width_error=max_width_error,run_qualification_frozen_before_block_QC=True,new_postQC_180_gate=False,accepted_median_absolute_max=float(accepted[['high_signed_median_db','low_signed_median_db']].abs().to_numpy().max()),source_sha256={str(SRC/f):sha(SRC/f) for f in ['paired_eligible_measured_epoch_source.parquet','paired_exact60_blocks.csv','panel_c_paired_frequency_segments.csv']},summary_by_display_latitude=summary,scientific_changes='Only the three declared QC gates reapplied to existing residuals and frozen paired blocks. No refit, re-centering, new runs, re-anchoring, or synthetic observations.',output_sha256={str(ROOT/'data'/f):sha(ROOT/'data'/f) for f in ['paired_exact60_blocks.csv','paired_exact60_block_QC_ledger.csv','panel_c_paired_frequency_segments.csv','paired_all_original_points_QC_ledger.csv']})
(ROOT/'qa/paired_strict_QC_R4.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps(qa,indent=2))
