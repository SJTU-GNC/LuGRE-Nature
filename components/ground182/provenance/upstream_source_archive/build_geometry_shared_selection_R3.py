"""Stream exact shared Ground masks; preserve near-space geometry distributions."""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
import os,sys,json,hashlib,datetime,argparse,time,gc
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,'E:/LUGRE/python_deps/pyarrow_py312')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
OLD=Path('E:/LUGRE/figures_R3/Ground141_common_selection_R3_20261028/geometry')
RESOLUTION=Path('E:/LUGRE/figures_R3/Ground_fractional_CN0_R3_20261028/Ground_resolution_selection_R3.json')
SUPPORT=Path('E:/LUGRE/figures_R3/Geometry_Extended_Data_R3_20261027/epoch_recalculation/nearspace/canonical_band_UTC_support_epoch_R3.csv')
BANDS=['Equatorial','Mid-latitude','Polar']
ANGLES=['ray_B_acute_deg','scan_Bperp_acute_deg','magnetic_meridian_acute_deg']
SPEEDS=['point_speed_mps','scan_speed_mps','fresnel_crossing_time_s']
GRID=np.arange(901,dtype=float)/10
FIELDS=['source_unit','receiver_identity','satellite','signal','fit_arc_id','utc_s','latitude_band','ipp_CD_magnetic_latitude_deg','elevation_deg','cn0_dbhz']+ANGLES+SPEEDS
VERSION='shared-masks-streamed-geometry-v1'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def writej(p,data):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def readcsv(p):return pd.read_csv(p,float_precision='round_trip')
class Counts:
    def __init__(self):
        self.rows=0;self.bands=Counter();self.stations=Counter();self.signals=Counter()
        self.cdf=defaultdict(lambda:np.zeros(901,dtype=np.int64));self.den=Counter();self.angle_invalid=Counter()
        self.hist=defaultdict(Counter);self.speed={};self.clusters=defaultdict(set)
    def add(self,d):
        self.rows+=len(d);self.stations.update(d.source_unit.value_counts().to_dict());self.signals.update(d.signal.value_counts().to_dict())
        for band,g in d.groupby('latitude_band',observed=True):
            self.bands[band]+=len(g)
            for metric in ANGLES:
                key=band+'|'+metric;x=g[metric].to_numpy(float);a=x[np.isfinite(x)]
                assert ((a>=0)&(a<=90)).all(),(band,metric,'angle outside0to90')
                count=np.searchsorted(np.sort(a),GRID,side='right')
                # Independent threshold-index accumulation checks the CDF inclusion convention.
                idx=np.searchsorted(GRID,a,side='left');reference=np.bincount(idx,minlength=901).cumsum()[:901]
                np.testing.assert_array_equal(count,reference)
                self.cdf[key]+=count;self.den[key]+=len(a);self.angle_invalid[key]+=len(x)-len(a)
            for metric in SPEEDS:
                key=band+'|'+metric;x=g[metric].to_numpy(float);good=np.isfinite(x)&(x>=0);a=x[good];step=.25 if metric=='fresnel_crossing_time_s' else 5.
                st=self.speed.setdefault(key,dict(n_rows=0,n_valid=0,n_missing_or_nonfinite=0,finite_zero_count=0,minimum=None,maximum=None))
                st['n_rows']+=len(x);st['n_valid']+=len(a);st['n_missing_or_nonfinite']+=int((~good).sum());st['finite_zero_count']+=int((a==0).sum())
                if len(a):
                    bins=np.floor(a/step).astype(np.int64);assert bins.max()<1000000,'Unexpected extreme geometry; inspect source rather than clip'
                    ids,n=np.unique(bins,return_counts=True)
                    reference,_=np.histogram(a,bins=np.arange(int(bins.max())+2)*step)
                    np.testing.assert_array_equal(np.bincount(bins,minlength=len(reference)),reference)
                    self.hist[key].update(dict(zip(map(int,ids),map(int,n))))
                    st['minimum']=min(st['minimum'],float(a.min())) if st['minimum'] is not None else float(a.min())
                    st['maximum']=max(st['maximum'],float(a.max())) if st['maximum'] is not None else float(a.max())
                    cl=g.loc[good,['source_unit','receiver_identity','satellite','fit_arc_id']].drop_duplicates().astype(str)
                    self.clusters[key].update(json.dumps(list(r),separators=(',',':')) for r in cl.itertuples(index=False,name=None))
    def dump(self):
        return dict(rows=self.rows,bands=dict(self.bands),stations=dict(self.stations),signals=dict(self.signals),cdf={k:v.tolist() for k,v in self.cdf.items()},den=dict(self.den),angle_invalid=dict(self.angle_invalid),hist={k:dict(v) for k,v in self.hist.items()},speed=self.speed,clusters={k:sorted(v) for k,v in self.clusters.items()})
    def merge(self,z):
        self.rows+=z['rows'];self.bands.update(z['bands']);self.stations.update(z['stations']);self.signals.update(z['signals']);self.den.update(z['den']);self.angle_invalid.update(z['angle_invalid'])
        for k,v in z['cdf'].items():self.cdf[k]+=np.array(v,dtype=np.int64)
        for k,v in z['hist'].items():self.hist[k].update({int(i):int(n) for i,n in v.items()})
        for k,v in z['speed'].items():
            a=self.speed.setdefault(k,dict(n_rows=0,n_valid=0,n_missing_or_nonfinite=0,finite_zero_count=0,minimum=None,maximum=None))
            for n in ['n_rows','n_valid','n_missing_or_nonfinite','finite_zero_count']:a[n]+=v[n]
            if v['minimum'] is not None:a['minimum']=min(a['minimum'],v['minimum']) if a['minimum'] is not None else v['minimum']
            if v['maximum'] is not None:a['maximum']=max(a['maximum'],v['maximum']) if a['maximum'] is not None else v['maximum']
        for k,v in z['clusters'].items():self.clusters[k].update(v)
    def frames(self):
        c=[];h=[];s=[]
        for band in BANDS:
            for metric in ANGLES:
                key=band+'|'+metric;n=self.den[key];a=self.cdf[key]
                if not n:continue
                assert a[-1]==n and (np.diff(a)>=0).all()
                c.extend(dict(system='Ground',latitude_band=band,metric=metric,angle_deg=float(x),cumulative_count=int(y),n_valid=n,cumulative_fraction=float(y/n)) for x,y in zip(GRID,a))
            for metric in SPEEDS:
                key=band+'|'+metric;st=self.speed.get(key)
                if not st or not st['n_valid']:continue
                n=st['n_valid'];step=.25 if metric=='fresnel_crossing_time_s' else 5.;bins=self.hist[key]
                assert sum(bins.values())==n and st['n_rows']==n+st['n_missing_or_nonfinite']
                s.append(dict(system='Ground',latitude_band=band,metric=metric,**st,valid_source_clusters=len(self.clusters[key]),valid_weight=n))
                h.extend(dict(system='Ground',latitude_band=band,metric=metric,bin_left=i*step,bin_right=(i+1)*step,weighted_count=bins[i],weighted_proportion_percent=100*bins[i]/n) for i in range(max(bins)+1))
        return pd.DataFrame(c),pd.DataFrame(h),pd.DataFrame(s)
def part_counts(part,cache,intervals,baseids,excluded,batch_size,buffer_limit):
    path=Path(part['source']);maskpath=Path(part['selection_mask']);assert sha(maskpath)==part['selection_mask_sha256']
    spec=dict(source_part=part['source_part'],source=part['source'],source_sha256=part['source_sha256'],mask=part['selection_mask'],mask_sha256=part['selection_mask_sha256'],source_rows=part['source_rows'],retained_epochs=part['retained_epochs'],base_stations=sorted(baseids),version=VERSION)
    key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest();cp=cache/(key+'.json')
    stat=path.stat()
    if cp.exists():
        old=readj(cp)
        assert old['identity']==spec and old['source_size']==stat.st_size and old['source_mtime_ns']==stat.st_mtime_ns,'Cached source changed'
        return old
    with np.load(maskpath,allow_pickle=False) as z:
        keep=np.unpackbits(z['packed'],bitorder='little')[:int(part['source_rows'])].astype(bool)
        if 'source_rows' in z:assert int(z['source_rows'])==part['source_rows']
        if 'source_sha256' in z:assert str(z['source_sha256'])==part['source_sha256']
    assert keep.sum()==part['retained_epochs']
    raw=None
    if stat.st_size<=buffer_limit:
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==part['source_sha256'];pf=pq.ParquetFile(pa.BufferReader(raw));mode='bounded whole-compressed-file buffer'
    else:
        assert sha(path)==part['source_sha256'];pf=pq.ParquetFile(path);mode='column-batch streaming from file, above buffer cap'
    assert pf.metadata.num_rows==part['source_rows'];assert set(FIELDS)<=set(pf.schema_arrow.names)
    groups={'base122':Counts(),'added':Counts()};offset=0;count=0
    for batch in pf.iter_batches(batch_size=batch_size,columns=FIELDS,use_threads=False):
        n=batch.num_rows;select=keep[offset:offset+n];offset+=n
        if not select.any():continue
        d=batch.filter(pa.array(select)).to_pandas();count+=len(d)
        assert not set(d.source_unit)&excluded,'Separated1dB product present in final shared mask'
        assert 'KIRG01KIR' not in set(d.source_unit),'Shared-antenna reference should not enter primary figures'
        assert d.elevation_deg.ge(30).all() and d.cn0_dbhz.ge(30).all();assert set(d.signal)<={'GPS L1','Galileo E1'}
        lat=d.ipp_CD_magnetic_latitude_deg.abs().to_numpy();assert np.isfinite(lat).all()
        np.testing.assert_array_equal(np.where(lat<20,BANDS[0],np.where(lat<60,BANDS[1],BANDS[2])),d.latitude_band)
        for band,g in d.groupby('latitude_band',observed=True):
            t=g.utc_s.to_numpy(float);iv=intervals[band];i=np.searchsorted(iv[:,0],t,side='right')-1;j=np.maximum(i,0)
            assert ((i>=0)&(t>=iv[j,0])&(t+1<=iv[j,1])).all(),'Full1s same-bandUTC mismatch'
        base=d.source_unit.isin(baseids)
        if base.any():groups['base122'].add(d.loc[base])
        if (~base).any():groups['added'].add(d.loc[~base])
        del d,batch
    assert offset==part['source_rows'] and count==part['retained_epochs']
    result=dict(identity=spec,source_size=stat.st_size,source_mtime_ns=stat.st_mtime_ns,read_mode=mode,batch_size=batch_size,all_selected_UTC_IPP_threshold_checks=True,independent_CDF_threshold_index_checks=True,independent_histogram_edge_checks=True,groups={k:v.dump() for k,v in groups.items()})
    writej(cp,result);del pf,raw,keep;gc.collect();return result
def compare_tables(before,after,kind):
    rows=[]
    for (band,metric),g in after.groupby(['latitude_band','metric']):
        a=before[before.latitude_band.eq(band)&before.metric.eq(metric)]
        if kind=='cdf':
            a=a.sort_values('angle_deg');g=g.sort_values('angle_deg');np.testing.assert_array_equal(a.angle_deg,g.angle_deg)
            x=g.angle_deg.to_numpy();p=a.cumulative_fraction.to_numpy();q=g.cumulative_fraction.to_numpy()
            rows.append(dict(latitude_band=band,metric=metric,before_valid=int(a.n_valid.iloc[0]),after_valid=int(g.n_valid.iloc[0]),before_median_grid_deg=float(x[np.searchsorted(p,.5)]),after_median_grid_deg=float(x[np.searchsorted(q,.5)]),max_absolute_CDF_change=float(abs(q-p).max())))
        else:
            a=a.set_index('bin_left');g=g.set_index('bin_left');idx=a.index.union(g.index).sort_values();p=a.weighted_proportion_percent.reindex(idx,fill_value=0);q=g.weighted_proportion_percent.reindex(idx,fill_value=0)
            rows.append(dict(latitude_band=band,metric=metric,before_valid=int(a.weighted_count.sum()),after_valid=int(g.weighted_count.sum()),before_median_bin_left=float(idx[np.searchsorted(p.cumsum(),50)]),after_median_bin_left=float(idx[np.searchsorted(q.cumsum(),50)]),max_absolute_bin_percentage_change=float(abs(q-p).max())))
    return pd.DataFrame(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT.parent/'Ground_selection_R3.json');p.add_argument('--batch-size',type=int,default=65536);p.add_argument('--buffer-mib',type=int,default=64);p.add_argument('--cache-only',action='store_true',help='Compute verified per-part caches only; do not freeze final population or export figure tables.');args=p.parse_args()
    assert args.manifest.exists(),'Final shared Ground selection has not yet been supplied'
    m=readj(args.manifest);assert 'parts' in m and 'Ground_epochs' in m and 'latitude_counts' in m and 'station_count' in m
    manifest_sha=sha(args.manifest)
    if not args.cache_only:assert m['status']=='FROZEN_SHARED_GROUND_SELECTION','Final shared scientific selection must be frozen before plotting data are exported'
    assert readj(ROOT/'figure_contract_R3.json')['backend'].startswith('Python')
    out=ROOT/'plot_data';cache=ROOT/'cache_parts';comp=ROOT/'comparison'
    for folder in ([cache] if args.cache_only else [out,cache,comp]):folder.mkdir(exist_ok=True)
    finalfreeze=out/'GEOMETRY_FREEZE.json'
    if not args.cache_only and finalfreeze.exists():raise FileExistsError('Final data already frozen; preserve it and choose a new output directory for a changed selection')
    resolution=readj(RESOLUTION);baseids=set(resolution['included_stations']);excluded=set(m.get('excluded_station_ids',[]))
    assert baseids and not baseids&excluded
    support_sha=sha(SUPPORT);assert m.get('support_sha256',support_sha)==support_sha
    st=readcsv(SUPPORT);intervals={b:st.loc[st.latitude_band.eq(b),['start_s','end_s']].to_numpy(float) for b in BANDS}
    states={'base122':Counts(),'added':Counts()};checks=[];start=time.monotonic()
    assert sum(p['retained_epochs'] for p in m['parts'])==m['Ground_epochs']
    assert len({p['source_part'] for p in m['parts']})==len(m['parts']),'Repeated source_part identity'
    progress_path=cache/'cache_progress.json' if args.cache_only else ROOT/'progress.json'
    for i,part in enumerate(m['parts']):
        z=part_counts(part,cache,intervals,baseids,excluded,args.batch_size,args.buffer_mib*1024**2)
        for k in states:states[k].merge(z['groups'][k])
        checks.append({k:v for k,v in z.items() if k!='groups'})
        writej(progress_path,dict(stage='STREAM_PART_CACHE_ONLY' if args.cache_only else 'STREAM_SHARED_MASK_GEOMETRY',source_manifest=str(args.manifest),parts_completed=i+1,total_parts=len(m['parts']),selected_epochs=sum(v.rows for v in states.values()),elapsed_s=time.monotonic()-start))
        print(json.dumps(dict(part=i+1,total=len(m['parts']),epochs=sum(v.rows for v in states.values()))),flush=True)
    total=Counts()
    for state in states.values():total.merge(state.dump())
    assert total.rows==m['Ground_epochs'] and total.bands==Counter(m['latitude_counts']) and len(total.stations)==m['station_count']
    assert sha(args.manifest)==manifest_sha,'Shared scientific selection changed during computation'
    if args.cache_only:
        proof=dict(status='PASS_PART_CACHES_ONLY_NOT_FINAL_POPULATION',source_manifest=str(args.manifest),source_manifest_sha256=sha(args.manifest),Ground_epochs=total.rows,Ground_stations=len(total.stations),latitude_counts=dict(total.bands),source_parts=checks,final_tables_written=False,final_figures_written=False,elapsed_s=time.monotonic()-start)
        writej(cache/'Base122_cache_completion_R3.json',proof);writej(progress_path,{k:v for k,v in proof.items() if k!='source_parts'});print(json.dumps({k:v for k,v in proof.items() if k!='source_parts'},indent=2));return
    fullframes=total.frames();baseframes=states['base122'].frames();newframes=states['added'].frames()
    filenames=['geometry_angle_CDF_curves_R3.csv','geometry_speed_bins_R3.csv','geometry_speed_support_R3.csv'];exports={};nearproof={}
    for name,ground,base,added in zip(filenames,fullframes,baseframes,newframes):
        original=readcsv(OLD/'plot_data'/name);near=original.loc[original.system.ne('Ground')].copy()
        combined=pd.concat([ground[original.columns],near],ignore_index=True);combined.to_csv(out/name,index=False,float_format='%.17g')
        back=readcsv(out/name);pd.testing.assert_frame_equal(near.reset_index(drop=True),back.loc[back.system.ne('Ground')].reset_index(drop=True),check_exact=True,check_dtype=False)
        ground.to_csv(out/('Ground_'+name),index=False,float_format='%.17g');base.to_csv(comp/('Historical_base_'+name),index=False,float_format='%.17g');added.to_csv(comp/('Added_'+name),index=False,float_format='%.17g')
        nearproof[name]=True;exports['data/'+name]=sha(out/name)
    compare_tables(baseframes[0],fullframes[0],'cdf').to_csv(comp/'Historical_base_vs_final_angle_CDF_R3.csv',index=False,float_format='%.17g')
    compare_tables(baseframes[1],fullframes[1],'hist').to_csv(comp/'Historical_base_vs_final_speed_hist_R3.csv',index=False,float_format='%.17g')
    counts=pd.DataFrame([dict(station=s,epochs=n,cohort='base122' if s in baseids else 'added') for s,n in sorted(total.stations.items())]);counts.to_csv(out/'Ground_station_epoch_counts_R3.csv',index=False)
    freeze=dict(status='PASS_SHARED_FINAL_GROUND_GEOMETRY_TABLES',Ground_retained_epochs=total.rows,Ground_contributing_stations=len(total.stations),Ground_latitude_counts=dict(total.bands),base122_epochs=states['base122'].rows,added_epochs=states['added'].rows,base122_station_count=len(states['base122'].stations),baseline_note='Historical fractional-resolution base identifiers, retained only for exact cache reuse and before/after summaries; final membership is determined by the supplied selection manifest',added_station_count=len(states['added'].stations),source_manifest=str(args.manifest),source_manifest_sha256=sha(args.manifest),resolution_base_sha256=sha(RESOLUTION),canonical_support_sha256=support_sha,weights='one per actual valid retained native epoch',CDF_grid_step_deg=.1,Ground_speed_bin_mps=5.,Ground_crossing_time_bin_s=.25,nearspace_speed_bin_mps=50.,nearspace_time_bin_s=.005,geometry_recomputed=False,residuals_refitted=False,nearspace_statistics_exactly_preserved=nearproof,Ground_angle_valid_counts=dict(total.den),Ground_angle_missing_counts=dict(total.angle_invalid),Ground_speed_support=fullframes[2].to_dict('records'),redraw_input_sha256=exports,source_parts=checks,independent_numeric_checks='CDFsort/searchsorted equals independent threshold-index cumulative counts; histogram floor counts equal numpy histogram for every batch and metric',source_quality=m.get('source_quality',m.get('source_admission_note')),selection_description=m.get('selection',m.get('selection_rationale')),program_sha256=sha(__file__),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    writej(finalfreeze,freeze);writej(ROOT/'progress.json',dict(stage='GEOMETRY_TABLES_COMPLETE',Ground_epochs=total.rows,Ground_stations=len(total.stations),elapsed_s=time.monotonic()-start));print(json.dumps({k:v for k,v in freeze.items() if k not in ['source_parts','Ground_speed_support']},indent=2))
if __name__=='__main__':main()
