"""Enrich the authoritative expanded windows using real native epochs."""
from pathlib import Path
import sys, os, json, hashlib, shutil, importlib.util, time, math
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/LUGRE/python_deps/pyarrow_py312')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent
OLD=Path('E:/LUGRE/figures_R3/Ground_F_expansion_R3_20261028/illumination')
SEL=ROOT.parent/'Ground_selection_R3.json'
KEYS=['source_unit','satellite','signal','fit_arc_id','qc_segment_id','latitude_band','first_utc_s','last_utc_s','n_points']
GROUP=['source_unit','satellite','signal','fit_arc_id','latitude_band','utc_minute_start_s']
COORDS=['reference_latitude_deg','reference_longitude_deg','reference_height_km']
XYZ=['geometry_point_ecef_m_x','geometry_point_ecef_m_y','geometry_point_ecef_m_z']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def independent_check(t,row,solar):
    """Recheck selected windows with scalar dot products and quadratic roots."""
    times=t.utc_s.to_numpy(float)
    sun=solar.sun_ecef(times)
    angles=[];local=[];sunlit=[]
    axes=np.array([6378.137,6378.137,6378.137*(1-1/298.257223563)])
    for j,obs in enumerate(t.itertuples()):
        p=np.array([getattr(obs,k) for k in XYZ])/1000
        u=sun[j]-p;u=u/np.linalg.norm(u)
        la,lo=math.radians(obs.reference_latitude_deg),math.radians(obs.reference_longitude_deg)
        normal=np.array([math.cos(la)*math.cos(lo),math.cos(la)*math.sin(lo),math.sin(la)])
        angles.append(math.degrees(math.acos(float(np.clip(np.dot(normal,u),-1,1)))))
        slon=math.degrees(math.atan2(sun[j,1],sun[j,0]))
        local.append((12+(obs.reference_longitude_deg-slon)/15)%24)
        ps=p/axes;us=u/axes
        a=float(np.dot(us,us));b=2*float(np.dot(ps,us));c=float(np.dot(ps,ps)-1)
        disc=b*b-4*a*c
        blocked=c<0 or (disc>=0 and (-b+math.sqrt(max(0,disc)))/(2*a)>=0)
        sunlit.append(not blocked)
    rep=int(np.argmin(np.abs(times-np.median(times))))
    err={
      'representative_utc_s':abs(times[rep]-row.representative_utc_s),
      'representative_solar_zenith_deg':abs(angles[rep]-row.representative_solar_zenith_deg),
      'solar_zenith_min_deg':abs(min(angles)-row.solar_zenith_min_deg),
      'solar_zenith_max_deg':abs(max(angles)-row.solar_zenith_max_deg),
      'solar_zenith_median_deg':abs(float(np.median(angles))-row.solar_zenith_median_deg),
      'representative_solar_local_time_h':abs((local[rep]-row.representative_solar_local_time_h+12)%24-12),
      'sunlit_fraction':abs(float(np.mean(sunlit))-row.sunlit_fraction),
      'width_db':abs(float(t.delta_poly4_db.quantile(.95)-t.delta_poly4_db.quantile(.05))-row.width_db)}
    assert max(err.values())<1e-10,err
    assert bool(sunlit[rep])==bool(row.representative_sunlit)
    assert int(t.iloc[rep].source_file_id)==int(row.representative_source_file_id)
    return dict(window_id=int(row.window_id),station=str(row.source_unit),native_epochs=len(t),maximum_error=max(err.values()),errors=err)

def enrich(part,w,solar):
    path=Path(part['source']);assert sha(path)==part['source_sha256']
    fields=list(dict.fromkeys(GROUP[:-1]+['utc_s','delta_poly4_db','ipp_CD_magnetic_latitude_deg','source_file_id']+COORDS+XYZ))
    d=pq.ParquetFile(path).read(columns=fields,use_threads=False).to_pandas()
    if part.get('selection_mask'):
        assert sha(part['selection_mask'])==part['selection_mask_sha256']
        with np.load(part['selection_mask'],allow_pickle=False) as z:
            assert str(z['source_sha256'])==part['source_sha256'] and int(z['source_rows'])==len(d)
            keep=np.unpackbits(z['packed'],bitorder='little')[:len(d)].astype(bool)
        assert int(keep.sum())==part['retained_epochs']
        d=d.loc[keep].copy()
    else:assert len(d)==part['retained_epochs']
    d['utc_minute_start_s']=np.floor(d.utc_s/60).astype(np.int64)*60
    d=d.merge(w[GROUP+['window_id','n_points','first_utc_s','last_utc_s']],on=GROUP,how='inner',validate='many_to_many')
    d=d.loc[d.utc_s.ge(d.first_utc_s)&d.utc_s.le(d.last_utc_s)].sort_values(['window_id','utc_s']).reset_index(drop=True)
    assert not d.duplicated(['window_id','utc_s']).any()
    assert not d.duplicated(GROUP[:-1]+['utc_s']).any(), 'Native epoch assigned to overlapping windows'
    g=d.groupby('window_id',sort=True);counts=g.size();expected=w.set_index('window_id').loc[counts.index]
    assert len(counts)==len(w) and np.array_equal(counts,expected.n_points)
    assert np.array_equal(g.utc_s.min(),expected.first_utc_s) and np.array_equal(g.utc_s.max(),expected.last_utc_s)
    # Do not create or fill missing seconds. Accepted windows must reproduce their actual source support.
    assert np.all(np.diff(d.utc_s.to_numpy())[d.window_id.to_numpy()[1:]==d.window_id.to_numpy()[:-1]]==1)
    q=g.delta_poly4_db.quantile([.05,.5,.95]).unstack()
    errors={c:float(np.max(np.abs(q[p]-expected[c]))) for p,c in [(.05,'signed_p05_db'),(.5,'signed_median_db'),(.95,'signed_p95_db')]}
    errors['width_db']=float(np.max(np.abs(q[.95]-q[.05]-expected.width_db)))
    assert max(errors.values())<1e-11,errors
    sf=solar.solar_fields(d.utc_s.to_numpy(),d.reference_latitude_deg.to_numpy(),d.reference_longitude_deg.to_numpy(),d.reference_height_km.to_numpy(),d[XYZ].to_numpy()/1000)
    for k,v in sf.items():d[k]=v
    assert np.max(np.abs(d.reference_height_km-350))<1e-6
    center=d.loc[d.utc_s.eq(d.first_utc_s+(d.n_points-1)//2)].set_index('window_id')
    assert len(center)==len(w) and center.index.is_unique
    g=d.groupby('window_id',sort=True)
    stats=g.agg(represented_epoch_count=('utc_s','size'),sunlit_fraction=('sunlit','mean'),
      solar_zenith_min_deg=('solar_zenith_deg','min'),solar_zenith_max_deg=('solar_zenith_deg','max'),solar_zenith_median_deg=('solar_zenith_deg','median'),
      reference_height_min_km=('reference_height_km','min'),reference_height_max_km=('reference_height_km','max'),reference_height_median_km=('reference_height_km','median'),
      reference_latitude_min_deg=('reference_latitude_deg','min'),reference_latitude_max_deg=('reference_latitude_deg','max'),reference_latitude_median_deg=('reference_latitude_deg','median'))
    for field,period,out in [('reference_longitude_deg',360,'reference_longitude_circular_median_deg'),('solar_local_time_h',24,'solar_local_time_circular_median_h')]:
        radians=d[field].to_numpy()*2*np.pi/period;d['_sin']=np.sin(radians);d['_cos']=np.cos(radians)
        gg=d.groupby('window_id',sort=True)
        circ=np.arctan2(gg._sin.mean(),gg._cos.mean())*period/(2*np.pi)
        centers=d.window_id.map(circ).to_numpy();d['_offset']=(d[field]-centers+period/2)%period-period/2
        value=(circ+d.groupby('window_id',sort=True)._offset.median())%period
        if period==360:value=(value+180)%360-180
        stats[out]=value
    mapping={'utc_s':'utc_s','reference_latitude_deg':'reference_latitude_deg','reference_longitude_deg':'reference_longitude_deg',
      'reference_height_km':'reference_height_WGS84_km','solar_zenith_deg':'solar_zenith_deg','solar_local_time_h':'solar_local_time_h',
      'sunlit':'sunlit','ipp_CD_magnetic_latitude_deg':'mlat_centered_dipole_deg','source_file_id':'source_file_id',
      'sun_ecef_longitude_deg':'sun_ecef_longitude_deg','sun_ecef_latitude_geocentric_deg':'sun_ecef_latitude_geocentric_deg'}
    for src,dst in mapping.items():stats['representative_'+dst]=center[src]
    stats['representative_geometry_unix_s']=center.utc_s
    stats['representative_epoch_identity']=center.source_unit.astype(str)+'|'+center.satellite.astype(str)+'|'+center.signal.astype(str)+'|'+center.utc_s.astype(str)
    stats['representative_observation_id']=pd.Series(pd.NA,index=stats.index,dtype='Int64')
    d['_LT_offset']=(d.solar_local_time_h-d.window_id.map(center.solar_local_time_h)+12)%24-12
    stats['solar_local_time_offset_from_representative_min_h']=d.groupby('window_id')._LT_offset.min()
    stats['solar_local_time_offset_from_representative_max_h']=d.groupby('window_id')._LT_offset.max()
    stats['representative_rule']='actual epoch nearest median UTC; ties within 1e-7 s choose earlier epoch'
    result=w.merge(stats,left_on='window_id',right_index=True,validate='one_to_one')
    aliases={'plot_solar_local_time_h':'representative_solar_local_time_h','plot_sza_deg':'representative_solar_zenith_deg',
      'plot_latitude_deg':'representative_reference_latitude_deg','plot_longitude_deg':'representative_reference_longitude_deg',
      'plot_height_km':'representative_reference_height_WGS84_km','plot_utc_s':'representative_utc_s'}
    for dst,src in aliases.items():result[dst]=result[src]
    pd.testing.assert_frame_equal(result[w.columns].reset_index(drop=True),w.reset_index(drop=True),check_exact=True)
    checks=[]
    for row in result.iloc[np.unique([0,len(result)//2,len(result)-1])].itertuples():
        checks.append(independent_check(d.loc[d.window_id.eq(row.window_id)],row,solar))
    return result,dict(source_part=str(part['source_part']),source=str(path),source_sha256=sha(path),windows=len(w),native_epochs=len(d),width_errors=errors,independent_checks=checks)

def main():
    start=time.time()
    selection=json.loads(SEL.read_text(encoding='utf8'))
    assert selection['status']=='FROZEN_SHARED_GROUND_SELECTION'
    source=Path(selection['windows_source']);assert sha(source)==selection['windows_sha256']
    assert not (ROOT/'ground/Ground_solar_windows_R3.parquet').exists(),'Completed solar output already exists; preserve it'
    core=pd.read_parquet(source)
    assert core.window_id.is_unique and not core.duplicated(KEYS).any()
    assert len(core)==selection['Ground_windows'] and set(core.source_unit.astype(str))<=set(selection['stations'])
    assert 'KIRI00KIR' in set(core.source_unit) and 'KIRG01KIR' not in set(core.source_unit)
    oldpath=OLD/'ground/Ground_solar_windows_R3.parquet';old=pd.read_parquet(oldpath)
    assert not old.duplicated(KEYS).any()
    idx=pd.MultiIndex.from_frame(old[KEYS]).get_indexer(pd.MultiIndex.from_frame(core[KEYS]));reuse=idx>=0
    oldselection=json.loads((OLD.parent/'Ground_selection_R3.json').read_text(encoding='utf8'))
    oldhash={str(x['source_part']):x['source_sha256'] for x in oldselection['parts']}
    newhash={str(x['source_part']):x['source_sha256'] for x in selection['parts']}
    bookkeeping={'window_id','source_part','cohort','continuous_run_id','parent_solar_window_id'}
    science=[c for c in core if c in old and c not in bookkeeping]
    candidate_positions=np.flatnonzero(reuse)
    if len(candidate_positions):
        candidate=old.iloc[idx[reuse]].reset_index(drop=True)
        current=core.loc[reuse].reset_index(drop=True)
        equal=np.ones(len(current),dtype=bool)
        for c in science:
            a=current[c].astype(object);b=candidate[c].astype(object)
            equal&=(a.eq(b)|(a.isna()&b.isna())).to_numpy(dtype=bool)
        equal&=np.array([newhash[str(a)]==oldhash[str(b)] for a,b in zip(current.source_part,candidate.source_part)])
        reuse[candidate_positions[~equal]]=False
    selected=old.iloc[idx[reuse]].reset_index(drop=True)
    pd.testing.assert_frame_equal(core.loc[reuse,science].reset_index(drop=True),selected[science],check_exact=True,check_dtype=False,check_categorical=False)
    extras=[c for c in old if c not in core]
    preserved=core.loc[reuse].copy().reset_index(drop=True)
    for c in extras:preserved[c]=selected[c]
    pd.testing.assert_frame_equal(preserved[extras],selected[extras],check_exact=True)
    for name in ['ground','ground/parts','nearspace','figures','captions']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    for name in ['nearspace_solar_windows_R3.parquet','solar_geometry_R3.py']:
        dst=ROOT/'nearspace'/name;src=OLD/'nearspace'/name
        if dst.exists():assert sha(dst)==sha(src)
        else:shutil.copy2(src,dst)
    solar=load('solar_expanded_native',ROOT/'nearspace/solar_geometry_R3.py')
    frozen={str(p):sha(p) for p in [SEL,source,oldpath,OLD/'nearspace/nearspace_solar_windows_R3.parquet',solar.ORIGINAL_SOLAR_SOURCE]}
    newcore=core.loc[~reuse].copy();newparts={str(x['source_part']):x for x in selection['parts']}
    newcore['source_part']=newcore.source_part.astype(str)
    frames=[preserved];reports=[]
    for part_id,w in newcore.groupby('source_part',sort=True):
        part=newparts[str(part_id)]
        result,report=enrich(part,w,solar)
        output=ROOT/'ground/parts'/f'Ground_solar_part_{hashlib.sha256(str(part_id).encode()).hexdigest()[:16]}_R3.parquet'
        result.to_parquet(output,index=False,compression='zstd')
        report.update(output=str(output),output_sha256=sha(output));reports.append(report);frames.append(result)
        write(ROOT/'progress_R3.json',dict(parts_complete=len(reports),new_windows=sum(x['windows'] for x in reports),elapsed_seconds=time.time()-start))
        print('Solar part',part_id,'windows',len(w),flush=True)
    merged=pd.concat(frames,ignore_index=True).set_index('window_id').loc[core.window_id].reset_index()
    result=core.copy()
    for c in merged:
        if c not in result:result[c]=merged[c]
    assert len(result)==len(core) and result.window_id.is_unique
    pd.testing.assert_frame_equal(result[core.columns],core,check_exact=True)
    pd.testing.assert_frame_equal(result.loc[reuse,extras].reset_index(drop=True),selected[extras],check_exact=True,check_dtype=False,check_categorical=False)
    assert result.groupby('latitude_band').size().to_dict()==selection['window_counts']
    assert result.represented_epoch_count.sum()==core.n_points.sum()
    target=ROOT/'ground/Ground_solar_windows_R3.parquet'
    result.to_parquet(target,index=False,compression='zstd')
    assert all(sha(p)==h for p,h in frozen.items())
    qa=dict(status='PASS_EXPANDED_GROUND_SOLAR_EXACT_SHARED_WINDOWS',selection_manifest=str(SEL),selection_sha256=sha(SEL),windows_source=str(source),windows_sha256=sha(source),
      Ground_stations=int(result.source_unit.nunique()),Ground_windows=len(result),window_counts=selection['window_counts'],represented_native_epochs=int(result.n_points.sum()),
      reused_windows=int(reuse.sum()),new_windows=int((~reuse).sum()),recomputed_station_ids=sorted(newcore.source_unit.unique()),new_stations=sorted(set(core.source_unit)-set(old.source_unit)),old_solar_values_bitexact=True,all_core_columns_bitexact=True,
      native_epoch_illumination_computed=True,representative_values_not_broadcast_to_other_epochs=True,interpolation=False,added_masks=False,nearspace_unchanged=True,
      solar_model='Unchanged ERFA epv00/c2t06a; TT approximates TDB, UT1 approximates UTC, zero polar motion; solar-centre ray and solid WGS84 ellipsoid',
      parts=reports,source_sha256=frozen,outputs={str(target):sha(target)},elapsed_seconds=time.time()-start)
    write(ROOT/'ground/Ground_expanded_solar_QA_R3.json',qa)
    print(json.dumps({k:v for k,v in qa.items() if k not in ['parts','source_sha256']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
