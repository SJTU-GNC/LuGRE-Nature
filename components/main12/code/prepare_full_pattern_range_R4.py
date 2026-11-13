"""Recover full reference-cut range while preserving the archived ED3 centre.

Reads original sources only; all generated files remain beside this script on E:.
GPS: every actual available observed-SVN x azimuth cut, normalized per boresight.
Galileo: all 360 actual GRAP population-model azimuth cuts, not per-spacecraft.
"""
from pathlib import Path
import os, sys, json, hashlib
sys.dont_write_bytecode = True
COMPONENT = Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'antenna_raw_check'
ROOT.mkdir(parents=True,exist_ok=True)
PACKAGE=COMPONENT/'data/antenna'
ARCHIVE=PACKAGE/'R4_plot_assets'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent/'mplconfig'))
sys.path.insert(0,str(COMPONENT/'code'))
import numpy as np
import pandas as pd
import plot_main_lobe_gps_galileo as core
import plot_main_lobe_previous_ab_gps_galileo as previous

GRID = np.arange(0.,76.,1.)
inputs = [ARCHIVE/'selected_lunar_offboresight_points_R3.csv',
          ARCHIVE/'reference_pattern_envelopes_R3.csv',
          ARCHIVE/'reference_first_valley_R3.csv',
          PACKAGE/'GPS_antenna_patterns/igs_satellite_metadata.snx',
          COMPONENT/'provenance/antenna_source_catalog.csv',
          COMPONENT/'code/plot_main_lobe_gps_galileo.py',
          COMPONENT/'code/plot_main_lobe_previous_ab_gps_galileo.py',
          COMPONENT/'code/plot_main_lobe_grap_revised.py']
for folder,suffixes in [('GPS_antenna_patterns',['*.pptx','*.xlsx','*.zip']),('GRAP_metadata',['*.xlsx'])]:
    for suffix in suffixes:
        inputs.extend(sorted((PACKAGE/folder).glob(suffix)))

def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

hashes = {str(p):sha(p) for p in inputs}
points = pd.read_csv(inputs[0],usecols=['signal_name','sat'])
archive = pd.read_csv(inputs[1])
valleys = pd.read_csv(inputs[2]).set_index('signal').first_valley_aggregate_median_deg.to_dict()
mapping = core.load_prn_svn_map()
rows, raw_rows, catalog, qa_signals = [], [], [], []

for signal in core.SIGNALS:
    vectors, ids = [], []
    if signal.startswith('GPS'):
        patterns, observed_svns, missing_svns = previous.load_gps_pattern_set(signal,points,mapping)
        for svn,cuts in sorted(patterns.items()):
            for cut_index,(theta,gain) in enumerate(cuts):
                valid=np.isfinite(theta)&np.isfinite(gain)
                theta,gain=theta[valid],gain[valid]
                if len(theta)<5:
                    continue
                order=np.argsort(theta); theta,gain=theta[order],gain[order]
                normalized=gain-np.interp(0.,theta,gain)
                row=np.interp(GRID,theta,normalized,left=np.nan,right=np.nan)
                vectors.append(row)
                identity=dict(signal=signal,curve_id=f'{signal.replace(" ","_")}_SVN{svn:03d}_cut{cut_index:03d}',
                              source_kind='NAVCEN measured spacecraft angular cut',svn=svn,cut_index=cut_index,azimuth_deg=np.nan)
                ids.append(identity)
                catalog.append(dict(**identity,original_theta_min_deg=float(theta.min()),original_theta_max_deg=float(theta.max())))
        reproduced=core.summarize_satellite_patterns(patterns,'NAVCEN measured SV patterns; equal weight per patterned SVN')
        qa=dict(signal=signal,source_kind='all original available observed-SVN angular cuts',
                observed_svns=observed_svns,patterned_svns=sorted(patterns),missing_svns=missing_svns,
                n_source_cuts=len(vectors),cuts_per_svn={str(k):len(v) for k,v in patterns.items()},
                missing_sources_not_substituted=True,
                centre_definition='archived equal-SVN azimuth-quantile mixture median, copied exactly')
    else:
        filename='GRAP_File_E1_.xlsx' if signal=='Galileo E1' else 'GRAP_File_E5a_.xlsx'
        grap=core.load_grap_band(core.GRAP_DIR/filename)
        keep=grap['theta']<=75
        assert np.array_equal(grap['theta'][keep],GRID)
        normalized=grap['eirp'][:,keep]-grap['eirp'][:,[0]]
        for i,(azimuth,row) in enumerate(zip(grap['azimuth'],normalized)):
            vectors.append(row)
            identity=dict(signal=signal,curve_id=f'{signal.replace(" ","_")}_GRAP_az{int(azimuth):03d}',
                          source_kind='GRAP FOC population-model azimuth cut; not a spacecraft',
                          svn=np.nan,cut_index=i,azimuth_deg=azimuth)
            ids.append(identity)
            catalog.append(dict(**identity,original_theta_min_deg=float(grap['theta'].min()),original_theta_max_deg=float(grap['theta'].max())))
        reproduced=core.load_galileo_summary(signal)
        qa=dict(signal=signal,source_kind='all 360 GRAP population-model azimuth cuts; not per-spacecraft',
                n_source_cuts=len(vectors),duplicate_360_degree_cut_omitted=True,
                model_confidence_bounds_not_used_as_minmax=True,
                centre_definition='archived median across boresight-normalized azimuth cuts, copied exactly')
    matrix=np.vstack(vectors)
    low=np.nanmin(matrix,axis=0); high=np.nanmax(matrix,axis=0); n=np.isfinite(matrix).sum(axis=0)
    old=archive[archive.signal.eq(signal)].sort_values('theta_deg')
    assert np.array_equal(old.theta_deg.to_numpy(),GRID)
    median=old.median_delta_eirp_db.to_numpy(copy=True)
    p05=old.p05_delta_eirp_db.to_numpy();p95=old.p95_delta_eirp_db.to_numpy()
    centre_error=float(np.max(np.abs(np.asarray(reproduced.median)-median)))
    assert centre_error<1e-10
    assert np.all(low<=p05+1e-10) and np.all(high>=p95-1e-10)
    assert np.all(low<=median+1e-10) and np.all(high>=median-1e-10)
    for j,theta in enumerate(GRID):
        rows.append(dict(signal=signal,theta_deg=theta,n_cuts=int(n[j]),min_delta_eirp_db=low[j],
                         median_delta_eirp_db=median[j],max_delta_eirp_db=high[j],first_valley_deg=valleys[signal],
                         legacy_p05_delta_eirp_db=p05[j],legacy_p95_delta_eirp_db=p95[j]))
    for identity,row in zip(ids,matrix):
        for theta,value in zip(GRID,row):
            raw_rows.append(dict(**identity,theta_deg=theta,relative_eirp_db=value))
    qa.update(centre_reproduced_max_abs_difference_db=centre_error,archived_centre_copied_exactly=True,
              full_range_contains_archived_p05_p95_at_all_angles=True,full_range_contains_archived_centre_at_all_angles=True,
              first_valley_deg=valleys[signal],full_min_over_all_source_values_db=float(np.nanmin(matrix)),
              full_max_over_all_source_values_db=float(np.nanmax(matrix)),
              source_values_below_minus40=int((matrix<-40).sum()),source_values_above_plus5=int((matrix>5).sum()),
              summary_angles_min_below_minus40=int((low<-40).sum()),
              n_finite_cuts_at_0_20_24_60_75_deg={str(a):int(n[a]) for a in [0,20,24,60,75]},
              no_extrapolation=True,all_source_values_preserved=True)
    qa_signals.append(qa)
    print(json.dumps(qa,ensure_ascii=False),flush=True)

summary=pd.DataFrame(rows)
long=pd.DataFrame(raw_rows)
catalog=pd.DataFrame(catalog)
paths={'reference_full_range_summary_R3.csv':summary,'original_normalized_cuts_R3.csv.gz':long,'original_cut_catalog_R3.csv':catalog}
for filename,frame in paths.items():
    frame.to_csv(ROOT/filename,index=False,float_format='%.17g',compression='gzip' if filename.endswith('.gz') else None)

# Round-trip parsed centres and aggregate the long source table independently.
read_summary=pd.read_csv(ROOT/'reference_full_range_summary_R3.csv',float_precision='round_trip')
read_long=pd.read_csv(ROOT/'original_normalized_cuts_R3.csv.gz',float_precision='round_trip')
independent=read_long.groupby(['signal','theta_deg']).relative_eirp_db.agg(['count','min','max']).reset_index()
joined=read_summary.merge(independent,on=['signal','theta_deg'],validate='one_to_one')
assert np.array_equal(joined.n_cuts,joined['count'])
assert np.array_equal(joined.min_delta_eirp_db,joined['min'])
assert np.array_equal(joined.max_delta_eirp_db,joined['max'])
expected=archive.sort_values(['signal','theta_deg']).median_delta_eirp_db.to_numpy()
restored=read_summary.sort_values(['signal','theta_deg']).median_delta_eirp_db.to_numpy()
assert np.array_equal(expected,restored)
assert not read_long.duplicated(['curve_id','theta_deg']).any()
for p,h in hashes.items():
    assert sha(p)==h,p
qa=dict(status='COMPLETE_FULL_SOURCE_RANGE_ARCHIVED_CENTRE_PRESERVED',signals=qa_signals,
        source_hashes=hashes,all_original_inputs_unchanged=True,
        range_definition='pointwise min/max of all finite, independently boresight-normalized original cuts; no azimuth-median collapse and no percentile clipping',
        centre_definition='exact archived median; not recomputed from the min/max or changed to per-SVN medians',
        first_valleys_restored_deg={k:float(v) for k,v in valleys.items()},
        CSV_roundtrip_and_independent_minmax_count_check=True,centre_bitwise_equal_after_CSV_roundtrip=True,
        normalization='same per-original-cut boresight normalization and 1-degree no-extrapolation grid as the archived parser',
        outputs={name:dict(rows=len(frame),sha256=sha(ROOT/name)) for name,frame in paths.items()},
        caution='GPS source cuts correspond to available observed SVNs and reference azimuths; Galileo cuts are population-model azimuths, not spacecraft. Full range includes deep nulls below the -40dB display limit; source values must not be filtered by display limits.')
(ROOT/'full_pattern_range_QA_R3.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
