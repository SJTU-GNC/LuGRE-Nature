"""Verify recomputed R4 statistics against the selected numerical values."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
C=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=C/'outputs')
a=p.parse_args();out=a.output.resolve();expected=C/'data/r4/expected'
comparisons=[]
def compare(name,keys,values):
    aa=pd.read_csv(out/'fig2/data'/name).sort_values(keys).reset_index(drop=True)
    bb=pd.read_csv(expected/name).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(aa[keys+values],bb[keys+values],check_dtype=False,rtol=0,atol=1e-10)
    comparisons.append({'table':name,'rows':len(aa),'keys':keys,'verified_values':values,'absolute_tolerance':1e-10})
compare('all_60s_windows.csv',['track_uid','window_idx'],['n_points','signed_p05_db','signed_median_db','signed_p95_db','signed_p95_p05_width_db','is_event'])
compare('spatial_amplitude_cells.csv',['lat_bin','lon_bin'],['metric','n_samples_1hz'])
compare('fig3_panel_a_psd_no_time_interpolation.csv',['signal_name','lat_group','frequency_hz'],['median_psd_dbhz2_per_hz','n_runs'])
compare('panel_c_paired_frequency_segments.csv',['joint_run_id'],['n_exact_60sample_blocks','high_width_60sample_block_median_db','low_width_60sample_block_median_db'])
compare('paired_exact60_blocks.csv',['joint_run_id','block_index'],['high_width_db','low_width_db','n_valid_paired_seconds'])
d=pd.read_csv(out/'fig2/data/all_60s_windows.csv')
rates=d.groupby('lat_group').agg(events=('is_event','sum'),valid=('is_event','size')).to_dict('index')
assert rates=={'Equatorial':{'events':21,'valid':515},'Mid-lat':{'events':26,'valid':1001},'Polar':{'events':207,'valid':680}}
background=None
b=out/'qc_background/qa/background_QC_R4.json'
if b.exists():
    background=json.loads(b.read_text('utf-8'))
    assert background['highalt_retained_windows']==9178
    assert .48<background['highalt_median_W_db']<.50
report={'status':'PASS_R4_NUMERICAL_REPRODUCTION','comparisons':comparisons,'event_rates':rates,'background_windows':background['highalt_retained_windows'] if background else None,'background_median_W_db':background['highalt_median_W_db'] if background else None,'inputs_are_package_relative':True,'pixel_identity_required':False}
(out/'numerical_verification_r4.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
