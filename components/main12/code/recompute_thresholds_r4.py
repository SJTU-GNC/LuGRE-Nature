"""Fixed-cohort R4 event-threshold sensitivity; no refitting or cohort changes."""
from pathlib import Path
import os,csv,io,json,hashlib
from collections import Counter
import numpy as np,pandas as pd
C=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('MAIN12_OUTPUT',str(C/'outputs')))
SOURCE=BASE/'fig2/data/all_60s_windows.csv'
OUT=BASE/'thresholds';OUT.mkdir(parents=True,exist_ok=True)
raw=SOURCE.read_bytes();d=pd.read_csv(io.BytesIO(raw),float_precision='round_trip')
BANDS=['Equatorial','Mid-lat','Polar'];THRESHOLDS=[.75,1.,1.5];W='signed_p95_p05_width_db'
assert len(d)==2196 and not d.duplicated(['track_uid','window_idx']).any()
assert d.groupby('lat_group').size().to_dict()=={'Equatorial':515,'Mid-lat':1001,'Polar':680}
assert d.n_points.between(30,60).all() and d.signed_median_db.abs().le(1).all()
np.testing.assert_allclose(d[W],d.signed_p95_db-d.signed_p05_db,atol=1e-12,rtol=0)
manual=Counter()
for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
    for threshold in THRESHOLDS:
        if float(row[W])>=threshold:manual[(threshold,row['lat_group'])]+=1
rows=[];wide=[]
for threshold in THRESHOLDS:
    event=d[W].ge(threshold);total=int(event.sum());polar=int((event&d.lat_group.eq('Polar')).sum());share=100*polar/total
    out=dict(threshold_db=threshold,nominal_background_width_db=.5,multiplier_of_nominal_background=threshold/.5,valid_windows_total=len(d),event_windows_total=total,polar_share_of_all_event_windows_percent=share)
    for band in BANDS:
        z=d.lat_group.eq(band);n=int(z.sum());e=int((event&z).sum());assert e==manual[(threshold,band)]
        rows.append(dict(threshold_db=threshold,multiplier_of_nominal_background=threshold/.5,latitude_band=band,valid_windows=n,event_windows=e,event_fraction_percent=100*e/n,non_event_windows=n-e,all_event_windows=total,polar_share_of_all_event_windows_percent=share))
        name={'Equatorial':'equatorial','Mid-lat':'mid_latitude','Polar':'polar'}[band]
        out[name+'_events']=e;out[name+'_valid_windows']=n;out[name+'_event_fraction_percent']=100*e/n
    wide.append(out)
assert [r['event_windows'] for r in rows[-3:]]==[21,26,207]
pd.DataFrame(rows).to_csv(OUT/'threshold_sensitivity_by_latitude_R4.csv',index=False,float_format='%.17g')
pd.DataFrame(wide).to_csv(OUT/'threshold_sensitivity_summary_R4.csv',index=False,float_format='%.17g')
expected={(.75,'Equatorial'):69,(.75,'Mid-lat'):111,(.75,'Polar'):364,(1.,'Equatorial'):43,(1.,'Mid-lat'):61,(1.,'Polar'):289,(1.5,'Equatorial'):21,(1.5,'Mid-lat'):26,(1.5,'Polar'):207}
assert dict(manual)==expected
report=dict(status='PASS_R4_FIXED_COHORT_THRESHOLD_COMPARISON',source_sha256=hashlib.sha256(raw).hexdigest(),valid_windows=2196,thresholds_db=THRESHOLDS,comparison=wide,independent_plain_CSV_count_agrees=True,
    interpretation='The polar event fraction remains highest at all tested thresholds. Rates vary with threshold; multipliers of the nominal median background width are not sigma levels or calibrated false-alarm probabilities.')
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report),flush=True)
