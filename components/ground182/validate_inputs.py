"""Validate current182 membership, QC masks and recompute Fig.4 plot summaries."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    selected=read(ROOT/'provenance/R3_Ground182_current_version.json')
    source=ROOT/'provenance/frozen_original_Ground_selection_R3.json'
    assert sha(source)==selected['selection_sha256']=='07214f1264ee81a24c80ff475a5c3d516539042cada62413f1df791eedde7b32'
    original=read(source)
    total=0
    for i,part in enumerate(original['parts']):
        path=ROOT/f'provenance/selection_masks/part_{i:04d}.npz'
        assert sha(path)==part['selection_mask_sha256']
        with np.load(path,allow_pickle=False) as z:
            assert str(z['source_sha256'])==part['source_sha256']
            assert int(z['source_rows'])==part['source_rows']
            count=int(np.unpackbits(z['packed'],bitorder='little')[:int(z['source_rows'])].sum())
        assert count==part['retained_epochs']
        total+=count
    assert total==selected['Ground_epochs']==29803114
    ground=pd.read_parquet(ROOT/'fig34/response_source/Ground_response_windows_R3.parquet')
    assert len(ground)==498430 and ground.window_id.is_unique and ground.source_unit.nunique()==182
    assert set(ground.source_unit)==set(original['stations'])
    assert ground.groupby('latitude_band').size().to_dict()==selected['latitude_window_counts']
    solar_qa=read(ROOT/'illumination/ground/Ground_expanded_solar_QA_R3.json')
    assert ground.n_points.sum()==solar_qa['represented_native_epochs']==28810247
    windows=pd.read_csv(ROOT/'fig34/response_source/common_60s_response_windows_R3.csv',float_precision='round_trip',
        usecols=['platform','signal','latitude_band','width_db','ray_B_acute_deg','scan_Bperp_acute_deg','magnetic_meridian_acute_deg'])
    bins=pd.read_csv(ROOT/'fig34/response_source/angle_bin_median_IQR_counts_R3.csv',float_precision='round_trip')
    checks=0
    for row in bins.itertuples():
        group=windows[windows.platform.eq(row.platform)&windows.latitude_band.eq(row.latitude_band)]
        subset=group[group[row.angle].ge(row.bin_left_deg)&(group[row.angle].le(90) if row.bin_right_deg==90 else group[row.angle].lt(row.bin_right_deg))]
        assert len(subset)==row.n_windows
        if len(subset):np.testing.assert_allclose(subset.width_db.quantile([.5,.25,.75]),[row.median_width_db,row.q25_width_db,row.q75_width_db],rtol=0,atol=4e-15)
        checks+=1
    assert checks==162
    hist=pd.read_csv(ROOT/'fig34/amplitude_source/histogram_source_data.csv',float_precision='round_trip')
    quant=pd.read_csv(ROOT/'fig34/amplitude_source/quantiles.csv',float_precision='round_trip')
    for row in quant.itertuples():
        h=hist[hist.platform.eq(row.platform)&hist.latitude_band.eq(row.latitude_band)]
        assert int(h['count'].sum())==row.n_records
    solar=pd.read_parquet(ROOT/'illumination/ground/Ground_solar_windows_R3.parquet')
    assert len(solar)==498430 and solar.n_points.sum()==28810247
    assert set(solar.source_unit)==set(original['stations'])
    freeze=read(ROOT/'geometry/plot_data/GEOMETRY_FREEZE.json')
    assert freeze['source_manifest_sha256']==selected['selection_sha256']
    assert freeze['Ground_retained_epochs']==29803114 and freeze['Ground_contributing_stations']==182
    coords=pd.read_csv(ROOT/'context/frozen/Ground_final_contributors_R3.csv')
    assert set(coords.station)==set(original['stations']) and len(coords)==182
    qa={'status':'PASS','source_selection_sha256':sha(source),'Ground_epochs':total,'Ground_stations':182,
        'Ground_windows':len(ground),'selection_masks_checked':len(original['parts']),
        'independently_recomputed_Fig4_bin_summaries':checks,'histogram_group_counts_checked':len(quant),
        'solar_window_membership_checked':True,'geometry_binding_checked':True,'station_coordinates_membership_checked':True}
    (ROOT/'validation').mkdir(exist_ok=True)
    (ROOT/'validation/input_checks.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
