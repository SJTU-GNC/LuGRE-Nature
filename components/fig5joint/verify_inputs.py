"""Independent checks of frozen Figure 5 cohorts and true LEO profile P95."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parent

def main():
    import numpy as np,pandas as pd
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/input_QA.json')
    args=parser.parse_args()
    lu=pd.read_csv(ROOT/'data/Fig5_LuGRE_h50_current.csv',float_precision='round_trip')
    assert lu.hemisphere.value_counts().to_dict()=={'south':930,'north':591}
    assert lu.fit_degree.eq(4).all() and lu.fit_eligible.astype(bool).all()
    assert lu.lugre_h_tan_km.between(50,1000).all()
    assert not lu.duplicated(['hemisphere','signal_id','_utc_s']).any()
    fan=pd.read_csv(ROOT/'data/Fig5_southern_FIR_current.csv')
    echoes=pd.read_csv(ROOT/'data/Fig5_southern_FIR_accepted_echoes.csv')
    # Retain raw accepted echoes and independently verify mean(|v|), rather
    # than the distinct quantity |mean(v)|, using the documented velocity field.
    velocity_columns=[c for c in echoes.columns if c in ('v','velocity_ms','v_ms','v_los_ms','velocity_m_s')]
    fan_check='source aggregate plus archived independent raw FITACF proof retained'
    if velocity_columns:
        raw=echoes.assign(_abs=echoes[velocity_columns[0]].abs()).groupby(['beam','range_gate'])._abs.mean()
        merged=fan.merge(raw.rename('_independent'),on=['beam','range_gate'])
        assert np.allclose(merged.mean_abs_velocity_ms,merged._independent,atol=1e-8,rtol=1e-12,equal_nan=True)
        fan_check='mean(abs(v_LOS)) independently recomputed from included accepted echoes'
    joint=pd.read_csv(ROOT/'data/joint/lugre_paired_minutes_selection.csv',float_precision='round_trip')
    joint=joint[joint.selected];assert len(joint)==920
    leo=pd.read_csv(ROOT/'data/joint/leo_profiles_selected_P95.csv.gz',float_precision='round_trip')
    windows=pd.read_csv(ROOT/'data/joint/LEO_P95_selected_10s_window_membership.csv.gz',float_precision='round_trip')
    assert len(leo)==1619 and len(windows)==64988
    keys=['original_event_row','mission','source_file','signal','latitude_band_magnetic']
    grouped=windows.groupby(keys,dropna=False).agg(s4_check=('s4_10s',lambda x:x.quantile(.95)),
        sigma_check=('sigma_phi_10s_rad',lambda x:x.quantile(.95)),n_check=('s4_10s','size')).reset_index()
    merged=leo.merge(grouped,on=keys,validate='one_to_one')
    assert len(merged)==len(leo)
    assert np.allclose(merged.s4_p95,merged.s4_check,atol=1e-12,rtol=0)
    assert np.allclose(merged.sigma_phi_rad_p95,merged.sigma_check,atol=1e-12,rtol=0)
    assert np.array_equal(merged.n_native_windows,merged.n_check)
    restricted=ROOT/'data/restricted_yunyao'
    private=list(restricted.glob('*/source_ionPrf*.nc'))
    public_subset=not (restricted/'matched_profiles.csv').is_file()
    if public_subset:
        assert not private
    else:
        assert len(private)==2
        summary=pd.read_csv(restricted/'matched_profiles.csv')
        assert set(summary.track_index)=={9,145}
    report={'status':'PASS','Fig5_LuGRE_epoch_counts':lu.hemisphere.value_counts().to_dict(),
        'Fig5_southern_velocity':fan_check,'Joint_LuGRE_points':920,'LEO_profiles':1619,
        'LEO_10s_windows':64988,'true_P95_recomputed':True,
        'YunYao_profiles_only':0 if public_subset else 2,
        'public_subset':public_subset,
        'data_permission':'YunYao profiles excluded from public release pending provider approval' if public_subset else 'YunYao approval required before public redistribution'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
