"""Recompute antenna evidence; preserve the separately authored schematic.

The selected artwork was manually assembled. The left schematic is an authored
bitmap, not a numerical result. Right panels are newly drawn from measured
off-boresight samples and the archived NAVCEN/JRC pattern inputs. Scientific
statistics, rather than the manually assembled final image bytes, are the
reproducibility target for this figure.
"""
from pathlib import Path
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
from PIL import Image
import generate_panels_bc as bc
import generate_panel_c_individual as individual
import plot_main_lobe_previous_ab_gps_galileo as dist
import plot_main_lobe_gps_galileo as core

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('MAIN12_OUTPUT',str(ROOT/'outputs')))/'antenna'
OUT.mkdir(parents=True,exist_ok=True)
points,summaries,gps,e1,e5,missing=bc.build_inputs()
expected_patterns=pd.read_csv(ROOT/'data/antenna/derived_data/lugre_panel_c_derived_pattern_envelopes.csv')
expected_geometry=pd.read_csv(ROOT/'data/antenna/derived_data/lugre_panel_c_signal_summary.csv')
for signal,summary in summaries.items():
    frozen=expected_patterns[expected_patterns.signal.eq(signal)].sort_values('theta_deg')
    np.testing.assert_allclose(summary.theta,frozen.theta_deg.to_numpy(float),atol=1e-10,rtol=0)
    for actual,key in [(summary.median,'median_delta_eirp_db'),(summary.p05,'p05_delta_eirp_db'),(summary.p95,'p95_delta_eirp_db')]:
        np.testing.assert_allclose(actual,frozen[key].to_numpy(float),atol=1e-10,rtol=0,equal_nan=True)
    obs=points[points.signal_name.eq(signal)]
    frozen_geometry=expected_geometry[expected_geometry.signal_name.eq(signal)].iloc[0]
    assert len(obs)==int(frozen_geometry.point_count)
    np.testing.assert_allclose([obs.tx_off_boresight_deg.min(),obs.tx_off_boresight_deg.median(),obs.tx_off_boresight_deg.max()],
        [frozen_geometry.angle_min_deg,frozen_geometry.angle_median_deg,frozen_geometry.angle_max_deg],atol=1e-10,rtol=0)
fig=plt.figure(figsize=(4510/600,2860/600),facecolor='white')
plt.rcParams.update({'font.size':6.0,'axes.labelsize':6.0,'xtick.labelsize':5.7,'ytick.labelsize':5.7})
with Image.open(ROOT/'reference/Extended_Data_GNSS_antenna_geometry.png') as artwork:
    schematic=artwork.crop((0,0,2350,2860)).convert('RGB')
left=fig.add_axes([0,0,2350/4510,1]);left.imshow(schematic,aspect='auto');left.axis('off')
right=fig.add_gridspec(2,1,left=.555,right=.978,bottom=.065,top=.91,height_ratios=[.82,1.47],hspace=.42)
ax,box,amin,amax=dist.draw_previous_panel_a(fig,right[0,0],points)
# Match the selected composite's panel lettering; source function formerly used a.
for text in ax.texts:
    if text.get_text().startswith('a  LuGRE'):
        text.set_text('b. LuGRE off-boresight distributions at 50–1,000 km tangent heights')
        text.set_fontsize(7.2)
lower=GridSpecFromSubplotSpec(2,2,subplot_spec=right[1,0],wspace=.18,hspace=.43)
valleys={'GPS L1':gps['GPS L1'],'GPS L5':gps['GPS L5'],'Galileo E1':e1,'Galileo E5a':e5}
rows=[]
for i,signal in enumerate(core.SIGNALS):
    axis=fig.add_subplot(lower[i//2,i%2])
    obs=points[points.signal_name.eq(signal)]
    handles=individual.draw_signal(axis,signal,summaries[signal],obs,valleys[signal],individual.Y_SIDES[signal])
    for t in axis.texts:t.set_fontsize(5.6)
    axis.title.set_fontsize(6.8);axis.xaxis.label.set_fontsize(5.8);axis.yaxis.label.set_fontsize(5.8)
    axis.tick_params(labelsize=5.3)
    axis.legend(handles=handles[:2],loc='lower right',fontsize=4.6,frameon=False,handlelength=1.6)
    rows.append({'signal':signal,'n':len(obs),'minimum_off_boresight_deg':float(obs.tx_off_boresight_deg.min()),
                 'maximum_off_boresight_deg':float(obs.tx_off_boresight_deg.max()),
                 'median_off_boresight_deg':float(obs.tx_off_boresight_deg.median()),'first_valley':valleys[signal]})
    if i==0:fig.text(axis.get_position().x0,axis.get_position().y1+.038,'c. GNSS antenna transmit pattern',fontsize=7.2,fontweight='bold')
fig.savefig(OUT/'Extended_Data_GNSS_antenna_geometry.png',dpi=600)
plt.close(fig)
pd.DataFrame([{k:v for k,v in row.items() if k!='first_valley'} for row in rows]).to_csv(OUT/'recomputed_signal_geometry_summary.csv',index=False)
patterns=[]
for signal,s in summaries.items():
    for i,angle in enumerate(s.theta):patterns.append({'signal':signal,'off_boresight_deg':angle,'median_db':s.median[i],'p05_db':s.p05[i],'p95_db':s.p95[i]})
pd.DataFrame(patterns).to_csv(OUT/'recomputed_pattern_envelopes.csv',index=False)
def json_value(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f'Unsupported QA metadata type: {type(value).__name__}')

(OUT/'reproduction_QA.json').write_text(json.dumps({'status':'PASS_NUMERICAL_SOURCE_REPRODUCTION',
    'reproduction_class':'source statistics and plotting; left schematic reused as authored artwork',
    'final_manual_composite_pixel_identity_not_claimed':True,'missing_public_GPS_pattern_SVNs':missing,
    'pattern_envelopes_and_geometry_counts_verified_against_frozen_source_tables':True,
    'tangent_height_selection_km':[50,1000],'signals':rows},indent=2,default=json_value),encoding='utf-8')
print(OUT/'Extended_Data_GNSS_antenna_geometry.png')
