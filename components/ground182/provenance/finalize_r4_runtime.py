"""Assembly-time adaptation of portable interfaces; not a runtime dependency."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
p=R/'reproduce.py';s=p.read_text('utf-8');prefix=s[:s.index('def run(task,output):')]
prefix=prefix.replace('"""Regenerate the six selected Ground182 figures from the bundled frozen inputs."""','"""Reconstruct R4 LuGRE QC and regenerate the six Ground182 component figures offline."""')
tail=r'''
def run(task,output,numeric=None):
    output.mkdir(parents=True,exist_ok=True)
    amp=(numeric/'amplitude_source') if numeric else ROOT/'fig34/amplitude_source'
    response=(numeric/'response_source') if numeric else ROOT/'fig34/response_source'
    if task=='fig3':
        command=[ROOT/'fig34/plotting/plot_Fig3_R3.py','--data-dir',amp,'--output-dir',output,'--expected-status','COMPLETE_FIG3_R4_THREE_CONDITION_QC']
    elif task=='fig4':
        command=[ROOT/'fig34/plotting/plot_Fig4_R3.py','--data-dir',response,'--output-dir',output,'--expected-status','PASS_FIG4_R4_THREE_CONDITION_QC']
    elif task=='geometry':
        command=[ROOT/'geometry/plot_geometry_R3.py','--data-dir',ROOT/'geometry/plot_data','--output-dir',output]
    elif task=='illumination':command=[ROOT/'illumination/plot_LT_illumination_R3.py']
    elif task=='context':return run_context(output)
    else:raise ValueError(task)
    env=os.environ.copy();env['LUGRE_FIGURE_OUTPUT']=str(output)
    if task=='illumination' and numeric:env['LUGRE_NEARSPACE_SOLAR']=str(numeric/'illumination/nearspace_solar_windows_R3.parquet')
    env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUTF8']='1'
    subprocess.run([sys.executable,'-B',*map(str,command)],cwd=ROOT,env=env,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',nargs='?',default='all',choices=['all','recompute',*TASKS])
    p.add_argument('--output-dir',type=Path,default=OUTPUT)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    tasks=TASKS if a.task=='all' else [a.task]
    numeric=out/'numerical_r4'
    if a.task=='recompute' or any(t in ['fig3','fig4','illumination'] for t in tasks):
        subprocess.run([sys.executable,'-B',str(ROOT/'code/recompute_r4.py'),'--output-dir',str(numeric)],cwd=ROOT,check=True)
    if a.task!='recompute':
        for task in tasks:
            print('Rendering',task,flush=True);run(task,out,numeric if task in ['fig3','fig4','illumination'] else None)
    print('Outputs:',out)
if __name__=='__main__':main()
'''
p.write_text(prefix+tail,encoding='utf-8')
p=R/'validate_inputs.py';s=p.read_text('utf-8');s=s[:s.rindex("if __name__=='__main__':main()")]
s+=r'''

def validate_r4():
    sys.path.insert(0,str(ROOT/'code'))
    from recompute_r4 import rebuild
    result=rebuild(ROOT/'validation/reconstructed_r4',verify=True)
    p=ROOT/'validation/input_checks.json';base=read(p)
    base.update(revision='R4',R4_LuGRE_reconstruction_status=result['status'],
        LuGRE_Fig3_epochs=result['Fig3_LuGRE_epochs'],LuGRE_quality_windows=result['Fig3_quality_windows'],
        LuGRE_Fig4_LT_windows=result['Fig4_LT_LuGRE_windows'],
        every_rebuilt_R4_table_matches_packaged_reference=True)
    p.write_text(json.dumps(base,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(base,indent=2))
if __name__=='__main__':
    main();validate_r4()
'''
p.write_text(s,encoding='utf-8')
# R4 data dictionary describes the released computational boundary and distinct units.
dictionary={
 'revision':'R4','component':'ground182',
 'scope':'Processed-data scientific reproduction of Fig3,Fig4 and four Extended Data figures; raw multi-station reprocessing is not included.',
 'fixed_Ground':{'stations':182,'accepted_1Hz_receiver_satellite_epochs':29803114,'accepted_minute_windows':498430,'selection_manifest':'provenance/frozen_original_Ground_selection_R3.json','selection_masks':'provenance/selection_masks'},
 'LuGRE':{'original_comparison_epochs':82761,'original_UTC_candidate_windows':1654,'quality_windows':1354,'Fig3_epochs':77945,'Fig4_and_LT_windows':1342,'signals':['GPS L1','Galileo E1'],
  'QC':{'point_CN0_ge25_dBHz':True,'N_valid_ge30_actual_1Hz_samples':True,'abs_signed_residual_median_le1_dB':True,'failed_window_excluded_in_full':True,'no_extra_native_segment_or_local_CN0_gate':True},
  'window_definition':'Half-open UTC clock minutes within original source/satellite/signal/fit/QC segments,continuous runs(gap>1.5s) and per-epoch centered-dipole latitude bands. Distinct from Fig2 event-window partition.',
  'geometry_difference':'The original complete-angle-geometry condition is retained for Fig4/LT,not added to Fig3 amplitude data.',
  'magnetic_latitude':'Fixed centered-dipole pole80.65N,72.68W; Equatorial abs<20,Mid20<=abs<60,Polar abs>=60; each epoch classified at its own tangent point.',
  'height_km':'WGS84 tangent height50–1000km; original approved reference geometry unchanged.',
  'detrending':'Existing independent E23-method fourth-order polynomial residuals; no new fit or recentering in this component.'},
 'variables':{
  'utc_s':{'unit':'s since1970-01-01 UTC','meaning':'Actual observed epoch; no synthesized observations.'},
  'delta_cn0_db':{'unit':'dB','meaning':'Signed1Hz measurement minus its original qualified quartic background.'},
  'cn0_dbhz':{'unit':'dB-Hz','meaning':'Receiver-reported native carrier-to-noise density ratio.'},
  'N_valid':{'unit':'count','meaning':'Actual point-qualified1Hz samples in the original comparison minute/run/band partition.'},
  'width_db':{'unit':'dB','meaning':'P95-P05 of signed delta within each accepted minute window,not a scintillation index.'},
  'P95_abs_db':{'unit':'dB','meaning':'95th percentile of absolute1Hz delta for the full retained platform/band population,not minute width.'},
  'ray_B_acute_deg':{'unit':'degree','meaning':'Acute ray–IGRF14 magnetic-field angle0–90.'},
  'scan_Bperp_acute_deg':{'unit':'degree','meaning':'Acute geometric transverse scan–projected magnetic-field angle0–90; plasma drift not included.'},
  'magnetic_meridian_acute_deg':{'unit':'degree','meaning':'Acute ray azimuth relative to the local magnetic meridian0–90.'},
  'point_speed_mps':{'unit':'m/s','meaning':'Geometric IPP/tangent-point speed from the existing source geometry.'},
  'scan_speed_mps':{'unit':'m/s','meaning':'Transverse geometric scan speed.'},
  'fresnel_crossing_time_s':{'unit':'s','meaning':'Geometric r1/vscan crossing time,excluding irregularity drift.'},
  'plot_solar_local_time_h':{'unit':'hour','meaning':'Original actual representative epoch solar local time.'},
  'plot_sza_deg':{'unit':'degree','meaning':'Solar zenith angle at Ground350km IPP or LEO/LuGRE tangent point.'}},
 'geometry_ED_population':'Frozen original geometry-only selection and bins remain unchanged; do not interpret this as the reduced Fig3 amplitude-quality cohort.',
 'context_categories':'182 native1Hz contributing sites are distinct from the lower-rate polar GNSS,high-rate scintillation,radar,and track context layers; categories overlap.',
 'legacy_internal_filenames':'R3 in some processed input basenames is a stable interface name. This component metadata,selected output filenames and reference figures identify R4.',
 'entrypoints':['validate_inputs.py','reproduce.py','verify_outputs.py','code/recompute_r4.py'],
 'third_party_rights':'Existing LICENSE_DATA and provider credits apply; no new redistribution permissions are invented.'}
(R/'DATA_DICTIONARY_R4.json').write_text(json.dumps(dictionary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(R/'README.md').write_text('''# R4 Ground182 reproduction component

This component regenerates R4 Fig.3, Fig.4 and the observing-context, geometry-angle,
scan-speed/Fresnel-time and local-time/illumination Extended Data figures. All data
lookups are relative to this directory. After installing the supplied dependencies,
it runs without the original E/H/D drives or a network connection.

```text
python validate_inputs.py
python reproduce.py all
python verify_outputs.py
```

Use `python reproduce.py fig3`, `fig4`, `illumination`, `geometry`, or `context`
for a subset. `python reproduce.py recompute` reconstructs only the numerical R4
LuGRE QC and summaries. `--output-dir PATH` selects a separate output location.
The component-level requirements and package-level offline wheels provide the
software environment. No original-machine dependency directories are imported.

## What is recomputed

`inputs_r4/` contains the original 82,761 processed LuGRE comparison epochs,
1,654 original UTC-minute/run/band candidate windows, and the original solar
geometry/metadata. The supported entrypoint reapplies point C/N0 >=25 dB-Hz,
at least30 valid1Hz samples and absolute signed-residual median <=1 dB. Failed
windows are excluded in full. It reconstructs the retained samples, quantiles,
height envelopes, angle-bin summaries and solar-window selection, then checks
those results against the frozen selected R4 tables before plotting.

Fig.3 uses77,945 actual epochs from1,354 quality windows. Fig.4 and illumination
use1,342 windows, preserving their original additional complete-geometry condition.
Their difference is intentional; unavailable scan-angle geometry does not remove
an otherwise valid amplitude window from Fig.3. The comparison UTC partition
must not be substituted with the different source-origin event windows of Fig.2.

Ground remains182 contributing stations,29,803,114 accepted native1Hz epochs and
498,430 accepted windows. Its selection masks and all LEO processed inputs remain
unchanged. Their existing quartic/rolling-median methods are not replaced.
Geometry-only CDFs and speed/time histograms also keep their established source
population and bins. The context map retains the approved182 station identities
and the separately defined polar/scintillation/radar/observing-track layers.

## Files and computational boundary

- `inputs_r4/`: original processed LuGRE inputs for actual QC reconstruction.
- `code/recompute_r4.py`: portable numerical reconstruction and comparison.
- `fig34/amplitude_source/`, `fig34/response_source/`: selected R4 figure tables
  and unchanged Ground/LEO processed inputs.
- `illumination/`: package-local actual solar geometry and width inputs.
- `geometry/plot_data/`: exact geometry CDF and speed/time histograms.
- `context/`: frozen station/track tables, satellite illustrations and offline
  Natural Earth map resources.
- `reference/`: unchanged copies of the six selected formal R4 PNGs.
- `DATA_DICTIONARY_R4.json`: variables, units, cohorts and counting boundaries.
- `provenance/`: source checksums and original selection evidence. Assembly and
  historical upstream scripts here deliberately retain original paths and are
  not supported portable entrypoints.

The release reproduces these results from processed observations and frozen
near-Earth products. It is not a full raw multi-network acquisition or orbit/
geomagnetic processing rerun. No missing samples are filled, no residual is
refitted/recentered, and no new station or window is admitted during reproduction.
Numeric tables are verified; font-dependent pixel identity is not required.

All new exports are600dpi PNG. Arial is used when installed and DejaVu Sans is
available as a Matplotlib fallback; no proprietary fonts are redistributed.
Legacy `R3` input basenames are stable internal interface names, not a claim
that the selected scientific version is R3. Main outputs are `Fig3_R4.png` and
`Fig4_R4.png`; descriptive Extended Data names remain stable.

Provider credits and licences already supplied with the package remain in force.
This component does not invent additional licences for third-party observations,
mission images or map resources.
''',encoding='utf-8')
print('R4_RUNTIME_AND_DOCUMENTATION_READY')
