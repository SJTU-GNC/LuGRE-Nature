"""Assembly provenance only: run on the original machine, never a portable entrypoint."""
from pathlib import Path
import ast,csv,json,hashlib,shutil,sys
sys.path.insert(0,'E:/LUGRE/python_deps/pyarrow_py312')
import pandas as pd
R=Path('E:/LUGRE/Zenodo_R4_release_20261113/components/ground182')
assert R.is_dir() and R.resolve()!=Path('E:/LUGRE/Zenodo_R3_Ground182_20261030/components/ground182').resolve()
S=Path('E:/LUGRE/figures_R4/Fig34_LT_three_condition_QC_20261111')
H=Path('E:/LUGRE/figures_R3/LuGRE_E23_h50_R3_20261028/derived_figures')
F=Path('E:/LUGRE/LuGRE_Latex_R4')
I=R/'inputs_r4';C=R/'code';I.mkdir(exist_ok=True);C.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
origins=[]
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 origins.append({'package_path':str(dst.relative_to(R)).replace('\\','/'),'sha256':sha(dst),'original_source':str(src),'purpose':'Numerical input or selected R4 plotting data; original source is provenance, never a runtime dependency.'})
# Small original LuGRE inputs permit a genuine R4 window-QC reconstruction.
cp(H/'data/LuGRE_h50_epochs_window_keys_R3.parquet',I/'LuGRE_comparison_epochs_before_QC.parquet')
cp(H/'response_source/LuGRE_h50_window_eligibility_R3.parquet',I/'LuGRE_comparison_candidate_windows_before_QC.parquet')
cp(R/'illumination/nearspace/nearspace_solar_windows_R3.parquet',I/'nearspace_solar_windows_before_QC.parquet')
baseline=R/'fig34/response_source/common_60s_response_windows_R3.csv'
with baseline.open('rb') as src,(I/'LuGRE_original_response_rows.csv').open('wb') as dst:
 header=src.readline();dst.write(header);n=0
 for line in src:
  if line.startswith(b'LuGRE,'):dst.write(line);n+=1
assert n==1352
origins.append({'package_path':'inputs_r4/LuGRE_original_response_rows.csv','sha256':sha(I/'LuGRE_original_response_rows.csv'),'original_source':'Previous self-contained ground182 common response table; LuGRE rows only','purpose':'Preserve pre-QC geometry and metadata while independently recomputing the R4 acceptance mask and W.'})
# Promote the independently verified R4 numerical selection without altering Ground/LEO.
for p in (S/'amplitude_source').iterdir():
 if p.is_file():cp(p,R/'fig34/amplitude_source'/p.name)
for name in ['common_60s_response_windows_R3.csv','angle_bin_median_IQR_counts_R3.csv','response_window_manifest_R3.json']:
 cp(S/'response_source'/name,R/'fig34/response_source'/name)
cp(S/'illumination/nearspace_solar_windows_R3.parquet',R/'illumination/nearspace/nearspace_solar_windows_R3.parquet')
cp(S/'baseline_source_contract_R3.json',R/'baseline_source_contract_R3.json')
for name in ['Fig34_LT_QC_data_R4.json','independent_Fig34_LT_QC_R4.json','independent_science_summary_R4.json']:
 cp(S/'qa'/name,R/'provenance/R4_QC_validation'/name)
# Copy only the two unchanged numerical helper functions, with source attribution.
source=Path('E:/LUGRE/imports/Ground_strictQC_20261027/strict_upper/source/amplitude_response/portable.py')
t=source.read_text('utf-8-sig');tree=ast.parse(t)
functions=[ast.get_source_segment(t,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_amplitude_tables','build_angle_bins']]
assert len(functions)==2
(C/'numerical_helpers.py').write_text('"""Unchanged numerical helpers from the accepted amplitude-response implementation."""\nimport numpy as np\nimport pandas as pd\nBANDS=["Equatorial","Mid-latitude","Polar"]\nPLATFORMS=["Ground","LEO-RO","LuGRE"]\nANGLES=["ray_B_acute_deg","scan_Bperp_acute_deg","magnetic_meridian_acute_deg"]\n\n'+'\n\n'.join(functions)+'\n',encoding='utf-8')
origins.append({'package_path':'code/numerical_helpers.py','sha256':sha(C/'numerical_helpers.py'),'original_source_sha256':sha(source),'purpose':'Exact _amplitude_tables and build_angle_bins algorithms, with no project-path imports.'})
# Approved reference PNGs are untouched copies, distinct from regenerated figures.
for name in ['Fig3_R4.png','Fig4_R4.png','Extended_Data_Local_time_illumination.png','Extended_Data_Geometry_angles.png','Extended_Data_Scan_speed_Fresnel_time.png','Extended_Data_Observing_context.png']:
 cp(F/name,R/'reference'/name)
# Remove only stale main-figure references from the new independent release copy.
for name in ['Fig3_R3.png','Fig4_R3.png']:
 p=R/'reference'/name
 if p.is_file():p.unlink()
# Retain supported portable renderers and apply the reviewed R4 local layout changes.
p=R/'geometry/plot_geometry_R3.py';s=p.read_text('utf-8')
s=s.replace('"Ray azimuth to magnetic meridian,\\n" + r"$\\theta_{\\rm mer}$ (°)"','"Ray azimuth to magnetic meridian, " + r"$\\theta_{\\rm mer}$ (°)"')
s=s.replace('SPEED_HEIGHT_MM = 210.','ORIGINAL_MAIN_HEIGHT_MM=210.*(.92-.085)/(3.+2.*.62)\nSPEED_HEIGHT_MM=210.-2.*(.62-.50)*ORIGINAL_MAIN_HEIGHT_MM')
s=s.replace('fig.subplots_adjust(left=.09, right=.985, bottom=.085, top=.92, wspace=.37, hspace=.62)','fig.subplots_adjust(left=.09,right=.985,bottom=(210.*.085)/height_mm,top=1.-(210.*.08)/height_mm,wspace=.37,hspace=.50)')
s=s.replace('rect=(.53,.66,.42,.28) if r<2 else (.25,.66,.42,.28)','rect=(.53,.648,.42,.28) if r<2 else (.25,.648,.42,.28)')
s=s.replace('inset.set_title("Ground",fontsize=6.2,pad=1.7)','inset.set_title("Ground",fontsize=6.2,pad=.7)')
s=s.replace('bbox_to_anchor=(.545,.988)','bbox_to_anchor=(.545,1.-210.*.012/SPEED_HEIGHT_MM)')
p.write_text(s,encoding='utf-8')
p=R/'context/surface_only_layout.py';s=p.read_text('utf-8')
s=s.replace("'TSX':(1.63,665)","'TSX':(1.63,645)")
s=s.replace("text.set_position((0.018,0.5))\n                    text.set_ha('left')","text.set_position((-.02,.5))\n                    text.set_ha('right')\n                    text.set_clip_on(False)")
old='        for ax in (d,e,f):\n'
new="""        for latitude in (60,70,80):
            label_n=next(t for t in a.texts if t.get_text()==rf'${latitude}^\\circ$N')
            label_s=next(t for t in b.texts if t.get_text()==rf'${latitude}^\\circ$S')
            label_s.set_position((-160.,-float(latitude)))
            label_s.set_ha(label_n.get_ha());label_s.set_va(label_n.get_va())
        for ax in (d,e,f):
"""
assert old in s;s=s.replace(old,new,1);p.write_text(s,encoding='utf-8')
# Preserve the current figure design and change the public main-figure names only.
for fig in [3,4]:
 p=R/f'fig34/plotting/plot_Fig{fig}_R3.py';s=p.read_text('utf-8');s=s.replace(f"'Fig{fig}_R3.png'",f"'Fig{fig}_R4.png'");p.write_text(s,encoding='utf-8')
# The solar module uses already computed local tables and never fetches trajectories.
p=R/'illumination/plot_LT_illumination_R3.py';s=p.read_text('utf-8')
s=s.replace("ROOT/'nearspace/nearspace_solar_windows_R3.parquet'","Path(os.environ.get('LUGRE_NEARSPACE_SOLAR',str(ROOT/'nearspace/nearspace_solar_windows_R3.parquet')))",1)
p.write_text(s,encoding='utf-8')
# R4 metadata retains stable internal R3 filenames solely for interface compatibility.
meta=json.loads((R/'component.json').read_text('utf-8'))
meta.update(revision='R4',description='Six selected R4 figures; frozen Ground182/LEO inputs with independently reconstructed LuGRE three-condition window QC',LuGRE_Fig3_epochs=77945,LuGRE_quality_windows=1354,LuGRE_Fig4_LT_windows=1342,LEO_windows=1161,
 r4_numeric_reconstruction='code/recompute_r4.py',data_dictionary='DATA_DICTIONARY_R4.json',
 runtime_data_boundary='Processed original LuGRE epochs, candidate windows and original solar geometry; frozen Ground182/LEO processed inputs; no raw multi-network reprocessing.')
for task in meta['tasks']:
 task['output_names']=[n.replace('Fig3_R3.png','Fig3_R4.png').replace('Fig4_R3.png','Fig4_R4.png') for n in task['output_names']]
 if task['id'] in ['fig3','fig4','illumination']:task['reproduction_class']='recompute_R4_LuGRE_QC_and_plot_with_frozen_Ground_LEO'
meta['archival_python_not_supported_entrypoints'].append('provenance/assemble_r4_component.py')
(R/'component.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
(R/'provenance/R4_input_origin_manifest.json').write_text(json.dumps({'revision':'R4','entries':origins,'runtime_paths_are_component_relative':True,'original_absolute_sources_are_provenance_only':True},indent=2)+'\n',encoding='utf-8')
print('ASSEMBLY_DATA_AND_LOCAL_STYLE_READY',R)
