"""One-time packaging recipe; historical machine paths are provenance only."""
from pathlib import Path
import ast, shutil, json, hashlib, re

ROOT=Path(__file__).resolve().parents[1]
PAPER=Path('D:/Reference/Paper/2026/Nature_LuGRE')
H50=Path('E:/LUGRE/figures_R3/LuGRE_E23_h50_R3_20261028')
FORMAL=PAPER/'LuGRE_Latex_R3'
for d in ['code','data/fig2','data/e21','data/space_weather','data/antenna','assets/cartopy','assets/antenna','reference','outputs','provenance/original_code','source_artwork']:
    (ROOT/d).mkdir(parents=True,exist_ok=True)
ledger=[]
def copy(src,dest):
    src=Path(src);dest=ROOT/dest;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dest)
    ledger.append({'source':str(src),'package':str(dest.relative_to(ROOT)).replace('\\','/'),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    return dest
def write(path,text):
    dest=ROOT/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text,encoding='utf-8',newline='\n')
def extracted(path,functions=(),constants=()):
    source=Path(path).read_text(encoding='utf-8-sig');tree=ast.parse(source)
    selected=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in functions:selected.append(ast.get_source_segment(source,n))
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets):selected.append(ast.get_source_segment(source,n))
    return '\n\n'.join(selected)+'\n'

# Immutable currently selected artwork references.
for name in ['Fig1_R3.png','Fig2_R3.png','Extended_Data_CN0_detrending_OP38_E21.png','Extended_Data_Space_weather_index.png','Extended_Data_GNSS_antenna_geometry.png']:
    copy(FORMAL/name,'reference/'+name)
copy('E:/LUGRE/figures_R3/Fig1_R3_export_20261013/Fig1_R3_source_snapshot.pptx','source_artwork/Fig1_R3.pptx')
copy('E:/LUGRE/figures_R3/Fig1_R3_export_20261013/Fig1_R3_final_png_QA.json','provenance/Fig1_original_export_QA.json')
source=Path('E:/LUGRE/figures_R3/Fig1_R3_export_20261013/export_Fig1_R3.ps1').read_text(encoding='utf-8-sig')
copy('E:/LUGRE/figures_R3/Fig1_R3_export_20261013/export_Fig1_R3.ps1','provenance/original_code/export_Fig1_R3.ps1')
source=source.replace("$figSource = 'D:/Reference/Paper/2026/Nature_LuGRE/Fig1_R3.pptx'", "$figRoot = Split-Path -Parent $PSScriptRoot\n$figSource = Join-Path $figRoot 'source_artwork/Fig1_R3.pptx'")
source=source.replace("$figProject = (Resolve-Path -LiteralPath 'H:/LUGRE').Path\n$figBuild = Join-Path $figProject 'figures_R3/Fig1_R3_export_20261013'", "$figBuild = Join-Path $figRoot 'outputs/fig1'\nNew-Item -ItemType Directory -Path $figBuild -Force | Out-Null")
write('code/export_fig1.ps1',source)

# Figure 2: current full renderer, exact frozen h50 plotting tables and styles.
plot_tables={'all_60s_windows.csv','spatial_amplitude_cells.csv','spatial_roti_cells.csv',
    'fig3_panel_a_psd_no_time_interpolation.csv','panel_c_paired_frequency_segments.csv',
    'paired_geometry.csv','fig3_panel_b_event_level_points.csv','fig3_panel_d_altitude_layer_heatmap.csv',
    'OP76_G01_measured_samples.csv','OP38_E21_measured_samples.csv','paired_exact60_blocks.csv',
    'psd_per_run_frequency_bins.csv','psd_excluded_runs.csv','psd_retained_runs.csv'}
for p in (H50/'fig2_adapter/derived_source_data').glob('*.csv'):
    if p.name in plot_tables:copy(p,'data/fig2/'+p.name)
copy(H50/'data/updated_points.csv','data/fig2/updated_points.csv')
summary=PAPER/'Figure/main_figures_lugre_reference_update_20260622/Fig1_replace_1_OP76G01_replace_6_OP38E25_new_reference_ecef_summary.csv'
trackqa=PAPER/'Figure/Fig2_polar_track_panels_wgs84_update_20260702/data/fig2_selected_tracks_wgs84_height_latlon_QA.csv'
copy(summary,'data/fig2/selected_track_summary.csv');copy(trackqa,'data/fig2/selected_track_geometry_QA.csv')
base=PAPER/'Program/plot_lugre_ionosphere_tangent_tracks_by_phase.py'
polar=PAPER/'Program/plot_lugre_polar_abs_nonuniform_polarcoords.py'
track=PAPER/'Figure/main_text_panel_png_exports_20260630/redraw_fig1_source_panels_no_labels.py'
write('code/polar_primitives.py','from __future__ import annotations\n'+extracted(base,constants={'TRACK_KEYS','CN0_RESID_COL','CN0_ABS_RESID_COL'})+extracted(polar,functions={'wrap_lon','iter_lines','polar_radius','setup_axis','kept_runs','split_geo_piece','draw_land','draw_coastlines','split_track','add_tracks'},constants={'POLAR_LAT_DEG','MAP_LAT_MIN','CMAX_DBHZ','BOUNDARIES'}))
write('code/selected_track_primitives.py','from __future__ import annotations\n'+extracted(track,functions={'split_track_hemi','draw_selected_track_on_polar','add_native_label'}))
for src in [base,polar,track,H50/'fig2_adapter/rendered_Fig2_source.py',H50/'fig2_adapter/rendered_polar_helper_source.py']:
    copy(src,'provenance/original_code/'+src.name)
helper=(H50/'fig2_adapter/rendered_polar_helper_source.py').read_text(encoding='utf-8-sig')
a=helper.index("PROJECT = Path(");b=helper.index('SOURCE_FILES =',a)
helper=helper[:a]+'''COMPONENT = Path(__file__).resolve().parents[1]
PROGRAM = COMPONENT/'code'
POLAR_SCRIPT = PROGRAM/'polar_primitives.py'
POLAR_BASE_SCRIPT = POLAR_SCRIPT
TRACK_SCRIPT = PROGRAM/'selected_track_primitives.py'
POINTS = COMPONENT/'data/fig2/updated_points.csv'
SUMMARY = COMPONENT/'data/fig2/selected_track_summary.csv'
TRACK_QA = COMPONENT/'data/fig2/selected_track_geometry_QA.csv'
'''+helper[b:]
write('code/polar_helper.py',helper)
source=(H50/'fig2_adapter/rendered_Fig2_source.py').read_text(encoding='utf-8-sig')
a=source.index('ROOT = Path(');b=source.index("os.environ['MPLCONFIGDIR']",a)
source=source[:a]+'''COMPONENT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'fig2'
ROOT.mkdir(parents=True,exist_ok=True)
PROJECT = COMPONENT
DATA = COMPONENT/'data/fig2'
'''+source[b:]
source=source.replace("cartopy.config['data_dir'] = Path('C:/Users/YR/.local/share/cartopy')", "cartopy.config['data_dir'] = COMPONENT/'assets/cartopy'\ncartopy.config['pre_existing_data_dir'] = COMPONENT/'assets/cartopy'")
source=source.replace("LOCAL=Path('C:/Users/YR/Documents/Nature_LuGRE/outputs')","LOCAL=DATA")
a=source.index('paths={');b=source.index('\ndef sha(',a)
names={'amplitude':'spatial_amplitude_cells.csv','roti':'spatial_roti_cells.csv','psd':'fig3_panel_a_psd_no_time_interpolation.csv','paired':'panel_c_paired_frequency_segments.csv','paired_geometry':'paired_geometry.csv','events':'fig3_panel_b_event_level_points.csv','height':'fig3_panel_d_altitude_layer_heatmap.csv','north':'OP76_G01_measured_samples.csv','south':'OP38_E21_measured_samples.csv'}
source=source[:a]+'paths={'+','.join(repr(k)+':DATA/'+repr(v) for k,v in names.items())+'}\n'+source[b:]
source=source.replace("'source_values_recomputed':True","'source_values_recomputed':False,'frozen_current_source_tables':True")
source=source.replace('All eligible amplitude statistics recomputed; raw C/N0/ROTI unchanged; frozen previous QC and cohorts.','Redraw current h50 E23-method source tables; scientific statistics were computed in the original processing recorded under provenance.')
write('code/render_fig2.py',source)
for scale in ['110m','50m']:
    for feature in ['land','coastline']:
        source_dir=Path.home()/'.local/share/cartopy/shapefiles/natural_earth/physical'
        found=list(source_dir.glob(f'ne_{scale}_{feature}.*'))
        if not found:raise RuntimeError(f'Missing Natural Earth offline cache: {scale} {feature}')
        for p in found:copy(p,'assets/cartopy/shapefiles/natural_earth/physical/'+p.name)

# E21: exact archived E23 robust quartic; independently reconstructed geometry.
for p in (H50/'E21_example/data').glob('*.csv'):copy(p,'data/e21/'+p.name)
copy(H50/'E21_example/qa/E21_independent_geometry_QA.json','data/e21/E21_independent_geometry_QA.json')
fit=PAPER/'Figure/op38_e23_front_arc_example_package_20260616_145440/code/GPS_simulator/Build_QuadraticDetrendExample.py'
template=PAPER/'Figure/extended_data_fig4_wgs84_tangent_height_update_20260701/code/build_op38_e23_nature_style_extract_wgs84_candidate_plot.py'
style=PAPER/'Figure/op40_e02_package_20260615_whole_arc_cheb4/code/Build_OP40E02Extract.py'
write('code/e21_fit.py','from __future__ import annotations\n'+extracted(fit,functions={'median_abs_deviation','binned_median_fit_points','fit_poly_trend'}))
write('code/e21_style.py','from __future__ import annotations\n'+extracted(template,functions={'set_panel_header','time_segments','map_extent','plot_time_pieces_local','geometry_ionosphere_spans','merge_time_spans'})+extracted(style,functions={'style_nature_axis','format_lon','format_lat','draw_nature_map_base'}))
copy(PAPER/'Figure/op38_e23_front_arc_example_package_20260616_145440/data/op38_e23_anomaly_focused_cn0_data.csv','data/e21/E23_archived_validation.csv')
copy(H50/'E21_example/code/archived_check_and_build_E21_whole_arc.py','provenance/original_code/archived_check_and_build_E21_whole_arc.py')
source=(H50/'E21_example/code/build_E21_whole_arc_h50.py').read_text(encoding='utf-8-sig')
copy(H50/'E21_example/code/build_E21_whole_arc_h50.py','provenance/original_code/build_E21_whole_arc_h50.py')
source=source.replace("ROOT = Path(__file__).resolve().parents[1]", "COMPONENT = Path(__file__).resolve().parents[1]\nROOT = Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'e21'\nROOT.mkdir(parents=True,exist_ok=True)")
source=re.sub(r"sys.path.insert\(0, 'D:/[^\n]+\n",'',source)
a=source.index("PAPER = Path(");b=source.index('\ndef extract(',a)
source=source[:a]+'''E23_PACKAGE = COMPONENT/'data/e21'
FIT = COMPONENT/'code/e21_fit.py'
TEMPLATE = COMPONENT/'code/e21_style.py'
HELPER = TEMPLATE
SOURCE = COMPONENT/'data/e21/E21_plotted_observations_R3.csv'
'''+source[b:]
source=source.replace("E23_PACKAGE / 'data/op38_e23_anomaly_focused_cn0_data.csv'","E23_PACKAGE/'E23_archived_validation.csv'")
source=source.replace("ROOT/'data/E21_independent_WGS84_geometry.csv'","COMPONENT/'data/e21/E21_independent_WGS84_geometry.csv'")
source=source.replace("ROOT/'qa/E21_independent_geometry_QA.json'","COMPONENT/'data/e21/E21_independent_geometry_QA.json'")
source=source.replace("ROOT/'code/archived_check_and_build_E21_whole_arc.py'","COMPONENT/'provenance/original_code/archived_check_and_build_E21_whole_arc.py'")
source=source.replace("'status': 'h50_user_requested_correction_not_yet_promoted'","'status': 'current_selected_R3_h50_reproduction'")
write('code/render_e21.py',source)

# Space weather: preserve current drawing functions and the saved plotting grid.
for name in ['unchanged_space_weather_context.csv','op_mean_all_latitude_and_polar_h50.csv','all_eligible_60s_windows_h50.csv']:
    copy(H50/'Space_weather_sync'/name,'data/space_weather/'+name)
source=(H50/'Space_weather_sync/rendered_source.py').read_text(encoding='utf-8-sig')
copy(H50/'Space_weather_sync/rendered_source.py','provenance/original_code/space_weather_rendered_source.py')
a=source.index('BASE = Path(');b=source.index('OUT_STEM =',a)
source=source[:a]+'''COMPONENT = Path(__file__).resolve().parents[1]
DATA = COMPONENT/'data/space_weather'
RENDER_SCALE = 3
'''+source[b:]
a=source.index('def font(');b=source.index('\ndef scaled_font(',a)
source=source[:a]+'''def font(size: int, bold: bool = False):
    from matplotlib.font_manager import FontProperties, findfont
    name = findfont(FontProperties(family=['Arial','DejaVu Sans'],weight='bold' if bold else 'normal'))
    return ImageFont.truetype(name,size=size)
'''+source[b:]
a=source.index('def load_context(');b=source.index('\ndef clean_series(',a)
source=source[:a]+'''def load_context():
    return pd.read_csv(DATA/'unchanged_space_weather_context.csv',index_col=0,parse_dates=True)

def load_op_cn0_stats():
    windows=pd.read_csv(DATA/'all_eligible_60s_windows_h50.csv')
    original=pd.read_csv(DATA/'op_mean_all_latitude_and_polar_h50.csv',parse_dates=['start','end','center'])
    all_stats=windows.groupby('op_id',as_index=False).agg(all_windows=('signed_p95_p05_width_db','size'),all_mean_width_db=('signed_p95_p05_width_db','mean'))
    polar=windows[windows.lat_group.eq('Polar')].groupby('op_id',as_index=False).agg(polar_windows=('signed_p95_p05_width_db','size'),polar_mean_width_db=('signed_p95_p05_width_db','mean'))
    computed=all_stats.merge(polar,on='op_id',how='left')
    computed['polar_windows']=computed.polar_windows.fillna(0).astype(int)
    result=original[['op_id','start','end','center','display_label']].merge(computed,on='op_id',validate='one_to_one').sort_values('center')
    for col in ['all_mean_width_db','polar_mean_width_db']:
        test=result[['op_id',col]].merge(original[['op_id',col]],on='op_id')
        np.testing.assert_allclose(test[col+'_x'],test[col+'_y'],atol=1e-12,rtol=0,equal_nan=True)
    assert len(windows)==2244 and int(windows.is_event.sum())==266
    return result
'''+source[b:]
source=source.replace('def draw_figure() -> Path:', 'def draw_figure():')
a=source.index('if __name__ == "__main__":')
source=source[:a]+'''if __name__ == '__main__':
    import os
    out=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'space_weather'
    out.mkdir(parents=True,exist_ok=True)
    PANELS[-1]=Panel('lugre','f','Mean\\nfluctuation\\nwidth (dB)',0,3,(0,1,2,3))
    image=draw_figure().resize((4320,4344),Image.Resampling.LANCZOS)
    image.save(out/'Extended_Data_Space_weather_index.png',dpi=(600,600))
    load_op_cn0_stats().to_csv(out/'recomputed_op_means.csv',index=False)
    print(out/'Extended_Data_Space_weather_index.png')
'''
write('code/render_space_weather.py',source)

# Antenna data and original numerical processing. Left-panel artwork is manual.
antenna=PAPER/'Nature_LuGRE_Latex/figure_b_c_reproducible_package_20260727'
for name in ['lugre_gnss_main_lobe_point_check.csv','lugre_full_geometry_noise_windows_refined.csv']:
    copy(antenna/name,'data/antenna/'+name)
for folder in ['GPS_antenna_patterns','GRAP_metadata','derived_data']:
    for p in (antenna/folder).rglob('*'):
        if p.is_file():copy(p,'data/antenna/'+str(p.relative_to(antenna)).replace('\\','/'))
for name in ['plot_main_lobe_gps_galileo.py','plot_main_lobe_grap_revised.py','plot_main_lobe_previous_ab_gps_galileo.py','plot_main_lobe_nature_figures.py']:
    source=(antenna/name).read_text(encoding='utf-8-sig')
    copy(antenna/name,'provenance/original_code/'+name)
    source=source.replace('ROOT = Path(__file__).resolve().parent',"ROOT = Path(__file__).resolve().parents[1]/'data/antenna'")
    source=re.sub(r'USER_EARTH_IMAGE = .*',"USER_EARTH_IMAGE = ROOT/'earth_user_reference_globe.png'",source)
    write('code/'+name,source)
for name in ['earth_user_reference_globe.png','gps_iii_satellite_cutout.png']:
    copy(antenna/name,'data/antenna/'+name)

write('provenance/copied_input_manifest.json',json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'files':len(ledger),'copied_bytes':sum(x['bytes'] for x in ledger),'root':str(ROOT)}))
