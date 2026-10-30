"""Assembly receipt only; not a supported reproduction command on a new computer."""
from pathlib import Path
import hashlib, json, re, shutil

ROOT=Path(__file__).resolve().parents[1]
PIPE=Path('E:/LUGRE/analysis_R3/Ground_full_native_background_QC_R3_20261029/interval_scope_review_20261030/episode_downstream_v1/figure_pipeline')
GEO=Path('E:/LUGRE/figures_R3/Ground182_selected_six_R3_20261030/geometry')
CTX=Path('E:/LUGRE/figures_R3/Observing_context_labels_R3_20261026/LuGRE_190sites_sampling_source')
FINAL=Path('E:/LUGRE/figures_R3/Ground_F_expansion_R3_20261028/observing_context/final182_candidate_30s')
FORMAL=Path('D:/Reference/Paper/2026/Nature_LuGRE/LuGRE_Latex_R3')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[]
def copy(source,target):
    source=Path(source);dest=ROOT/target;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest);assert sha(source)==sha(dest)
    files.append(dict(source=str(source),package_path=target,bytes=dest.stat().st_size,sha256=sha(dest)))
def tree(source,target):
    for p in Path(source).rglob('*'):
        if p.is_file():copy(p,(Path(target)/p.relative_to(source)).as_posix())
def modify(relative,old,new,count=1):
    path=ROOT/relative;text=path.read_text(encoding='utf-8')
    assert text.count(old)==count,(relative,old,text.count(old))
    path.write_text(text.replace(old,new),encoding='utf-8',newline='\n')

# Frozen numerical inputs and all source-derived plotting tables.
tree(PIPE/'fig34/amplitude_source','fig34/amplitude_source')
tree(PIPE/'fig34/response_source','fig34/response_source')
tree(PIPE/'fig34/comparison','provenance/fig34_statistics')
tree(PIPE/'fig34/captions','captions')
for name in ['plot_Fig3_R3.py','plot_Fig4_R3.py','smoothing.py']:
    copy(PIPE/'fig34/plotting'/name,'fig34/plotting/'+name)
for name in ['plot_Fig3_R3.py','plot_Fig4_R3.py']:
    rel='fig34/plotting/'+name
    modify(rel,"sys.path[:0] = [str(p) for name in ('matplotlib_py312','scipy_py312','pyarrow_py312')\n               if (p := Path('E:/LUGRE/python_deps')/name).is_dir()]",'# Dependencies are installed through requirements.txt; no external project paths.')
for name in ['Ground_solar_windows_R3.parquet','Ground_expanded_solar_QA_R3.json']:
    copy(PIPE/'illumination/ground'/name,'illumination/ground/'+name)
for name in ['nearspace_solar_windows_R3.parquet','solar_geometry_R3.py']:
    copy(PIPE/'illumination/nearspace'/name,'illumination/nearspace/'+name)
copy(PIPE/'illumination/plot_LT_illumination_R3.py','illumination/plot_LT_illumination_R3.py')
copy(PIPE/'baseline_source_contract_R3.json','baseline_source_contract_R3.json')
tree(PIPE/'illumination/captions','captions')
modify('illumination/plot_LT_illumination_R3.py',"sys.path[:0]=['E:/LUGRE/python_deps/matplotlib_py312','E:/LUGRE/python_deps/pyarrow_py312',str(ROOT/'nearspace')]","sys.path.insert(0,str(ROOT/'nearspace'))")
modify('illumination/plot_LT_illumination_R3.py',"OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)","OUT=Path(os.environ.get('LUGRE_FIGURE_OUTPUT',str(ROOT.parent/'outputs'))).resolve();OUT.mkdir(parents=True,exist_ok=True)")
tree(GEO/'plot_data','geometry/plot_data')
copy(GEO/'plot_geometry_R3.py','geometry/plot_geometry_R3.py')
modify('geometry/plot_geometry_R3.py',"sys.path[:0] = [str(p) for name in (\"matplotlib_py312\", \"scipy_py312\", \"pyarrow_py312\")\n               if (p := Path(\"E:/LUGRE/python_deps\") / name).is_dir()]",'# Dependencies are installed through requirements.txt; no external project paths.')
modify('geometry/plot_geometry_R3.py','FONT = font_manager.findfont("Arial", fallback_to_default=False)','''try:
    FONT = font_manager.findfont("Arial", fallback_to_default=False)
except ValueError:
    mpl.rcParams["font.family"] = "DejaVu Sans"
    FONT = font_manager.findfont("DejaVu Sans", fallback_to_default=False)''')

# Context: self-contained map assets, tracks, station inventory and Natural Earth cache.
for p in (CTX/'data').rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(CTX/'data')
    if rel.parts[0]=='availability_audit' or rel.name.startswith('ground_1hz_'):continue
    if rel.name in ['asset_contact_sheet.png','MetOp_raw.jpg','Sentinel6_raw.png','Sentinel_raw_satellite.png']:continue
    copy(p,'context/data/'+rel.as_posix())
for name in ['surface_only_scope.py','surface_only_layout.py','reproduce_figure.py']:
    copy(CTX/name,'context/'+name)
for name in ['create_global_plus_polar_zoom_station_map.py','create_global_plus_polar_zoom_station_map_v5_fig3_consistent_time_matched.py','dependency_create_combined_polar_source_maps.py']:
    copy(CTX/'code'/name,'context/code/'+name)
copy('E:/LUGRE/figures_R3/Observing_context_contributing143_R3_20261027/render_contributing143_R3.py','context/render_context.py')
copy(CTX/'results/panel_summary_R3.csv','context/results/panel_summary_R3.csv')
for name in ['Ground_final_contributors_R3.csv','Station_selection_provenance_R3.json','panel_summary.csv','lunar_surface_selection.json','lunar_surface_track_ledger.csv']:
    copy(FINAL/name,'context/frozen/'+name)
modify('context/render_context.py',"ROOT = Path('E:/LUGRE/figures_R3/Observing_context_labels_R3_20261026/LuGRE_190sites_sampling_source')",'ROOT = LOCAL_ROOT')
# Optional live-source branches are redirected to absent local placeholders so caches are mandatory.
path=ROOT/'context/code/create_global_plus_polar_zoom_station_map.py'
text=path.read_text(encoding='utf-8')
start=text.index('SUPERDARN_INVENTORY_CSV = Path(');end=text.index('\n\nOUT_DIR = FIGURE_DIR',start)
text=text[:start]+'''# Offline release: live acquisition paths are intentionally unavailable.
SUPERDARN_INVENTORY_CSV = DATA_DIR / "unavailable_live_superdarn.csv"
ESWUA_ROOT = DATA_DIR / "unavailable_live_eswua"
ESWUA_STATION_CSV = ESWUA_ROOT / "station_manifest.csv"
ESWUA_AVAILABILITY_CSV = ESWUA_ROOT / "availability_manifest.csv"
CHAIN_IPP_META = DATA_DIR / "unavailable_live_chain.js"
LUNA_TRACK_ZIP = DATA_DIR / "All_CTLS_recomputed_J2000_tracks_50_1000km.zip"
LUNA_TRACK_ZIP_MEMBER = "All_CTLS_recomputed_J2000_tracks_50_1000km/data/All_CTLS_recomputed_J2000_tracks_50_1000km_tangent_points.csv"
LUNA_TRACK_CSV = DATA_DIR / "unavailable_live_luna.csv"
GLOBAL_OCCULTATION_SCRIPT = DATA_DIR / "unavailable_live_occultation.py"
'''+text[end:]
start=text.index('GNSS_ROTI_RINEX_DIR_CANDIDATES = [');end=text.index('\n\nPOLAR_LAT_LIMIT',start)
text=text[:start]+'GNSS_ROTI_RINEX_DIR_CANDIDATES = []\nGNSS_ROTI_SUMMARY_CANDIDATES = []'+text[end:]
path.write_text(text,encoding='utf-8',newline='\n')
path=ROOT/'context/code/create_global_plus_polar_zoom_station_map_v5_fig3_consistent_time_matched.py'
text=path.read_text(encoding='utf-8')
text=re.sub(r'^SOURCE_TABLE = .*$', 'SOURCE_TABLE = PACKAGE_ROOT / "data/extended_data_fig7_leoro_panel_c_sources_20260617.csv"',text,flags=re.M)
text=re.sub(r'^METOP_WINDOWS_CSV = .*$', 'METOP_WINDOWS_CSV = PACKAGE_ROOT / "data/unavailable_live_metop_windows.csv"',text,flags=re.M)
path.write_text(text,encoding='utf-8',newline='\n')
path=ROOT/'context/code/dependency_create_combined_polar_source_maps.py'
text=path.read_text(encoding='utf-8')
start=text.index('ROOT = Path(');end=text.index('\n\nPOLAR_LAT_LIMIT',start)
text=text[:start]+'''ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT / "data/unavailable_live_polar_sources"
CHAIN_STATIONS = ROOT / "data/global_plus_polar_eswua_chain_ismr_stations.csv"
COHERENT_RADARS = ROOT / "data/global_plus_polar_current_superdarn_radars.csv"
OUT = ROOT / "unused_combined_station_polar_maps.png"'''+text[end:]
path.write_text(text,encoding='utf-8',newline='\n')
modify('context/surface_only_layout.py',"font_manager.findfont('Arial', fallback_to_default=False)","""try:
        font_manager.findfont('Arial', fallback_to_default=False)
        portable_font = 'Arial'
    except ValueError:
        portable_font = 'DejaVu Sans'""")
modify('context/surface_only_layout.py',"'font.family':'Arial','font.sans-serif':['Arial']","'font.family':portable_font,'font.sans-serif':[portable_font]")
modify('context/surface_only_layout.py',"'mathtext.fontset':'custom','mathtext.rm':'Arial',\n            'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'","'mathtext.fontset':'custom','mathtext.rm':portable_font,\n            'mathtext.it':portable_font+':italic','mathtext.bf':portable_font+':bold'")

# Exact selected release and upstream scientific decisions are immutable provenance.
current=json.loads((FORMAL/'R3_Ground182_current_version.json').read_text(encoding='utf-8'))
for name in ['R3_Ground182_current_version.json','R3_h50_current_version.json']:
    copy(FORMAL/name,'provenance/'+name)
selection=Path(current['selection_manifest']);sel=json.loads(selection.read_text(encoding='utf-8'))
assert sha(selection)==current['selection_sha256']
copy(selection,'provenance/frozen_original_Ground_selection_R3.json')
for item in current['images']:copy(item['formal_png'],'reference/'+item['name'])
for i,p in enumerate(sel['parts']):
    copy(p['selection_mask'],f'provenance/selection_masks/part_{i:04d}.npz')
for category in ['background_QC','interval_QC']:
    for key,value in sel[category].items():
        if isinstance(value,str) and Path(value).is_file() and Path(value).suffix.lower() in ['.json','.parquet','.csv']:
            if Path(value).stat().st_size<20_000_000:copy(value,'provenance/scientific_QC/'+Path(value).name)
for sub in ['fig34','illumination']:
    builder=PIPE/sub/('build_fig34_R3.py' if sub=='fig34' else 'build_solar_R3.py')
    copy(builder,'provenance/upstream_source_archive/'+builder.name)
copy(GEO/'build_geometry_shared_selection_R3.py','provenance/upstream_source_archive/build_geometry_shared_selection_R3.py')
copy(PIPE/'THREE_FIGURE_NUMERIC_VERIFICATION_R3.json','provenance/THREE_FIGURE_NUMERIC_VERIFICATION_R3.json')
(ROOT/'provenance/assembly_source_map.json').write_text(json.dumps({'source_selection_sha256':current['selection_sha256'],'files':files},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files),'root':str(ROOT)}))
