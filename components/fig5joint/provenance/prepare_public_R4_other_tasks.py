from pathlib import Path
import ast, hashlib, json, shutil, sys
sys.dont_write_bytecode=True
RELEASE=Path('E:/LUGRE/Zenodo_R4_release_20261113').resolve()
OLD=Path('E:/LUGRE/Zenodo_R3_Ground182_20261030').resolve()
FORMAL=Path('E:/LUGRE/LuGRE_Latex_R4').resolve()
M=RELEASE/'components/main12'; F=RELEASE/'components/fig5joint'
assert RELEASE.is_dir() and RELEASE!=OLD and FORMAL.is_dir()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,text):
 p=Path(p); assert p.resolve().is_relative_to(RELEASE);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
def cp(src,dst):
 dst=Path(dst); assert dst.resolve().is_relative_to(RELEASE);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(src)==sha(dst)
def dump(p,x):put(p,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
# Preserve authored figures; their R4 pixels are unchanged from the frozen R3 selection.
for current,dest in [('Fig1_R4.png',M/'reference/Fig1_R4.png'),('Fig5_R4.png',F/'reference/Fig5_R4.png'),('Extended_Data_Joint_observables.png',F/'reference/Extended_Data_Joint_observables_R4.png'),('Extended_Data_Space_weather_index.png',M/'reference/Extended_Data_Space_weather_index_R4_reference.png'),('Extended_Data_GNSS_antenna_geometry.png',M/'reference/Extended_Data_GNSS_antenna_geometry_R4_reference.png')]:cp(FORMAL/current,dest)
cp(OLD/'components/main12/source_artwork/Fig1_R3.pptx',M/'source_artwork/Fig1_R4.pptx')
fig1=(OLD/'components/main12/code/compute_fig1_fresnel.py').read_text().replace('source_artwork/Fig1_R3.pptx','source_artwork/Fig1_R4.pptx')
put(M/'code/compute_fig1_fresnel_R4.py',fig1)
# Current accepted R4 space-weather layout, with frozen independent context cohort.
base_source=Path('E:/LUGRE/figures_R4/Extended_Data_style_unification_20261110/ed2_8/render_ED2_ED8_style_R4.py')
base=base_source.read_text(encoding='utf-8-sig').split('\ndef drawing_functions():')[0]
base=base.replace("sys.path[:0]=['E:/LUGRE/python_deps/matplotlib_py312','E:/LUGRE/python_deps/pyarrow_py312']",'')
base=base.replace('ROOT=Path(__file__).resolve().parent',"COMPONENT=Path(__file__).resolve().parents[1]\nROOT=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'space_weather'\nROOT.mkdir(parents=True,exist_ok=True)")
base=base.replace("FROZEN=Path('E:/LUGRE/figures_R3/LuGRE_E23_h50_R3_20261028')\nSPACE=FROZEN/'Space_weather_sync'\nJOINT=FROZEN/'derived_figures/joint_h50'\nFORMAL=Path('D:/Reference/Paper/2026/Nature_LuGRE/LuGRE_Latex_R4')", "SPACE=COMPONENT/'data/space_weather'\nFORMAL=COMPONENT/'reference'")
base=base.replace("SPACE/'rendered_source.py'", "COMPONENT/'code/space_weather_native_R4.py'")
base=base.replace("FORMAL/'Extended_Data_Space_weather_index.png'", "FORMAL/'Extended_Data_Space_weather_index_R4_reference.png'")
put(M/'code/space_weather_native_R4.py',base)
review_source=Path('E:/LUGRE/figures_R4/Extended_Data_ED2_review_20261110/build_ED2_review_R4.py')
s=review_source.read_text(encoding='utf-8-sig')
a=s.index("sys.path[:0]");b=s.index('from matplotlib.lines import Line2D')
s=s[:a]+"COMPONENT=Path(__file__).resolve().parents[1]\nROOT=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'space_weather'\nROOT.mkdir(parents=True,exist_ok=True)\nRECIPE=COMPONENT/'code/space_weather_native_R4.py'\nspec=importlib.util.spec_from_file_location('frozen_ED2_recipe',RECIPE)\nbase=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)\nbase.ROOT=ROOT;base.OUT=ROOT\n"+s[b:]
put(M/'code/render_space_weather_R4.py',s)
# Copy only the accepted antenna recipe's finite dependency closure.
A=M/'data/antenna/R4_plot_assets'; A.mkdir(parents=True,exist_ok=True)
S=Path('E:/LUGRE/figures_R3/Extended_Data_revision_20261109/antenna')
P=Path('E:/LUGRE/figures_R3/Extended_Data_ED3_balance_columns_20261110')
PAT=Path('E:/LUGRE/figures_R3/Extended_Data_ED3_full_pattern_range_flat_20261110/pattern_sources')
files=[(P/'A_clean.png','A_clean.png'),(Path('E:/LUGRE/figures_R3/Extended_Data_ED3_font_only_20261110/density_display_source_R3.csv'),'density_display_source_R3.csv'),(S/'selected_lunar_offboresight_summary_R3.csv','selected_lunar_offboresight_summary_R3.csv'),(S/'selected_lunar_offboresight_points_R3.csv','selected_lunar_offboresight_points_R3.csv'),(S/'reference_pattern_envelopes_R3.csv','reference_pattern_envelopes_R3.csv'),(S/'reference_first_valley_R3.csv','reference_first_valley_R3.csv'),(PAT/'reference_full_range_summary_R3.csv','reference_full_range_summary_R3.csv'),(PAT/'original_normalized_cuts_R3.csv.gz','original_normalized_cuts_R3.csv.gz'),(PAT/'original_cut_catalog_R3.csv','original_cut_catalog_R3.csv'),(PAT/'full_pattern_range_QA_R3.json','full_pattern_range_QA_R3.json'),(Path('E:/LUGRE/figures_R3/Extended_Data_revision_20261109/manuscript_before/Extended_Data_GNSS_antenna_geometry.png'),'original_schematic_source.png'),(OLD/'components/main12/data/antenna/gps_iii_satellite_cutout.png','gps_iii_satellite_cutout.png')]
for src,name in files:cp(src,A/name)
ant=(P/'build_ED3_balance_columns.py').read_text(encoding='utf-8-sig')
ant=ant.replace('R=Path(__file__).resolve().parent',"COMPONENT=Path(__file__).resolve().parents[1]\nASSETS=COMPONENT/'data/antenna/R4_plot_assets'\nR=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'antenna'\nR.mkdir(parents=True,exist_ok=True)")
a=ant.index("S=Path('E:");b=ant.index("SIGNALS=",a)
ant=ant[:a]+"S=ASSETS\nPREV=ASSETS\nORIG=ASSETS/'original_schematic_source.png'\nICON=ASSETS/'gps_iii_satellite_cutout.png'\n"+ant[b:]
ant=ant.replace("PATTERN_ROOT=Path('E:/LUGRE/figures_R3/Extended_Data_ED3_full_pattern_range_flat_20261110/pattern_sources')",'PATTERN_ROOT=ASSETS')
ant=ant.replace("R/'A_clean.png'","ASSETS/'A_clean.png'")
a=ant.index("D=Path('D:");b=ant.index('density=pd.read_csv',a)
ant=ant[:a]+ant[b:]
ant=ant.replace("for name,h in protected.items():assert sha(D/name)==h,('Active manuscript changed',name)",'')
ant=ant.replace("(R/'FIGURE_QA_R3.json')","(R/'antenna_QA_R4.json')")
put(M/'code/render_antenna_R4.py',ant)
raw=(PAT/'prepare_full_pattern_range.py').read_text(encoding='utf-8-sig')
raw=raw.replace('ROOT = Path(__file__).resolve().parent',"COMPONENT = Path(__file__).resolve().parents[1]\nROOT=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'antenna_raw_check'\nROOT.mkdir(parents=True,exist_ok=True)")
raw=raw.replace("PACKAGE = Path('D:/Reference/Paper/2026/Nature_LuGRE/Nature_LuGRE_Latex/figure_b_c_reproducible_package_20260727')", "PACKAGE=COMPONENT/'data/antenna'")
raw=raw.replace("ARCHIVE = Path('E:/LUGRE/figures_R3/Extended_Data_revision_20261109/antenna')", "ARCHIVE=PACKAGE/'R4_plot_assets'")
raw=raw.replace('sys.path.insert(0,str(PACKAGE))',"sys.path.insert(0,str(COMPONENT/'code'))")
raw=raw.replace("PACKAGE/'provenance/source_catalog.csv'","COMPONENT/'provenance/antenna_source_catalog.csv'")
raw=raw.replace("PACKAGE/'plot_main_lobe_gps_galileo.py'","COMPONENT/'code/plot_main_lobe_gps_galileo.py'")
raw=raw.replace("PACKAGE/'plot_main_lobe_previous_ab_gps_galileo.py'","COMPONENT/'code/plot_main_lobe_previous_ab_gps_galileo.py'")
raw=raw.replace("PACKAGE/'plot_main_lobe_grap_revised.py'","COMPONENT/'code/plot_main_lobe_grap_revised.py'")
put(M/'code/prepare_full_pattern_range_R4.py',raw)
# Runtime deliberately ignores restricted files unless the caller explicitly opts in.
rt=(OLD/'components/fig5joint/code/runtime_fig5.py').read_text(encoding='utf-8-sig')
rt=rt.replace('def render(output):','def render(output, *, include_restricted=False, restricted_dir=None):')
needle='    current=pd.read_csv'
i=rt.index(needle)
insert='''    private_root=Path(restricted_dir) if restricted_dir is not None else ROOT/'data/restricted_yunyao'
    required=[private_root/'matched_profiles.csv',private_root/'track_145_Y038_E05/ionprf_position_time_ne_profile.csv',private_root/'track_145_Y038_E05/iri_profile_50_1000km_median_location.csv',private_root/'track_009_Y040_C19/ionprf_position_time_ne_profile.csv',private_root/'track_009_Y040_C19/iri_profile_50_1000km_median_location.csv']
    private_available=include_restricted and all(p.is_file() for p in required)
    if include_restricted and not private_available:
        raise FileNotFoundError('Authorized YunYao files are incomplete: '+', '.join(str(p) for p in required if not p.is_file()))
    if private_available:
        v37.YUNYAO_PROFILE_PACKAGE=private_root
        v37.YUNYAO_SUMMARY_SOURCE=private_root/'matched_profiles.csv'
    else:
        # No raw or extracted YunYao values, matched-location IRI values or
        # profile-track geometry are loaded by the default public entry point.
        v37.load_yunyao_summary=lambda:pd.DataFrame(columns=['track_index'])
        original_ne=v37.draw_yunyao_ne_panel
        def public_ne(ax,index,label,**kwargs):
            result=original_ne(ax,index,label,**kwargs)
            ax.text(.97,.08,'YunYao / matched IRI\\nnot redistributed',transform=ax.transAxes,
                    ha='right',va='bottom',fontsize=5.2,color='#636C76')
            return result
        v37.draw_yunyao_ne_panel=public_ne
'''
rt=rt[:i]+insert+rt[i:]
rt=rt.replace("'private_YunYao_profiles':2", "'private_YunYao_profiles':2 if private_available else 0,'restricted_profiles_included':private_available,'public_redraw_incomplete_panels':['g','i'] if not private_available else [],'YunYao_map_tracks_included':private_available")
rt=rt.replace('All scientific panels redrawn from included numerical inputs; PowerPoint assembly and font rasterization are not claimed pixel-exact.', 'Public mode redraws all unrestricted inputs but intentionally omits the two YunYao Ne profiles, their matched-location IRI curves and profile map tracks. The official reference is kept separately; this is not a complete Fig.5 reproduction.')
put(F/'code/runtime_fig5_R4.py',rt)
run='''"""Portable R4 public entry point; restricted YunYao inputs are opt-in."""
from pathlib import Path
import argparse,os,sys,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'code'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--figure',choices=['all','Fig5','ED9','ED8'],default='all')
 p.add_argument('--output-dir',type=Path,default=ROOT/'results_R4')
 p.add_argument('--with-restricted',action='store_true',help='Use externally supplied YunYao profiles only after provider permission')
 p.add_argument('--restricted-dir',type=Path,help='Directory with the documented authorized file layout')
 a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);results=[]
 if a.figure in ('all','Fig5'):
  from runtime_fig5_R4 import render
  name='Fig5_R4.png' if a.with_restricted else 'Fig5_R4_public_partial.png'
  results.append(render(a.output_dir/name,include_restricted=a.with_restricted,restricted_dir=a.restricted_dir))
 if a.figure in ('all','ED9','ED8'):
  from runtime_joint import render
  results.append(render(a.output_dir/'Extended_Data_Joint_observables.png'))
 (a.output_dir/'reproduction_QA_R4.json').write_text(json.dumps(results,indent=2)+'\\n',encoding='utf-8')
 print(json.dumps({'results':results,'restricted_opt_in':a.with_restricted},indent=2))
if __name__=='__main__':main()
'''
put(F/'run_R4.py',run)
verify=(OLD/'components/fig5joint/verify_inputs.py').read_text(encoding='utf-8-sig')
a=verify.index("    private=list(");b=verify.index("    report=",a)
verify=verify[:a]+"    # Restricted profile files are excluded from this public verifier.\n"+verify[b:]
verify=verify.replace("'YunYao_profiles_only':2,", "'YunYao_profiles_expected_external':2,'restricted_data_required_for_public_checks':False,")
verify=verify.replace("default=ROOT/'results/input_QA.json'", "default=ROOT/'results_R4/input_QA_R4.json'")
put(F/'verify_inputs_R4.py',verify)
restricted=[]
for p in sorted((OLD/'components/fig5joint/data/restricted_yunyao').rglob('*')):
 if p.is_file():restricted.append({'release_relative_path':str(p.relative_to(OLD)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size,'source_preserved':str(p),'runtime_full_mode_dependency':not p.name.endswith('.nc'),'public_default_loaded':False})
dump(F/'provenance/YunYao_public_exclusion_manifest_R4.json',{'status':'PUBLIC_DEFAULT_DOES_NOT_LOAD_RESTRICTED_FILES','public_distribution_exclude_all_below':'components/fig5joint/data/restricted_yunyao/','provider_permission_status':'not yet established; root will finalize after user reply','files':restricted,'raw_or_extracted_values_outside_restricted_folder':'none found; PFISR profiles are a separate public radar source','reference_artifacts':'Approved Fig.5 publication PNG retained, never read as runtime input'})
put(F/'README_R4_PUBLIC.md','''# Figure 5 and Extended Data 9: R4 public reproduction

The approved Fig5_R4.png and joint-observable figure are byte-identical to the frozen R3 targets. These figures retain their separate case-study and high-rate populations, not the Ground182 Fig3/4 cohort.

Run `python verify_inputs_R4.py`, then `python run_R4.py --figure all --output-dir results_R4`.

The default public entry point does not read YunYao files even if a local private copy is present. It writes Fig5_R4_public_partial.png, with missing-data labels in panels g/i. The two YunYao electron-density profiles, their matched-location IRI curves and YunYao track overlays cannot be reproduced from the public subset. PFISR and all other included numerical panels still redraw. The approved complete publication PNG is a separate reference, never used as a runtime data background. ED9 redraw and its true-P95 checks are unaffected.

For a complete numerical Fig.5 redraw, obtain provider permission and the two profiles through https://open.tjyyspace.com/#/dataDownload, then run `python run_R4.py --figure Fig5 --with-restricted --restricted-dir /path/to/authorized/layout --output-dir results_authorized`. The expected relative file paths are recorded in provenance/YunYao_public_exclusion_manifest_R4.json. Raw NetCDF files are documented source provenance; the plotting entry point reads the extracted profiles and match table, not NetCDF. No raw-to-extracted conversion is claimed here.

Public archive preparation must exclude the entire data/restricted_yunyao directory, including extracted profiles and match-location IRI tables, unless permission is explicitly established. Local originals and the older review bundle remain preserved.

The scientific redrawing target is the included numerical population. The historical PowerPoint montage and fonts are not claimed pixel-exact. Current approved PNGs remain reference targets. All third-party data/artwork retain their source terms; authors' code licensing does not relicense them.
''')
put(M/'R4_OTHER_TASKS_README.md','''# R4 Fig1, space-weather and antenna tasks

- Fig1: unchanged authored schematic; compute_fig1_fresnel_R4.py independently verifies the six PPTX finite-source Fresnel series. The whole schematic is a reference export, not a data-to-pixels reconstruction.
- Space weather: render_space_weather_R4.py redraws the current approved R4 six-row layout from the frozen independent context cohort: 2,244 windows, 693 polar. These are historical OP-context summaries, not the 2,196 post-QC Fig2 event windows. Keep the populations distinct.
- Antenna: render_antenna_R4.py redraws the approved A-over-B / C-at-right layout from frozen off-boresight density, range and source-cut summaries. C shades actual full min/max across raw SVN-by-azimuth cuts and keeps the archived equal-SVN reference median. Galileo is the GRAP population-model azimuth set, not individual spacecraft. Optional prepare_full_pattern_range_R4.py recomputes and checks these reference curves from the included official source pattern files.

Scripts use MAIN12_OUTPUT as the output parent and never write to source/manuscript folders. PNG-only 600dpi, exact approved layout preserved. Arial is not redistributed; a locally installed font or Matplotlib fallback may change rasterized pixels. Original artwork and official third-party antenna source files retain their own rights; no blanket CC-BY/MIT claim is made for them.
''')
put(F/'THIRD_PARTY_ACCESS_R4.md','''# Third-party data and public scope (R4)

YunYao observations were supplied on request and were not part of the provider's public release when analysed. Provider approval for redistribution is not established by this package. Exclude data/restricted_yunyao in full from the public archive unless explicit permission is confirmed. This includes raw NetCDF, extracted Ne/position profiles, the match table and IRI curves evaluated at those private match locations. All local original copies remain preserved. Requests: https://open.tjyyspace.com/#/dataDownload.

The public runtime does not substitute synthetic profiles. It clearly labels omitted Ne context and reports the missing profile/map-track dependencies. Publication-reference PNGs are retained separately and are not data inputs to the scientific renderer.

PFISR/Madrigal, SuperDARN, GNSS and radio-occultation data retain source citations and original terms. Natural Earth vector maps retain their public-domain source attribution. IRI model output is not itself a YunYao observation; the matched-location subset is withheld conservatively with the private match metadata. Code licensing covers author-generated code only, and does not relicense third-party source code, artwork, models or observations.
''')
proposal={'revision':'R4','main12_tasks':[
 {'id':'fig1','numeric_script':'code/compute_fig1_fresnel_R4.py','reference':'reference/Fig1_R4.png','output_name':'fig1/Fig1_R4.png','source_artwork':'source_artwork/Fig1_R4.pptx','classification':'authored reference plus independently checked theoretical curves'},
 {'id':'space_weather','script':'code/render_space_weather_R4.py','output_name':'space_weather/Extended_Data_Space_weather_index.png','scope':'independent frozen context cohort:2244 windows/693polar, not2196event windows'},
 {'id':'antenna','script':'code/render_antenna_R4.py','output_name':'antenna/Extended_Data_GNSS_antenna_geometry.png','optional_raw_check':'code/prepare_full_pattern_range_R4.py','reference':'reference/Extended_Data_GNSS_antenna_geometry_R4_reference.png'}],
 'fig5joint_entrypoint':'run_R4.py','fig5joint_verifier':'verify_inputs_R4.py','fig5joint_public_outputs':['Fig5_R4_public_partial.png','Extended_Data_Joint_observables.png'],'fig5joint_restricted_default':'not read even if present','fig5joint_public_complete':False,
 'third_party_licensing':'Do not automatically relicense NAVCEN/JRC/NaturalEarth/mission/network data, artwork, fonts or dependencies; preserve sources and individual licence notices.',
 'formal_reference_hashes':{n:sha(FORMAL/n) for n in ['Fig1_R4.png','Fig5_R4.png','Extended_Data_Joint_observables.png','Extended_Data_Space_weather_index.png','Extended_Data_GNSS_antenna_geometry.png']}}
dump(M/'R4_other_tasks_proposal.json',proposal)
print(json.dumps({'status':'NEW_R4_TASKS_PREPARED','main12_run_and_component_untouched':True,'restricted_staging_files_not_deleted':True,'proposal':str(M/'R4_other_tasks_proposal.json')},indent=2))
