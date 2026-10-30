"""Regenerate the six selected Ground182 figures from the bundled frozen inputs."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, subprocess, sys

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'outputs'
TASKS=('fig3','fig4','illumination','geometry','context')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
sys.dont_write_bytecode=True

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def run_context(out):
    import csv
    path=ROOT/'context/frozen/Ground_final_contributors_R3.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:stations=list(csv.DictReader(f))
    selection=json.loads((ROOT/'provenance/R3_Ground182_current_version.json').read_text(encoding='utf-8'))
    original=json.loads((ROOT/'context/frozen/Station_selection_provenance_R3.json').read_text(encoding='utf-8'))
    assert len(stations)==len({r['station'] for r in stations})==selection['station_count']==182
    assert set(r['station'] for r in stations)==set(original['stations'])
    for r in stations:r['lat']=float(r['lat']);r['lon']=float(r['lon'])
    north=sorted(r['station'] for r in stations if r['lat']>=60)
    south=sorted(r['station'] for r in stations if r['lat']<=-60)
    ctx=module('ground182_context_renderer',ROOT/'context/render_context.py')
    ctx.STATION_LEGEND=tuple((key,name,'30 s' if key=='GNSS/ROTI station' else cadence) for key,name,cadence in ctx.STATION_LEGEND)
    provenance={'status':'CURRENT182_MEMBERSHIP_WITH_UNCHANGED_FROZEN_COORDINATES',
        'station_count':182,'Ground_epochs':selection['Ground_epochs'],
        'selection_sha256':selection['selection_sha256'],
        'coordinate_source':'context/frozen/Ground_final_contributors_R3.csv',
        'station_count_is_not_IPP_band_count':True,
        'historical_coordinate_table_epoch_totals_not_used_for_analysis':True}
    provenance_path=out/'context_current_binding.json'
    out.mkdir(parents=True,exist_ok=True)
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    def load_final(csv_path,provenance_path_ignored=None):
        report={'csv':str(path),'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'input_rows':182,'eligible_unique_stations':182,'north_unique_stations':len(north),
            'south_unique_stations':len(south),'mid_low_latitude_unique_stations':182-len(north)-len(south),
            'station_ids':sorted(r['station'] for r in stations),'north_station_ids':north,'south_station_ids':south,
            'inventory_provenance':str(provenance_path),'inventory_provenance_sha256':hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            'admission':'Frozen user-selected182 contribution; coordinates unchanged after QC interval exclusions.',
            'accepted_epoch_count':selection['Ground_epochs'],'coordinate_policy':'One unchanged marker per accepted station ID.',
            'category_overlap':'Polar GNSS, scintillation and1Hz site categories can overlap.'}
        return stations,report
    ctx.load_ground_stations=load_final
    return ctx.render(path,out,inventory_provenance=provenance_path)

def run(task,output):
    output.mkdir(parents=True,exist_ok=True)
    if task=='fig3':
        command=[ROOT/'fig34/plotting/plot_Fig3_R3.py','--data-dir',ROOT/'fig34/amplitude_source','--output-dir',output,'--expected-status','COMPLETE_FIG3_R3_FINAL_SELECTION']
    elif task=='fig4':
        command=[ROOT/'fig34/plotting/plot_Fig4_R3.py','--data-dir',ROOT/'fig34/response_source','--output-dir',output,'--expected-status','PASS_FIG4_R3_FINAL_SELECTION']
    elif task=='geometry':
        command=[ROOT/'geometry/plot_geometry_R3.py','--data-dir',ROOT/'geometry/plot_data','--output-dir',output]
    elif task=='illumination':
        command=[ROOT/'illumination/plot_LT_illumination_R3.py']
    elif task=='context':
        return run_context(output)
    else:raise ValueError(task)
    env=os.environ.copy();env['LUGRE_FIGURE_OUTPUT']=str(output)
    env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUTF8']='1'
    subprocess.run([sys.executable,'-B',*map(str,command)],cwd=ROOT,env=env,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',nargs='?',default='all',choices=['all',*TASKS])
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args()
    for task in (TASKS if args.task=='all' else [args.task]):
        print('Rendering',task,flush=True)
        run(task,args.output_dir.resolve())
    print('Outputs:',args.output_dir.resolve())

if __name__=='__main__':main()
