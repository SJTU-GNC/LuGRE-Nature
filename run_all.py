"""Run the packaged figure workflows with the Python interpreter invoking this file."""
from pathlib import Path
import argparse, datetime, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parent

def restricted_yunyao_available():
    root=ROOT/'components'/'fig5joint'/'data'/'restricted_yunyao'
    required=[
        root/'matched_profiles.csv',
        root/'track_145_Y038_E05'/'ionprf_position_time_ne_profile.csv',
        root/'track_009_Y040_C19'/'ionprf_position_time_ne_profile.csv',
    ]
    return all(path.is_file() for path in required)

def task_index():
    tasks=[]
    for component in ('main12','ground182','fig5joint'):
        meta=ROOT/'components'/component/'component.json'
        if not meta.exists(): raise FileNotFoundError('Incomplete component: '+str(meta))
        d=json.loads(meta.read_text(encoding='utf-8'))
        for task in d['tasks']:
            tasks.append(dict(task,component=component))
    ids=[t['id'] for t in tasks]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate task identifiers')
    return tasks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',nargs='+',default=['all'],help='Task IDs, or all')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    args=parser.parse_args()
    tasks=task_index()
    if args.list:
        for t in tasks:print(t['id']+': '+t['reproduction_class'])
        return
    requested=set(args.tasks)
    if 'all' not in requested:
        unknown=requested-{t['id'] for t in tasks}
        if unknown:parser.error('Unknown tasks: '+', '.join(sorted(unknown)))
        if 'fig5' in requested and not restricted_yunyao_available():
            parser.error('Fig.5 requires separately authorised YunYao profiles that are excluded from the public release.')
        tasks=[t for t in tasks if t['id'] in requested]
    elif not restricted_yunyao_available():
        print('NOTICE: skipping fig5 because separately authorised YunYao profiles are not included in the public release.',flush=True)
        tasks=[t for t in tasks if t['id']!='fig5']
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    logs=output/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update({'MPLBACKEND':'Agg','MPLCONFIGDIR':str(output/'cache/matplotlib'),
        'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','OPENBLAS_NUM_THREADS':'1',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_MAX_THREADS':'1',
        'TEMP':str(output/'tmp'),'TMP':str(output/'tmp')})
    (output/'tmp').mkdir(exist_ok=True)
    report=[]
    for task in tasks:
        component=task['component'];component_root=ROOT/'components'/component
        component_output=output/component
        flag='--output' if component=='main12' else '--output-dir'
        command=[sys.executable,'-B',*task['command'],flag,str(component_output)]
        log=logs/(task['id']+'.log')
        print('Running '+task['id']+' ...',flush=True)
        start=time.monotonic()
        with log.open('w',encoding='utf-8') as f:
            result=subprocess.run(command,cwd=component_root,env=env,stdout=f,stderr=subprocess.STDOUT)
        expected=[component_output/name for name in task['output_names']]
        missing=[str(p) for p in expected if not p.is_file()]
        record={'task':task['id'],'component':component,'returncode':result.returncode,
            'seconds':round(time.monotonic()-start,3),'reproduction_class':task['reproduction_class'],
            'notes':task.get('notes',''),'outputs':[str(p) for p in expected],
            'missing_outputs':missing,'log':str(log)}
        report.append(record)
        (output/'RUN_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        if result.returncode or missing:
            print('FAILED '+task['id']+'; see '+str(log),file=sys.stderr)
            raise SystemExit(result.returncode or 1)
        print('Finished '+task['id'],flush=True)
    print('All requested tasks completed. Report: '+str(output/'RUN_REPORT.json'))

if __name__=='__main__':main()
