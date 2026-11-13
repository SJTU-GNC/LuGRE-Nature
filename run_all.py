"""Run the packaged figure workflows with the Python interpreter invoking this file."""
from pathlib import Path
import argparse, datetime, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parent

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
        tasks=[t for t in tasks if t['id'] in requested]
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    logs=output/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update({'MPLBACKEND':'Agg','MPLCONFIGDIR':str(output/'cache/matplotlib'),
        'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','OPENBLAS_NUM_THREADS':'1',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_MAX_THREADS':'1',
        'TEMP':str(output/'tmp'),'TMP':str(output/'tmp')})
    (output/'tmp').mkdir(exist_ok=True)
    report=[]
    if 'all' in requested:
        # Each all-component driver reuses its one numerical reconstruction.
        groups=[('main12','main12_other',['run_other_r4.py','--task','all'],[t for t in tasks if t['id'] in {'fig1','space_weather','antenna'}]),
                ('main12','main12_processing',['run_processing_r4.py','--task','all'],[t for t in tasks if t['id'] in {'fig2','e21','qc_background','thresholds'}]),
                ('ground182','ground182_all',['reproduce.py','all'],[t for t in tasks if t['component']=='ground182']),
                ('fig5joint','fig5joint_public',['run_R4.py','--figure','all'],[t for t in tasks if t['component']=='fig5joint'])]
    else:
        groups=[(t['component'],t['id'],t['command'],[t]) for t in tasks]
    for component,group_id,group_command,covered in groups:
        component_root=ROOT/'components'/component
        component_output=output/component
        flag='--output' if component=='main12' else '--output-dir'
        command=[sys.executable,'-B',*group_command,flag,str(component_output)]
        log=logs/(group_id+'.log')
        print('Running '+group_id+' ...',flush=True)
        start=time.monotonic()
        with log.open('w',encoding='utf-8') as f:
            result=subprocess.run(command,cwd=component_root,env=env,stdout=f,stderr=subprocess.STDOUT)
        missing_all=[]
        for task in covered:
            expected=[component_output/name for name in task['output_names']]
            missing=[str(p) for p in expected if not p.is_file()]
            missing_all.extend(missing)
            report.append({'task':task['id'],'component':component,'execution_group':group_id,'returncode':result.returncode,
                'group_seconds':round(time.monotonic()-start,3),'reproduction_class':task['reproduction_class'],
                'notes':task.get('notes',''),'outputs':[str(p) for p in expected],
                'missing_outputs':missing,'log':str(log)})
        (output/'RUN_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        if result.returncode or missing_all:
            print('FAILED '+group_id+'; see '+str(log),file=sys.stderr)
            raise SystemExit(result.returncode or 1)
        print('Finished '+group_id,flush=True)
    print('All requested tasks completed. Report: '+str(output/'RUN_REPORT.json'))

if __name__=='__main__':main()
