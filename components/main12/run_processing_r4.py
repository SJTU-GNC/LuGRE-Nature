"""Recompute and render the R4 Fig.2, E21, QC/background and threshold tasks.

All scientific inputs are relative to this component. No network access is
needed when the declared Python requirements are installed.
"""
from pathlib import Path
import argparse,subprocess,os,sys,json
C=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['all','fig2','e21','qc_background','thresholds'],default='all')
    p.add_argument('--output',type=Path,default=C/'outputs')
    p.add_argument('--refit-results',type=Path,help='Optional native full-refit parquet for QC/background recomputation.')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['MAIN12_OUTPUT']=str(out);env['PYTHONDONTWRITEBYTECODE']='1';env['OPENBLAS_NUM_THREADS']='1'
    if a.refit_results:env['MAIN12_REFIT_RESULTS']=str(a.refit_results.resolve())
    tasks=['fig2','e21','qc_background','thresholds'] if a.task=='all' else [a.task]
    reports=[];fig2_ready=False
    def run(name,*extra):
        subprocess.run([sys.executable,'-B',str(C/'code'/name),*extra],env=env,cwd=C,check=True)
    for task in tasks:
        if task in ['fig2','thresholds'] and not fig2_ready:
            run('recompute_fig2_qc_r4.py');run('recompute_paired_qc_r4.py');fig2_ready=True
        if task=='fig2':run('render_fig2_r4.py')
        elif task=='e21':run('render_e21_r4.py')
        elif task=='qc_background':run('recompute_background_qc_r4.py');run('render_qc_background_r4.py')
        elif task=='thresholds':run('recompute_thresholds_r4.py')
        reports.append({'task':task,'status':'PASS_RECOMPUTED_AND_RENDERED' if task not in ['thresholds'] else 'PASS_RECOMPUTED'})
    if fig2_ready:run('verify_processing_r4.py','--output',str(out))
    (out/'run_processing_r4_report.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
