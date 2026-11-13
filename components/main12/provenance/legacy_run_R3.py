"""Portable plotting entry point; inputs are immutable and package relative."""
from pathlib import Path
import argparse, subprocess, os, sys, json, shutil
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['all','fig1','fig2','e21','space_weather','antenna'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['MAIN12_OUTPUT']=str(out);env['PYTHONDONTWRITEBYTECODE']='1'
    env['MPLCONFIGDIR']=str(out/'mplconfig');env['OPENBLAS_NUM_THREADS']='1'
    tasks=['fig1','fig2','e21','space_weather','antenna'] if args.task=='all' else [args.task]
    reports=[]
    for task in tasks:
        if task=='fig1':
            dest=out/'fig1';dest.mkdir(exist_ok=True)
            subprocess.run([sys.executable,str(ROOT/'code/compute_fig1_fresnel.py'),'--output',str(dest)],env=env,check=True,cwd=ROOT)
            shutil.copyfile(ROOT/'reference/Fig1_R3.png',dest/'Fig1_R3.png')
            reports.append({'task':task,'status':'REFERENCE_AUTHORED_SCHEMATIC',
                'note':'The illustrative Fresnel chart is independently recomputed and verified; the authored schematic PNG is copied explicitly. Editable PPTX and optional native PowerPoint exporter are supplied.'})
            continue
        script={'fig2':'render_fig2.py','e21':'render_e21.py','space_weather':'render_space_weather.py','antenna':'render_antenna.py'}[task]
        subprocess.run([sys.executable,str(ROOT/'code'/script)],env=env,check=True,cwd=ROOT)
        reports.append({'task':task,'status':'RENDERED_FROM_SOURCE_DATA'})
    (out/'run_report.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
