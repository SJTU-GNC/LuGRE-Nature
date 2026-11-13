"""Portable R4 Fig1, space-weather and antenna tasks (PNG-only)."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--task',choices=['all','fig1','space_weather','antenna'],default='all')
 p.add_argument('--output',type=Path,default=ROOT/'outputs_R4_other')
 a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(MAIN12_OUTPUT=str(out),MPLCONFIGDIR=str(out/'mplconfig'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 tasks=['fig1','space_weather','antenna'] if a.task=='all' else [a.task]
 result=[]
 for task in tasks:
  if task=='fig1':
   folder=out/'fig1'
   subprocess.run([sys.executable,'-B',str(ROOT/'code/compute_fig1_fresnel_R4.py'),'--output',str(folder)],env=env,cwd=ROOT,check=True)
   shutil.copyfile(ROOT/'reference/Fig1_R4.png',folder/'Fig1_R4.png')
   result.append({'task':task,'status':'AUTHORED_SCHEMATIC_REFERENCE_WITH_INDEPENDENT_NUMERICAL_VERIFICATION'})
  else:
   script='render_space_weather_R4.py' if task=='space_weather' else 'render_antenna_R4.py'
   subprocess.run([sys.executable,'-B',str(ROOT/'code'/script)],env=env,cwd=ROOT,check=True)
   result.append({'task':task,'status':'REDRAWN_FROM_NUMERICAL_INPUTS_CURRENT_APPROVED_LAYOUT'})
 (out/'run_other_R4_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
