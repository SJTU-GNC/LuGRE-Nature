"""Compatibility entrypoint forwarding all supported figure tasks to current R4 workflows."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--task',choices=['all','fig1','fig2','e21','qc_background','space_weather','antenna','thresholds'],default='all')
 p.add_argument('--output',type=Path,default=ROOT/'outputs')
 a=p.parse_args()
 if a.task=='all':groups=[('run_other_r4.py','all'),('run_processing_r4.py','all')]
 elif a.task in {'fig1','space_weather','antenna'}:groups=[('run_other_r4.py',a.task)]
 else:groups=[('run_processing_r4.py',a.task)]
 for script,task in groups:
  subprocess.run([sys.executable,'-B',str(ROOT/script),'--task',task,'--output',str(a.output.resolve())],cwd=ROOT,check=True)
if __name__=='__main__':main()
