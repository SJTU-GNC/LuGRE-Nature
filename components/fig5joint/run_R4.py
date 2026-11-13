"""Portable R4 public entry point; restricted YunYao inputs are opt-in."""
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
 if not a.with_restricted:
  def public_access_guard(event,args):
   if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
    name=os.fsdecode(args[0]).replace('\\','/').lower()
    if '/restricted_yunyao/' in name:
     raise PermissionError('Public mode must not read restricted YunYao files: '+name)
  sys.addaudithook(public_access_guard)
 if a.figure in ('all','Fig5'):
  from runtime_fig5_R4 import render
  name='Fig5_R4.png' if a.with_restricted else 'Fig5_R4_public_partial.png'
  results.append(render(a.output_dir/name,include_restricted=a.with_restricted,restricted_dir=a.restricted_dir))
 if a.figure in ('all','ED9','ED8'):
  from runtime_joint import render
  results.append(render(a.output_dir/'Extended_Data_Joint_observables.png'))
 (a.output_dir/'reproduction_QA_R4.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'results':results,'restricted_opt_in':a.with_restricted},indent=2))
if __name__=='__main__':main()
