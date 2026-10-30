"""Portable entry point. All inputs are resolved relative to this file."""
from pathlib import Path
import argparse, os, sys, json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
sys.path.insert(0,str(ROOT/'code'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--figure',choices=['all','Fig5','ED8'],default='all')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    if args.figure in ('all','Fig5'):
        restricted=ROOT/'data'/'restricted_yunyao'
        required=[restricted/'matched_profiles.csv',
            restricted/'track_145_Y038_E05'/'ionprf_position_time_ne_profile.csv',
            restricted/'track_009_Y040_C19'/'ionprf_position_time_ne_profile.csv']
        if not all(path.is_file() for path in required):
            parser.error('Fig.5 requires separately authorised YunYao profiles that are excluded from the public release. Run ED8 alone or supply authorised inputs at the documented paths.')
        from runtime_fig5 import render
        results.append(render(args.output_dir/'Fig5_R3.png'))
    if args.figure in ('all','ED8'):
        from runtime_joint import render
        results.append(render(args.output_dir/'Extended_Data_Joint_observables.png'))
    (args.output_dir/'reproduction_QA.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output_dir':str(args.output_dir.resolve()),'results':results},indent=2))
if __name__=='__main__':main()
