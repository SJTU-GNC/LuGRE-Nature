"""Verify figure dimensions/DPI and report rendering differences from approved PNGs."""
from pathlib import Path
import argparse, hashlib, json
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    p.add_argument('--require-identical-pixels',action='store_true')
    args=p.parse_args()
    metadata=json.loads((ROOT/'component.json').read_text(encoding='utf-8'))
    rows=[]
    for name in [name for task in metadata['tasks'] for name in task['output_names']]:
        output=args.output_dir/name;reference=ROOT/'reference'/name
        assert output.is_file() and output.stat().st_size>0,name
        with Image.open(output) as actual, Image.open(reference) as expected:
            assert actual.size==expected.size,(name,actual.size,expected.size)
            dpi=actual.info.get('dpi');assert dpi and min(dpi)>=599,name
            a=actual.convert('RGB');b=expected.convert('RGB')
            d=ImageChops.difference(a,b)
            identical=d.getbbox() is None
            h=d.histogram()
            mean=sum((i%256)*n for i,n in enumerate(h))/(a.width*a.height*3)
            a.close();b.close();d.close()
            if args.require_identical_pixels:assert identical,name
            rows.append({'file':name,'pixels':list(actual.size),'dpi':list(dpi),
                'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                'approved_reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
                'pixels_identical':identical,'mean_absolute_RGB_difference':mean})
    qa={'status':'PASS_IMAGE_DIMENSIONS_AND_DPI','source_selection_sha256':metadata['source_selection_sha256'],
        'images':rows,'all_pixels_identical':all(x['pixels_identical'] for x in rows),
        'typography_note':'Pixel differences may arise from Matplotlib/font/OS versions; validate_inputs.py separately checks scientific quantities.'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'image_verification.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(qa,indent=2))
if __name__=='__main__':main()
