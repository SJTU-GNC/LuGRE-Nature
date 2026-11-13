"""Verify the merged R4 code/data release and its selected 14-figure snapshot."""
from pathlib import Path
import argparse,csv,hashlib,json
ROOT=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-only',action='store_true')
    args=parser.parse_args()
    snap=json.loads((ROOT/'provenance/MANUSCRIPT_FIGURE_SNAPSHOT.json').read_text(encoding='utf-8'))
    assert snap['revision']=='R4' and snap['station_count']==182 and len(snap['figures'])==14
    for item in snap['figures']+snap['manuscripts']:
        p=ROOT/item['file']
        assert p.is_file() and p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],str(p)
    checked=len(snap['figures'])+len(snap['manuscripts'])
    if not args.snapshot_only:
        manifest=ROOT/'FILE_MANIFEST.csv'
        if not manifest.exists():raise SystemExit('FILE_MANIFEST.csv missing; extract both data and code ZIPs into the same parent folder.')
        with manifest.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            rel=Path(row['path'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
            p=ROOT/rel
            assert p.is_file() and not p.is_symlink(),str(p)
            assert p.stat().st_size==int(row['bytes']) and digest(p)==row['sha256'],str(p)
        checked=len(rows)
    print(json.dumps({'status':'PASS','revision':'R4','station_count':182,'official_figures':14,'files_verified':checked}))
if __name__=='__main__':main()
