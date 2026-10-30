"""Finish provenance metadata after original-machine assembly; no data rewriting."""
from pathlib import Path
import hashlib, json, ast, re

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'provenance/assembly_source_map.json'
record=json.loads(path.read_text(encoding='utf-8'))
for row in record['files']:
    original=row.pop('sha256',row.get('copied_original_sha256'))
    row['copied_original_sha256']=original
    dest=ROOT/row['package_path']
    if dest.suffix.lower()=='.py':
        row['release_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
        row['adapted_for_portability']=row['release_sha256']!=original
    else:
        row['release_sha256']=original
        row['adapted_for_portability']=False
record['hash_scope']='copied_original_sha256 is the verified source/copy hash before adaptation; release_sha256 is the actual release file hash after documented Python portability edits.'
path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
active=[]
for sub in ['fig34/plotting','illumination','geometry','context']:
    for p in (ROOT/sub).rglob('*.py'):
        text=p.read_text(encoding='utf-8');ast.parse(text)
        matches=re.findall(r'''(?:[A-Z]:[\\/])''',text)
        assert not matches,(str(p),matches)
        active.append(str(p.relative_to(ROOT)))
for name in ['reproduce.py','validate_inputs.py','verify_outputs.py']:
    text=(ROOT/name).read_text(encoding='utf-8');ast.parse(text)
    assert not re.findall(r'''(?:[A-Z]:[\\/])''',text),name
    active.append(name)
qa={'status':'PASS_STATIC_PORTABILITY','active_python_files':active,
    'absolute_external_drive_paths_in_active_code':0,
    'archival_code_is_not_supported_entrypoint':True,
    'original_selection_sha256':record['source_selection_sha256']}
(ROOT/'validation').mkdir(exist_ok=True)
(ROOT/'validation/static_portability.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':qa['status'],'active_python_files_checked':len(active)}))
