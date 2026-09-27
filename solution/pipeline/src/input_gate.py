"""Fail-closed verification before launching a candidate/model stage."""
from pathlib import Path
import hashlib,json

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def verify_generation(manifest_path):
    path=Path(manifest_path);m=json.loads(path.read_text())
    if m.get('status')!='APPROVED':raise ValueError('Generation has not passed its audit gate')
    if not m.get('normalization_code_sha256') or not m.get('normalization_config_sha256'):raise ValueError('Missing code/config fingerprints')
    if set(m.get('sources',{}))!={f'{s}_s{i}' for s in ['train','test'] for i in [1,2,3]}:raise ValueError('All six sources required')
    records=list(m['sources'].values())+list(m.get('combined_targets',{}).values())
    if len(records)!=8:raise ValueError('Both combined targets required')
    for item in records:
        if item.get('generation_id')!=m['generation_id']:raise ValueError('Mixed generation IDs')
        p=path.parent/item['path']
        if not p.is_file() or digest(p)!=item['sha256']:raise ValueError(f'Input fingerprint mismatch: {p}')
        if int(item.get('rows',0))<=0:raise ValueError('Missing row count')
    for split in ['train','test']:
        s=m['sources'];c=m['combined_targets'][split]
        expected=[s[f'{split}_s2']['sha256'],s[f'{split}_s3']['sha256']]
        if c.get('component_sha256')!=expected:raise ValueError('Stale target components')
        if c['rows']!=s[f'{split}_s2']['rows']+s[f'{split}_s3']['rows']:raise ValueError('Target count mismatch')
    return m
