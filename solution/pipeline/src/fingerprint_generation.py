from pathlib import Path
import hashlib,json
from input_gate import digest

def main():
    out=Path('artifacts')
    code=digest(Path('src/normalization_v2.py'))
    cfg=digest(out/'normalization_maps_policy.json')
    raw=json.loads((out/'data_manifest.json').read_text())
    seed=json.dumps({'raw':raw,'code':code,'config':cfg},sort_keys=True).encode()
    gid=hashlib.sha256(seed).hexdigest()[:24]
    m={'status':'APPROVED','generation_id':gid,'normalization_version':'normalization-audit-v2',
       'normalization_code_sha256':code,'normalization_config_sha256':cfg,'sources':{},'combined_targets':{}}
    for split in ['train','test']:
      for i in [1,2,3]:
        p=out/f'{split}_s{i}.norm.tsv'; m['sources'][f'{split}_s{i}']={'path':p.name,'rows':sum(1 for _ in p.open()),'sha256':digest(p),'generation_id':gid}
      p2=out/f'{split}_s2.norm.tsv';p3=out/f'{split}_s3.norm.tsv';c=out/f'{split}_targets.norm.tsv'
      c.write_bytes(p2.read_bytes()+p3.read_bytes())
      m['combined_targets'][split]={'path':c.name,'rows':m['sources'][f'{split}_s2']['rows']+m['sources'][f'{split}_s3']['rows'],'sha256':digest(c),'component_sha256':[m['sources'][f'{split}_s2']['sha256'],m['sources'][f'{split}_s3']['sha256']],'generation_id':gid}
    (out/'generation.json').write_text(json.dumps(m,indent=2))
    print(json.dumps(m,indent=2))
if __name__=='__main__': main()
