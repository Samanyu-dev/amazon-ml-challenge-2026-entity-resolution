from pathlib import Path
import argparse,json,time,collections,os
from concurrent.futures import ProcessPoolExecutor
import duckdb
import importlib
MAPS={};VIEWS=None
def init(maps,version='v2'):
    global MAPS,VIEWS
    MAPS=maps
    VIEWS=importlib.import_module(f'normalization_{version}')

def process(batch):
    lines=[]
    for eid,country,name,address,owner,fold in batch:
        n,c,a,p,script,missing=VIEWS.normalized(name,address,MAPS)
        fields=[eid,country,n,c,a,p,str(script),str(missing),str(owner),str(fold)]
        # v3 appends the separately retained legal form as an 11th column.
        if VIEWS.VERSION!='normalization-audit-v2':fields.append(VIEWS.views(name,'','')['legal_form'])
        lines.append('\t'.join(fields)+'\n')
    return ''.join(lines)

def learn(db,out):
    # Learned aliases are quarantined after the normalization audit.  The
    # audited v2 normalizer accepts only its reviewed static rules.
    maps={'name':{},'address':{}}
    (out/'normalization_maps.json').write_text(json.dumps(maps,ensure_ascii=False,indent=2))
    (out/'normalization_maps_policy.json').write_text(json.dumps({
        'version':'normalization-audit-v2','learned_aliases':'QUARANTINED',
        'approved_learned_alias_count':0
    },ensure_ascii=False,indent=2))
    print('maps quarantined; approved learned aliases=0',flush=True)
    return maps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--split',choices=['train','test','all'],default='all');ap.add_argument('--version',choices=['v2','v3'],default='v2');ap.add_argument('--out',help='output dir (default: --work)')
    a=ap.parse_args();out=Path(a.out or a.work);out.mkdir(parents=True,exist_ok=True);db=duckdb.connect(str(Path(a.work)/'records.duckdb'),read_only=True);db.execute("SET memory_limit='2GB'");db.execute('SET threads=2')
    maps=learn(db,out)
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(maps,a.version)) as ex:
        for split in ['train','test'] if a.split=='all' else [a.split]:
            for src in [1,2,3]:
                name=f'{split}_s{src}';path=out/f'{name}.norm.tsv'
                if path.exists() and (out/f'{name}.norm.done').exists():continue
                if src==1:
                    q=f'SELECT entity_id,country,business_name,business_address,rid AS owner,fold FROM {name} ORDER BY rid'
                elif split=='train':
                    q=f'SELECT t.entity_id,t.country,t.business_name,t.business_address,l.aid,l.fold FROM {name} t JOIN {name}_labels l ON l.rid=t.rid ORDER BY t.rid'
                else:q=f'SELECT entity_id,country,business_name,business_address,-1 AS owner,-1 AS fold FROM {name} ORDER BY rid'
                cur=db.execute(q);n=0;t=time.time();pending=[]
                with open(str(path)+'.tmp','w') as f:
                    while True:
                        rows=cur.fetchmany(10000)
                        if rows:pending.append(ex.submit(process,rows))
                        if len(pending)>=a.workers*2 or not rows:
                            if pending:
                                text=pending.pop(0).result();f.write(text);n+=text.count('\n')
                                if n%200000==0:print(name,n,round(time.time()-t,1),flush=True)
                        if not rows:
                            for fu in pending:
                                text=fu.result();f.write(text);n+=text.count('\n')
                            break
                os.replace(str(path)+'.tmp',path);(out/f'{name}.norm.done').write_text(json.dumps({'rows':n,'seconds':time.time()-t}))
                print('NORMALIZED',name,n,time.time()-t,flush=True)
    db.close()
if __name__=='__main__':main()
