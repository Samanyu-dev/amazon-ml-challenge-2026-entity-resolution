"""Score every genuine candidate, preserving lineage and per-target alternatives."""
from pathlib import Path
import argparse,json,time,os
import numpy as np
from catboost import CatBoostClassifier
from train_model import pair_dtype,feature_count
SF=np.dtype([('tid','<u4'),('aid','<u4'),('p','<f4')])
BFIDX=[0,1,8,13,14,16,24,26,37,42,43,52]
BD=np.dtype([('aid','<i4'),('aid2','<i4'),('owner','<i4'),('afold','<i4'),('tfold','<i4'),('p','<f4'),('p2','<f4'),('cos','<f4'),('route','<i4'),('f','<f4',(len(BFIDX),))])

def merge_best(old,new):
    first=new['p']>old['p'];result=old.copy();result[first]=new[first]
    result['p2']=np.where(first,np.maximum(old['p'],new['p2']),np.maximum(old['p2'],new['p']))
    result['aid2']=np.where(first,np.where(old['p']>=new['p2'],old['aid'],new['aid2']),np.where(new['p']>=old['p2'],new['aid'],old['aid2']))
    return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pairs',nargs='+',required=True);ap.add_argument('--model',required=True);ap.add_argument('--out',required=True);ap.add_argument('--targets',type=int,required=True);ap.add_argument('--anchors',type=int,required=True);ap.add_argument('--threads',type=int,default=6);ap.add_argument('--dense-features',action='store_true');ap.add_argument('--artifacts');ap.add_argument('--split',choices=['train','test'])
 ap.add_argument('--max-lexical-rank',type=int,help='blocking cut: keep lexical candidates with retrieval_rank below this')
 ap.add_argument('--dense-min-dice',type=float,help='blocking cut: keep ANN-only candidates whose name or address trigram dice reaches this');a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():raise RuntimeError('Output already complete; select a new run directory to avoid mixed versions')
 model=CatBoostClassifier();model.load_model(a.model)
 lookup=None
 if a.dense_features:
  from embedding_features import EmbeddingLookup
  lookup=EmbeddingLookup(a.artifacts,a.split)
 best=np.memmap(out/'best.bin',mode='w+',dtype=BD,shape=(a.targets,));best[:]=np.zeros((),dtype=BD);best['aid']=-1;best['aid2']=-1;best['owner']=-1;best['afold']=-1;best['tfold']=-1
 nparts=(a.anchors+99999)//100000;handles=[open(out/f'candidates-{i:03}.bin','wb') for i in range(nparts)];total=0;groups=0;start=time.time()
 files=[p for d in a.pairs for p in sorted(Path(d).glob('pairs-*.bin'))]
 nfeat=feature_count(a.pairs[0]);assert all(feature_count(d)==nfeat for d in a.pairs);DT=pair_dtype(nfeat)
 names=json.loads((Path(a.pairs[0])/'features.json').read_text());fi={n:i for i,n in enumerate(names)}
 def blocking_keep(z):
  # Rule-based blocking on retrieval-time features only (no model score), so the kept rows are the candidate set.
  keep=np.ones(len(z),bool);x=z['x']
  dense=x[:,fi['dense_route']]>0 if 'dense_route' in fi else np.zeros(len(z),bool)
  if a.max_lexical_rank is not None:keep&=dense|(x[:,fi['retrieval_rank']]<a.max_lexical_rank)
  if a.dense_min_dice is not None:keep&=~dense|(np.maximum(x[:,fi['name_trigram_dice']],x[:,fi['address_trigram_dice']])>=a.dense_min_dice)
  return keep
 for file in files:
  assert file.stat().st_size%DT.itemsize==0
  with file.open('rb') as stream:
   carry=np.empty(0,dtype=DT)
   while True:
    z=np.fromfile(stream,dtype=DT,count=200000)
    eof=len(z)==0
    if len(carry):z=np.concatenate([carry,z])
    if not len(z):break
    if not eof:
     boundary=np.searchsorted(z['tid'],z['tid'][-1],side='left')
     carry=z[boundary:].copy();z=z[:boundary]
     if not len(z):continue
    else:carry=np.empty(0,dtype=DT)
    dense_route='dense_pairs' in str(file.parent)
    z=z[blocking_keep(z)]
    if not len(z):
     if eof:break
     continue
    X=lookup.features(z,dense_route=dense_route) if lookup else np.ascontiguousarray(z['x'])
    p=model.predict_proba(X,thread_count=a.threads)[:,1].astype(np.float32)
    assert np.isfinite(p).all() and np.all(z['aid']<a.anchors) and np.all(z['tid']<a.targets)
    rows=np.empty(len(z),dtype=SF);rows['tid']=z['tid'];rows['aid']=z['aid'];rows['p']=p
    buckets=z['aid']//100000
    for b in np.unique(buckets):rows[buckets==b].tofile(handles[b])
    starts=np.r_[0,np.flatnonzero(np.diff(z['tid']))+1];ends=np.r_[starts[1:],len(z)];sizes=ends-starts
    pmax=np.maximum.reduceat(p,starts);indices=np.arange(len(z));win=np.minimum.reduceat(np.where(p==np.repeat(pmax,sizes),indices,len(z)),starts)
    pother=p.copy();pother[win]=-1;psecond=np.maximum.reduceat(pother,starts)
    win2=np.minimum.reduceat(np.where(pother==np.repeat(psecond,sizes),indices,len(z)),starts);psecond=np.maximum(psecond,0)
    selected=z[win];tid=selected['tid'];b=np.empty(len(win),dtype=BD)
    for field in ['aid','owner','afold','tfold']:b[field]=selected[field]
    b['aid2']=np.where(sizes>1,z['aid'][win2].astype(np.int32),-1)
    b['p']=pmax;b['p2']=psecond;b['cos']=X[win,nfeat] if lookup else 0;b['route']=int(dense_route);b['f']=selected['x'][:,BFIDX]
    best[tid]=merge_best(best[tid],b);total+=len(z);groups+=len(win)
    if total//2000000!=(total-len(z))//2000000:print('SCORED',total,'targets',groups,'seconds',round(time.time()-start,1),flush=True)
    if eof:break
  best.flush()
 for f in handles:f.close()
 best.flush();report={'candidate_pairs':total,'target_groups':groups,'target_universe':a.targets,'anchor_universe':a.anchors,'best_feature_indices':BFIDX,'best_dtype':BD.descr,'seconds':time.time()-start,'model':a.model,'source_pair_dirs':a.pairs,'max_lexical_rank':a.max_lexical_rank,'dense_min_dice':a.dense_min_dice}
 (out/'complete.json').write_text(json.dumps(report,indent=2));print('SCORE_COMPLETE',report,flush=True)
if __name__=='__main__':main()
