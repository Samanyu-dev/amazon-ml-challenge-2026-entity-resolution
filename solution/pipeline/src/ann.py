"""Country-generic, compressed multilingual ANN retrieval with exact reranking."""
from pathlib import Path
import argparse,json,time
import numpy as np,faiss

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work',required=True);ap.add_argument('--split',choices=['train','test'],required=True);ap.add_argument('--build',action='store_true');ap.add_argument('--source',type=int,choices=[2,3]);ap.add_argument('--threads',type=int,default=5);ap.add_argument('--topk',type=int,default=4);a=ap.parse_args()
 root=Path(a.work);emb=root/'embeddings';out=root/f'{a.split}_ann';out.mkdir(exist_ok=True);faiss.omp_set_num_threads(a.threads)
 meta=json.loads((emb/f'{a.split}_s1.json').read_text());assert meta['done']==meta['rows'];d=meta['dim'];vec=np.memmap(emb/f'{a.split}_s1.f16',mode='r',dtype=np.float16,shape=(meta['rows'],d));countries=np.load(root/f'{a.split}_anchor_countries.npy');manifest=json.loads((root/'data_manifest.json').read_text())
 if a.build:
  for country in np.unique(countries):
   idxfile=out/f'{country}.faiss'
   if idxfile.exists():continue
   ids=np.flatnonzero(countries==country);rng=np.random.default_rng(42);sample=rng.choice(ids,size=min(200000,len(ids)),replace=False);x=np.asarray(vec[sample],dtype=np.float32);faiss.normalize_L2(x)
   nlist=min(4096,max(64,len(ids)//200));quant=faiss.IndexFlatIP(d);index=faiss.IndexIVFPQ(quant,d,nlist,48,8,faiss.METRIC_INNER_PRODUCT)
   index.cp.niter=12;index.pq.cp.niter=12;start=time.time();index.train(x);del x
   for st in range(0,len(ids),20000):
    chunk=ids[st:st+20000];x=np.asarray(vec[chunk],dtype=np.float32);faiss.normalize_L2(x);index.add_with_ids(x,chunk.astype(np.int64))
   faiss.write_index(index,str(idxfile));print('INDEXED',country,len(ids),'seconds',time.time()-start,flush=True)
  return
 assert a.source is not None
 src=a.source;info=json.loads((emb/f'{a.split}_s{src}.json').read_text());assert info['done']==info['rows'];queries=np.memmap(emb/f'{a.split}_s{src}.f16',mode='r',dtype=np.float16,shape=(info['rows'],d))
 # Country and rid arrays are generated directly from the supplied source table.
 import duckdb
 db=duckdb.connect(str(root/'records.duckdb'),read_only=True);db.execute("SET memory_limit='1GB'")
 offset=0 if src==2 else manifest[f'{a.split}_s2']['rows'];dtype=np.dtype([('tid','<u4'),('aid','<u4'),('cos','<f4')])
 for country in np.unique(countries):
  dest=out/f's{src}-{country}.bin'
  if Path(str(dest)+'.done').exists():continue
  index=faiss.read_index(str(out/f'{country}.faiss'));index.nprobe=24
  cur=db.execute(f'SELECT rid FROM {a.split}_s{src} WHERE country=? ORDER BY rid',[str(country)]);n=0;start=time.time()
  with dest.open('wb') as f:
   while rows:=cur.fetchmany(2048):
    tids=np.array([r[0] for r in rows],dtype=np.int64);q=np.asarray(queries[tids],dtype=np.float32);faiss.normalize_L2(q);_,ids=index.search(q,32)
    safe=np.maximum(ids,0);ref=np.asarray(vec[safe],dtype=np.float32);sims=np.einsum('ijk,ik->ij',ref,q);sims[ids<0]=-10
    take=np.argsort(-sims,axis=1)[:,:a.topk];aa=np.take_along_axis(ids,take,axis=1);ss=np.take_along_axis(sims,take,axis=1)
    z=np.empty(aa.size,dtype=dtype);z['tid']=np.repeat(tids+offset,a.topk);z['aid']=aa.ravel();z['cos']=ss.ravel();z=z[aa.ravel()>=0];z.tofile(f);n+=len(tids)
    if n%102400<2048:print('ANN',a.split,src,country,n,'qps',round(n/(time.time()-start),1),flush=True)
  Path(str(dest)+'.done').write_text(json.dumps({'queries':n,'seconds':time.time()-start,'topk':a.topk}));del index
 db.close()
if __name__=='__main__':main()
