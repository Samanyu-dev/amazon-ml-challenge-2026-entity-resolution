"""Train split-safe CatBoost on retrieved hard negatives; no test fitting."""
from pathlib import Path
import argparse,json,time,gc
import numpy as np
from catboost import CatBoostClassifier,Pool
NF=54
def pair_dtype(nf):return np.dtype([('tid','<u4'),('aid','<u4'),('owner','<i4'),('tfold','<i4'),('afold','<i4'),('x','<f4',(nf,))])
DT=pair_dtype(NF)
def feature_count(pair_dir):return len(json.loads((Path(pair_dir)/'features.json').read_text()))
def batches(files,n=100000,DT=DT):
    for p in files:
        if p.stat().st_size%DT.itemsize:raise ValueError(f'Incomplete feature file: {p}')
        with p.open('rb') as f:
            while len(a:=np.fromfile(f,dtype=DT,count=n)):yield a

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pairs',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--artifacts',required=True);ap.add_argument('--iterations',type=int,default=1200);ap.add_argument('--pilot',action='store_true');ap.add_argument('--threads',type=int,default=6);ap.add_argument('--dense-features',action='store_true')
    a=ap.parse_args()
    if (Path(a.artifacts)/'NORMALIZATION_AUDIT_HOLD.json').exists():
        raise RuntimeError('Normalization audit hold: no matcher training permitted')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    files=[p for d in a.pairs for p in sorted(Path(d).glob('pairs-*.bin'))]
    lookup=None
    if a.dense_features:
        from embedding_features import EmbeddingLookup
        lookup=EmbeddingLookup(a.artifacts,'train')
    nfeat=feature_count(a.pairs[0]);assert all(feature_count(d)==nfeat for d in a.pairs);DTX=pair_dtype(nfeat)
    nf=nfeat+2 if lookup else nfeat
    counts=np.load(Path(a.artifacts)/'train_truth_counts.npy');stats={'train':0,'dev':0,'positive_train':0,'negative_train':0};t=time.time()
    handles={k:open(out/(k+'.bin'),'wb') for k in ['train_x','train_y','train_w','dev_x','dev_y']}
    for file in files:
      for z in batches([file],DT=DTX):
          y=(z['aid'].astype(np.int32)==z['owner']).astype(np.uint8)
          eligibility=(z['tfold']<80)&(z['afold']<80)
          h=(z['tid'].astype(np.uint64)*2654435761+z['aid'].astype(np.uint64)*2246822519)%4294967291
          rank=z['x'][:,42]
          keep=(y==1)|((rank==0)&(h%2==0))|((rank==1)&(h%8==0))|(h%64==0)
          train=eligibility&(z['afold']<75)&((z['tfold']<75)|(z['tfold']==-1))&keep
          dev=(z['afold']>=75)&(z['afold']<80)&((z['tfold']>=75)|(z['tfold']==-1))&(z['tfold']<80)&((y==1)|(h%8==0))
          for key,mask in [('train',train),('dev',dev)]:
              x=lookup.features(z[mask],dense_route='dense_pairs' in str(file.parent)) if lookup else np.ascontiguousarray(z['x'][mask],dtype=np.float32);yy=y[mask]
              x.tofile(handles[key+'_x']);yy.tofile(handles[key+'_y']);stats[key]+=len(yy)
              if key=='train':
                  # Mild business balancing; full calibration will correct sampling prevalence.
                  w=(1/np.sqrt(np.maximum(1,counts[z['aid'][mask]]))).astype(np.float32)
                  w.tofile(handles['train_w']);stats['positive_train']+=int(yy.sum());stats['negative_train']+=int(len(yy)-yy.sum())
    for h in handles.values():h.close()
    print('SAMPLES',stats,'seconds',time.time()-t,flush=True)
    for key in ['train','dev']:
        X=np.memmap(out/(key+'_x.bin'),dtype=np.float32,mode='r',shape=(stats[key],nf))
        y=np.memmap(out/(key+'_y.bin'),dtype=np.uint8,mode='r')
        w=np.memmap(out/'train_w.bin',dtype=np.float32,mode='r') if key=='train' else None
        pool=Pool(X,label=y,weight=w,thread_count=a.threads)
        if key=='train':
            pool.quantize(border_count=96)
            pool.save_quantization_borders(str(out/'borders.tsv'))
        else:
            pool.quantize(input_borders=str(out/'borders.tsv'))
        pool.save(str(out/(key+'.quantized')))
        del X,y,w,pool;gc.collect();print('QUANTIZED',key,flush=True)
    train=Pool('quantized://'+str((out/'train.quantized').resolve()));dev=Pool('quantized://'+str((out/'dev.quantized').resolve()))
    model=CatBoostClassifier(iterations=a.iterations,depth=7,learning_rate=.06,l2_leaf_reg=8,loss_function='Logloss',eval_metric='Logloss',random_seed=42,thread_count=a.threads,verbose=100,allow_writing_files=False)
    model.fit(train,eval_set=dev,early_stopping_rounds=100)
    model.save_model(str(out/'model.cbm'))
    importance=model.get_feature_importance();names=json.loads((Path(a.pairs[0])/'features.json').read_text())
    if lookup:names+=['multilingual_cosine','dense_retrieval_route']
    stats.update({'trees':model.tree_count_,'seconds':time.time()-t,'features':names,'feature_importance':sorted(zip(names,map(float,importance)),key=lambda z:-z[1]),'evaluation_scope':'Internal training development (folds 75-79), not final macro validation'})
    (out/'training.json').write_text(json.dumps(stats,indent=2))
    print('TRAIN_COMPLETE',stats['trees'],stats['seconds'],flush=True)
if __name__=='__main__':main()
