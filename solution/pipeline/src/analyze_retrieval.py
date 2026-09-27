from pathlib import Path
import argparse,json,collections
import numpy as np
from train_model import DT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pairs',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 files=sorted(Path(a.pairs).glob('pairs-*.bin')); rows=targets=truth=found=0; ranks=[]; countries=collections.defaultdict(lambda:[0,0]); counts=[]; scores=[]; missed=[]
 for p in files:
  with p.open('rb') as f:
   while len(z:=np.fromfile(f,dtype=DT,count=200000)):
    rows+=len(z); order=np.argsort(z['tid'],kind='stable');z=z[order]
    for tid,g in itertools.groupby(z, key=lambda x:int(x['tid'])):
     g=list(g); targets+=1; counts.append(len(g)); owner=int(g[0]['owner']); country=str(int(g[0]['tfold']))
     if owner>=0:
      truth+=1; hit=[i for i,x in enumerate(g) if int(x['aid'])==owner]
      if hit: found+=1; ranks.append(hit[0]+1)
      else: missed.append({'tid':tid,'owner':owner})
 report={'candidate_pairs':rows,'targets':targets,'truth_links':truth,'overall_recall':found/truth if truth else 0,
  'rank_recall':{str(k):sum(r<=k for r in ranks)/truth if truth else 0 for k in [1,5,10,20,50,100,200]},
  'rank_histogram':dict(collections.Counter(ranks)),'candidate_count':{'mean':float(np.mean(counts)) if counts else 0,'median':float(np.median(counts)) if counts else 0,'p95':float(np.percentile(counts,95)) if counts else 0,'max':max(counts) if counts else 0},'missed_examples':missed[:100]}
 Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 import itertools
 main()
