from pathlib import Path
import argparse,json,zipfile,pickle
import numpy as np
from score_pairs import BD
SF=np.dtype([('tid','<u4'),('aid','<u4'),('p','<f4')])

def verify_probabilities(best, probabilities, calibrator, chunk=200000):
 from calibrate import calibration_features
 if probabilities.shape != (len(best),):
  raise ValueError('Probability shape does not match scored targets')
 for start in range(0,len(best),chunk):
  z=best[start:start+chunk]
  expected=calibrator.predict_proba(calibration_features(z))[:,1].astype(np.float32)
  expected[z['aid']<0]=0
  actual=probabilities[start:start+len(z)]
  if not np.isfinite(actual).all() or not np.allclose(actual,expected,rtol=0,atol=1e-7):
   raise ValueError(f'Probability/model-calibrator mismatch at target batch {start}')

def ids(path):
 out=[]
 with Path(path).open() as f:
  next(f)
  for line in f:
   out.append(line.split('\t',1)[0])
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scores',required=True);ap.add_argument('--artifacts',required=True);ap.add_argument('--out',required=True);ap.add_argument('--threshold',type=float,required=True);ap.add_argument('--margin',type=float,required=True);ap.add_argument('--pair-scores',required=True);ap.add_argument('--probabilities',required=True);ap.add_argument('--calibrator',required=True);ap.add_argument('--test-dir',required=True);a=ap.parse_args()
 art=Path(a.artifacts);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if (out/'matching_results.tsv').exists():raise ValueError('Refusing to overwrite an existing submission')
 meta=json.loads((Path(a.scores)/'complete.json').read_text());b=np.memmap(Path(a.scores)/'best.bin',mode='r',dtype=BD,shape=(meta['target_universe'],));prob=np.load(a.probabilities,mmap_mode='r')
 with open(a.calibrator,'rb') as f:cal=pickle.load(f)
 verify_probabilities(b,prob,cal)
 print('PROBABILITY_LINEAGE_VERIFIED',len(b),flush=True)
 s1=ids(Path(a.test_dir)/'test_source1.tsv')
 targets=ids(Path(a.test_dir)/'test_source2.tsv')+ids(Path(a.test_dir)/'test_source3.tsv')
 if len(s1)!=meta['anchor_universe'] or len(targets)!=len(b):raise ValueError('Source cardinality mismatch')
 matches=[[] for _ in s1]
 for tid,row in enumerate(b):
  if row['aid']>=0 and prob[tid]>=a.threshold and float(row['p']-row['p2'])>=a.margin:matches[int(row['aid'])].append(targets[tid])
 with (out/'matching_results.tsv').open('w') as f:
  f.write('source1_entity_id\tmatched_entity_ids\n')
  for eid,vals in zip(s1,matches):f.write(eid+'\t'+','.join(dict.fromkeys(vals))+'\n')
 # Reconstruct the exact final candidate set from score_pairs' anchor buckets.
 # Each bucket contains every candidate row actually passed through the model.
 cand=[[] for _ in s1]
 for fp in sorted(Path(a.pair_scores).glob('candidates-*.bin')):
  z=np.fromfile(fp,dtype=SF)
  if not len(z):continue
  order=np.lexsort((z['tid'],z['aid']));z=z[order]
  keep=np.r_[True,(z['aid'][1:]!=z['aid'][:-1])|(z['tid'][1:]!=z['tid'][:-1])]
  z=z[keep];starts=np.r_[0,np.flatnonzero(np.diff(z['aid']))+1];ends=np.r_[starts[1:],len(z)]
  for st,en in zip(starts,ends):cand[int(z['aid'][st])].extend(targets[int(t)] for t in z['tid'][st:en])
 with (out/'candidate_pairs.tsv').open('w') as f:
  f.write('source1_entity_id\tcandidate_entity_ids\n')
  for eid,vals in zip(s1,cand):f.write(eid+'\t'+','.join(dict.fromkeys(vals))+'\n')
 from input_gate import digest
 (out/'submission_metadata.json').write_text(json.dumps({'threshold':a.threshold,'minimum_raw_margin':a.margin,'candidate_policy':'all_scored_candidates','matching_rows':len(s1),'scores':str(Path(a.scores).resolve()),'probabilities':str(Path(a.probabilities).resolve()),'calibrator':str(Path(a.calibrator).resolve()),'sha256':{k:digest(p) for k,p in {'best_scores':Path(a.scores)/'best.bin','probabilities':a.probabilities,'calibrator':a.calibrator,'model':meta['model'],'matching':out/'matching_results.tsv','candidates':out/'candidate_pairs.tsv'}.items()}},indent=2))
 zip_path=out/'amazon_ml_submission.zip'
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
  z.write(out/'matching_results.tsv','output/matching_results.tsv');z.write(out/'candidate_pairs.tsv','output/candidate_pairs.tsv')
 print('SUBMISSION_BUILT',zip_path,'rows',len(s1),'nonempty',sum(bool(x) for x in matches))
if __name__=='__main__':main()
