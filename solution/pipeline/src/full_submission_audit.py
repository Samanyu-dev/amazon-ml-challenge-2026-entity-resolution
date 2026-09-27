"""Exact set, byte lineage, source-order, and archive checks for corrected V2."""
from pathlib import Path
import argparse,json,zipfile,itertools,collections
import numpy as np
from train_model import DT
from score_pairs import SF,BD
from input_gate import digest,verify_generation
from audit_matching_content import key

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifacts',required=True);p.add_argument('--test-root',required=True);p.add_argument('--submission',required=True);p.add_argument('--report',required=True);a=p.parse_args()
 art=Path(a.artifacts);data=Path(a.test_root);out=Path(a.submission);report={}
 m=verify_generation(art/'generation.json');assert digest('src/normalization_v2.py')==m['normalization_code_sha256'];assert digest(art/'normalization_maps_policy.json')==m['normalization_config_sha256']
 manifest=json.loads((out/'submission_metadata.json').read_text())
 paths={'best_scores':art/'test_scores_dense/best.bin','probabilities':art/'test_calibration_dense/probabilities.npy','calibrator':art/'calibration_dense/calibrator.pkl','model':art/'model_dense_lexical/model.cbm','matching':out/'matching_results.tsv','candidates':out/'candidate_pairs.tsv'}
 for name,path in paths.items():assert digest(path)==manifest['sha256'][name],name
 report['submission_lineage_hashes']='PASS'
 raw=json.loads((art/'data_manifest.json').read_text())
 for table,info in raw.items():
  source=data/info['file'];assert digest(source)==info['sha256']
  rows=0
  with source.open() as f,(art/f'{table}.norm.tsv').open() as g:
   header=next(f).rstrip('\n').split('\t');ci=header.index('country')
   for left,right in itertools.zip_longest(f,g):
    assert left is not None and right is not None
    lf=left.rstrip('\n').split('\t');rf=right.rstrip('\n').split('\t')
    assert len(rf)==10 and lf[0]==rf[0] and lf[ci]==rf[1],(table,rows)
    rows+=1
  assert rows==info['rows'];print('SOURCE_ORDER_AND_HASH_PASS',table,rows,flush=True)
 report['raw_normalized_order_and_hash']='PASS';report['generation']=m['generation_id']
 meta=json.loads((art/'test_scores_dense/complete.json').read_text());n=meta['candidate_pairs']
 expected=np.empty(n,dtype=np.uint64);cursor=0
 for path in sorted((art/'test_pairs_audited').glob('pairs-*.bin')):
  assert path.stat().st_size%DT.itemsize==0
  with path.open('rb') as f:
   while len(z:=np.fromfile(f,dtype=DT,count=200000)):
    assert np.all(z['aid']<meta['anchor_universe']) and np.all(z['tid']<meta['target_universe'])
    expected[cursor:cursor+len(z)]=(z['aid'].astype(np.uint64)<<32)|z['tid'];cursor+=len(z)
 assert cursor==n;expected.sort();assert np.all(expected[1:]!=expected[:-1])
 actual=np.empty(n,dtype=np.uint64);cursor=0
 for path in sorted((art/'test_scores_dense').glob('candidates-*.bin')):
  with path.open('rb') as f:
   while len(z:=np.fromfile(f,dtype=SF,count=200000)):
    actual[cursor:cursor+len(z)]=(z['aid'].astype(np.uint64)<<32)|z['tid'];cursor+=len(z)
 assert cursor==n;actual.sort();assert np.array_equal(expected,actual)
 print('RETRIEVED_VS_SCORED_EXACT_SET_PASS',n,flush=True)
 def target_keys():
  for source in (2,3):
   with (data/f'test/test_source{source}.tsv').open() as f:
    next(f)
    for line in f:yield key(line.split('\t',1)[0])
 keys=np.fromiter(target_keys(),dtype=np.uint64,count=meta['target_universe']);order=np.argsort(keys);keys=keys[order]
 assert np.all(keys[1:]!=keys[:-1]);cursor=0;rows=0
 with (out/'candidate_pairs.tsv').open() as f,(data/'test/test_source1.tsv').open() as s1:
  assert next(f)=='source1_entity_id\tcandidate_entity_ids\n';next(s1)
  for aid,(line,ref) in enumerate(itertools.zip_longest(f,s1)):
   assert line is not None and ref is not None
   eid,vals=line.rstrip('\n').split('\t');assert eid==ref.split('\t',1)[0]
   ids=vals.split(',') if vals else [];encoded=np.fromiter((key(v) for v in ids),dtype=np.uint64,count=len(ids));pos=np.searchsorted(keys,encoded)
   assert np.all(pos<len(keys)) and np.array_equal(keys[pos],encoded)
   actual[cursor:cursor+len(ids)]=(np.uint64(aid)<<np.uint64(32))|order[pos].astype(np.uint64);cursor+=len(ids);rows+=1
   if rows%400000==0:print('CANDIDATE_TSV_ROWS',rows,flush=True)
 assert cursor==n and rows==meta['anchor_universe'];actual.sort();assert np.array_equal(expected,actual)
 report['candidate_exact_set']={'retrieval_rows':n,'scored_rows':n,'tsv_pairs':cursor,'tsv_anchor_rows':rows,'status':'PASS'}
 del actual,expected,keys,order
 original=np.memmap(art/'test_scores_dense/best.bin',dtype=BD,mode='r');repeat=np.memmap(art/'test_scores_dense_reaudit/best.bin',dtype=BD,mode='r')
 assert len(original)==len(repeat)==meta['target_universe']
 assert digest(art/'test_scores_dense/best.bin')==digest(art/'test_scores_dense_reaudit/best.bin')
 report['full_repeat_inference']='BYTE_IDENTICAL'
 hashes={name:digest(out/name) for name in ('matching_results.tsv','candidate_pairs.tsv')}
 with zipfile.ZipFile(out/'prediction_files_only.zip') as z:
  import hashlib
  for name,want in hashes.items():
   h=hashlib.sha256()
   with z.open('output/'+name) as f:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
   assert h.hexdigest()==want
 report['archive_contents_match']=True;report['sha256']=hashes
 Path(a.report).write_text(json.dumps(report,indent=2));print('FULL_AUDIT_PASS',json.dumps(report),flush=True)
if __name__=='__main__':main()
