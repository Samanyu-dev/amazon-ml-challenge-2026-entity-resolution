"""Independently check the emitted ID mapping against raw IDs and model decisions."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from score_pairs import BD

def key(value):
 source,number=value.split('-')
 if source not in ('S2','S3') or not number.isdigit():raise ValueError('Invalid target ID')
 n=int(number)
 if n>=2**60:raise ValueError('Target ID too large')
 return n+(int(source[1])<<60)

def main():
 p=argparse.ArgumentParser();p.add_argument('--matching',required=True);p.add_argument('--scores',required=True);p.add_argument('--probabilities',required=True);p.add_argument('--selection',required=True);p.add_argument('--test-dir',required=True);p.add_argument('--report',required=True);a=p.parse_args()
 test=Path(a.test_dir);meta=json.loads((Path(a.scores)/'complete.json').read_text());sel=json.loads(Path(a.selection).read_text())
 b=np.memmap(Path(a.scores)/'best.bin',dtype=BD,mode='r',shape=(meta['target_universe'],));prob=np.load(a.probabilities,mmap_mode='r')
 expected=(b['aid']>=0)&(prob>=sel['threshold'])&((b['p']-b['p2'])>=sel['minimum_raw_margin'])
 def source_ids():
  for source in (2,3):
   with (test/f'test_source{source}.tsv').open() as f:
    next(f)
    for line in f:yield key(line.split('\t',1)[0])
 keys=np.fromiter(source_ids(),dtype=np.uint64,count=len(b));order=np.argsort(keys);sorted_keys=keys[order]
 assert np.all(sorted_keys[1:]>sorted_keys[:-1]),'Raw target IDs not unique'
 seen=np.zeros(len(b),dtype=bool);stats=collections.Counter();countries=collections.defaultdict(collections.Counter);h=hashlib.sha256()
 with open(a.matching,'rb') as f,(test/'test_source1.tsv').open() as raw:
  header=f.readline();h.update(header);assert header==b'source1_entity_id\tmatched_entity_ids\n'
  raw_header=next(raw).rstrip('\n').split('\t');ci=raw_header.index('country')
  for aid,line in enumerate(raw):
   fields=line.rstrip('\n').split('\t');country=fields[ci]
   output=f.readline();h.update(output)
   assert output.endswith(b'\n') and output.count(b'\t')==1
   assert b'\r' not in output and b'\x00' not in output
   eid,vals=output[:-1].decode('ascii').split('\t');assert eid==fields[0],'S1 coverage/order mismatch'
   mids=vals.split(',') if vals else []
   assert len(mids)==len(set(mids)),'Duplicate list entries'
   if mids:
    encoded=np.array([key(v) for v in mids],dtype=np.uint64);pos=np.searchsorted(sorted_keys,encoded)
    assert np.all(pos<len(sorted_keys)) and np.array_equal(sorted_keys[pos],encoded),'Unknown target IDs'
    tid=order[pos];assert not seen[tid].any(),'Target assigned more than once'
    seen[tid]=True
    stats['wrong_model_decisions']+=int(np.count_nonzero(~expected[tid]|(b['aid'][tid]!=aid)))
   stats['rows']+=1;stats['links']+=len(mids);stats['empty_rows']+=not mids
   countries[country]['rows']+=1;countries[country]['links']+=len(mids);countries[country]['empty_rows']+=not mids
   if aid%400000==0:print('AUDITED',aid,'/',meta['anchor_universe'],flush=True)
  assert f.read()==b'','Unexpected extra rows'
 stats['missing_model_decisions']=int(np.count_nonzero(expected&~seen))
 report={'format_and_raw_ids_pass':True,'model_decisions_pass':stats['wrong_model_decisions']==stats['missing_model_decisions']==0,'statistics':dict(stats),'by_country':{k:dict(v) for k,v in countries.items()},'sha256':h.hexdigest(),'bytes':Path(a.matching).stat().st_size,'probabilities':str(Path(a.probabilities).resolve()),'selection':sel['threshold']}
 Path(a.report).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
