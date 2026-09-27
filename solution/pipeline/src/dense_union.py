"""Merge ANN pairs with lexical pairs, retaining only genuinely new candidates.

The output is sorted by global target id and is consumed by the forced-feature
stage.  It deliberately contains no labels or target metadata.
"""
from pathlib import Path
import argparse
import json
import numpy as np

PAIR = np.dtype([('tid','<u4'),('aid','<u4'),('owner','<i4'),('tfold','<i4'),('afold','<i4'),('x','<f4',(54,))])
ANN = np.dtype([('tid','<u4'),('aid','<u4'),('cos','<f4')])
OUT = np.dtype([('tid','<u4'),('aid','<u4'),('cos','<f4')])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--lex',required=True); ap.add_argument('--ann',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); lex=Path(a.lex); ann=Path(a.ann); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    seen=set(); nlex=0
    for p in sorted(lex.glob('pairs-*.bin')):
        z=np.memmap(p,mode='r',dtype=PAIR); nlex += len(z)
        # Pair key is compact and independent of model features.
        seen.update((int(t),int(x)) for t,x in zip(z['tid'],z['aid']))
    rows=[]; nann=0
    for p in sorted(ann.glob('s*.bin')):
        z=np.memmap(p,mode='r',dtype=ANN)
        for t,x,c in zip(z['tid'],z['aid'],z['cos']):
            nann += 1; k=(int(t),int(x))
            if k not in seen: rows.append((int(t),int(x),float(c)))
    arr=np.array(rows,dtype=OUT); arr.sort(order=['tid','aid'])
    tmp=Path(str(out)+'.tmp'); arr.tofile(tmp); tmp.replace(out)
    Path(str(out)+'.done').write_text(json.dumps({'lex_pairs':nlex,'ann_pairs':nann,'new_pairs':len(arr),'dtype':OUT.descr},indent=2))
    print(json.dumps({'lex_pairs':nlex,'ann_pairs':nann,'new_pairs':len(arr)}))
if __name__=='__main__': main()
