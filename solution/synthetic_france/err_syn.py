import sys, json, re
sys.path.insert(0, 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
exec(open('sweep_syn.py').read().split("print('current knobs'")[0])
raw = pd.concat([pd.read_csv(D + f'test/test_source{k}.tsv', sep='\t', dtype=str, quoting=3, keep_default_na=False) for k in (2, 3)], ignore_index=True)
import re
src = open('/private/tmp/claude-501/fable_review/synth_compare.py').read(); exec('SYL' + src.split('SYL', 1)[1].split('out = {}')[0])
OPS = ops(raw)
qq = q * 1.4 / (q * 1.4 + 1 - q); m = decode(aid, qq, mg, **k) & (aid >= 0)
miss = (own >= 0) & ~(m & (aid == own)); false = m & (aid != own)
print(f'synthetic France at x1.4: true copies {int((own >= 0).sum()):,}, missed {int(miss.sum()):,}, false links {int(false.sum()):,}')
rows = []
for name, f in OPS.items():
    f = f.fillna(False).values; rows.append((name, int((f & (own >= 0)).sum()), int((f & miss).sum()), (f & miss).sum() / max(1, (f & (own >= 0)).sum()), int((f & false).sum())))
for r in sorted(rows, key=lambda r: -r[2])[:14]: print(f'  {r[0]:32s} true {r[1]:7,d}  missed {r[2]:6,d} ({r[3]:.3f})  false links {r[4]:5,d}')
