import sys, re, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np, pandas as pd
from decode import load, decode
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z', 'legal']
rdn = lambda f: pd.read_csv(W + f'artifacts_v3/train_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[1, 3, 4, 10])
s1 = rdn('s1'); tg = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
k1 = s1.ctry + '|' + s1.core; vc = k1.value_counts(); u = k1[k1.map(vc) == 1]; look = pd.Series(u.index.values, index=u.values)
j = (tg.ctry + '|' + tg.core).map(look).fillna(-1).astype(np.int64).values; j[tg.core.values == ''] = -1
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
def num(x):
    x = re.sub(r'(?<=\d)[-/ ](?=\d)', '', x); mm = re.search(r'\d+', x); return mm.group(0).lstrip('0') if mm else ''
res = []
for t in np.flatnonzero((j >= 0) & (tg.addr.values != '')):
    a, b = num(tg.addr.values[t]), num(s1.addr.values[j[t]])
    if not a or not b or a == b or not (a in b or b in a): continue
    lt, ls = tg.legal.values[t], s1.legal.values[j[t]]
    if lt != '' and lt != ls: continue
    res.append((t, own[t] == j[t], m[t] and aid[t] == j[t], m[t]))
r = pd.DataFrame(res, columns=['t', 'true', 'linked_there', 'linked_any'])
print(f'v9 rule on train: all pairs {len(r):,} true {r.true.mean():.4f} | we link them {r.linked_there.mean():.4f}')
un = r[~r.linked_any]; print(f'  RESIDUE (unlinked by our model): {len(un):,} -> true {un.true.mean():.4f}')
