import sys, json, math
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np, pandas as pd
from collections import Counter
from decode import load, decode, evaluate
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z', 'legal']
rdn = lambda f: pd.read_csv(W + f'artifacts_v3/train_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[3, 4])
s1 = rdn('s1'); tg = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
jac = lambda a, b: len(a & b) / max(1, len(a | b))
# ---- 1. kill-switch on LINKED pairs
lk = np.flatnonzero(m & (aid >= 0)); kill = np.zeros(len(aid), bool)
for t in lk:
    ta, sa = set(tg.addr.values[t].split()), set(s1.addr.values[aid[t]].split())
    if len(ta) >= 3 and jac(ta, sa) >= 0.9 and jac(set(tg.core.values[t].split()), set(s1.core.values[aid[t]].split())) <= 0.2: kill[t] = True
tp = int((kill & (own == aid)).sum()); fp = int((kill & (own != aid)).sum())
print(f'1. KILL-SWITCH (address Jaccard>=0.9, name Jaccard<=0.2) on linked pairs: fires {kill.sum():,}: destroys TRUE {tp:,} vs kills FALSE {fp:,}  (ratio FP:TP = {fp / max(tp, 1):.3f})')
m2 = m & ~kill; r0, f0 = evaluate(m, aid, own, folds, truth, 90, 100); r1, f1 = evaluate(m2, aid, own, folds, truth, 90, 100); g = f1 - f0
print(f'   val effect {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}')
# ---- 2. rare-token anchoring on UNLINKED targets (top candidate)
df = Counter(w for x in s1.core.values for w in set(x.split())); N = len(s1)
idf = lambda w: math.log(N / (1 + df.get(w, 0)))
un = np.flatnonzero(~m & (aid >= 0)); rows = []
for t in un:
    sw = s1.core.values[aid[t]].split()
    if not sw: continue
    r = max(sw, key=idf)
    if r in set(tg.core.values[t].split()): rows.append((idf(r), own[t] == aid[t]))
R = pd.DataFrame(rows, columns=['idf', 'true'])
print(f'2. RARE-TOKEN ANCHOR on unlinked targets (S1 rarest core word present in target): {len(R):,}')
for th in (6, 8, 10, 12, 13):
    s = R[R.idf >= th]; print(f'   IDF >= {th}: n {len(s):,} precision {s.true.mean() if len(s) else 0:.3f}')
