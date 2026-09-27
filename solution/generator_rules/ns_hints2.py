"""Namesake tie-break hints for blank-address copies (val misses where owner has exact-core-name namesakes)."""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode
from fingerprints import rd
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True); s1 = rd('train/train_source1.tsv')
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
core1 = pd.read_csv(W + 'artifacts/train_s1.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[3]).core.values
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
fo = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.bincount(own[own >= 0], minlength=len(s1))
idn = s1.entity_id.str.slice(3).astype(np.int64).values
grp = pd.Series(np.arange(len(s1))).groupby(core1).apply(np.array).to_dict()
linked = np.bincount(aid[m & (aid >= 0)], minlength=len(s1))
ae = tg.business_address.values == ''
u = np.flatnonzero((own >= 0) & ae & ~(m & (aid == own)) & (fo[np.maximum(own, 0)] >= 90))
res = {h: [] for h in ['first row', 'last row', 'smallest id', 'largest id', 'most linked', 'fewest linked', 'most TRUE copies (oracle-ish)', 'random']}
for t in u:
    g = grp[core1[own[t]]]
    if len(g) < 2: continue
    o = own[t]
    def pick(vals, fn):
        b = g[vals == fn(vals)]; return (o in b) / len(b)
    res['first row'].append(pick(g, np.min)); res['last row'].append(pick(g, np.max))
    res['smallest id'].append(pick(idn[g], np.min)); res['largest id'].append(pick(idn[g], np.max))
    res['most linked'].append(pick(linked[g], np.max)); res['fewest linked'].append(pick(linked[g], np.min))
    res['most TRUE copies (oracle-ish)'].append(pick(truth[g], np.max)); res['random'].append(1 / len(g))
for h, v in res.items(): print(f'{h:30s} owner hit {np.mean(v):.3f}  (n={len(v):,})')
# group-size distribution & source
print('group size median', np.median([len(grp[core1[own[t]]]) for t in u]))
