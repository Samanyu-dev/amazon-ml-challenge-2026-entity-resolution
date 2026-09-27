"""Dump raw-text samples of validation errors (folds 90-99, leak-free stage 2)."""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np, pandas as pd
from decode import load, decode
R = '/Users/apple/Downloads/student_resource/dataset/train/'
S = W + 'artifacts_v3/exp_owner_split/'
d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid, q, mg = d['aid'].astype(np.int64), d['q'].astype(np.float64), d['margin']
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy')
m = decode(aid, q, mg, **k)
rd = lambda f: pd.read_csv(R + f, sep='\t', dtype=str, quoting=3, keep_default_na=False)
s1 = rd('train_source1.tsv'); tg = pd.concat([rd('train_source2.tsv'), rd('train_source3.tsv')], ignore_index=True)
assert len(tg) == len(aid), (len(tg), len(aid))
ids1 = pd.read_csv(W + 'artifacts/train_s1.norm.tsv', sep='\t', header=None, usecols=[0], quoting=3)[0].values
assert (ids1[:5] == s1.entity_id.values[:5]).all()
def rec(df, i): r = df.iloc[i]; return f'{r.business_name} | {r.business_address} [{r.country}]'
rng = np.random.default_rng(1)
def show(title, rows, n=40):
    print(f'\n######## {title}  (total {len(rows):,})')
    for t in rng.choice(rows, min(n, len(rows)), replace=False):
        o, a = own[t], aid[t]
        print(f'-- q={q[t]:.3f} margin={mg[t]:.3f} linked={m[t]}')
        print(f'   TARGET : {rec(tg, t)}')
        if a >= 0: print(f'   TOP    : {rec(s1, a)}')
        if o >= 0 and o != a: print(f'   OWNER  : {rec(s1, o)}')
fa = np.where(aid >= 0, folds[np.maximum(aid, 0)], -1); fo = np.where(own >= 0, folds[np.maximum(own, 0)], -1)
show('FALSE LINKS (linked, wrong)', np.flatnonzero(m & (aid != own) & (fa >= 90)))
show('RIGHT TOP BUT REJECTED', np.flatnonzero(~m & (aid == own) & (own >= 0) & (fo >= 90)))
show('WRONG ENTITY TOP (owner exists)', np.flatnonzero((aid != own) & (own >= 0) & (aid >= 0) & (fo >= 90)))
