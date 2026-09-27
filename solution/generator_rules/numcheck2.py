"""Label-free precision proxy: for linked pairs where both addresses have a house number, how often does it match?
Train (true links vs false links) vs test per country. Also split by target-name operation."""
import sys, re, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode
from fingerprints import rd
def nums(x):
    x = re.sub(r'(?<=\d)[-/ ](?=\d)', '', x)
    return set(n.lstrip('0') for n in re.findall(r'\d+', x) if n.lstrip('0'))
def close(a, b):
    if a == b: return True
    if len(a) >= 2 and len(b) >= 2 and (a in b or b in a): return True
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1: return True
    return False
def stats(sp, aid, m, sel):
    s1 = rd(f'{sp}/{sp}_source1.tsv').business_address.values
    tg = pd.concat([rd(f'{sp}/{sp}_source2.tsv'), rd(f'{sp}/{sp}_source3.tsv')], ignore_index=True).business_address.values
    t = np.flatnonzero(m & (aid >= 0) & sel); r = np.zeros(len(t), np.int8)   # 0 no number, 1 share, 2 disjoint
    for i, x in enumerate(t):
        a, b = nums(tg[x]), nums(s1[aid[x]])
        r[i] = 0 if not a or not b else (1 if any(close(u, v) for u in a for v in b) else 2)
    return t, r
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
tc = np.load(W + 'artifacts/train_anchor_countries.npy', allow_pickle=True)
t, r = stats('train', aid, m, np.ones(len(aid), bool)); ok = own[t] == aid[t]
for c in ('US', 'India'):
    s = tc[aid[t]] == c
    print(f'TRAIN {c:6s} linked {s.sum():,}: TRUE links -> numbers disjoint {np.mean(r[s & ok & (r > 0)] == 2):.4f} | FALSE links -> disjoint {np.mean(r[s & ~ok & (r > 0)] == 2):.4f} | precision {ok[s].mean():.4f}')
    print(f'        if precision were P, disjoint rate = {np.mean(r[s & ok & (r > 0)] == 2):.4f}*P + {np.mean(r[s & ~ok & (r > 0)] == 2):.4f}*(1-P); observed {np.mean(r[s & (r > 0)] == 2):.4f}')
ctry = np.load(W + 'artifacts/test_anchor_countries.npy', allow_pickle=True); odds = lambda q, r: q * r / (q * r + 1 - q)
for S, rr, c in (('stage2_v7e_usin_owner_meta', 1, 'US'), ('stage2_v7e_usin_owner_meta', 1, 'India'), ('stage2_v6_pp', .5, 'France')):
    d = np.load(W + f'artifacts_v3/{S}/test_decisions.npz'); k = json.load(open(W + f'artifacts_v3/{S}/stage2_report.json'))['knobs']
    a2 = d['aid'].astype(np.int64); m2 = decode(a2, odds(d['q'].astype(np.float64), rr), d['margin'], **k)
    t2, r2 = stats('test', a2, m2, ctry[np.maximum(a2, 0)] == c)
    print(f'TEST  {c:6s} linked {len(t2):,}: numbers disjoint {np.mean(r2[r2 > 0] == 2):.4f}  (no-number share {np.mean(r2 == 0):.3f})')
    np.save(f'/private/tmp/claude-501/fable_review/numcheck2_{c}.npy', np.column_stack([t2, a2[t2], r2]))
