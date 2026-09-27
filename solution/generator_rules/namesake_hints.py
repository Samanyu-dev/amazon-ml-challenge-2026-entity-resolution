import sys, re, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode
from fingerprints import rd
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True); s1 = rd('train/train_source1.tsv')
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
t = np.flatnonzero(own >= 0); o = own[t]
# 1) ID / row-order leakage
tid_num = tg.entity_id.str.slice(3).astype(np.int64).values; s1_num = s1.entity_id.str.slice(3).astype(np.int64).values
print('corr(target row, owner row):', np.corrcoef(t, o)[0, 1].round(4), ' corr(target id, owner id):', np.corrcoef(tid_num[t], s1_num[o])[0, 1].round(4))
print('  same last 3 digits:', np.mean(tid_num[t] % 1000 == s1_num[o] % 1000).round(4), ' (chance 0.001)')
r = np.random.default_rng(0).permutation(len(t))[:200000]
print('  id difference spread (should be huge if random):', np.percentile(np.abs(tid_num[t][r] - s1_num[o][r]), [1, 50, 99]))
# 2) address-empty copies: how many per owner, per source
ae = (tg.business_address.values == '')
src3 = np.arange(len(tg)) >= 5034616
cnt = pd.Series(o[ae[t]]).value_counts(); print('address-empty copies per owner:', cnt.value_counts().sort_index().to_dict())
cnt2 = pd.Series(list(zip(o[ae[t]], src3[t][ae[t]]))).value_counts(); print('  per owner per source:', cnt2.value_counts().sort_index().to_dict())
# 3) copy-count balance among exact-name namesakes (normalised core name)
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
core1 = pd.read_csv(W + 'artifacts/train_s1.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[3]).core.values
S = W + 'artifacts_v3/exp_owner_split/'; d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
linked = np.bincount(aid[m & (aid >= 0)], minlength=len(s1)); truec = np.bincount(o, minlength=len(s1))
grp = pd.Series(np.arange(len(s1))).groupby(core1).apply(np.array); big = {c: g for c, g in grp.items() if len(g) >= 2}
fo = np.load(W + 'artifacts/train_anchor_folds.npy')
u = t[ae[t] & ~m[t] & (fo[o] >= 90)]
hit = tot = rnd = 0; ties = 0
for x in u:
    g = big.get(core1[own[x]])
    if g is None: continue
    lk = linked[g]; best = g[lk == lk.min()]
    tot += 1; rnd += 1 / len(g); hit += (own[x] in best) / len(best); ties += len(best) > 1
print(f'unlinked address-empty val copies whose owner has exact-name namesakes: {tot:,} (of {len(u):,})')
print(f'  pick namesake with FEWEST linked copies -> owner {hit / max(tot, 1):.3f}   random pick {rnd / max(tot, 1):.3f}   (ties {ties / max(tot, 1):.2f})')
