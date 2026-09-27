import sys, glob
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load
from fingerprints import rd
aid, own, _, q, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy')
SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')])
has = np.zeros(len(own), bool)
for f in sorted(glob.glob(W + 'artifacts_v3/train_scores/candidates-*.bin')):
    c = np.fromfile(f, dtype=SF); ok = own[c['tid']] == c['aid'].astype(np.int64); has[c['tid'][ok]] = True
miss = np.flatnonzero((own >= 0) & ~has & (folds[np.maximum(own, 0)] >= 90))
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True); s1 = rd('train/train_source1.tsv')
print('blocking misses (val):', len(miss))
ae = (tg.business_address.values[miss] == ''); print('  address empty share', ae.mean())
rng = np.random.default_rng(3)
for t in rng.choice(miss, 60, replace=False):
    o = own[t]; r = tg.iloc[t]; s = s1.iloc[o]
    print(f'T: {r.business_name} | {r.business_address} [{r.country}]\n O: {s.business_name} | {s.business_address}')
    if aid[t] >= 0: x = s1.iloc[aid[t]]; print(f' TOP(q={q[t]:.2f}): {x.business_name} | {x.business_address}')

import re
nm = tg.business_name.values[miss]; ad = tg.business_address.values[miss]
nonlat = np.array([bool(re.search(r'[^\x00-\u024F]', x)) for x in nm])
print(f'\nblocking misses {len(miss)}: addr empty {ae.sum()}  | non-Latin name w/ addr {(nonlat & ~ae).sum()}  | Latin name w/ addr {(~nonlat & ~ae).sum()}')
dig = re.compile(r'\d+')
oa = s1.business_address.values[own[miss]]
share = np.array([len(set(dig.findall(a)) & set(dig.findall(b))) > 0 for a, b in zip(ad, oa)])
print(f'  non-Latin w/ addr sharing a number with owner: {(nonlat & ~ae & share).sum()}   Latin w/ addr sharing number: {(~nonlat & ~ae & share).sum()}')
# how many train targets overall are non-Latin with address, and their blocking recall
allt = np.flatnonzero((own >= 0) & (folds[np.maximum(own, 0)] >= 90))
nl = tg.business_name.str.contains(r'[^\x00-\u024F]', regex=True).values
e = tg.business_address.values == ''
for nmx, m in (('nonLatin+addr', nl & ~e), ('Latin+addr', ~nl & ~e), ('empty addr', e)):
    s = allt[m[allt]]; print(f'  {nmx:14s} true targets {len(s):7,d}  blocking recall {has[s].mean():.4f}')
