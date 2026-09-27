import sys, glob, re, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode
from fingerprints import rd
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
S = W + 'artifacts_v3/exp_owner_split/'; d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
folds = np.load(W + 'artifacts/train_anchor_folds.npy')
SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')]); has = np.zeros(len(own), bool)
for f in sorted(glob.glob(W + 'artifacts_v3/train_scores/candidates-*.bin')):
    c = np.fromfile(f, dtype=SF); ok = own[c['tid']] == c['aid'].astype(np.int64); has[c['tid'][ok]] = True
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
n = tg.business_name
pats = {'l_for_I': r'\bl[bcdfgjkmnpqrstvwxz]', '5_for_S': r'\b5[a-z]', '0_for_O': r'[a-zA-Z]0|0[a-zA-Z]', '1_for_l': r'[a-zA-Z]1[a-zA-Z]'}
v = (own >= 0) & (folds[np.maximum(own, 0)] >= 90)
good = m & (aid == own)
base = v & ~np.any([n.str.contains(p, regex=True).values for p in pats.values()], axis=0)
print(f'clean     n={base.sum():7,d} recall-in-cands {has[base].mean():.4f} linked-correct {good[base].mean():.4f}')
for name, p in pats.items():
    s = v & n.str.contains(p, regex=True).values
    print(f'{name:9s} n={s.sum():7,d} recall-in-cands {has[s].mean():.4f} linked-correct {good[s].mean():.4f}')
