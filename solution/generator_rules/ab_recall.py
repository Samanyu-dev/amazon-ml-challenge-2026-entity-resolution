import sys, glob, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from decode import load
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
fo = np.load(W + 'artifacts/train_anchor_folds.npy')
SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')]); has = np.zeros(len(own), bool)
for f in sorted(glob.glob(W + 'artifacts_v3/train_scores/candidates-*.bin')):
    c = np.fromfile(f, dtype=SF); ok = own[c['tid']] == c['aid'].astype(np.int64); has[c['tid'][ok]] = True
Z = np.load('/private/tmp/claude-501/fable_review/addr_block_train.npz'); t, a = Z['tid'], Z['aid']
hit = own[t] == a
T = np.unique(t); v = (own[T] >= 0) & (fo[np.maximum(own[T], 0)] >= 90)
found = np.zeros(len(own), bool); found[t[hit]] = True
print(f'scope targets {len(T):,}; val true {v.sum():,}; owner in new top-5 {found[T[v]].mean():.3f}')
miss = T[v & ~has[T]]
print(f'val blocking misses in scope {len(miss):,}; recovered by new top-5 {found[miss].mean():.3f} ({found[miss].sum():,})')
st = np.r_[0, np.flatnonzero(np.diff(t)) + 1]; rank = np.arange(len(t)) - np.repeat(st, np.diff(np.r_[st, len(t)]))
mm = hit & np.isin(t, miss); print('rank of recovered owner:', np.bincount(rank[mm], minlength=5))
print('median sim rank0 hit vs non-hit:', np.median(Z['sim'][(rank == 0) & hit]), np.median(Z['sim'][(rank == 0) & ~hit]))
