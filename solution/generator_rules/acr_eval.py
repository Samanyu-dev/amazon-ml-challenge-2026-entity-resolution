"""Gate for the acronym rule on train: precision (all folds, it is a fixed rule) and paired F gain on folds 90-99."""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from decode import load, decode, evaluate
Z = np.load('/private/tmp/claude-501/fable_review/acr_train.npz'); t, a, s, rank, match, gap = Z['tid'], Z['aid'], Z['sim'], Z['rank'], Z['match'], Z['gap']
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
ctry = np.load(W + 'artifacts/train_anchor_countries.npy', allow_pickle=True)
S = W + 'artifacts_v3/exp_owner_split/'; d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
T = np.unique(t); print(f'acronym targets {len(T):,}: true dup {np.mean(own[T] >= 0):.3f}; currently linked {m[T].mean():.3f}, linked correct {np.mean(m[T] & (aid[T] == own[T])):.3f}')
print(f'  owner among address top-10 {np.isin(T, t[own[t] == a]).mean():.3f}; owner initials == acronym {np.mean(match[own[t] == a]):.3f}')
nm = np.bincount(np.searchsorted(T, t[match]), minlength=len(T))   # matches per target
for name, rule in [(f'rank0 sim>={x} gap>={y}', (rank == 0) & (s >= x) & (gap >= y)) for x in (0.3, 0.5, 0.7) for y in (0.0, 0.1, 0.3)] + [('match any (first)', match)]:
    idx = np.flatnonzero(rule); _, f1 = np.unique(t[idx], return_index=True); idx = idx[f1]
    prec = np.mean(own[t[idx]] == a[idx]); cov = len(idx) / len(T)
    un = idx[~m[t[idx]]]
    na, nmk = aid.copy(), m.copy(); na[t[un]] = a[un]; nmk[t[un]] = True
    r0, f0 = evaluate(m, aid, own, folds, truth, 90, 100, ctry); r1, f1_ = evaluate(nmk, na, own, folds, truth, 90, 100, ctry); g = f1_ - f0
    print(f'{name:26s} fires {len(idx):6,d} ({cov:.3f}) precision {prec:.4f} | on unlinked: {len(un):5,d} precision {np.mean(own[t[un]] == a[un]) if len(un) else 0:.4f} | val gain {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}')
