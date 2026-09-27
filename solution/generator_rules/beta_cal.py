"""Beta calibration of stage-2 q (leak-free), knobs retuned on 85-89 for each variant, report on 90-99."""
import sys, json, itertools
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from sklearn.linear_model import LogisticRegression
from decode import load, decode, evaluate
S = W + 'artifacts_v3/' + (sys.argv[1] if len(sys.argv) > 1 else 'exp_owner_split') + '/'
d = np.load(S + 'train_decisions.npz'); k0 = json.load(open(S + 'stage2_report.json'))['knobs']
aid, q, mg = d['aid'].astype(np.int64), d['q'].astype(np.float64), d['margin']
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
fa = np.where(aid >= 0, folds[np.maximum(aid, 0)], -1)
band = (aid >= 0) & (q > 1e-4) & (q < 1 - 1e-4)
tune = band & (fa >= 85) & (fa < 90); y = (own == aid).astype(int)
qc = np.clip(q, 1e-6, 1 - 1e-6)
def feats(kind, x):
    if kind == 'platt': return np.log(x / (1 - x))[:, None]
    return np.column_stack([np.log(x), -np.log(1 - x)])
variants = {'raw': q}
for kind in ('platt', 'beta'):
    m = LogisticRegression(C=100, max_iter=1000).fit(feats(kind, qc[tune]), y[tune])
    v = q.copy(); v[band] = m.predict_proba(feats(kind, qc[band]))[:, 1]; variants[kind] = v
    print(kind, m.coef_, m.intercept_)
grid = list(itertools.product([0.05, 0.1, 0.2, 0.3], [3, 4, 6, 8, 10], [0.0, 0.01, 0.015, 0.02]))
res = {}
for name, v in variants.items():
    best = max(grid, key=lambda g: evaluate(decode(aid, v, mg, g[0], g[1], g[2]), aid, own, folds, truth, 85, 90)[0]['macro_f05'])
    rt, _ = evaluate(decode(aid, v, mg, *best), aid, own, folds, truth, 85, 90)
    r, f = evaluate(decode(aid, v, mg, *best), aid, own, folds, truth, 90, 100); res[name] = f
    print(f'{name:6s} knobs {best} tune {rt["macro_f05"]:.6f} | report {r["macro_f05"]:.6f} P {r["precision"]:.5f} R {r["recall"]:.5f}')
for n in ('platt', 'beta'):
    g = res[n] - res['raw']; print(f'{n} vs raw {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}')
g = res['beta'] - res['platt']; print(f'beta vs platt {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}')
