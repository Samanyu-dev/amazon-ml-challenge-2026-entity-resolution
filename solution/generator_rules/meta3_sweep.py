import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np
from catboost import CatBoostClassifier
from decode import load, decode, evaluate
S = W + 'artifacts_v3/stage2_v7e_usin_owner/'
d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid, q, mg = d['aid'].astype(np.int64), d['q'].astype(np.float64), d['margin'].astype(np.float64)
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
X = np.load('/private/tmp/claude-501/fable_review/meta3_X_train.npy'); nx2 = X.shape[1] - 8
cols = [c for c in range(nx2) if c != 2 + 14] + list(range(nx2, X.shape[1]))
band = (aid >= 0) & (q > .003) & (q < .998); fa = np.where(aid >= 0, folds[np.maximum(aid, 0)], -1); y = (own == aid).astype(int)
tr = band & (fa >= 80) & (fa < 90)
M = CatBoostClassifier(iterations=800, depth=5, learning_rate=0.05, verbose=0, thread_count=8, random_seed=0).fit(X[tr][:, cols], y[tr])
v = q.copy(); v[band] = M.predict_proba(X[band][:, cols])[:, 1]
r0, f0 = evaluate(decode(aid, v, mg, **k), aid, own, folds, truth, 90, 100); print(f'meta3 x1.0: {r0["macro_f05"]:.6f}')
for o in (0.5, 0.6, 0.7, 0.8, 0.9, 1.2):
    vv = v * o / (v * o + 1 - v); r, f = evaluate(decode(aid, vv, mg, **k), aid, own, folds, truth, 90, 100); g = f - f0
    print(f'  meta3 x{o}: {r["macro_f05"]:.6f}  vs x1.0 {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}')
