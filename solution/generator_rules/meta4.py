"""meta4 = meta3 features + query-relative competition features from stage-2 pair probs; gate on folds 90-99 (and wider fit 75-89)."""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from catboost import CatBoostClassifier
from decode import load, decode, evaluate
def qrel(pp, n):
    t, p = pp['tid'].astype(np.int64), pp['p'].astype(np.float64); o = np.lexsort((-p, t)); t, p = t[o], p[o]
    st = np.r_[0, np.flatnonzero(np.diff(t)) + 1]; sz = np.diff(np.r_[st, len(t)]); g = np.repeat(np.arange(len(st)), sz)
    p1 = p[st]; p2 = np.where(sz > 1, p[np.minimum(st + 1, len(p) - 1)], 0); p3 = np.where(sz > 2, p[np.minimum(st + 2, len(p) - 1)], 0)
    within = lambda d: np.bincount(g, weights=(p >= p1[g] - d).astype(float))
    tot = np.add.reduceat(p, st); pn = p / np.maximum(tot[g], 1e-9); ent = -np.add.reduceat(pn * np.log(np.clip(pn, 1e-12, 1)), st)
    eff = 1 / np.maximum(np.add.reduceat(pn ** 2, st), 1e-9)
    Q = np.full((n, 11), -1, np.float32)
    Q[t[st]] = np.column_stack([p1 - p2, p2 - p3, p1 - p3, within(.01), within(.03), within(.05), p2 / np.maximum(tot, 1e-9), (p2 + p3) / np.maximum(tot, 1e-9), ent / np.log(np.maximum(sz, 2)), eff, sz])
    return Q
S = W + 'artifacts_v3/stage2_v7e_usin_owner/'
d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid, q, mg = d['aid'].astype(np.int64), d['q'].astype(np.float64), d['margin'].astype(np.float64)
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
X3 = np.load('/private/tmp/claude-501/fable_review/meta3_X_train.npy'); nx2 = X3.shape[1] - 8
c3 = [c for c in range(nx2) if c != 16] + list(range(nx2, X3.shape[1]))
X = np.column_stack([X3, qrel(np.load(S + 'train_pair_probs.npz'), len(aid))]); c4 = c3 + list(range(X3.shape[1], X.shape[1]))
band = (aid >= 0) & (q > .003) & (q < .998); fa = np.where(aid >= 0, folds[np.maximum(aid, 0)], -1); y = (own == aid).astype(int)
res = {}
for name, cols, lo in (('meta3 (fit 80-89)', c3, 80), ('meta4 (fit 80-89)', c4, 80), ('meta4 (fit 75-89)', c4, 75)):
    tr = band & (fa >= lo) & (fa < 90)
    M = CatBoostClassifier(iterations=800, depth=5, learning_rate=0.05, verbose=0, thread_count=5, random_seed=0).fit(X[tr][:, cols], y[tr])
    v = q.copy(); v[band] = M.predict_proba(X[band][:, cols])[:, 1]
    r, f = evaluate(decode(aid, v, mg, **k), aid, own, folds, truth, 90, 100); res[name] = f
    print(f'{name:20s} val F {r["macro_f05"]:.6f} P {r["precision"]:.5f} R {r["recall"]:.5f}', flush=True)
    if name.startswith('meta4 (fit 75'): M.save_model('/private/tmp/claude-501/fable_review/meta4.cbm')
b = res['meta3 (fit 80-89)']
for nm in ('meta4 (fit 80-89)', 'meta4 (fit 75-89)'):
    g = res[nm] - b; print(f'  {nm} vs meta3: {g.mean():+.6f} ± {g.std() / np.sqrt(len(g)):.1e}')
