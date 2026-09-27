import sys, json, itertools
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from decode import load, decode, evaluate
S = W + 'artifacts_v3/stage2_v7e_usin_owner/'
d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid, mg = d['aid'].astype(np.int64), d['margin']
v = np.load('/private/tmp/claude-501/fable_review/meta3_v_train.npy')
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
ev = lambda kk, lo, hi: evaluate(decode(aid, v, mg, **kk), aid, own, folds, truth, lo, hi)
b_t, _ = ev(k, 90, 95); b_r, f0 = ev(k, 95, 100); print('current knobs', k, f'tune {b_t["macro_f05"]:.6f} report {b_r["macro_f05"]:.6f}', flush=True)
best = (b_t['macro_f05'], k)
for miss, es, mm in itertools.product((0.05, 0.1, 0.25, 0.4, 0.6), (3, 4, 5, 6, 8), (0.0, 0.015, 0.03)):
    kk = dict(k); kk.update(missing=miss, empty_scale=es, min_margin=mm); r, _ = ev(kk, 90, 95)
    if r['macro_f05'] > best[0]: best = (r['macro_f05'], kk)
print('best on tune 90-94:', best, flush=True)
r, f1 = ev(best[1], 95, 100); g = f1 - f0
print(f'REPORT 95-99: {b_r["macro_f05"]:.6f} -> {r["macro_f05"]:.6f}  gain {g.mean():+.6f} ± {g.std() / np.sqrt(len(g)):.1e}')
