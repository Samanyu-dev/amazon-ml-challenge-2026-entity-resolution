"""Rescue model for Indian-script targets via address-number blocking + e5-small CE.
Train on candidate-anchor folds 80-84, tune threshold on 85-89, report paired gain on 90-99 vs the leak-free baseline.
Only adds links for targets the baseline leaves unlinked."""
import sys, json, os
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from catboost import CatBoostClassifier
from decode import load, decode, evaluate
FR = '/private/tmp/claude-501/fable_review/'
BLOCK = os.environ.get('BLOCK', 'addr_block'); TAG = os.environ.get('TAG', '')
def table(sp, ce_files, d, knobs):
    Z = np.load(FR + f'{BLOCK}_{sp}.npz'); t, a = Z['tid'].astype(np.int64), Z['aid'].astype(np.int64)
    ce = [np.load(f).astype(np.float32) for f in ce_files]
    aid, q = d['aid'].astype(np.int64), d['q'].astype(np.float64); m = decode(aid, q, d['margin'], **knobs)
    load_ = np.bincount(aid[m & (aid >= 0)], minlength=a.max() + 1)
    st = np.r_[0, np.flatnonzero(np.diff(t)) + 1]; sz = np.diff(np.r_[st, len(t)]); g = np.repeat(np.arange(len(st)), sz)
    rank = np.arange(len(t)) - st[g]
    cols = [Z['sim'], Z['ov'], Z['ncand'], rank]
    for c in ce:
        mx = np.maximum.reduceat(c, st)[g]; cols += [c, mx - c]
        o = np.lexsort((-c, g)); cr = np.empty(len(c)); cr[o] = np.arange(len(c)) - st[g[o]]; cols.append(cr)
    cols += [q[t], (aid[t] == a).astype(float), m[t].astype(float), load_[a]]
    return t, a, np.column_stack(cols).astype(np.float32), m, aid
if __name__ == '__main__':
    S = W + 'artifacts_v3/exp_owner_split/'; d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
    _, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
    folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
    ce_files = sys.argv[1:] or [FR + 'ab_ce_train.npy']
    t, a, X, m, aid = table('train', ce_files, d, k); y = (own[t] == a).astype(int); fa = folds[a]
    elig = ~m[t]                                   # only targets the baseline leaves unlinked
    tr = elig & (fa >= 80) & (fa < 85)
    M = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, verbose=0, thread_count=4, random_seed=0).fit(X[tr], y[tr])
    p = M.predict_proba(X)[:, 1]; p[~elig] = 0
    from sklearn.metrics import roc_auc_score
    ev = elig & (fa >= 90); print(f'pairs {len(t):,} eligible {elig.sum():,}; eval AUC {roc_auc_score(y[ev], p[ev]):.4f}; positives in eval {y[ev].sum():,}')
    # best candidate per target
    o = np.lexsort((-p, t)); first = np.r_[True, np.diff(t[o]) != 0]; bt, ba, bp = t[o][first], a[o][first], p[o][first]
    def apply(th):
        na, nm = aid.copy(), m.copy(); s = bp >= th; na[bt[s]] = ba[s]; nm[bt[s]] = True; return nm, na
    base_t, _ = evaluate(m, aid, own, folds, truth, 85, 90); base_r, f0 = evaluate(m, aid, own, folds, truth, 90, 100)
    best = None
    for th in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97):
        nm, na = apply(th); r, _ = evaluate(nm, na, own, folds, truth, 85, 90); g = r['macro_f05'] - base_t['macro_f05']
        print(f'  th {th:.2f} tune gain {g:+.6f} added {int((bp >= th).sum())}')
        if best is None or g > best[1]: best = (th, g)
    nm, na = apply(best[0]); r, f1 = evaluate(nm, na, own, folds, truth, 90, 100); g = f1 - f0
    print(f'REPORT th {best[0]}: {base_r["macro_f05"]:.6f} -> {r["macro_f05"]:.6f}  gain {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}  P {r["precision"]:.5f}')
    M.save_model(FR + f'rescue{TAG}.cbm'); json.dump({'th': best[0], 'ce': ce_files}, open(FR + f'rescue{TAG}.json', 'w'))
