"""Made-up-name rescue: address-only candidates + e5-small CE. Train folds 80-84, tune 85-89, report 90-99 (leak-free base).
With --apply: score test and save additions (unlinked targets only)."""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from catboost import CatBoostClassifier
from decode import load, decode, evaluate
FR = '/private/tmp/claude-501/fable_review/'
def table(sp, d, knobs):
    Z = np.load(FR + f'synth_block_{sp}.npz'); t, a, s = Z['tid'].astype(np.int64), Z['aid'].astype(np.int64), Z['sim']
    ce = np.load(FR + f'sy_ce_{sp}.npy').astype(np.float32)
    aid, q = d['aid'].astype(np.int64), d['q'].astype(np.float64); m = decode(aid, q, d['margin'], **knobs)
    ld = np.bincount(aid[m & (aid >= 0)], minlength=max(a.max(), aid.max()) + 1)
    st = np.r_[0, np.flatnonzero(np.diff(t)) + 1]; sz = np.diff(np.r_[st, len(t)]); g = np.repeat(np.arange(len(st)), sz)
    rank = np.arange(len(t)) - st[g]; smax = s[st][g]; s2 = np.where(sz > 1, s[np.minimum(st + 1, len(s) - 1)], 0)[g]
    cmax = np.maximum.reduceat(ce, st)[g]; o = np.lexsort((-ce, g)); cr = np.empty(len(ce)); cr[o] = np.arange(len(ce)) - st[g[o]]
    csum = np.add.reduceat(ce, st)[g]
    X = np.column_stack([s, rank, smax - s, smax - s2, ce, cmax - ce, cr, csum, q[t], (aid[t] == a), ld[a]]).astype(np.float32)
    return t, a, X, m, aid
S = W + 'artifacts_v3/exp_owner_split/'; d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
t, a, X, m, aid = table('train', d, k); y = (own[t] == a).astype(int); fa = folds[a]; elig = ~m[t]
tr = elig & (fa >= 80) & (fa < 85)
M = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, verbose=0, thread_count=6, random_seed=0).fit(X[tr], y[tr])
p = M.predict_proba(X)[:, 1]; p[~elig] = 0
from sklearn.metrics import roc_auc_score
ev = elig & (fa >= 90); print(f'pairs {len(t):,}; eval AUC {roc_auc_score(y[ev], p[ev]):.4f}; eval positives {y[ev].sum():,}; owner in cands (unlinked true targets) {np.isin(np.unique(t[(own[t] >= 0)]), t[y == 1]).mean():.3f}')
o = np.lexsort((-p, t)); first = np.r_[True, np.diff(t[o]) != 0]; bt, ba, bp = t[o][first], a[o][first], p[o][first]
def apply(th):
    na, nm = aid.copy(), m.copy(); s_ = bp >= th; na[bt[s_]] = ba[s_]; nm[bt[s_]] = True; return nm, na
b_t, _ = evaluate(m, aid, own, folds, truth, 85, 90); b_r, f0 = evaluate(m, aid, own, folds, truth, 90, 100)
best = None
for th in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
    nm, na = apply(th); r, _ = evaluate(nm, na, own, folds, truth, 85, 90); g = r['macro_f05'] - b_t['macro_f05']
    sel = (bp >= th) & (folds[ba] >= 85) & (folds[ba] < 90); print(f'  th {th} tune gain {g:+.6f}  added(tune) {sel.sum()} precision {np.mean(own[bt[sel]] == ba[sel]):.4f}')
    if best is None or g > best[1]: best = (th, g)
nm, na = apply(best[0]); r, f1 = evaluate(nm, na, own, folds, truth, 90, 100); g = f1 - f0
sel = (bp >= best[0]) & (folds[ba] >= 90)
print(f'REPORT th {best[0]}: {b_r["macro_f05"]:.6f} -> {r["macro_f05"]:.6f} gain {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}; added {sel.sum()} precision {np.mean(own[bt[sel]] == ba[sel]):.4f}')
if '--apply' in sys.argv:
    ctry = np.load(W + 'artifacts/test_anchor_countries.npy', allow_pickle=True); odds = lambda q, r: q * r / (q * r + 1 - q)
    out = []
    for S2, r_, cs in (('stage2_v7e_usin', 1.0, ('US', 'India')), ('stage2_v6_pp', 0.5, ('France',))):
        dt = dict(np.load(W + f'artifacts_v3/{S2}/test_decisions.npz')); dt['q'] = odds(dt['q'].astype(np.float64), r_)
        kt = json.load(open(W + f'artifacts_v3/{S2}/stage2_report.json'))['knobs']
        tt, at, Xt, mt, _ = table('test', dt, kt); pt = M.predict_proba(Xt)[:, 1]; pt[mt[tt]] = 0
        o = np.lexsort((-pt, tt)); f_ = np.r_[True, np.diff(tt[o]) != 0]; bt_, ba_, bp_ = tt[o][f_], at[o][f_], pt[o][f_]
        for c in cs:
            s_ = (bp_ >= best[0]) & (ctry[ba_] == c); out.append(np.column_stack([bt_[s_], ba_[s_]]))
            print(f'test {c}: additions {s_.sum():,} (mean p {bp_[s_].mean():.3f})')
    np.save(FR + 'sy_add.npy', np.concatenate(out))
