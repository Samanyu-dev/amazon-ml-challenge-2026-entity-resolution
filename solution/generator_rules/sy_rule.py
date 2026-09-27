"""Made-up-name rule: best address match (TF-IDF) is clear (sim>=X, gap to 2nd >= Y). Train gate + test application."""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from decode import load, decode, evaluate
FR = '/private/tmp/claude-501/fable_review/'
def feats(sp):
    Z = np.load(FR + f'synth_block_{sp}.npz'); t, a, s = Z['tid'], Z['aid'], Z['sim']
    st = np.r_[0, np.flatnonzero(np.diff(t)) + 1]; sz = np.diff(np.r_[st, len(t)])
    s2 = np.where(sz > 1, s[np.minimum(st + 1, len(s) - 1)], 0)
    return t[st], a[st], s[st], s[st] - s2
t, a, s, g = feats('train')
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
S = W + 'artifacts_v3/exp_owner_split/'; d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
r0, f0 = evaluate(m, aid, own, folds, truth, 90, 100)
best = None
for X in (0.5, 0.6, 0.7, 0.8):
    for Y in (0.1, 0.2, 0.3):
        f = (s >= X) & (g >= Y) & ~m[t]
        tune = f & (folds[a] >= 85) & (folds[a] < 90); prec_t = np.mean(own[t[tune]] == a[tune]) if tune.any() else 0
        na, nm = aid.copy(), m.copy(); na[t[f]] = a[f]; nm[t[f]] = True
        rt, _ = evaluate(nm, na, own, folds, truth, 85, 90); rt0, _ = evaluate(m, aid, own, folds, truth, 85, 90)
        r1, f1 = evaluate(nm, na, own, folds, truth, 90, 100); gg = f1 - f0; rep = f & (folds[a] >= 90)
        print(f'sim>={X} gap>={Y}: fires {f.sum():5,d} precision all-folds {np.mean(own[t[f]] == a[f]):.4f} | tune gain {rt["macro_f05"] - rt0["macro_f05"]:+.6f} | report gain {gg.mean():+.6f} ± {gg.std()/np.sqrt(len(gg)):.1e} (added {rep.sum()}, prec {np.mean(own[t[rep]] == a[rep]):.4f})')
        if best is None or rt['macro_f05'] - rt0['macro_f05'] > best[2]: best = (X, Y, rt['macro_f05'] - rt0['macro_f05'])
print('chosen on tune folds:', best)
X, Y, _ = best
tt, at, st_, gt = feats('test'); ctry = np.load(W + 'artifacts/test_anchor_countries.npy', allow_pickle=True)
odds = lambda q, r: q * r / (q * r + 1 - q); out = []
for S2, r_, cs in (('stage2_v7e_usin', 1.0, ('US', 'India')), ('stage2_v6_pp', 0.5, ('France',))):
    dt = np.load(W + f'artifacts_v3/{S2}/test_decisions.npz'); kt = json.load(open(W + f'artifacts_v3/{S2}/stage2_report.json'))['knobs']
    a2 = dt['aid'].astype(np.int64); m2 = decode(a2, odds(dt['q'].astype(np.float64), r_), dt['margin'], **kt)
    for c in cs:
        f = (st_ >= X) & (gt >= Y) & ~m2[tt] & (ctry[at] == c); out.append(np.column_stack([tt[f], at[f]]))
        print(f'test {c}: additions {f.sum():,}; rule pick == our stage-2 top anchor {np.mean(a2[tt[f]] == at[f]):.3f}')
np.save(FR + 'sy_rule_add.npy', np.concatenate(out))
