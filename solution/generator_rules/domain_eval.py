import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from decode import load, decode, evaluate
Z = np.load('/private/tmp/claude-501/fable_review/domain_train.npz'); t, a, nc = Z['tid'], Z['aid'], Z['ncand']
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
ok = own[t] == a
print(f'rule fires {len(t):,}: precision {ok.mean():.4f} (unique key {ok[nc == 1].mean():.4f} n={int((nc == 1).sum()):,}; number tie-break {ok[nc > 1].mean() if (nc > 1).any() else 0:.4f} n={int((nc > 1).sum()):,})')
lk = m[t]; print(f'  already linked {lk.sum():,} (agree {np.mean(aid[t][lk] == a[lk]):.4f}); unlinked {(~lk).sum():,} precision {ok[~lk].mean():.4f}')
for nm, sel in (('unique only', ~lk & (nc == 1)), ('all', ~lk)):
    na, nm_ = aid.copy(), m.copy(); na[t[sel]] = a[sel]; nm_[t[sel]] = True
    r0, f0 = evaluate(m, aid, own, folds, truth, 90, 100); r1, f1 = evaluate(nm_, na, own, folds, truth, 90, 100); g = f1 - f0
    rt0, _ = evaluate(m, aid, own, folds, truth, 85, 90); rt1, _ = evaluate(nm_, na, own, folds, truth, 85, 90)
    print(f'  {nm:12s} val gain {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}   tune gain {rt1["macro_f05"] - rt0["macro_f05"]:+.6f}')
