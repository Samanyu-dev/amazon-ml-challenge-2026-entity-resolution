"""Apply the Indian-script rescue model to test (base = leak-free US/India stage 2); save additions for unlinked targets."""
import sys, json, os
sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np
from catboost import CatBoostClassifier
from rescue import table, W, FR, TAG
cfg = json.load(open(FR + f'rescue{TAG}.json')); M = CatBoostClassifier(); M.load_model(FR + f'rescue{TAG}.cbm')
S = W + 'artifacts_v3/stage2_v7e_usin_owner/'; d = np.load(S + 'test_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
t, a, X, m, aid = table('test', [FR + os.environ.get('CE_TEST', 'ab_ce_test.npy')], d, k)
p = M.predict_proba(X)[:, 1]; p[m[t]] = 0
o = np.lexsort((-p, t)); f = np.r_[True, np.diff(t[o]) != 0]; bt, ba, bp = t[o][f], a[o][f], p[o][f]
s = bp >= cfg['th']; np.save(FR + os.environ.get('OUT', 'ab_add.npy'), np.column_stack([bt[s], ba[s]]))
print(f'test Indic rescue: targets scored {len(bt):,}; additions {s.sum():,} (th {cfg["th"]}); share of scope {s.mean():.4f}')
