"""Gate: do target raw-text fingerprints add signal beyond stage-2 q? Meta CatBoost on folds 80-89 (out-of-sample
for stage 2), eval 90-99. Control = same model on [q, margin] only."""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from catboost import CatBoostClassifier
from decode import load, decode, evaluate
S = W + 'artifacts_v3/exp_owner_split/'
d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid, q, mg = d['aid'].astype(np.int64), d['q'].astype(np.float64), d['margin'].astype(np.float64)
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
ctry = np.load(W + 'artifacts/train_anchor_countries.npy', allow_pickle=True)
FP = np.load('/private/tmp/claude-501/fable_review/train_fp.npy').astype(np.float32)
fa = np.where(aid >= 0, folds[np.maximum(aid, 0)], -1)
band = (aid >= 0) & (q > 0.003) & (q < 0.998)
y = (own == aid).astype(int); lg = np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
tr = band & (fa >= 80) & (fa < 90); ev = band & (fa >= 90)
print(f'band rows: train {tr.sum():,} eval {ev.sum():,}')
def fit(X):
    m = CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, verbose=0, thread_count=8, random_seed=0)
    m.fit(X[tr], y[tr]); v = q.copy(); v[band] = m.predict_proba(X[band])[:, 1]; return v, m
base = np.column_stack([lg, mg])
v0, _ = fit(base); v1, m1 = fit(np.column_stack([base, FP]))
from sklearn.metrics import roc_auc_score
for n, v in (('q', q), ('ctrl', v0), ('fp', v1)): print(f'{n:5s} eval AUC {roc_auc_score(y[ev], v[ev]):.5f}')
r = {}
for n, v in (('q', q), ('ctrl', v0), ('fp', v1)):
    res, f = evaluate(decode(aid, v, mg, **k), aid, own, folds, truth, 90, 100, ctry); r[n] = f
    print(f'{n:5s} F {res["macro_f05"]:.6f} P {res["precision"]:.5f} R {res["recall"]:.5f} {res["by_country"]}')
for a, b in (('ctrl', 'q'), ('fp', 'ctrl')):
    g = r[a] - r[b]; print(f'{a} vs {b}: {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}')
print(dict(zip(['lg', 'mg'] + ['addr_empty','addr_upper','addr_hash','addr_null','addr_numletter','addr_www','addr_nonlatin','name_bracket','name_paren','name_dblspace','name_phone','name_www','name_upper','name_lower','name_accent','name_nonlatin','name_hyphen_glue','name_prefix','name_suffix_word','name_one_word','name_punct_end'], np.round(m1.get_feature_importance(), 2))))
