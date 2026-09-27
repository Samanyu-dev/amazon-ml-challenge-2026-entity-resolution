"""Offline gates for the Qwen2.5-7B pair scores (no files for upload are written).

Gate 1: US/India. Fit a 3-parameter logistic fusion of stage-2 pair probability and LLM score on
        tuning folds 85-89 (labels known), apply to holdout folds 90-99, re-decide + decode,
        paired macro F0.5 vs the leak-free baseline (exp_owner_split). Bar: >= 0.9905, precision >= 0.993.
Gate 2: France. Distribution of LLM scores on uncertain French pairs, and how fused decisions differ
        from v7d's France (v6 stack, odds x0.5) - counts only, France has no labels.
"""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np
from sklearn.linear_model import LogisticRegression
from decode import load, decode, evaluate
from stack_ce import redecide
import os; L = os.environ.get('LLM_OUT', '/private/tmp/claude-501/llm_out/'); Bd = '/private/tmp/claude-501/llm_bundle/'
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def pair_p(pp, t, a):
    k = pp['tid'].astype(np.int64) << 32 | pp['aid'].astype(np.int64); o = np.argsort(k); ks = k[o]
    q = t.astype(np.int64) << 32 | a.astype(np.int64); pos = np.clip(np.searchsorted(ks, q), 0, len(ks) - 1)
    hit = ks[pos] == q; out = np.full(len(t), np.nan); out[hit] = pp['p'][o][pos[hit]]; return out

def fuse(pp, llm_pairs, llm_s, clf, use_llm=True):
    """replace stage-2 pair probabilities of LLM-scored pairs by the fused probability"""
    t, a = llm_pairs['tid'].astype(np.int64), llm_pairs['s1_row'].astype(np.int64)
    base = pair_p(pp, t, a); ok = ~np.isnan(base)
    X = np.column_stack([lg(base[ok]), lg(llm_s[ok])]) if use_llm else lg(base[ok])[:, None]; fused = clf.predict_proba(X)[:, 1]
    return t[ok], a[ok], base[ok], fused

def gate1():
    S = W + 'artifacts_v3/exp_owner_split/'
    d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']; pp = np.load(S + 'train_pair_probs.npz')
    _, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
    folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
    ctry = np.load(W + 'artifacts/train_anchor_countries.npy', allow_pickle=True)
    P = np.load(Bd + 'llm_infer_train.npz'); s = np.load(L + 'llm_infer_train_scores.npy').astype(np.float64)
    t, a = P['tid'].astype(np.int64), P['s1_row'].astype(np.int64); y = (own[t] == a).astype(int)
    base = pair_p(pp, t, a); ok = ~np.isnan(base); tune = ok & (folds[a] >= 85) & (folds[a] < 90)
    print(f'LLM-scored US/India pairs {len(t):,}; with stage-2 prob {ok.sum():,}; tuning rows {tune.sum():,}')
    from sklearn.metrics import roc_auc_score
    hv = ok & (folds[a] >= 90)
    print(f'   holdout pair AUC: stage-2 {roc_auc_score(y[hv], base[hv]):.4f}  LLM alone {roc_auc_score(y[hv], s[hv]):.4f}')
    clf = LogisticRegression(C=1.0).fit(np.column_stack([lg(base[tune]), lg(s[tune])]), y[tune])
    clf0 = LogisticRegression(C=1.0).fit(lg(base[tune])[:, None], y[tune])  # recalibration only (control)
    print(f'   fusion weights: stage2 {clf.coef_[0][0]:+.3f}  llm {clf.coef_[0][1]:+.3f}  bias {clf.intercept_[0]:+.3f}')
    base_mask = decode(d['aid'], d['q'].astype(np.float64), d['margin'], **k)
    r0, f0 = evaluate(base_mask, d['aid'], own, folds, truth, 90, 100, ctry)
    def run(c, use):
        ft, fa, _, fp = fuse(pp, P, s, c, use)
        pt, pa, pv = pp['tid'].astype(np.int64), pp['aid'].astype(np.int64), pp['p'].astype(np.float64).copy()
        k1 = pt << 32 | pa; o = np.argsort(k1); ks = k1[o]; pos = np.searchsorted(ks, ft << 32 | fa); pv[o[pos]] = fp
        a1, q1 = redecide(pt, pa, pv, d['aid'].astype(np.int64), d['q'].astype(np.float64))
        return evaluate(decode(a1, q1, d['margin'], **k), a1, own, folds, truth, 90, 100, ctry)
    rc, fc = run(clf0, False); r1, f1 = run(clf, True)
    g, gl = f1 - f0, f1 - fc
    print(f'GATE 1  baseline {r0["macro_f05"]:.6f} (P {r0["precision"]:.5f}) | recalibration-only control {rc["macro_f05"]:.6f} (P {rc["precision"]:.5f}) | with LLM {r1["macro_f05"]:.6f} (P {r1["precision"]:.5f}, R {r1["recall"]:.5f})')
    print(f'        LLM effect (vs control) {gl.mean():+.6f} ± {gl.std() / np.sqrt(len(gl)):.1e}   total vs baseline {g.mean():+.6f} ± {g.std() / np.sqrt(len(g)):.1e}   by country {r1["by_country"]}')
    print(f'        bar: macro >= 0.9905 and precision >= 0.993 -> {"PASS" if r1["macro_f05"] >= .9905 and r1["precision"] >= .993 else "FAIL"}')
    return clf, clf0

def gate2(clf, clf0):
    P = np.load(Bd + 'llm_infer_test_fr.npz'); s = np.load(L + 'llm_infer_test_fr_scores.npy').astype(np.float64)
    h = np.histogram(s, bins=[0, .05, .2, .4, .6, .8, .95, 1.0001])[0] / len(s)
    print(f'GATE 2  French uncertain pairs {len(s):,}; LLM score distribution: ' + '  '.join(f'{lo}-{hi}: {v:.3f}' for (lo, hi), v in zip([(0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1)], h)))
    print(f'        decisive (<0.05 or >0.95): {np.mean((s < .05) | (s > .95)):.3f}   undecided 0.4-0.6: {np.mean((s >= .4) & (s <= .6)):.3f}')
    S = W + 'artifacts_v3/stage2_v6_pp/'
    d = np.load(S + 'test_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']; pp = np.load(S + 'test_pair_probs.npz')
    odds = lambda q, r: q * r / (q * r + 1 - q)
    def dec(c, use):
        ft, fa, fb, fp = fuse(pp, P, s, c, use)
        pt, pa, pv = pp['tid'].astype(np.int64), pp['aid'].astype(np.int64), pp['p'].astype(np.float64).copy()
        k1 = pt << 32 | pa; o = np.argsort(k1); ks = k1[o]; pos = np.searchsorted(ks, ft << 32 | fa); pv[o[pos]] = fp
        a1, q1 = redecide(pt, pa, pv, d['aid'].astype(np.int64), d['q'].astype(np.float64))
        return np.where(decode(a1, odds(q1, .5), d['margin'], **k) & (a1 >= 0), a1, -1)
    A, B = dec(clf0, False), dec(clf, True)
    ct = np.load(W + 'artifacts/test_anchor_countries.npy', allow_pickle=True)
    fr = ((A >= 0) & (ct[np.maximum(A, 0)] == 'France')) | ((B >= 0) & (ct[np.maximum(B, 0)] == 'France'))
    ch = fr & (A != B)
    print(f'        French decisions changed vs recalibration-only control: {ch.sum():,}  (adds {int((ch & (A < 0)).sum()):,}, removes {int((ch & (B < 0)).sum()):,}, switches company {int((ch & (A >= 0) & (B >= 0)).sum()):,})')

if __name__ == '__main__':
    clf, clf0 = gate1(); gate2(clf, clf0)
