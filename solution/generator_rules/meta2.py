"""Stage-3 meta model over stage-2 decisions: q + target fingerprints + candidate competition + anchor-level + raw pair
features. Fit on folds 80-89 (out-of-sample for stage 2), gate on 90-99 vs control [q, margin].
usage: meta2.py STACK [--apply]   (--apply also scores test and writes STACK_meta/test_decisions.npz for US/India)"""
import sys, json, re, os
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from decode import load, decode, evaluate
from fingerprints import flags, rd
STACK = sys.argv[1]; S = W + 'artifacts_v3/' + STACK + '/'

def feats(split, d, pp):
    aid, q, mg = d['aid'].astype(np.int64), d['q'].astype(np.float64), d['margin'].astype(np.float64)
    n = len(aid); lg = np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    tg = pd.concat([rd(f'{split}/{split}_source2.tsv'), rd(f'{split}/{split}_source3.tsv')], ignore_index=True)
    s1 = rd(f'{split}/{split}_source1.tsv'); assert len(tg) == n
    FP = flags(tg).values.astype(np.float32)
    # candidate competition from stage-2 pair probs
    t, p = pp['tid'].astype(np.int64), pp['p'].astype(np.float64)
    o = np.lexsort((-p, t)); t, p = t[o], p[o]
    st = np.r_[0, np.flatnonzero(np.diff(t)) + 1]; tt = t[st]
    cnt = np.diff(np.r_[st, len(t)]); p1 = p[st]; p2 = np.where(cnt > 1, p[np.minimum(st + 1, len(p) - 1)], 0)
    sp = np.add.reduceat(p, st); ent = -np.add.reduceat(p * np.log(np.clip(p, 1e-9, 1)), st)
    C = np.full((n, 5), -1, np.float32); C[tt] = np.column_stack([cnt, p1, p2, sp, ent])
    # anchor-level
    ok = aid >= 0; a = np.maximum(aid, 0); na = int(a.max()) + 1
    c5 = np.bincount(a[ok & (q > .5)], minlength=na); c1 = np.bincount(a[ok & (q > .1)], minlength=na)
    sq = np.bincount(a[ok], weights=q[ok], minlength=na)
    idx = np.flatnonzero(ok); o = np.lexsort((-q[idx], a[idx])); idx = idx[o]
    gs = np.r_[0, np.flatnonzero(np.diff(a[idx])) + 1]; rank = np.arange(len(idx)) - np.repeat(gs, np.diff(np.r_[gs, len(idx)]))
    rk = np.full(n, -1, np.float32); rk[idx] = rank
    A = np.column_stack([c5[a], c1[a], sq[a] - q, rk]).astype(np.float32); A[~ok] = -1
    # raw pair features vs top anchor
    tn = tg.business_name.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip().values
    ta = tg.business_address.str.lower().values
    an = s1.business_name.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip().values
    aa = s1.business_address.str.lower().values
    band = ok & (q > .003) & (q < .998); P = np.full((n, 4), -1, np.float32)
    dig = re.compile(r'\d+')
    for i in np.flatnonzero(band):
        x, y = tn[i], an[a[i]]; dx, dy = set(dig.findall(ta[i])), set(dig.findall(aa[a[i]]))
        P[i] = [x == y, (x in y) or (y in x), len(dx & dy) / max(1, len(dx | dy)), len(dx - dy)]
    return np.column_stack([lg, mg, FP, C, A, P]), band, aid, q, mg

def fit_eval(X, band, aid, q, mg, own, folds, truth, ctry, knobs, cols):
    fa = np.where(aid >= 0, folds[np.maximum(aid, 0)], -1); y = (own == aid).astype(int)
    tr = band & (fa >= 80) & (fa < 90); out = {}; models = {}
    for name, c in (('ctrl', [0, 1]), ('meta', cols)):
        m = CatBoostClassifier(iterations=800, depth=5, learning_rate=0.05, verbose=0, thread_count=8, random_seed=0)
        m.fit(X[tr][:, c], y[tr]); v = q.copy(); v[band] = m.predict_proba(X[band][:, c])[:, 1]; models[name] = (m, c)
        r, f = evaluate(decode(aid, v, mg, **knobs), aid, own, folds, truth, 90, 100, ctry); out[name] = f
        rt, _ = evaluate(decode(aid, v, mg, **knobs), aid, own, folds, truth, 85, 90)
        print(f'{name:5s} report F {r["macro_f05"]:.6f} P {r["precision"]:.5f} R {r["recall"]:.5f} tune {rt["macro_f05"]:.6f} {r["by_country"]}')
    r, f = evaluate(decode(aid, q, mg, **knobs), aid, own, folds, truth, 90, 100, ctry); out['q'] = f
    print(f'q     report F {r["macro_f05"]:.6f}')
    for x, b in (('ctrl', 'q'), ('meta', 'ctrl'), ('meta', 'q')):
        g = out[x] - out[b]; print(f'  {x} vs {b}: {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}')
    return models

if __name__ == '__main__':
    d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
    _, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
    folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
    ctry = np.load(W + 'artifacts/train_anchor_countries.npy', allow_pickle=True)
    X, band, aid, q, mg = feats('train', d, np.load(S + 'train_pair_probs.npz'))
    np.save(f'/private/tmp/claude-501/fable_review/meta2_X_{STACK}.npy', X)
    cols = list(range(X.shape[1]))
    models = fit_eval(X, band, aid, q, mg, own, folds, truth, ctry, k, cols)
    nofp_accent = [c for c in cols if c != 2 + 14]   # drop name_accent (means something else in France)
    print('-- without name_accent:'); models = fit_eval(X, band, aid, q, mg, own, folds, truth, ctry, k, nofp_accent)
    if '--apply' in sys.argv:
        dt = np.load(S + 'test_decisions.npz'); Xt, bt, at, qt, mt = feats('test', dt, np.load(S + 'test_pair_probs.npz'))
        m, c = models['meta']; v = qt.copy(); v[bt] = m.predict_proba(Xt[bt][:, c])[:, 1]
        o = W + 'artifacts_v3/' + STACK + '_meta/'; os.makedirs(o, exist_ok=True)
        np.savez(o + 'test_decisions.npz', aid=dt['aid'], q=v, margin=dt['margin'])
        json.dump({'knobs': k, 'source': STACK}, open(o + 'stage2_report.json', 'w'))
        print('wrote', o)
