"""Transformation-likelihood ranker for blank-address word-drop groups (target words ⊂ candidate words, same country)."""
import sys, re, json, math
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd, unicodedata
from collections import defaultdict, Counter
from catboost import CatBoostClassifier
from decode import load, decode
from fingerprints import rd
LEG = set('sarl sas sa eurl ei sci snc ltd llc inc corp pvt private limited llp lp pllc pc plc co company corporation incorporated l c p'.split())
deacc = lambda s: unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
allw = lambda s: re.findall(r'[a-z0-9]+', deacc(s))
core = lambda ws: [w for w in ws if w not in LEG]
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True); s1 = rd('train/train_source1.tsv')
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); q = d['q']; m = decode(aid, q.astype(np.float64), d['margin'], **k)
fo = np.load(W + 'artifacts/train_anchor_folds.npy')
S1a = [allw(x) for x in s1.business_name.values]; S1c = [core(x) for x in S1a]
df_ = Counter(w for ws in S1c for w in set(ws)); N = len(S1c); idf = lambda w: math.log(N / (1 + df_.get(w, 0)))
inv = defaultdict(list)
for j, ws in enumerate(S1c):
    for w in set(ws): inv[w].append(j)
inv = {w: np.array(v) for w, v in inv.items()}
ctry1 = s1.country.values
def group(tw, c):
    if not tw or any(w not in inv for w in tw): return None
    ls = sorted((inv[w] for w in set(tw)), key=len); g = ls[0]
    for x in ls[1:]:
        g = np.intersect1d(g, x, assume_unique=True)
        if len(g) == 0: return None
    g = g[ctry1[g] == c]
    return g if 2 <= len(g) <= 60 else None
def feats(tw_all, tw, j):
    cw = S1c[j]; ca = S1a[j]; drop = [w for w in cw if w not in tw]
    pos = [i for i, w in enumerate(cw) if w not in tw]
    kept_idx = [i for i, w in enumerate(cw) if w in tw]
    contiguous = float(bool(kept_idx) and kept_idx == list(range(kept_idx[0], kept_idx[0] + len(kept_idx))))
    prefix = float(bool(kept_idx) and kept_idx[0] == 0 and contiguous == 1)
    order = float([w for w in cw if w in tw] == [w for w in tw if w in cw])
    tleg = [w for w in tw_all if w in LEG]; cleg = [w for w in ca if w in LEG]
    return [len(drop), len(cw), float(0 in pos), float(len(cw) - 1 in pos), float(any(0 < p < len(cw) - 1 for p in pos)),
            sum(idf(w) for w in drop), max([idf(w) for w in drop], default=0), min([idf(w) for w in drop], default=0),
            contiguous, prefix, order, float(bool(tleg)), float(bool(cleg)), float(tleg == cleg), float(bool(set(tleg) & set(cleg))),
            len(ca), len(s1.business_name.values[j])]
ae = tg.business_address.values == ''
T = np.flatnonzero(ae & (own >= 0))
rows = []; rng = np.random.default_rng(0)
for t in T:
    ta = allw(tg.business_name.values[t]); tw = core(ta); o = own[t]
    g = group(tw, ctry1[o])
    if g is None or o not in g: continue
    for j in g: rows.append((t, j, int(j == o), len(g), *feats(ta, tw, j)))
R = pd.DataFrame(rows); R.columns = ['t', 'j', 'y', 'gs'] + [f'f{i}' for i in range(R.shape[1] - 4)]
fold = fo[own[R.t.values]]; X = R.iloc[:, 3:].values.astype(np.float32); y = R.y.values
tr = fold < 80; ev = fold >= 90
M = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05, verbose=0, thread_count=6, random_seed=0).fit(X[tr], y[tr])
R['p'] = M.predict_proba(X)[:, 1]
E = R[ev].copy(); E['pn'] = E.p / E.groupby('t').p.transform('sum')
top = E.loc[E.groupby('t').pn.idxmax()]
print(f'word-drop groups: train {R[tr].t.nunique():,}  eval {E.t.nunique():,}; median group size {E.groupby("t").size().median():.0f}')
print(f'eval top-1 accuracy {top.y.mean():.3f}   random {np.mean(1 / E.groupby("t").size()):.3f}   fewest-dropped-words {E.loc[E.groupby("t").f0.idxmin()].y.mean():.3f}')
for th in (0.5, 0.6, 0.7, 0.8, 0.9):
    s = top[top.pn >= th]; print(f'  groups with normalized top prob >= {th}: {len(s) / len(top):.3f} of groups, precision {s.y.mean() if len(s) else 0:.3f}')
# on the groups we currently MISS (unlinked owner)
miss_t = set(np.flatnonzero(~(m & (aid == own))))
tm = top[top.t.isin(miss_t)]
for th in (0.7, 0.8, 0.9):
    s = tm[tm.pn >= th]; print(f'  CURRENTLY-MISSED groups, prob >= {th}: n={len(s):,} ({len(s) / max(len(tm), 1):.3f}), precision {s.y.mean() if len(s) else 0:.3f}')
names = ['n_drop', 'cand_words', 'drop_first', 'drop_last', 'drop_mid', 'idf_sum', 'idf_max', 'idf_min', 'contig', 'prefix', 'order', 't_leg', 'c_leg', 'leg_eq', 'leg_overlap', 'c_allwords', 'c_len', 'gs'][:X.shape[1]]
print(dict(zip(['gs'] + names, np.round(M.get_feature_importance(), 1))))
R.to_pickle('/private/tmp/claude-501/fable_review/xform_rows.pkl'); M.save_model('/private/tmp/claude-501/fable_review/xform.cbm')
