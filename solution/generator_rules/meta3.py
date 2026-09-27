"""Meta v3 = meta2 features + S2/S3 source + generator-op flags (synthetic, acronym, alias, domain) + S1 ambiguity
(anchor core-name group size, anchor address group size, target-core-name group size). Same gate as meta2."""
import sys, re, json, os
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode, evaluate
from meta2 import feats, fit_eval
from fingerprints import rd
SYL = ['onyx','kor','delta','xylo','wex','novi','brix','halo','gild','umbra','faye','nex','kelo','zeph','calo','aria','lum','jax','cira','dova','veo','nyla','tavo','mira','yuma','ecto','riza','lyra','flux','pyra','belo','drex','arc','sol','evo','vera','syn','orbi','iri','avi','vio','vantage','zeta','quo','io','x']
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z', 'legal']
def extra(sp, aid):
    tg = pd.concat([rd(f'{sp}/{sp}_source2.tsv'), rd(f'{sp}/{sp}_source3.tsv')], ignore_index=True); n2 = len(rd(f'{sp}/{sp}_source2.tsv'))
    n = tg.business_name; nl = n.str.lower().str.replace('1', 'l').str.replace('0', 'o').str.replace('5', 's').str.strip()
    ops = np.column_stack([np.arange(len(tg)) >= n2,
        nl.str.fullmatch('(?:' + '|'.join(SYL) + '){2,}').values, n.str.strip().str.fullmatch(r'[A-Z]{2,4}').values,
        n.str.lower().str.contains(r'\b(?:dba|d/b/a|aka|formerly|doing business as|t/a|f/k/a)\b', regex=True).values,
        n.str.contains(r'\.c[o0]m\b', regex=True, flags=re.I).values]).astype(np.float32)
    rdn = lambda f: pd.read_csv(W + f'artifacts_v3/{sp}_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[1, 3, 4])
    s1 = rdn('s1'); t = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
    k1 = s1.ctry + '|' + s1.core; ka = s1.ctry + '|' + s1.addr
    gname = k1.map(k1.value_counts()).values.astype(np.float32); gaddr = ka.map(ka.value_counts()).values.astype(np.float32)
    tgn = (t.ctry + '|' + t.core).map(k1.value_counts()).fillna(0).values.astype(np.float32)
    a = np.maximum(aid, 0); G = np.column_stack([np.log1p(gname[a]), np.log1p(gaddr[a]), np.log1p(tgn)]); G[aid < 0] = -1
    return np.column_stack([ops, G])
STACK = sys.argv[1]; S = W + 'artifacts_v3/' + STACK + '/'
d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
ctry = np.load(W + 'artifacts/train_anchor_countries.npy', allow_pickle=True)
Xp = f'/private/tmp/claude-501/fable_review/meta2_X_{STACK}.npy'
X2, band, aid, q, mg = (np.load(Xp), *feats('train', d, np.load(S + 'train_pair_probs.npz'))[1:]) if os.path.exists(Xp) else feats('train', d, np.load(S + 'train_pair_probs.npz'))
X = np.column_stack([X2, extra('train', aid)])
c2 = [c for c in range(X2.shape[1]) if c != 2 + 14]; c3 = c2 + list(range(X2.shape[1], X.shape[1]))
print('-- meta2 columns (reference):'); m2 = fit_eval(X, band, aid, q, mg, own, folds, truth, ctry, k, c2)
print('-- meta3 = meta2 + source/op flags + S1 ambiguity:'); m3 = fit_eval(X, band, aid, q, mg, own, folds, truth, ctry, k, c3)
np.save('/private/tmp/claude-501/fable_review/meta3_X_train.npy', X)

if '--apply' in sys.argv:
    dt = np.load(S + 'test_decisions.npz')
    Xt2, bt, at, qt, mt = feats('test', dt, np.load(S + 'test_pair_probs.npz'))
    Xt = np.column_stack([Xt2, extra('test', at)])
    M, cc = m3['meta']; v = qt.copy(); v[bt] = M.predict_proba(Xt[bt][:, cc])[:, 1]
    o = W + 'artifacts_v3/' + STACK + '_meta3/'; os.makedirs(o, exist_ok=True)
    np.savez(o + 'test_decisions.npz', aid=dt['aid'], q=v, margin=dt['margin']); json.dump({'knobs': k, 'source': STACK}, open(o + 'stage2_report.json', 'w'))
    print('wrote', o)
