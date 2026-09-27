"""Typo-tolerant name blocking for blank-address targets not confidently linked: char 3-gram TF-IDF over S1 core names
(n-grams in >0.5% of names dropped), top-5 per target within country. usage: bn_block.py train|test"""
import sys, json, time
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from decode import decode
sp = sys.argv[1]; K = 5; FR = '/private/tmp/claude-501/fable_review/'
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
rdn = lambda f: pd.read_csv(W + f'artifacts/{sp}_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[1, 3, 4])
s1 = rdn('s1'); tg = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
ST = W + 'artifacts_v3/' + ('exp_owner_split' if sp == 'train' else 'stage2_v7e_usin_owner') + '/'
d = np.load(ST + f'{sp}_decisions.npz'); k = json.load(open(ST + 'stage2_report.json'))['knobs']
da = d['aid'].astype(np.int64); dq = d['q'].astype(np.float64); dm = decode(da, dq, d['margin'], **k)
scope = (tg.addr.values == '') & ~(dm & (dq >= 0.5)) & (tg.core.values != '')
rt, ra, rs = [], [], []; t0 = time.time()
for c in ('US', 'India'):
    ia = np.flatnonzero(s1.ctry.values == c); T = np.flatnonzero(scope & (tg.ctry.values == c))
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), max_df=0.005, sublinear_tf=True).fit(s1.core.values[ia])
    A = v.transform(s1.core.values[ia]).T.tocsr(); X = v.transform(tg.core.values[T])
    for st in range(0, len(T), 1000):
        M = (X[st:st + 1000] @ A).tocsr()
        for r in range(M.shape[0]):
            lo, hi = M.indptr[r], M.indptr[r + 1]
            if hi == lo: continue
            dd, jj = M.data[lo:hi], M.indices[lo:hi]; o = np.argsort(-dd)[:K]
            rt.append(np.full(len(o), T[st + r])); ra.append(ia[jj[o]]); rs.append(dd[o])
    print(sp, c, 'targets', len(T), f'{time.time() - t0:.0f}s', flush=True)
t, a, s = np.concatenate(rt), np.concatenate(ra), np.concatenate(rs); o = np.lexsort((-s, t)); t, a, s = t[o], a[o], s[o]
np.savez(FR + f'bn_block_{sp}.npz', tid=t, aid=a, sim=s, ov=np.zeros(len(t)), ncand=np.full(len(t), K))
n2 = 5034616 if sp == 'train' else 4887273
np.savez(FR + f'bn_pairs_{sp}.npz', s1_row=a.astype(np.int32), src=np.where(t < n2, 2, 3).astype(np.int8), t_row=np.where(t < n2, t, t - n2).astype(np.int32), tid=t.astype(np.int32))
print('done', len(t))
