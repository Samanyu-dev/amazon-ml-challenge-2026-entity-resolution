"""Address-only blocking for synthetic-name targets the stage-2 decision leaves unlinked.
Per country: TF-IDF (word 1-2grams) over normalized S1 addresses; top-K S1 by cosine. usage: synth_block.py train|test"""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from decode import decode
sp = sys.argv[1]; K = 5; FR = '/private/tmp/claude-501/fable_review/'
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
rdn = lambda f: pd.read_csv(W + f'artifacts/{sp}_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[1, 4])
s1 = rdn('s1'); tg = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
syn = np.load(FR + f'synth_{sp}.npy')
out = {}
plan = [('exp_owner_split', ('US', 'India'))] if sp == 'train' else [('stage2_v7e_usin', ('US', 'India')), ('stage2_v6_pp', ('France',))]
for S, cs in plan:
    d = np.load(W + f'artifacts_v3/{S}/{sp}_decisions.npz'); k = json.load(open(W + f'artifacts_v3/{S}/stage2_report.json'))['knobs']
    m = decode(d['aid'].astype(np.int64), d['q'].astype(np.float64), d['margin'], **k)
    for c in cs:
        ia = np.flatnonzero(s1.ctry.values == c); T = np.flatnonzero(syn & ~m & (tg.ctry.values == c) & (tg.addr.values != ''))
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, token_pattern=r'\S+').fit(s1.addr.values[ia])
        A = v.transform(s1.addr.values[ia]).T.tocsr(); X = v.transform(tg.addr.values[T])
        rt, ra, rs = [], [], []
        for st in range(0, len(T), 2000):
            M = (X[st:st + 2000] @ A).tocsr()
            for r in range(M.shape[0]):
                lo, hi = M.indptr[r], M.indptr[r + 1]
                if hi == lo: continue
                dd, jj = M.data[lo:hi], M.indices[lo:hi]; o = np.argsort(-dd)[:K]
                rt.append(np.full(len(o), T[st + r])); ra.append(ia[jj[o]]); rs.append(dd[o])
        out[c] = (np.concatenate(rt), np.concatenate(ra), np.concatenate(rs)); print(sp, c, 'targets', len(T), flush=True)
t = np.concatenate([v[0] for v in out.values()]); a = np.concatenate([v[1] for v in out.values()]); s = np.concatenate([v[2] for v in out.values()])
o = np.lexsort((-s, t)); t, a, s = t[o], a[o], s[o]
n2 = 5034616 if sp == 'train' else 4887273
np.savez(FR + f'synth_block_{sp}.npz', tid=t, aid=a, sim=s)
np.savez(FR + f'sy_pairs_{sp}.npz', s1_row=a.astype(np.int32), src=np.where(t < n2, 2, 3).astype(np.int8), t_row=np.where(t < n2, t, t - n2).astype(np.int32), tid=t.astype(np.int32))
print('done', len(t))
