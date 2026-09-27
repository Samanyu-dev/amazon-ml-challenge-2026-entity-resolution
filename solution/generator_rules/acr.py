"""Acronym rule: target name is 2-4 capitals == initials of an S1 name (legal forms / small words dropped) at the
same address (address TF-IDF top-K within country). usage: acr.py train|test"""
import sys, re, json, unicodedata
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from fingerprints import rd
sp = sys.argv[1]; K = 10; FR = '/private/tmp/claude-501/fable_review/'
LEGAL = {'sarl', 'sas', 'sasu', 'sa', 'eurl', 'ei', 'sci', 'snc', 'scop', 'selarl', 'gie', 'ltd', 'llc', 'l.l.c.', 'inc', 'inc.', 'corp', 'corp.',
         'pvt', 'pvt.', 'private', 'limited', 'llp', 'lp', 'pllc', 'pc', 'p.c.', 'co', 'co.', 'company', 'corporation', 'incorporated', 'plc', 'ltd.'}
SMALL = {'de', 'la', 'le', 'les', 'du', 'des', "l'", "d'", 'et', '&', 'and', 'of', 'the', 'a', 'à', 'en', 'pour', 'sur', 'au', 'aux'}
def initials(name):
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\((france|india|usa|us)\)', ' ', s); s = s.replace("l'", "l' ").replace("d'", "d' ")
    w = [x for x in re.split(r'[\s,]+', s) if x and x not in LEGAL and x not in SMALL and re.search('[a-z]', x)]
    return ''.join(x.lstrip('(["')[0] for x in w if x.lstrip('(["')).upper()
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
rdn = lambda f: pd.read_csv(W + f'artifacts/{sp}_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[1, 4])
s1n = rdn('s1'); tgn = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
s1 = rd(f'{sp}/{sp}_source1.tsv'); tg = pd.concat([rd(f'{sp}/{sp}_source2.tsv'), rd(f'{sp}/{sp}_source3.tsv')], ignore_index=True)
acr = tg.business_name.str.strip().str.fullmatch(r'[A-Z]{2,4}').values & (tgn.addr.values != '')
ini = np.array([initials(x) for x in s1.business_name.values])
rt, ra, rs = [], [], []
name = tg.business_name.str.strip().values
for c in ('US', 'India', 'France'):
    ia = np.flatnonzero(s1n.ctry.values == c); T = np.flatnonzero(acr & (tgn.ctry.values == c))
    if len(T) == 0 or len(ia) == 0: continue
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r'\S+').fit(s1n.addr.values[ia])
    A = v.transform(s1n.addr.values[ia]).tocsr(); X = v.transform(tgn.addr.values[T]).tocsr()
    groups = pd.Series(np.arange(len(ia))).groupby(ini[ia]).apply(np.array).to_dict()
    for i, tt in enumerate(T):
        g = groups.get(name[tt])
        if g is None: continue
        sim = (A[g] @ X[i].T).toarray().ravel(); o = np.argsort(-sim)[:K]
        rt.append(np.full(len(o), tt)); ra.append(ia[g[o]]); rs.append(sim[o])
    print(sp, c, 'acronym targets', len(T), 'with initials group', len(set(rt[-1]) if rt else 0), flush=True)
t, a, s = np.concatenate(rt), np.concatenate(ra), np.concatenate(rs)
st = np.r_[0, np.flatnonzero(np.diff(t)) + 1]; sz = np.diff(np.r_[st, len(t)]); rank = np.arange(len(t)) - np.repeat(st, sz)
second = np.where(sz > 1, s[np.minimum(st + 1, len(s) - 1)], 0); gap = np.repeat(s[st] - second, sz)
match = np.ones(len(t), bool)   # every candidate here already has matching initials
np.savez(FR + f'acr_{sp}.npz', tid=t, aid=a, sim=s, rank=rank, gap=gap, match=match)
print('done', len(t))
