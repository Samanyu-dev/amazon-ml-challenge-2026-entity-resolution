"""One-word-swap at identical address: target name tokens == top S1 name tokens except exactly one substituted word
(legal forms ignored), and every target address token appears in the S1 address. Train: true rate + our linking; test: counts."""
import sys, re, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode
LEG = set('sarl sas sasu sa eurl ei sci snc ltd llc inc corp pvt private limited llp lp co cie compagnie company corporation incorporated group groupe'.split())
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
def run(sp, S, r=1.0):
    rdn = lambda f: pd.read_csv(W + f'artifacts/{sp}_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[1, 2, 4])
    s1 = rdn('s1'); tg = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
    d = np.load(W + f'artifacts_v3/{S}/{sp}_decisions.npz'); k = json.load(open(W + f'artifacts_v3/{S}/stage2_report.json'))['knobs']
    aid = d['aid'].astype(np.int64); q = d['q'].astype(np.float64); q = q * r / (q * r + 1 - q); m = decode(aid, q, d['margin'], **k)
    idx = np.flatnonzero((aid >= 0) & (tg.addr.values != ''))
    tn, an, ta, aa = tg.name.values, s1.name.values, tg.addr.values, s1.addr.values
    hit = np.zeros(len(aid), bool)
    for t in idx:
        x = [w for w in tn[t].split() if w not in LEG]; y = [w for w in an[aid[t]].split() if w not in LEG]
        if len(x) != len(y) or len(x) < 2: continue
        if sum(a != b for a, b in zip(x, y)) != 1 or set(x) == set(y): continue
        xa = ta[t].split()
        if len(xa) >= 3 and set(xa) <= set(aa[aid[t]].split()): hit[t] = True
    return hit, aid, m, tg.ctry.values
h, aid, m, c = run('train', 'exp_owner_split')
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
print(f'TRAIN one-word-swap same-address pairs {h.sum():,}: top is owner {np.mean(own[h] == aid[h]):.3f}; linked {m[h].mean():.3f}; unlinked&true {np.sum(h & ~m & (own == aid)):,} unlinked&false {np.sum(h & ~m & (own != aid)):,}')
h2, aid2, m2, c2 = run('test', 'stage2_v6_pp', 0.5)
for x in ('France',):
    s = h2 & (c2 == x); print(f'TEST {x}: swap pairs {s.sum():,}; linked {m2[s].mean():.3f}')
h3, aid3, m3, c3 = run('test', 'stage2_v7e_usin')
for x in ('US', 'India'):
    s = h3 & (c3 == x); print(f'TEST {x}: swap pairs {s.sum():,}; linked {m3[s].mean():.3f}')
