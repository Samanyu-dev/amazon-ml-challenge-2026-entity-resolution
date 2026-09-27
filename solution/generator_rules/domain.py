"""Domain-name rule: target name 'xxx.com' == generated key of an S1 name (initials of first i words + rest concatenated,
with/without legal words, word rotations). Unique key hit within country (address tie-break by house number). usage: domain.py train|test"""
import sys, re, json, unicodedata
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from collections import defaultdict
from fingerprints import rd
sp = sys.argv[1]; FR = '/private/tmp/claude-501/fable_review/'
LEG = set('sarl sas sasu sa eurl ei sci snc ltd llc inc corp pvt private limited llp lp pllc pc plc co company corporation incorporated l l c p'.split())
deacc = lambda s: unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
def keys(name):
    w = re.findall(r'[a-z0-9]+', deacc(name.replace('&', ' ')))
    out = set()
    for ws in (w, [x for x in w if x not in LEG]):
        if not ws: continue
        for r in range(len(ws)):
            v = ws[r:] + ws[:r]
            for i in range(len(v)): out.add(''.join(x[0] for x in v[:i]) + ''.join(v[i:]))
    return out
s1 = rd(f'{sp}/{sp}_source1.tsv'); tg = pd.concat([rd(f'{sp}/{sp}_source2.tsv'), rd(f'{sp}/{sp}_source3.tsv')], ignore_index=True)
dm = tg.business_name.str.strip().str.extract(r'^(?:www\.)?([^\s.]+)\.c[o0]m$', flags=re.I)[0]
T = np.flatnonzero(dm.notna().values)
D = {t: deacc(dm.values[t]).replace('0', 'o') for t in T}
need = set(D.values())
idx = defaultdict(list)
for j, (nm, c) in enumerate(zip(s1.business_name.values, s1.country.values)):
    for k in keys(nm):
        if k in need or k.replace('0', 'o') in need: idx[(c, k.replace('0', 'o'))].append(j)
num = lambda x: set(n.lstrip('0') for n in re.findall(r'\d+', re.sub(r'(?<=\d)[-/ ](?=\d)', '', x)) if n.lstrip('0'))
rt, ra, rn, rk = [], [], [], []
for t in T:
    c = idx.get((tg.country.values[t], D[t]))
    if not c: continue
    c = list(dict.fromkeys(c)); pick = c
    if len(c) > 1:   # tie-break: shared house number
        tn = num(tg.business_address.values[t]); pick = [j for j in c if tn & num(s1.business_address.values[j])] if tn else []
    if len(pick) == 1: rt.append(t); ra.append(pick[0]); rn.append(len(c)); rk.append(0)
print(sp, 'domain targets', len(T), 'unique hits', len(rt))
np.savez(FR + f'domain_{sp}.npz', tid=np.array(rt), aid=np.array(ra), ncand=np.array(rn))
