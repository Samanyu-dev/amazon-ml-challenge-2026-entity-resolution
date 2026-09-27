"""#1 accounting of blank-address, unique-core-name misses (val): where is the owner, and is the target name ambiguous
under simple word-drop (target words subset of >=2 S1 names in the same country)?"""
import sys, re, json, glob
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd, unicodedata
from collections import defaultdict
from decode import load, decode
from fingerprints import rd
LEG = set('sarl sas sa eurl ei sci snc ltd llc inc corp pvt private limited llp lp pllc pc plc co company corporation incorporated l c p'.split())
deacc = lambda s: unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
words = lambda s: [w for w in re.findall(r'[a-z0-9]+', deacc(s)) if w not in LEG]
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True); s1 = rd('train/train_source1.tsv')
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
core1 = pd.read_csv(W + 'artifacts/train_s1.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[3]).core.values
ns = pd.Series(core1).map(pd.Series(core1).value_counts()).values
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); q = d['q']; m = decode(aid, q.astype(np.float64), d['margin'], **k)
fo = np.load(W + 'artifacts/train_anchor_folds.npy')
SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')]); has = np.zeros(len(own), bool)
for f in sorted(glob.glob(W + 'artifacts_v3/train_scores/candidates-*.bin')):
    c = np.fromfile(f, dtype=SF); ok = own[c['tid']] == c['aid'].astype(np.int64); has[c['tid'][ok]] = True
ae = tg.business_address.values == ''
u = np.flatnonzero((own >= 0) & ae & ~(m & (aid == own)) & (fo[np.maximum(own, 0)] >= 90) & (ns[np.maximum(own, 0)] == 1))
# S1 word index for superset lookup
S1w = [set(words(x)) for x in s1.business_name.values]; inv = defaultdict(set)
for j, ws in enumerate(S1w):
    for w in ws: inv[w].add(j)
cat = []
for t in u:
    tw = set(words(tg.business_name.values[t])); o = own[t]
    sup = set.intersection(*[inv[w] for w in tw]) if tw and all(w in inv for w in tw) else set()
    sup = {j for j in sup if s1.country.values[j] == s1.country.values[o]}
    owner_sup = o in sup
    cat.append(('owner not in cands' if not has[t] else 'owner top, rejected' if aid[t] == o else 'owner in cands, not top',
                'target words ⊂ owner' if owner_sup else 'target has words NOT in owner', len(sup)))
df = pd.DataFrame(cat, columns=['where', 'words', 'n_superset'])
print(f'unique-core-name blank-address val misses: {len(u):,}')
print(pd.crosstab(df['where'], df['words']))
print('number of same-country S1 names containing ALL target words (when target ⊂ owner):')
x = df[df.words == 'target words ⊂ owner'].n_superset; print(x.describe()[['mean', '50%']].to_dict(), ' share with exactly 1 (=owner unique):', round((x == 1).mean(), 3))
