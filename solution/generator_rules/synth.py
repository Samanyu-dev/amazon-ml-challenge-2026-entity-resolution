import sys, re, json, glob
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from collections import Counter
from decode import load, decode
from fingerprints import rd
SYL = ['onyx','kor','delta','xylo','wex','novi','brix','halo','gild','umbra','faye','nex','kelo','zeph','calo','aria','lum','jax','cira','dova','veo','nyla','tavo','mira','yuma','ecto','riza','lyra','flux','pyra','belo','drex','arc','sol','evo','vera','syn','orbi','iri','avi','vio','vantage','zeta','quo','io','x']
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
n = tg.business_name.str.lower().str.replace('1', 'l').str.replace('0', 'o').str.replace('5', 's').str.strip()
one = ~n.str.contains(r'[\s.]', regex=True) & n.str.fullmatch(r'[a-z]{5,}')
pat = '(?:' + '|'.join(SYL) + ')'
syn = n.str.fullmatch(f'{pat}{{2,}}').values
# discover more syllables: strip known ones from one-word names that are mostly covered
left = Counter()
for x in n[one & ~syn].values[:2000000]:
    y = re.sub(pat, ' ', x).split()
    if len(y) == 1 and len(x) - len(y[0]) >= 4: left[y[0]] += 1
print('uncovered leftovers (top):', left.most_common(15))
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
S = W + 'artifacts_v3/exp_owner_split/'; d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
print(f'synthetic-name targets {syn.sum():,} ({syn.mean():.4f}); true-dup rate {np.mean(own[syn] >= 0):.3f}; has address {np.mean(tg.business_address.values[syn] != ""):.3f}')
t = syn & (own >= 0)
print(f'  of true dups: linked correctly {np.mean(m[t] & (aid[t] == own[t])):.3f}, linked wrong {np.mean(m[t] & (aid[t] != own[t])):.3f}, unlinked {np.mean(~m[t]):.3f}')
dcy = syn & (own < 0); print(f'  decoys {dcy.sum():,}: linked (false) {np.mean(m[dcy]):.3f}')
np.save('/private/tmp/claude-501/fable_review/synth_train.npy', syn)

SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')]); has = np.zeros(len(own), bool)
for f in sorted(glob.glob(W + 'artifacts_v3/train_scores/candidates-*.bin')):
    c = np.fromfile(f, dtype=SF); ok = own[c['tid']] == c['aid'].astype(np.int64); has[c['tid'][ok]] = True
u = t & ~m
print(f'  unlinked true synth {u.sum():,}: owner is top {np.mean(aid[u] == own[u]):.3f}, in cands not top {np.mean(has[u] & (aid[u] != own[u])):.3f}, absent {np.mean(~has[u]):.3f}')
fo = np.load(W + 'artifacts/train_anchor_folds.npy'); v = u & (fo[np.maximum(own, 0)] >= 90); print(f'  val (folds 90-99) unlinked true synth links: {v.sum():,}')
