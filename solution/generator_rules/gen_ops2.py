"""Generator-operation audit: for each detectable noise op on a target record, how many validation links do we lose?"""
import sys, re, json, glob
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode
from fingerprints import rd
SYL = ['onyx','kor','delta','xylo','wex','novi','brix','halo','gild','umbra','faye','nex','kelo','zeph','calo','aria','lum','jax','cira','dova','veo','nyla','tavo','mira','yuma','ecto','riza','lyra','flux','pyra','belo','drex','arc','sol','evo','vera','syn','orbi','iri','avi','vio','vantage','zeta','quo','io','x']
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
s1 = rd('train/train_source1.tsv')
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
S = W + 'artifacts_v3/exp_owner_split/'; d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
fo = np.load(W + 'artifacts/train_anchor_folds.npy')
n, a = tg.business_name, tg.business_address
nl = n.str.lower().str.replace('1', 'l').str.replace('0', 'o').str.replace('5', 's').str.strip()
on = pd.Series(s1.business_name.values[np.maximum(own, 0)]).str.lower()   # owner name (only meaningful for true dups)
words = lambda s: re.findall(r'[a-z]+', s)
init = pd.Series([''.join(w[0] for w in words(x) if w not in ('private', 'limited', 'pvt', 'ltd', 'inc', 'llc', 'corp', 'co', 'the')) for x in on])
ops = {
 'synthetic name': nl.str.fullmatch('(?:' + '|'.join(SYL) + '){2,}').values,
 'domain name': n.str.contains(r'\.c[o0]m\b|c[o0]m$', regex=True).values,
 'website in record': n.str.contains('www.', regex=False).values | a.str.contains('www.', regex=False).values,
 'indic script name': n.str.contains(r'[ऀ-෿]', regex=True).values,
 'address empty': (a.str.strip() == '').values,
 'address null/NA': a.str.contains(r'null|NULL|N/A', regex=True).values,
 'address ## / #': a.str.contains('#', regex=False).values,
 'address PMB/POBox': a.str.contains(r'\bPMB\b|PO Box|P\.O\. Box', regex=True).values,
 'address CITY/CDP/County': a.str.contains(r' CITY\b|\bCDP\b| County\b', regex=True).values,
 'house no. letter suffix': a.str.contains(r'\b\d+[A-Z]\b', regex=True).values,
 'house no. leading 0': a.str.contains(r'(^|[ ,#])0\d+', regex=True).values,
 'acronym name (<=4 caps)': n.str.fullmatch(r'[A-Z]{2,4}').values,
 'name==owner initials': (nl.str.replace(r'[^a-z]', '', regex=True) == init.values) & (init.str.len() >= 2).values,
 'legal form doubled': n.str.contains(r'(Limited|Ltd|LLC|Inc)\W+(Limited|Ltd|LLC|Inc)\b', regex=True).values,
 'bracketed legal': n.str.contains(r'[\[(](Limited|Ltd|LLC|L\.L\.C\.|Inc|Corp|Center|Partners)[\])]', regex=True).values,
 'appended filler word': n.str.contains(r'\b(Center|Enterprises|Services|Service|Partners|Trading|Authority|Group|Federation|Global)\W*$', regex=True).values,
 'honorific prefix': n.str.contains(r'^(Dr|Shri|Sri|Smt|Mr|Mrs|M/s)\b', regex=True).values,
 'legal form moved to front': n.str.contains(r'^(Private|Limited|Inc|LLC|Pvt)\b', regex=True).values,
 'id/phone suffix': n.str.contains(r'(- \d{6,}|#\d{4,})\s*$', regex=True).values,
 'injected accent (non-FR)': n.str.contains(r'[À-ɏ]', regex=True).values,
 'name UPPER': ((n.str.upper() == n) & n.str.contains('[A-Z]', regex=True)).values,
 'name lower': ((n.str.lower() == n) & n.str.contains('[a-z]', regex=True)).values,
}
v_true = (own >= 0) & (fo[np.maximum(own, 0)] >= 90) & (a.str.strip() != '').values & ~n.str.contains(r'[\u0900-\u0DFF]', regex=True).values
v_dec = (own < 0) & (aid >= 0) & (fo[np.maximum(aid, 0)] >= 90)
ok = m & (aid == own); wrong = m & (aid != own); unl = ~m
rows = []
for name, f in ops.items():
    t = v_true & f; dcy = v_dec & f
    rows.append((name, int(t.sum()), ok[t].mean(), unl[t].mean(), wrong[t].mean(), int((unl | wrong)[t].sum()), int(dcy.sum()), m[dcy].mean() if dcy.any() else 0,
                 f[own >= 0].mean() / max(f[own < 0].mean(), 1e-9)))
base = v_true & ~np.any(list(ops.values()), axis=0)
print(f'{"no op detected":28s} true {base.sum():8,d}  correct {ok[base].mean():.3f}')
print(f'{"operation":28s} {"true":>8s} {"correct":>8s} {"unlinked":>8s} {"wrong":>6s} {"LOST":>6s} | {"decoys":>6s} {"dec.linked":>10s} {"dup/decoy odds":>14s}')
for r in sorted(rows, key=lambda r: -r[5]):
    print(f'{r[0]:28s} {r[1]:8,d} {r[2]:8.3f} {r[3]:8.3f} {r[4]:6.3f} {r[5]:6,d} | {r[6]:6,d} {r[7]:10.3f} {r[8]:14.2f}')
