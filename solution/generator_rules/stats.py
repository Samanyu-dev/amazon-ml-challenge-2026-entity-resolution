"""Full dataset statistics -> ~/Desktop/DATASET_STATISTICS.md"""
import sys, re, json, glob, unicodedata
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load
from fingerprints import rd
OUT = []; P = lambda s='': OUT.append(s)
def table(df, fl='{:,.4g}'):
    cols = list(df.columns); P('| ' + ' | '.join([''] + [str(c) for c in cols]) + ' |'); P('|' + '---|' * (len(cols) + 1))
    for i, r in df.iterrows(): P('| ' + ' | '.join([str(i)] + [(f'{v:,}' if isinstance(v, (int, np.integer)) else (fl.format(v) if isinstance(v, float) else str(v))) for v in r]) + ' |')
    P()
D = {(sp, k): rd(f'{sp}/{sp}_source{k}.tsv') for sp in ('train', 'test') for k in (1, 2, 3)}
C = ['US', 'India', 'France']
P('# Amazon ML Challenge 2026 — dataset statistics'); P('Computed from the raw organiser files (train/test, sources 1–3) and our pipeline artifacts. 27 Sep 2026.'); P()
# 1 files
P('## 1. Files and rows'); rows = {}
for (sp, k), df in D.items():
    rows[f'{sp} S{k}'] = {c: int((df.country == c).sum()) for c in C} | {'total': len(df), 'columns': ', '.join(df.columns)}
table(pd.DataFrame(rows).T)
P('ID format: `S1-<9 digits>`, `S2-<9 digits>`, `S3-<9 digits>`; IDs unique within each file: ' + ', '.join(f'{sp} S{k} {df.entity_id.is_unique}' for (sp, k), df in D.items())); P()
# 2 density
P('## 2. Density (S2+S3 records per S1 entity)'); r = {}
for sp in ('train', 'test'):
    for c in C:
        n1 = (D[sp, 1].country == c).sum(); nt = (D[sp, 2].country == c).sum() + (D[sp, 3].country == c).sum()
        if n1: r[f'{sp} {c}'] = {'S1': int(n1), 'S2+S3': int(nt), 'per S1': nt / n1, 'S2 per S1': (D[sp, 2].country == c).sum() / n1, 'S3 per S1': (D[sp, 3].country == c).sum() / n1}
table(pd.DataFrame(r).T, '{:.3f}')
# 3 ground truth
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
n2 = len(D['train', 2]); tg = pd.concat([D['train', 2], D['train', 3]], ignore_index=True); s1 = D['train', 1]
truth = np.bincount(own[own >= 0], minlength=len(s1)); src3 = np.arange(len(own)) >= n2
t2 = np.bincount(own[(own >= 0) & ~src3], minlength=len(s1)); t3 = np.bincount(own[(own >= 0) & src3], minlength=len(s1))
P('## 3. Ground truth (train)')
P(f'- True links: **{int((own >= 0).sum()):,}** (S2 {int(((own >= 0) & ~src3).sum()):,}, S3 {int(((own >= 0) & src3).sum()):,}); mean per S1 **{truth.mean():.3f}**, median {np.median(truth):.0f}, max {truth.max()}')
P(f'- Singletons (S1 with no copy): **{np.mean(truth == 0):.4f}** ({int((truth == 0).sum()):,}); max S2 copies per S1 {t2.max()}, max S3 copies {t3.max()}')
P(f'- Decoys (S2/S3 records with no owner): **{np.mean(own < 0):.4f}** of records ({int((own < 0).sum()):,}); S2 {np.mean(own[~src3] < 0):.4f}, S3 {np.mean(own[src3] < 0):.4f}')
P(f'- Every record belongs to at most one S1; all links stay within one country.'); P()
ct1 = s1.country.values
dist = pd.DataFrame({c: pd.Series(truth[ct1 == c]).value_counts(normalize=True).sort_index() for c in ('US', 'India')}).fillna(0)
P('Copies per S1 entity (share of entities):'); table(dist.T.round(4), '{:.4f}')
P('S2 copies × S3 copies per entity (share):'); ct = pd.crosstab(np.minimum(t2, 5), np.minimum(t3, 6), normalize=True).round(4); table(ct, '{:.4f}')
tc = tg.country.values; P('Decoy share by country: ' + ', '.join(f'{c} {np.mean(own[tc == c] < 0):.4f}' for c in ('US', 'India'))); P()
# 4 text properties
def props(df):
    n, a = df.business_name, df.business_address
    return {'name empty': (n.str.strip() == '').mean(), 'address empty': (a.str.strip() == '').mean(),
            'name chars (mean)': n.str.len().mean(), 'name words (mean)': n.str.split().str.len().mean(),
            'address chars (mean)': a.str.len().mean(), 'address words (mean)': a.str.split().str.len().mean(),
            'name ASCII only': n.str.fullmatch(r'[\x00-\x7F]*').mean(), 'name accented Latin': n.str.contains(r'[À-ɏ]', regex=True).mean(),
            'name Indic script': n.str.contains(r'[ऀ-෿]', regex=True).mean(), 'address Indic script': a.str.contains(r'[ऀ-෿]', regex=True).mean(),
            'name UPPERCASE': ((n.str.upper() == n) & n.str.contains('[A-Z]', regex=True)).mean(), 'name lowercase': ((n.str.lower() == n) & n.str.contains('[a-z]', regex=True)).mean(),
            'address has digit': a.str.contains(r'\d', regex=True).mean()}
P('## 4. Text properties (share of records unless "mean")'); r = {}
for (sp, k), df in D.items():
    for c in C:
        x = df[df.country == c]
        if len(x): r[f'{sp} S{k} {c}'] = props(x)
table(pd.DataFrame(r).T, '{:.3f}')
SCR = {'Devanagari': r'[ऀ-ॿ]', 'Bengali': r'[ঀ-৿]', 'Gurmukhi': r'[਀-੿]', 'Gujarati': r'[઀-૿]', 'Odia': r'[଀-୿]', 'Tamil': r'[஀-௿]', 'Telugu': r'[ఀ-౿]', 'Kannada': r'[ಀ-೿]', 'Malayalam': r'[ഀ-ൿ]'}
P('Indic scripts in India target names (count):'); r = {}
for sp in ('train', 'test'):
    x = pd.concat([D[sp, 2], D[sp, 3]]); x = x[x.country == 'India'].business_name
    r[sp] = {s: int(x.str.contains(p, regex=True).sum()) for s, p in SCR.items()}
table(pd.DataFrame(r).T)
# 5 legal forms
LEG = r'\b(Private Limited|Pvt\.? Ltd\.?|Limited|Ltd\.?|LLP|LLC|L\.L\.C\.|Inc\.?|Incorporated|Corp\.?|Corporation|Co\.?|Company|PLLC|P\.C\.|LP|SARL|SAS|SASU|SA|EURL|EI|SCI|SNC|Sàrl|GmbH)\s*$'
P('## 5. Legal form at end of S1 name (share of S1)'); r = {}
for sp in ('train', 'test'):
    for c in C:
        x = D[sp, 1][D[sp, 1].country == c].business_name
        if len(x): r[f'{sp} {c}'] = x.str.extract(LEG, flags=re.I)[0].str.upper().str.replace('.', '', regex=False).fillna('(none)').value_counts(normalize=True).head(8).round(3).to_dict()
for kk, v in r.items(): P(f'- **{kk}**: ' + ', '.join(f'{a} {b}' for a, b in v.items()))
P()
# 6 namesakes & shared addresses
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
P('## 6. Namesakes and shared addresses in S1')
r = {}
for sp in ('train', 'test'):
    nm = pd.read_csv(W + f'artifacts/{sp}_s1.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[1, 3, 4])
    for c in C:
        x = nm[nm.ctry == c]
        if not len(x): continue
        g = x.core.map(x.core.value_counts()); a = x.addr.map(x.addr.value_counts())
        r[f'{sp} {c}'] = {'S1': len(x), 'in namesake group (same core name)': (g > 1).mean(), 'mean group size (if >1)': g[g > 1].mean(), 'max group': int(g.max()),
                          'distinct core names': x.core.nunique(), 'address shared with another S1': ((a > 1) & (x.addr != '')).mean(), 'S1 address empty': (x.addr == '').mean()}
table(pd.DataFrame(r).T, '{:.3f}')
# 7 generator operations
SYL = ['onyx','kor','delta','xylo','wex','novi','brix','halo','gild','umbra','faye','nex','kelo','zeph','calo','aria','lum','jax','cira','dova','veo','nyla','tavo','mira','yuma','ecto','riza','lyra','flux','pyra','belo','drex','arc','sol','evo','vera','syn','orbi','iri','avi','vio','vantage','zeta','quo','io','x']
def ops(df):
    n, a = df.business_name, df.business_address; nl = n.str.lower().str.replace('1', 'l').str.replace('0', 'o').str.replace('5', 's').str.strip()
    return {'blank address': (a.str.strip() == ''), 'made-up syllable name': nl.str.fullmatch('(?:' + '|'.join(SYL) + '){2,}'),
            'acronym name (2-4 caps)': n.str.strip().str.fullmatch(r'[A-Z]{2,4}'), 'domain name (.com)': n.str.contains(r'\.c[o0]m\b', regex=True, flags=re.I),
            'alias (dba/aka/formerly/t/a)': n.str.lower().str.contains(r'\b(?:dba|d/b/a|aka|formerly|doing business as|t/a|f/k/a)\b', regex=True),
            'Indic-script name': n.str.contains(r'[ऀ-෿]', regex=True), 'filler word appended': n.str.contains(r'\b(?:Center|Enterprises|Services|Partners|Trading|Group|Groupe|Holding|International|Distribution|Développement|Participations)\W*$', regex=True),
            'honorific prefix': n.str.contains(r'^(?:Dr|Shri|Sri|Smt|Mr|Mrs|M/s)\b', regex=True), 'legal form moved to front': n.str.contains(r'^(?:Private|Limited|Inc|LLC|Pvt|SARL|SAS|SA|EURL)\b', regex=True),
            'bracketed word': n.str.contains(r'[\[(]', regex=True), 'id/phone suffix': n.str.contains(r'(?:- \d{6,}|#\d{4,}|\(ID: \d+\))', regex=True),
            'OCR swap (lnc, 5ervice, c0m)': n.str.contains(r'\bl[bcdfgjkmnpqrstvwxz]|\b5[a-z]|[a-z]0[a-z]|[a-z]1[a-z]', regex=True),
            'address ##': a.str.contains('##', regex=False), 'address null/N/A': a.str.contains(r'null|NULL|N/A', regex=True),
            'address hyphenated number': a.str.contains(r'\b\d-\d{2,}\b', regex=True), 'address leading-zero number': a.str.contains(r'\b00?\d+', regex=True) & a.str.contains(r'\b0\d', regex=True),
            'address CITY/CDP/County': a.str.contains(r' CITY\b|\bCDP\b| County\b', regex=True), 'name UPPERCASE': (n.str.upper() == n) & n.str.contains('[A-Z]', regex=True)}
P('## 7. Generator operations on S2/S3 records')
P('Share of records carrying each operation; "true-copy rate" = share of train records with that operation that belong to an S1 (the rest are decoys).'); r = {}
tro = ops(tg)
for name, f in tro.items():
    f = f.fillna(False).values; row = {'train true-copy rate': float(np.mean(own[f] >= 0)) if f.any() else np.nan}
    for c in ('US', 'India'): row[f'train {c}'] = float(f[tc == c].mean())
    r[name] = row
te = pd.concat([D['test', 2], D['test', 3]], ignore_index=True); teo = ops(te)
for name, f in teo.items():
    f = f.fillna(False).values
    for c in C: r[name][f'test {c}'] = float(f[te.country.values == c].mean())
table(pd.DataFrame(r).T, '{:.4f}')
# 8 pipeline
P('## 8. Our pipeline on this data')
P('- Candidate pairs (blocking): 117,097,579 (67.6 per S1); v8 adds 1,333,971 from address-number and acronym blocking. Recall of true links in candidates: 98.48% (train).')
P('- Stage-2 uncertain band: q in [0.005, 0.999].')
P('- Validation (US/India, folds 90–99, 220,938 entities), leak-free: stage 2 0.988409 → + meta 0.988688; Indic rescue +0.000431; acronym rule +0.00002 (US/India).')
P('- Leaderboard: v7d 0.984931; v7d + acronym rule **0.985608**.')
v8 = [l.rstrip('\n').split('\t') for l in list(open(W + '../../outputs/final_submission_v8/matching_results.tsv'))[1:]]
cnt = np.array([len([x for x in m_.split(',') if x]) for _, m_ in v8]); c1 = D['test', 1].country.values
r = {c: {'S1': int((c1 == c).sum()), 'links': int(cnt[c1 == c].sum()), 'links per S1': cnt[c1 == c].mean(), 'empty (no link)': float(np.mean(cnt[c1 == c] == 0))} for c in C}
P('v8 submission, per country:'); table(pd.DataFrame(r).T, '{:.4f}')
P('Validation loss decomposition (leak-free, 2,561 entity-points lost): partial misses 61%; entities with copies but nothing predicted 21%; entities with a false link 15%; singletons wrongly linked 3%. Blank-address namesake copies ≈ 0.0051 of the 0.0116 loss.')
open('/Users/apple/Desktop/DATASET_STATISTICS.md', 'w').write('\n'.join(OUT)); print('\n'.join(OUT))
