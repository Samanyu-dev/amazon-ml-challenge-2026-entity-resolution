"""Label-free France diagnostic: link rate per generator operation, France (v7d France model) vs US/India (v7e)."""
import sys, re, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import decode
from fingerprints import rd
tg = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')], ignore_index=True)
n, a, c = tg.business_name, tg.business_address, tg.country.values
syn = np.load('/private/tmp/claude-501/fable_review/synth_test.npy')
odds = lambda q, r: q * r / (q * r + 1 - q)
L = {}
for S, r, cs in (('stage2_v7e_usin', 1.0, ('US', 'India')), ('stage2_v6_pp', 0.5, ('France',))):
    d = np.load(W + f'artifacts_v3/{S}/test_decisions.npz'); k = json.load(open(W + f'artifacts_v3/{S}/stage2_report.json'))['knobs']
    m = decode(d['aid'].astype(np.int64), odds(d['q'].astype(np.float64), r), d['margin'], **k)
    for x in cs: L[x] = m
ops = {
 'ALL records': np.ones(len(tg), bool),
 'synthetic name': syn,
 'address empty': (a.str.strip() == '').values,
 'domain name': n.str.contains(r'\.c[o0]m\b|c[o0]m$', regex=True).values,
 'appended filler word': n.str.contains(r'\b(?:Center|Enterprises|Services|Service|Partners|Trading|Authority|Group|Federation|Global)\W*$', regex=True).values,
 'name UPPER': ((n.str.upper() == n) & n.str.contains('[A-Z]', regex=True)).values,
 'name lower': ((n.str.lower() == n) & n.str.contains('[a-z]', regex=True)).values,
 'honorific prefix': n.str.contains(r'^(?:Dr|Shri|Sri|Smt|Mr|Mrs|M/s|M\.)\b', regex=True).values,
 'legal form moved to front': n.str.contains(r'^(?:Private|Limited|Inc|LLC|Pvt|SARL|SAS|SA|EURL)\b', regex=True).values,
 'bracketed legal': n.str.contains(r'[\[(](?:Limited|Ltd|LLC|L\.L\.C\.|Inc|Corp|Center|Partners|SARL|SAS|SA)[\])]', regex=True).values,
 'address null/NA': a.str.contains(r'null|NULL|N/A', regex=True).values,
 'address ## / #': a.str.contains('#', regex=False).values,
 'id/phone suffix': n.str.contains(r'(?:- \d{6,}|#\d{4,})\s*$', regex=True).values,
 'acronym name (<=4 caps)': n.str.fullmatch(r'[A-Z]{2,4}').values,
 'website in record': n.str.contains('www.', regex=False).values | a.str.contains('www.', regex=False).values,
 'house no. letter suffix': a.str.contains(r'\b\d+[A-Z]\b', regex=True).values,
 'address CITY/CDP/County': a.str.contains(r' CITY\b|\bCDP\b| County\b|\bCEDEX\b', regex=True).values,
}
print(f'{"operation":28s} {"US n":>8s} {"US link":>7s} {"IN n":>8s} {"IN link":>7s} {"FR n":>8s} {"FR link":>7s}  FR-US/IN gap')
for name, f in ops.items():
    row = []
    for x in ('US', 'India', 'France'):
        s = f & (c == x); row += [s.sum(), L[x][s].mean() if s.any() else np.nan]
    print(f'{name:28s} {row[0]:8,d} {row[1]:7.3f} {row[2]:8,d} {row[3]:7.3f} {row[4]:8,d} {row[5]:7.3f}  {row[5] - (row[1] + row[3]) / 2:+.3f}')
