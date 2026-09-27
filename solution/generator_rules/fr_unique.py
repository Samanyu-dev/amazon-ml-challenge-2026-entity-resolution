"""Targets whose normalized core name == a UNIQUE S1 core name (same country). Train: true rate, our link rate.
Test: link rate per country (v8), plus French unlinked samples."""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode
from fingerprints import rd
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z', 'legal']
def setup(sp):
    rdn = lambda f: pd.read_csv(W + f'artifacts_v3/{sp}_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[1, 3, 4, 10])
    s1 = rdn('s1'); tg = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
    key1 = s1.ctry + '|' + s1.core; vc = key1.value_counts(); uniq = key1[key1.map(vc) == 1]
    look = pd.Series(uniq.index.values, index=uniq.values)
    kt = tg.ctry + '|' + tg.core; hit = kt.map(look)
    j = hit.fillna(-1).astype(np.int64).values; j[tg.core.values == ''] = -1
    return s1, tg, j
s1, tg, j = setup('train')
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
h = j >= 0; ae = tg.addr.values == ''
for nm, s in (('address present', h & ~ae), ('address blank', h & ae)):
    print(f'TRAIN unique-exact-name, {nm}: {s.sum():,}; owner == that S1 {np.mean(own[s] == j[s]):.4f}; we link it {np.mean(m[s] & (aid[s] == j[s])):.4f}; unlinked&true {int(np.sum(s & ~m & (own == j))):,}')
s1t, tgt, jt = setup('test')
v8 = {}
for l in list(open(W + '../../outputs/final_submission_v8/matching_results.tsv'))[1:]:
    e, mm = l.rstrip('\n').split('\t')
    for x in mm.split(','):
        if x: v8[x] = e
tid = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')], ignore_index=True)
s1id = rd('test/test_source1.tsv').entity_id.values
linked_to = np.array([v8.get(x, '') for x in tid.entity_id.values]); ht = jt >= 0; aet = tgt.addr.values == ''
for c in ('US', 'India', 'France'):
    for nm, s in (('address present', ht & ~aet & (tgt.ctry.values == c)), ('address blank', ht & aet & (tgt.ctry.values == c))):
        ok = linked_to[s] == s1id[jt[s]]
        print(f'TEST {c:6s} unique-exact-name, {nm}: {s.sum():,}; v8 links to that S1 {ok.mean():.4f}; unlinked {np.mean(linked_to[s] == ""):.4f}; linked elsewhere {np.mean((linked_to[s] != "") & ~ok):.4f}')
fr = np.flatnonzero(ht & (tgt.ctry.values == 'France') & (linked_to == ''))
raw1 = rd('test/test_source1.tsv')
for t in []:
    print(f'  T: {tid.business_name[t]} | {tid.business_address[t]}\n     S1: {raw1.business_name[jt[t]]} | {raw1.business_address[jt[t]]}')
np.save('/private/tmp/claude-501/fable_review/fr_unique_unlinked.npy', np.column_stack([fr, jt[fr]]))

import re
def num(x):
    x = re.sub(r'(?<=\d)[-/ ](?=\d)', '', x); m_ = re.search(r'\d+', x); return m_.group(0).lstrip('0') if m_ else ''
def split(tg_, s1_, j_, sel):
    idx = np.flatnonzero(sel)
    lt = tg_.legal.values[idx]; ls = s1_.legal.values[j_[idx]]
    nt = np.array([num(x) for x in tg_.addr.values[idx]]); ns_ = np.array([num(x) for x in s1_.addr.values[j_[idx]]])
    legal = np.where(lt == '', 'target no legal', np.where(lt == ls, 'legal same', 'legal DIFFERENT'))
    number = np.where((nt == '') | (ns_ == ''), 'no number', np.where(nt == ns_, 'number same', 'number DIFFERENT'))
    return idx, legal, number
if False: s1r, tgr, jr = setup('train'); idx, lg, nb = split(tgr, s1r, jr, (jr >= 0) & (tgr.addr.values != ''))
df = pd.DataFrame({'legal': lg, 'number': nb, 'true': own[idx] == jr[idx], 'linked': m[idx] & (aid[idx] == jr[idx])})
print('\nTRAIN unique-exact-name with address: true-copy rate (count)')
print(df.groupby(['legal', 'number']).agg(n=('true', 'size'), true_rate=('true', 'mean'), we_link=('linked', 'mean')).round(3))
idx, lg, nb = split(tgt, s1t, jt, (jt >= 0) & (tgt.addr.values != '') & (tgt.ctry.values == 'France'))
dft = pd.DataFrame({'legal': lg, 'number': nb, 'v8_links': linked_to[idx] == s1id[jt[idx]]})
print('\nFRANCE unique-exact-name with address: v8 link rate')
print(dft.groupby(['legal', 'number']).agg(n=('v8_links', 'size'), v8_link_rate=('v8_links', 'mean')).round(3))

def ntype(a, b):
    if not a or not b: return 'none'
    if a == b: return 'same'
    if a in b or b in a: return 'digit drop/add'
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1: return 'one digit changed'
    try:
        if abs(int(a) - int(b)) <= 20: return 'nearby (|diff|<=20)'
    except ValueError: pass
    return 'unrelated'
def ntypes(tg_, s1_, j_, idx):
    return np.array([ntype(num(tg_.addr.values[i]), num(s1_.addr.values[j_[i]])) for i in idx])
if False: idx, lg, nb = split(tgr, s1r, jr, (jr >= 0) & (tgr.addr.values != ''))
sel = nb == 'number DIFFERENT'; tt = ntypes(tgr, s1r, jr, idx[sel])
df = pd.DataFrame({'legal': lg[sel], 'ntype': tt, 'true': own[idx[sel]] == jr[idx[sel]]})
print('\nTRAIN unique-exact-name, number DIFFERENT, by number-change type: true-copy rate')
print(df.groupby(['legal', 'ntype']).true.agg(['size', 'mean']).round(3))
idx, lg, nb = split(tgt, s1t, jt, (jt >= 0) & (tgt.addr.values != '') & (tgt.ctry.values == 'France'))
sel = nb == 'number DIFFERENT'; tt = ntypes(tgt, s1t, jt, idx[sel])
dft = pd.DataFrame({'legal': lg[sel], 'ntype': tt, 'v8': linked_to[idx[sel]] == s1id[jt[idx[sel]]]})
print('\nFRANCE unique-exact-name, number DIFFERENT, by number-change type: v8 link rate')
print(dft.groupby(['legal', 'ntype']).v8.agg(['size', 'mean']).round(3))

# v9 France additions: unique exact core name, digit drop/add house-number change, legal same or target without legal, currently unlinked
idx, lg, nb = split(tgt, s1t, jt, (jt >= 0) & (tgt.addr.values != '') & (tgt.ctry.values == 'France'))
sel = nb == 'number DIFFERENT'; ii = idx[sel]; tt = ntypes(tgt, s1t, jt, ii); ll = lg[sel]
pick = (tt == 'digit drop/add') & np.isin(ll, ['legal same', 'target no legal']) & (linked_to[ii] == '')
add = np.column_stack([ii[pick], jt[ii[pick]]]); np.save('/private/tmp/claude-501/fable_review/fr_digit_add.npy', add)
print(f'\nFRANCE digit-drop additions (unlinked): {len(add):,}')
