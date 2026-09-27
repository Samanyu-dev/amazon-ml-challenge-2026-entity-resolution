"""F0.5 loss attribution on validation (folds 90-99, leak-free + meta-free base): gain if ONLY category X errors were fixed."""
import sys, re, json, glob
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode
from fingerprints import rd
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
core1 = pd.read_csv(W + 'artifacts/train_s1.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[3]).core.values
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy').astype(np.int64)
n = len(folds); ns = pd.Series(core1).map(pd.Series(core1).value_counts()).values   # namesake group size by normalized core name
syn = np.load('/private/tmp/claude-501/fable_review/synth_train.npy')
ae = (tg.business_address.values == ''); indic = tg.business_name.str.contains(r'[ऀ-෿]', regex=True).values
def cat(t, anchor):
    c = np.full(len(t), 'other', object)
    c[ae[t] & (ns[anchor] > 1)] = 'blank addr, namesakes'; c[ae[t] & (ns[anchor] == 1)] = 'blank addr, unique name'
    c[~ae[t] & indic[t]] = 'indic script'; c[~ae[t] & syn[t]] = 'synthetic name'
    return c
lk = np.flatnonzero(m & (aid >= 0)); good = own[lk] == aid[lk]
tp = np.bincount(aid[lk][good], minlength=n).astype(float); fp_all = np.bincount(aid[lk][~good], minlength=n).astype(float)
t = np.flatnonzero(own >= 0); miss = t[~(m[t] & (aid[t] == own[t]))]
F = lambda tp_, fp_: np.where((truth == 0) & (tp_ + fp_ == 0), 1.0, np.where(tp_ == 0, 0.0, 1.25 * (tp_ / np.maximum(tp_ + fp_, 1)) * (tp_ / np.maximum(truth, 1)) / (0.25 * tp_ / np.maximum(tp_ + fp_, 1) + tp_ / np.maximum(truth, 1))))
v = folds >= 90; base = F(tp, fp_all)[v].mean(); print(f'val F {base:.6f}  loss {1 - base:.6f}')
mc = cat(miss, own[miss]); fl = lk[~good]; fc = cat(fl, aid[fl])
for c in ['blank addr, namesakes', 'blank addr, unique name', 'indic script', 'synthetic name', 'other']:
    add = np.bincount(own[miss[mc == c]], minlength=n); rem = np.bincount(aid[fl[fc == c]], minlength=n)
    g1 = F(tp + add, fp_all)[v].mean() - base; g2 = F(tp, fp_all - rem)[v].mean() - base
    print(f'{c:26s} misses {int(add[v].sum()):6,d} -> fix gain {g1:+.5f} | false links {int(rem[v].sum()):5,d} -> fix gain {g2:+.5f}')
# 'other' misses: owner in candidates? top?
o = miss[mc == 'other']; o = o[v[own[o]]]
print(f'other misses {len(o):,}: owner is our top {np.mean(aid[o] == own[o]):.3f}')
s1 = rd('train/train_source1.tsv'); q = d['q']
u = miss[mc == 'blank addr, unique name']; u = u[v[own[u]]]
SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')]); has = np.zeros(len(own), bool)
for f in sorted(glob.glob(W + 'artifacts_v3/train_scores/candidates-*.bin')):
    c = np.fromfile(f, dtype=SF); ok = own[c['tid']] == c['aid'].astype(np.int64); has[c['tid'][ok]] = True
print(f'\nblank-addr unique-name misses {len(u):,}: owner is top {np.mean(aid[u] == own[u]):.3f}; owner in candidates {has[u].mean():.3f}; owner-group size of TOP anchor >1: {np.mean(ns[aid[u]] > 1):.3f}')
for t in np.random.default_rng(3).choice(u, 30, replace=False):
    print(f'q={q[t]:.2f} T: {tg.business_name[t]}  | OWNER: {s1.business_name[own[t]]}  | TOP: {s1.business_name[aid[t]]}')
o = miss[mc == 'other']; o = o[v[own[o]]]
print('\n=== OTHER misses (address present, Latin, not made-up)')
for t in np.random.default_rng(9).choice(o, 30, replace=False):
    print(f'q={q[t]:.2f} {"TOP=OWNER" if aid[t] == own[t] else "         "} T: {tg.business_name[t]} | {tg.business_address[t]}\n        OWNER: {s1.business_name[own[t]]} | {s1.business_address[own[t]]}')
f_ = fl[fc == 'other']; f_ = f_[v[aid[f_]]]
print('\n=== OTHER false links')
for t in np.random.default_rng(9).choice(f_, 15, replace=False):
    print(f'q={q[t]:.2f} T: {tg.business_name[t]} | {tg.business_address[t]}\n        LINKED: {s1.business_name[aid[t]]} | {s1.business_address[aid[t]]}' + (f'\n        OWNER : {s1.business_name[own[t]]} | {s1.business_address[own[t]]}' if own[t] >= 0 else '  (decoy)'))
