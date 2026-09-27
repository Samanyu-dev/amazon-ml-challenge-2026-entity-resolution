"""TRACK 1: error classes of the CURRENT US/India pipeline (leak-free stage 2 + meta3), folds 90-99, oracle gain per class."""
import sys, json, re
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd, glob
from catboost import CatBoostClassifier
from decode import load, decode, evaluate
from fingerprints import rd
S = W + 'artifacts_v3/stage2_v7e_usin_owner/'
d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid, q, mg = d['aid'].astype(np.int64), d['q'].astype(np.float64), d['margin'].astype(np.float64)
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy').astype(np.int64)
X = np.load('/private/tmp/claude-501/fable_review/meta3_X_train.npy'); nx2 = X.shape[1] - 8
cols = [c for c in range(nx2) if c != 16] + list(range(nx2, X.shape[1]))
band = (aid >= 0) & (q > .003) & (q < .998); fa = np.where(aid >= 0, folds[np.maximum(aid, 0)], -1); y = (own == aid).astype(int)
tr = band & (fa >= 80) & (fa < 90)
M = CatBoostClassifier(iterations=800, depth=5, learning_rate=0.05, verbose=0, thread_count=5, random_seed=0).fit(X[tr][:, cols], y[tr])
v = q.copy(); v[band] = M.predict_proba(X[band][:, cols])[:, 1]; np.save('/private/tmp/claude-501/fable_review/meta3_v_train.npy', v)
m = decode(aid, v, mg, **k) & (aid >= 0)
r0, f0 = evaluate(m, aid, own, folds, truth, 90, 100); print(f'current US/India val F {r0["macro_f05"]:.6f}', flush=True)
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True); s1 = rd('train/train_source1.tsv')
cols_ = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z', 'legal']
c1 = pd.read_csv(W + 'artifacts_v3/train_s1.norm.tsv', sep='\t', header=None, names=cols_, dtype=str, quoting=3, keep_default_na=False, usecols=[3, 4])
ct = pd.concat([pd.read_csv(W + f'artifacts_v3/train_s{s}.norm.tsv', sep='\t', header=None, names=cols_, dtype=str, quoting=3, keep_default_na=False, usecols=[3, 4]) for s in (2, 3)], ignore_index=True)
gname = c1.core.map(c1.core.value_counts()).values; gaddr = c1.addr.map(c1.addr.value_counts()).values
SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')]); has = np.zeros(len(own), bool)
for f in sorted(glob.glob(W + 'artifacts_v3/train_scores/candidates-*.bin')):
    c = np.fromfile(f, dtype=SF); ok = own[c['tid']] == c['aid'].astype(np.int64); has[c['tid'][ok]] = True
ae = tg.business_address.values == ''; nm = tg.business_name
flags = {'blank address': ae, 'indic script': nm.str.contains(r'[ऀ-෿]', regex=True).values,
         'acronym (2-4 caps)': nm.str.strip().str.fullmatch(r'[A-Z]{2,4}').values, 'short name (<=1 core word)': ct.core.str.split().str.len().fillna(0).values <= 1,
         'numeric token in name': nm.str.contains(r'\d', regex=True).values, 'S3 source': np.arange(len(tg)) >= 5034616,
         'small margin (<0.2)': mg < 0.2}
def pairflags(t, a):
    same_core = ct.core.values[t] == c1.core.values[a]; same_addr = (ct.addr.values[t] == c1.addr.values[a]) & (ct.addr.values[t] != '')
    return {'exact core name': same_core, 'exact address': same_addr, 'namesake anchor (core shared)': gname[a] > 1, 'shared-address anchor': gaddr[a] > 1}
val_t = lambda t_anchor: folds[np.maximum(t_anchor, 0)] >= 90
miss = np.flatnonzero((own >= 0) & ~(m & (aid == own)) & val_t(own)); fl = np.flatnonzero(m & (aid != own) & val_t(aid))
n = len(folds); tp = np.bincount(aid[m & (aid == own)], minlength=n).astype(float); fp = np.bincount(aid[m & (aid != own)], minlength=n).astype(float)
F = lambda tp_, fp_: np.where((truth == 0) & (tp_ + fp_ == 0), 1.0, np.where(tp_ == 0, 0.0, 1.25 * (tp_ / np.maximum(tp_ + fp_, 1)) * (tp_ / np.maximum(truth, 1)) / (0.25 * tp_ / np.maximum(tp_ + fp_, 1) + tp_ / np.maximum(truth, 1))))
vm = folds >= 90; base = F(tp, fp)[vm].mean()
cause = np.where(~has[miss], 'blocking miss', np.where(aid[miss] != own[miss], 'wrong rank', 'rejected at rank 1'))
fcause = np.where(own[fl] < 0, 'false link: decoy', 'false link: other entity')
pm = pairflags(miss, own[miss]); pf = pairflags(fl, aid[fl])
rows = []
def add(name, msel, fsel):
    g1 = F(tp + np.bincount(own[miss[msel]], minlength=n), fp)[vm].mean() - base if msel.any() else 0
    g2 = F(tp, fp - np.bincount(aid[fl[fsel]], minlength=n))[vm].mean() - base if fsel.any() else 0
    rows.append((name, int(msel.sum()), int(fsel.sum()), g1, g2))
for c in ('blocking miss', 'wrong rank', 'rejected at rank 1'): add('MISS ' + c, cause == c, np.zeros(len(fl), bool))
for c in ('false link: decoy', 'false link: other entity'): add(c, np.zeros(len(miss), bool), fcause == c)
for nm_, f in flags.items(): add(nm_, f[miss], f[fl])
for nm_ in pm: add(nm_, pm[nm_], pf[nm_])
add('exact core name + namesake anchor', pm['exact core name'] & pm['namesake anchor (core shared)'], pf['exact core name'] & pf['namesake anchor (core shared)'])
add('shared addr + NOT exact name', pm['shared-address anchor'] & ~pm['exact core name'], pf['shared-address anchor'] & ~pf['exact core name'])
print(f'misses {len(miss):,}  false links {len(fl):,}  total loss {1 - base:.5f}')
print(f'{"class":36s} {"misses":>7s} {"falseL":>7s} {"oracle gain(miss)":>18s} {"oracle gain(FP)":>16s}')
for r in sorted(rows, key=lambda r: -(r[3] + r[4])): print(f'{r[0]:36s} {r[1]:7,d} {r[2]:7,d} {r[3]:18.5f} {r[4]:16.5f}')
