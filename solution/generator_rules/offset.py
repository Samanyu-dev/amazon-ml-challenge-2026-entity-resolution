import sys, json, re
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode, evaluate
from fingerprints import rd
def num(x):
    x = re.sub(r'(?<=\d)[-/ ](?=\d)', '', x); m = re.search(r'\d+', x); return int(m.group(0)) if m and len(m.group(0)) < 9 else -1
S = W + 'artifacts_v3/stage2_v7e_usin_owner/'
d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid, mg = d['aid'].astype(np.int64), d['margin']; v = np.load('/private/tmp/claude-501/fable_review/meta3_v_train.npy')
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
m = decode(aid, v, mg, **k) & (aid >= 0)
tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True).business_address.values
s1 = rd('train/train_source1.tsv').business_address.values
lk = np.flatnonzero(m); nt = np.array([num(tg[t]) for t in lk]); ns = np.array([num(s1[a]) for a in aid[lk]])
off = np.where((nt >= 0) & (ns >= 0), nt - ns, -99999); ok = own[lk] == aid[lk]
for nm_, s in (('offset 0', off == 0), ('offset +1..+20 (target higher)', (off >= 1) & (off <= 20)), ('offset -20..-1 (target lower)', (off <= -1) & (off >= -20)), ('offset > +20', off > 20), ('offset < -20', (off < -20) & (off > -99999))):
    print(f'TRAIN linked, {nm_:32s}: n {s.sum():8,d}  precision {ok[s].mean() if s.any() else 0:.4f}  false {int((~ok[s]).sum()):,}')
r0, f0 = evaluate(m, aid, own, folds, truth, 90, 100)
for lo, hi in ((1, 20), (1, 10), (1, 5), (1, 3)):
    rem = lk[(off >= lo) & (off <= hi)]; m2 = m.copy(); m2[rem] = False
    r1, f1 = evaluate(m2, aid, own, folds, truth, 90, 100); g = f1 - f0
    print(f'  remove links with offset +{lo}..+{hi}: val {r0["macro_f05"]:.6f} -> {r1["macro_f05"]:.6f} gain {g.mean():+.6f} ± {g.std() / np.sqrt(len(g)):.1e}')
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z', 'legal']
l1 = pd.read_csv(W + 'artifacts_v3/train_s1.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[10]).legal.values
lt = pd.concat([pd.read_csv(W + f'artifacts_v3/train_s{s}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[10]) for s in (2, 3)], ignore_index=True).legal.values
ldiff = (lt[lk] != '') & (l1[aid[lk]] != '') & (lt[lk] != l1[aid[lk]])
for nm_, s in (('+1..+20 & legal DIFF', (off >= 1) & (off <= 20) & ldiff), ('+1..+20 & legal same/absent', (off >= 1) & (off <= 20) & ~ldiff)):
    print(f'TRAIN linked {nm_:30s}: n {s.sum():,} precision {ok[s].mean() if s.any() else 0:.4f}')
    rem = lk[s]; m2 = m.copy(); m2[rem] = False; r1, f1 = evaluate(m2, aid, own, folds, truth, 90, 100); g = f1 - f0
    print(f'    remove -> gain {g.mean():+.6f} ± {g.std() / np.sqrt(len(g)):.1e}')
