"""TRACK 2: A = v6 family (French decisions), B = v7e family (e5-large + decoy feats), J = stage 1. Labelled US/India val:
who is right when A and B disagree, by pattern x slice. Real France: pattern counts (A at odds 1.4 = fr14)."""
import sys, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np, pandas as pd
from decode import load, decode
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z', 'legal']
def dec(S, split, odds=1.0):
    d = np.load(W + f'artifacts_v3/{S}/{split}_decisions.npz'); k = json.load(open(W + f'artifacts_v3/{S}/stage2_report.json'))['knobs']
    a = d['aid'].astype(np.int64); q = d['q'].astype(np.float64); q = q * odds / (q * odds + 1 - q)
    return np.where(decode(a, q, d['margin'], **k) & (a >= 0), a, -1), a, q
def slices(split, t, anchor):
    s1 = pd.read_csv(W + f'artifacts_v3/{split}_s1.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[3, 4])
    tg = pd.concat([pd.read_csv(W + f'artifacts_v3/{split}_s{k}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[3, 4]) for k in (2, 3)], ignore_index=True)
    a = np.maximum(anchor, 0); ae = tg.addr.values[t] == ''
    ex = tg.core.values[t] == s1.core.values[a]; ea = (tg.addr.values[t] == s1.addr.values[a]) & ~ae
    return np.where(ae, 'blank addr', np.where(ex & ea, 'exact name+addr', np.where(ex, 'exact name, addr differs', np.where(ea, 'same addr, name differs', 'both differ'))))
def pattern(A, B):
    return np.where((A >= 0) & (B < 0), 'A links, B rejects', np.where((A < 0) & (B >= 0), 'B links, A rejects', 'different owner'))
# ---- labelled US/India val
A, aA, qA = dec('exp_owner_v6cov', 'train'); B, aB, qB = dec('stage2_v7e_usin_owner', 'train')
_, own, _, q0, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy'); a0 = _
folds = np.load(W + 'artifacts/train_anchor_folds.npy')
ref = np.where(own >= 0, own, np.maximum(A, B)); val = (ref >= 0) & (folds[np.maximum(ref, 0)] >= 90)
dis = np.flatnonzero((A != B) & val); pat = pattern(A[dis], B[dis]); sl = slices('train', dis, np.where(B[dis] >= 0, B[dis], A[dis]))
okA = A[dis] == np.where(own[dis] >= 0, own[dis], -1); okB = B[dis] == np.where(own[dis] >= 0, own[dis], -1)
df = pd.DataFrame({'pattern': pat, 'slice': sl, 'A_right': okA, 'B_right': okB})
print(f'US/India val: A!=B on {len(dis):,} targets; A right {okA.mean():.3f}, B right {okB.mean():.3f}')
T = df.groupby(['pattern', 'slice']).agg(n=('A_right', 'size'), A_right=('A_right', 'mean'), B_right=('B_right', 'mean')).round(3)
# ---- France
At, _, _ = dec('stage2_v6_pp', 'test', 1.4); Bt, _, _ = dec('stage2_v7e_usin_owner', 'test')
ct = np.load(W + 'artifacts/test_anchor_countries.npy', allow_pickle=True)
fr = ((At >= 0) & (ct[np.maximum(At, 0)] == 'France')) | ((Bt >= 0) & (ct[np.maximum(Bt, 0)] == 'France'))
dt = np.flatnonzero(fr & (At != Bt)); dft = pd.DataFrame({'pattern': pattern(At[dt], Bt[dt]), 'slice': slices('test', dt, np.where(Bt[dt] >= 0, Bt[dt], At[dt]))})
T['FRANCE n'] = dft.groupby(['pattern', 'slice']).size(); T = T.fillna(0)
pd.set_option('display.width', 200); print(T.to_string())
np.savez('/private/tmp/claude-501/fable_review/t2_france_dis.npz', t=dt, A=At[dt], B=Bt[dt], pattern=dft.pattern.values, slice=dft.slice.values)
