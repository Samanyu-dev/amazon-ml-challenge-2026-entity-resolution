"""Acronym rule v2 = v1 OR (unique initials-matching candidate with the same first house number and sim>=0.3). Gate on train, apply to test."""
import sys, re, json, glob
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
from decode import load, decode, evaluate
from fingerprints import rd
FR = '/private/tmp/claude-501/fable_review/'
num = lambda x: (lambda m: m.group(1).lstrip('0') if m else '')(re.search(r'(\d+)', x))
def rule(sp):
    Z = np.load(FR + f'acr_{sp}.npz'); t, a, s, rank, gap = Z['tid'], Z['aid'], Z['sim'], Z['rank'], Z['gap']
    s1 = rd(f'{sp}/{sp}_source1.tsv').business_address.values; tg = pd.concat([rd(f'{sp}/{sp}_source2.tsv'), rd(f'{sp}/{sp}_source3.tsv')], ignore_index=True).business_address.values
    tn = np.array([num(tg[x]) for x in t]); an = np.array([num(s1[x]) for x in a])
    nm = (tn == an) & (tn != '') & (s >= 0.3)
    cnt = pd.Series(nm).groupby(t).transform('sum').values
    v1 = (rank == 0) & (s >= .3) & (gap >= .3); v2 = nm & (cnt == 1)
    fire = v1 | v2
    idx = np.flatnonzero(fire); _, f1 = np.unique(t[idx], return_index=True); idx = idx[f1]   # first firing candidate per target
    # if v1 and v2 disagree on a target, drop the target
    both = pd.DataFrame({'t': t[fire], 'a': a[fire]}).groupby('t').a.nunique(); bad = set(both[both > 1].index)
    idx = np.array([i for i in idx if t[i] not in bad]); return t, a, idx, v1
if __name__ == '__main__':
    t, a, idx, v1 = rule('train')
    _, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
    folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy')
    S = W + 'artifacts_v3/exp_owner_split/'; d = np.load(S + 'train_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
    aid = d['aid'].astype(np.int64); m = decode(aid, d['q'].astype(np.float64), d['margin'], **k)
    T = np.unique(t); only2 = idx[~v1[idx]]
    print(f'train: v2 fires {len(idx):,} of {len(T):,} ({len(idx)/len(T):.3f}); precision {np.mean(own[t[idx]] == a[idx]):.4f}; new-by-number-only {len(only2):,} precision {np.mean(own[t[only2]] == a[only2]):.4f}')
    un = idx[~m[t[idx]]]; na, nmk = aid.copy(), m.copy(); na[t[un]] = a[un]; nmk[t[un]] = True
    r0, f0 = evaluate(m, aid, own, folds, truth, 90, 100); r1, f1 = evaluate(nmk, na, own, folds, truth, 90, 100); g = f1 - f0
    print(f'  unlinked additions {len(un):,} precision {np.mean(own[t[un]] == a[un]):.4f}; val gain {g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e}')
    t, a, idx, v1 = rule('test'); ctry = np.load(W + 'artifacts/test_anchor_countries.npy', allow_pickle=True)
    odds = lambda q, r: q * r / (q * r + 1 - q); L = {}
    for S2, r_, cs in (('stage2_v7e_usin', 1.0, ('US', 'India')), ('stage2_v6_pp', 0.5, ('France',))):
        dt = np.load(W + f'artifacts_v3/{S2}/test_decisions.npz'); kt = json.load(open(W + f'artifacts_v3/{S2}/stage2_report.json'))['knobs']
        at = dt['aid'].astype(np.int64); mt = decode(at, odds(dt['q'].astype(np.float64), r_), dt['margin'], **kt)
        for c in cs: L[c] = (at, mt)
    out = []
    for c in ('US', 'India', 'France'):
        at, mt = L[c]; i = idx[ctry[a[idx]] == c]; lk = mt[t[i]]
        print(f'test {c:6s} fires {len(i):6,d}; agrees with existing link {np.mean(at[t[i]][lk] == a[i][lk]) if lk.any() else 0:.4f}; new {(~lk).sum():,}')
        out.append(np.column_stack([t[i][~lk], a[i][~lk]]))
    np.save(FR + 'acr2_add.npy', np.concatenate(out))
