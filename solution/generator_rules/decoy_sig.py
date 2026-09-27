"""Decoy signature: target name words (minus legal/filler) == linked S1 name words, target has >=1 filler word; split by
first-house-number equal/different. Train: false-link rate of such LINKED pairs; test: how many we link per country."""
import sys, re, json
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd, unicodedata
from collections import Counter
from decode import load, decode
from fingerprints import rd
LEG = set('sarl sas sasu sa eurl ei sci snc ltd llc inc corp pvt private limited llp lp co cie company corporation incorporated pc pllc plc s a l r u e c i d b k n f t'.split())
deacc = lambda s: unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
words = lambda s: re.findall(r'[a-z]+', deacc(s))
num = lambda x: (lambda m: m.group(1).lstrip('0') if m else '')(re.search(r'(\d+)', x))
def fill_list(sp):
    s1 = rd(f'{sp}/{sp}_source1.tsv'); tg = pd.concat([rd(f'{sp}/{sp}_source2.tsv'), rd(f'{sp}/{sp}_source3.tsv')], ignore_index=True)
    w1 = Counter(w for x in s1.business_name for w in set(words(x))); wt = Counter(w for x in tg.business_name for w in set(words(x)))
    return {w for w, c in wt.items() if c >= 300 and w1.get(w, 0) * 10 < c and w not in LEG}, s1, tg
def sig(s1, tg, t, a, FILL):
    out = np.zeros(len(t), np.int8)   # 0 none, 1 filler + same number, 2 filler + different number
    for i, (x, y) in enumerate(zip(t, a)):
        tw = words(tg.business_name.values[x]); sw = words(s1.business_name.values[y])
        f = [w for w in tw if w in FILL]
        if not f: continue
        if [w for w in tw if w not in FILL and w not in LEG] != [w for w in sw if w not in FILL and w not in LEG]: continue
        n1, n2 = num(tg.business_address.values[x]), num(s1.business_address.values[y])
        out[i] = 1 if (n1 == n2 or not n1 or not n2) else 2
    return out
if __name__ == '__main__':
    FILL, s1, tg = fill_list('train'); print('train filler words:', sorted(FILL)[:60])
    _, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
    d = np.load(W + 'artifacts_v3/exp_owner_split/train_decisions.npz'); k = json.load(open(W + 'artifacts_v3/exp_owner_split/stage2_report.json'))['knobs']
    aid = d['aid'].astype(np.int64); q = d['q']; m = decode(aid, q.astype(np.float64), d['margin'], **k)
    t = np.flatnonzero(aid >= 0); g = sig(s1, tg, t, aid[t], FILL)
    for v, nm in ((1, 'filler, same number'), (2, 'filler, DIFFERENT number')):
        s = t[g == v]; print(f'TRAIN {nm}: pairs {len(s):,} true {np.mean(own[s] == aid[s]):.3f} | we link {m[s].mean():.3f}, of linked false {np.mean(own[s][m[s]] != aid[s][m[s]]):.4f}')
    FILLt, s1, tg = fill_list('test'); print('test-only filler words:', sorted(FILLt - FILL)[:60])
    FILL |= FILLt
    odds = lambda q, r: q * r / (q * r + 1 - q); ctry = np.load(W + 'artifacts/test_anchor_countries.npy', allow_pickle=True)
    for S, r, c in (('stage2_v7e_usin', 1, 'US'), ('stage2_v7e_usin', 1, 'India'), ('stage2_v6_pp', .5, 'France')):
        d = np.load(W + f'artifacts_v3/{S}/test_decisions.npz'); k = json.load(open(W + f'artifacts_v3/{S}/stage2_report.json'))['knobs']
        a2 = d['aid'].astype(np.int64); m2 = decode(a2, odds(d['q'].astype(np.float64), r), d['margin'], **k)
        t = np.flatnonzero((a2 >= 0) & (ctry[np.maximum(a2, 0)] == c)); g = sig(s1, tg, t, a2[t], FILL)
        for v, nm in ((1, 'filler, same number'), (2, 'filler, DIFFERENT number')):
            s = t[g == v]; print(f'TEST {c:6s} {nm}: pairs {len(s):,} we link {m2[s].mean():.3f}')
