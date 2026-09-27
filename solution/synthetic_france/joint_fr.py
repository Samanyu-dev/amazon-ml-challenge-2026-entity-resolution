import sys, json
sys.path.insert(0, 'src')
import numpy as np
exec(open('sweep_syn.py').read().split("print('current knobs'")[0])   # synthetic truth + score()
L = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
dr = np.load(L + 'artifacts_v3/stage2_v6_pp/test_decisions.npz'); ar = dr['aid'].astype(np.int64); qr = dr['q'].astype(np.float64)
ctry = np.load(L + 'artifacts/test_anchor_countries.npy', allow_pickle=True); fr_ent = np.flatnonzero(ctry == 'France')
def real_profile(odds, **kk):
    qq = qr * odds / (qr * odds + 1 - qr); m = decode(ar, qq, dr['margin'], **kk) & (ar >= 0)
    c = np.bincount(ar[m], minlength=len(ctry))[fr_ent]; return c.mean(), np.mean(c == 0)
rows = []
for odds in (1.0, 1.4, 2.0, 2.5):
    for miss in (0.1, 0.25, 0.4):
        for es in (2, 3, 5):
            kk = dict(k); kk['missing'] = miss; kk['empty_scale'] = es
            f, _ = score(odds, **kk); lr, er = real_profile(odds, **kk)
            rows.append((f, odds, miss, es, lr, er)); print(f'odds {odds} miss {miss} es {es}: synth F {f:.5f} | REAL France links/S1 {lr:.3f} empty {er:.4f}', flush=True)
ok = [r for r in rows if 3.30 <= r[4] <= 3.46 and 0.050 <= r[5] <= 0.062]
print('BEST within labelled-country profile:', max(ok) if ok else None)
