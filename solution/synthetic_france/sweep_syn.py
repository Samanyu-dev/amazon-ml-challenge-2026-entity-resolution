import sys, json
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from decode import decode
D = '/private/tmp/claude-501/synth2/out3/'; S = 'artifacts_v3/stage2_syn_v6/'
rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, quoting=3, keep_default_na=False, usecols=['entity_id'])
s1 = rd(D + 'test/test_source1.tsv').entity_id.values; n1 = len(s1)
tg = np.concatenate([rd(D + f'test/test_source{k}.tsv').entity_id.values for k in (2, 3)]); tix = {e: i for i, e in enumerate(tg)}; s1ix = {e: i for i, e in enumerate(s1)}
g = pd.read_csv(D + 'synth_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
own = np.full(len(tg), -1, np.int64); truth = np.zeros(n1, np.int64)
for e, m in zip(g.source1_entity_id, g.matched_entity_ids):
    for x in m.split(','):
        if x: own[tix[x]] = s1ix[e]; truth[s1ix[e]] += 1
d = np.load(S + 'test_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); q = d['q'].astype(np.float64); mg = d['margin']
def score(odds, **kk):
    qq = q * odds / (q * odds + 1 - q); m = decode(aid, qq, mg, **kk) & (aid >= 0)
    lk = np.flatnonzero(m); good = own[lk] == aid[lk]
    tp = np.bincount(aid[lk][good], minlength=n1); pr = np.bincount(aid[lk], minlength=n1)
    P = np.where(pr > 0, tp / np.maximum(pr, 1), 1.0); R = np.where(truth > 0, tp / np.maximum(truth, 1), 1.0)
    F = np.where((truth == 0) & (pr == 0), 1.0, np.where(tp == 0, 0.0, 1.25 * P * R / np.maximum(.25 * P + R, 1e-12)))
    return F.mean(), pr.mean()
print('current knobs', k)
base = dict(k)
for odds in (0.5, 1.0, 1.4, 2.0, 3.0):
    f, l = score(odds, **base); print(f'odds x{odds}: F {f:.5f} links/S1 {l:.3f}', flush=True)
best = None
for odds in (1.0, 1.4, 2.0):
    for miss in (0.05, 0.1, 0.2, 0.4):
        for es in (2, 4, 6, 8):
            kk = dict(base); kk['missing'] = miss; kk['empty_scale'] = es
            f, l = score(odds, **kk)
            if best is None or f > best[0]: best = (f, odds, miss, es, l)
print('best on synthetic:', best, flush=True)
