"""Grade French stage-2 decisions on synthetic France: macro F0.5 per S1 (singletons count) at several odds."""
import sys, json
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from decode import decode
D = '/private/tmp/claude-501/synth2/out3/'; S = sys.argv[1] if len(sys.argv) > 1 else 'artifacts_v3/stage2_syn_v6/'
rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, quoting=3, keep_default_na=False, usecols=['entity_id'])
s1 = rd(D + 'test/test_source1.tsv').entity_id.values; tg = np.concatenate([rd(D + f'test/test_source{k}.tsv').entity_id.values for k in (2, 3)])
g = pd.read_csv(D + 'synth_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
truth = {e: set(x for x in m.split(',') if x) for e, m in zip(g.source1_entity_id, g.matched_entity_ids)}
d = np.load(S + 'test_decisions.npz'); k = json.load(open(S + 'stage2_report.json'))['knobs']
aid = d['aid'].astype(np.int64); q = d['q'].astype(np.float64)
def f05(p, t):
    if not p and not t: return 1.0
    tp = len(p & t)
    if tp == 0: return 0.0
    pr, rc = tp / len(p), tp / len(t); return 1.25 * pr * rc / (.25 * pr + rc)
for odds in [float(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else '0.25,0.35,0.5,0.7,1.0,1.4,2.0'.split(','))]:
    qq = q * odds / (q * odds + 1 - q); m = decode(aid, qq, d['margin'], **k)
    pred = {}
    for t in np.flatnonzero(m & (aid >= 0)): pred.setdefault(s1[aid[t]], set()).add(tg[t])
    f = np.array([f05(pred.get(e, set()), truth[e]) for e in s1]); tp = sum(len(pred.get(e, set()) & truth[e]) for e in s1); npred = sum(len(v) for v in pred.values()); nt = sum(len(v) for v in truth.values())
    print(f'odds x{odds:<5} synthetic-France macro F0.5 {f.mean():.5f} (SE {f.std() / np.sqrt(len(f)):.5f})  precision {tp / max(npred, 1):.4f} recall {tp / nt:.4f} links {npred:,}', flush=True)
