"""Orphan-aware validation: remove a random share of S1 anchors, keep their S2/S3 copies as
decoys, re-point each affected target to its best surviving candidate, decode, and score the
surviving validation anchors (folds 90-99). Also tests decoder odds scaling in that world.

Approximations: a re-pointed target gets its stage-2 pair probability when stage 2 scored the
runner-up (all uncertain targets), otherwise the runner-up's raw stage-1 score mapped through a
monotone raw->calibrated fit; group features are not recomputed (second-order).
"""
import sys, json, glob
sys.path.insert(0, '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/src')
import numpy as np
from sklearn.isotonic import IsotonicRegression
from decode import load, decode, evaluate
from score_pairs import BD

W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
S = W + 'artifacts_v3/stage2_v7e_usin_owner/'
SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')])

def main(share=0.19, seed=0, odds_grid=(0.3, 0.45, 0.55, 0.7, 0.85, 1.0, 1.25)):
    meta = json.load(open(W + 'artifacts_v3/train_scores/complete.json'))
    b = np.memmap(W + 'artifacts_v3/train_scores/best.bin', dtype=BD, mode='r', shape=(meta['target_universe'],))
    aid0, owner, margin0, q0, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
    aid2 = np.asarray(b['aid2']).astype(np.int64); p2 = np.asarray(b['p2']).astype(np.float64); p1 = np.asarray(b['p']).astype(np.float64)
    folds = np.load(W + 'artifacts/train_anchor_folds.npy'); truth = np.load(W + 'artifacts/train_truth_counts.npy').astype(np.int64)
    ctry = np.load(W + 'artifacts/train_anchor_countries.npy', allow_pickle=True)
    d = np.load(S + 'train_decisions.npz'); a1, q1, m1 = d['aid'].astype(np.int64), d['q'].astype(np.float64), d['margin'].astype(np.float64)
    knobs = json.load(open(S + 'stage2_report.json'))['knobs']
    pp = np.load(S + 'train_pair_probs.npz'); pt, pa, pq = pp['tid'].astype(np.int64), pp['aid'].astype(np.int64), pp['p'].astype(np.float64)
    # monotone raw->calibrated map from best pairs (sampled)
    rng = np.random.default_rng(seed); ok = np.flatnonzero(aid0 >= 0); smp = rng.choice(ok, 400_000, replace=False)
    iso = IsotonicRegression(out_of_bounds='clip').fit(p1[smp], q0[smp])
    n_anchor = len(folds)
    removed = rng.random(n_anchor) < share
    print(f'removed {removed.sum():,} of {n_anchor:,} anchors ({share:.0%})')
    # ---- baseline world: original decisions, evaluated on surviving val anchors
    keep_anchor = ~removed
    def score(aid, q, m, own, tr, label, odds=1.0):
        qq = q if odds == 1.0 else q * odds / (q * odds + (1 - q))
        mask = decode(aid, qq, m, **knobs)
        # evaluate only on surviving anchors: fold trick -> give removed anchors fold -1
        f = np.where(keep_anchor, folds, -1)
        r, fe = evaluate(mask, aid, own, f, tr, 90, 100, ctry)
        return r, fe
    r_base, f_base = score(a1, q1, m1, owner, truth, 'base')
    # ---- orphan world
    own2 = owner.copy(); own2[(owner >= 0) & removed[np.maximum(owner, 0)]] = -1   # copies of removed anchors are decoys now
    tr2 = truth.copy(); tr2[removed] = 0
    hit = (a1 >= 0) & removed[np.maximum(a1, 0)]                                  # targets currently pointing at a removed anchor
    print(f'targets pointing at removed anchors: {hit.sum():,}  (of which decided-linked in base: {(hit & decode(a1, q1, m1, **knobs)).sum():,})')
    a2, q2, m2 = a1.copy(), q1.copy(), m1.copy()
    # alternative 1: best surviving stage-2 candidate
    sel = hit[pt] & ~removed[pa]
    o = np.lexsort((-pq[sel], pt[sel])); t_s, a_s, q_s = pt[sel][o], pa[sel][o], pq[sel][o]
    first = np.r_[True, t_s[1:] != t_s[:-1]]
    got = np.zeros(len(a1), bool)
    a2[t_s[first]] = a_s[first]; q2[t_s[first]] = q_s[first]; m2[t_s[first]] = 1.0; got[t_s[first]] = True
    # alternative 2: stage-1 runner-up (aid2) if surviving
    rest = hit & ~got & (aid2 >= 0) & ~removed[np.maximum(aid2, 0)]
    a2[rest] = aid2[rest]; q2[rest] = iso.predict(p2[rest]); m2[rest] = 1.0; got[rest] = True
    none = hit & ~got; a2[none] = -1; q2[none] = 0
    print(f'  re-pointed via stage-2 pair {int((hit & got & (m2 == 1.0) & np.isin(np.arange(len(a1)), t_s[first])).sum()):,}, via stage-1 runner-up {int(rest.sum()):,}, no surviving candidate {int(none.sum()):,}')
    print(f'  q of re-pointed targets: >=0.5 {np.mean(q2[hit & got] >= .5):.3f}  >=0.9 {np.mean(q2[hit & got] >= .9):.3f}  >=0.99 {np.mean(q2[hit & got] >= .99):.3f}')
    out = {'share': share, 'seed': seed, 'base': r_base['macro_f05'], 'orphan': {}}
    print(f'\nBASE world  F0.5 {r_base["macro_f05"]:.6f}  P {r_base["precision"]:.5f} R {r_base["recall"]:.5f}  links {r_base["links"]}')
    for odds in odds_grid:
        r, fe = score(a2, q2, m2, own2, tr2, 'orphan', odds)
        g = fe - f_base
        out['orphan'][odds] = {k: r[k] for k in ('macro_f05', 'precision', 'recall', 'links', 'singleton_false_merges')}
        print(f'ORPHAN odds x{odds:<4} F0.5 {r["macro_f05"]:.6f}  vs base {g.mean():+.6f} (SE {g.std(ddof=1)/np.sqrt(len(g)):.1e})  P {r["precision"]:.5f} R {r["recall"]:.5f}  links {r["links"]}  singleton-FM {r["singleton_false_merges"]}')
    json.dump(out, open(f'/private/tmp/claude-501/fable_review/orphan_owner_{share}_{seed}.json', 'w'), indent=2)

if __name__ == '__main__':
    share = float(sys.argv[1]) if len(sys.argv) > 1 else 0.19; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(share, seed)
