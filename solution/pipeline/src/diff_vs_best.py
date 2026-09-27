"""Where does a candidate matching_results.tsv disagree with our best file, and how sure is
our model about each disputed link? Label-free: it cannot score a test file, only show
which side our (validated) model believes on the links that differ.

usage: diff_vs_best.py CANDIDATE.tsv [--best ../../outputs/final_submission_v7d/matching_results.tsv]
"""
import argparse, json
import numpy as np
from pathlib import Path
from build_submission import ids

T = '/Users/apple/Downloads/student_resource/dataset/test'

def links(path, s1_ix, tg_ix):
    out = set()
    with open(path) as f:
        next(f)
        for line in f:
            e, m = line.rstrip('\n').split('\t')
            a = s1_ix[e]
            for x in filter(None, m.split(',')): out.add((tg_ix[x], a))
    return out

def model_prob(pairs, usin, france):
    """Our probability that target t belongs to anchor a (stage-2 pair prob, else top-anchor q, else 0)."""
    ctry = np.load('artifacts/test_anchor_countries.npy', allow_pickle=True)
    P = {}
    for S, keep in ((usin, lambda a: ctry[a] != 'France'), (france, lambda a: ctry[a] == 'France')):
        d = np.load(Path(S) / 'test_decisions.npz'); pp = Path(S) / 'test_pair_probs.npz'
        top = {(int(t), int(a)): float(q) for t, a, q in zip(np.flatnonzero(d['aid'] >= 0), d['aid'][d['aid'] >= 0], d['q'][d['aid'] >= 0])}
        pair = {}
        if pp.exists():
            z = np.load(pp); pair = {(int(t), int(a)): float(p) for t, a, p in zip(z['tid'], z['aid'], z['p'])}
        for k in pairs:
            if keep(k[1]): P[k] = pair.get(k, top.get(k, 0.0))
    return np.array([P[k] for k in pairs]), ctry

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('candidate')
    ap.add_argument('--best', default='../../outputs/final_submission_v7d/matching_results.tsv')
    ap.add_argument('--usin', default='artifacts_v3/stage2_v7d_all'); ap.add_argument('--france', default='artifacts_v3/stage2_v6_pp')
    a = ap.parse_args()
    s1 = ids(f'{T}/test_source1.tsv'); tg = ids(f'{T}/test_source2.tsv') + ids(f'{T}/test_source3.tsv')
    s1_ix = {e: i for i, e in enumerate(s1)}; tg_ix = {e: i for i, e in enumerate(tg)}
    B, C = links(a.best, s1_ix, tg_ix), links(a.candidate, s1_ix, tg_ix)
    only_c, only_b = sorted(C - B), sorted(B - C)
    pc, ctry = model_prob(only_c, a.usin, a.france); pb, _ = model_prob(only_b, a.usin, a.france)
    print(f'links: best {len(B):,}  candidate {len(C):,}  shared {len(B & C):,}  only-candidate {len(only_c):,}  only-best {len(only_b):,}')
    for k in ('US', 'India', 'France'):
        mc = np.array([ctry[x[1]] == k for x in only_c], bool); mb = np.array([ctry[x[1]] == k for x in only_b], bool)
        print(f'  {k:6s} only-candidate {mc.sum():7,d} (our model: mean p {pc[mc].mean() if mc.any() else 0:.2f}, p<0.5 {np.mean(pc[mc] < .5) if mc.any() else 0:.0%})'
              f'   only-best {mb.sum():7,d} (mean p {pb[mb].mean() if mb.any() else 0:.2f})')
    # our model's expected link-count balance: true links gained minus false links added
    print(f'our model expects: candidate adds ~{pc.sum():,.0f} true / ~{(1 - pc).sum():,.0f} false links; drops ~{pb.sum():,.0f} true / ~{(1 - pb).sum():,.0f} false')

if __name__ == '__main__':
    main()
