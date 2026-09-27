"""French pseudo-label pair files for cross-encoder self-training (France has no labels).

round1: labelled train pairs (ce/ce_train_pairs.npz) + French test pairs from stage 1:
        positives  = q >= 0.999 and margin >= 0.3        (99.996% correct on labelled train)
        negatives  = rank-1/2 runner-ups of those targets, and targets with q <= 0.001 (99.99%)
round2: round1 file + hard French pairs re-decided by stage 2 (v6 decisions, q >= 0.999 /
        <= 0.001, measured 99.97% / 99.99% correct on labelled train) and their other candidates.
Only organiser data and our own model outputs are used.
"""
import argparse, glob, sys
import numpy as np
from decode import load

N2 = 4887273  # test Source-2 rows (targets are S2 rows then S3 rows)
SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')])

def pack(tid, s1, lab):
    return tid, s1, lab, np.where(tid < N2, 2, 3).astype(np.int8), np.where(tid < N2, tid, tid - N2).astype(np.int32)

def round1(rng):
    aid, _, margin, q, _ = load('artifacts_v3/test_scores', 'artifacts_v3/test_calibration/probabilities.npy')
    ctry = np.load('artifacts/test_anchor_countries.npy', allow_pickle=True); fr = (aid >= 0) & (ctry[np.maximum(aid, 0)] == 'France')
    pos_t = rng.choice(np.flatnonzero(fr & (q >= .999) & (margin >= .3)), 250000, replace=False)
    dec_t = rng.choice(np.flatnonzero(fr & (q <= .001)), 100000, replace=False)
    z = np.concatenate([np.fromfile(f, dtype=SF) for f in sorted(glob.glob('artifacts_v3/test_scores/candidates-*.bin'))])
    keep = np.zeros(len(aid), bool); keep[pos_t] = True; z = z[keep[z['tid']]]; z = z[np.lexsort((-z['p'], z['tid']))]
    g = np.r_[0, np.flatnonzero(np.diff(z['tid'])) + 1]; rank = np.arange(len(z)) - np.repeat(g, np.diff(np.r_[g, len(z)]))
    ru = z[(rank >= 1) & (rank <= 2) & (z['aid'] != aid[z['tid']])]; ru = ru[rng.permutation(len(ru))[:250000]]
    tid = np.r_[pos_t, ru['tid'].astype(np.int64), dec_t]; s1 = np.r_[aid[pos_t], ru['aid'].astype(np.int64), aid[dec_t]]
    lab = np.r_[np.ones(len(pos_t)), np.zeros(len(ru)), np.zeros(len(dec_t))].astype(np.int8)
    tr = np.load('ce/ce_train_pairs.npz'); tid, s1, lab, src, trow = pack(tid, s1, lab)
    out = {'s1_row': np.r_[tr['s1_row'], s1].astype(np.int32), 'src': np.r_[tr['src'], src].astype(np.int8), 't_row': np.r_[tr['t_row'], trow].astype(np.int32),
           'label': np.r_[tr['label'], lab].astype(np.int8), 'tid': np.r_[tr['tid'], tid].astype(np.int32),
           'split': np.r_[np.zeros(len(tr['label'])), np.ones(len(lab))].astype(np.int8)}
    perm = rng.permutation(len(out['label'])); return {k: v[perm] for k, v in out.items()}

def round2(rng, decisions):
    d = np.load(decisions); aid, q = d['aid'], d['q'].astype(np.float64)
    ctry = np.load('artifacts/test_anchor_countries.npy', allow_pickle=True); fr = (aid >= 0) & (ctry[np.maximum(aid, 0)] == 'France')
    files = ['ce/ce_infer_test.npz', 'ce/ce_infer4new_testa.npz', 'ce/ce_infer4new_testb.npz']
    inb = np.zeros(len(aid), bool)
    for f in files: inb[np.load(f)['tid']] = True
    hp = np.flatnonzero(fr & inb & (q >= .999)); hd = np.flatnonzero(fr & inb & (q <= .001))
    t_all = np.concatenate([np.load(f)['tid'].astype(np.int64) for f in files]); a_all = np.concatenate([np.load(f)['s1_row'].astype(np.int64) for f in files])
    ishp = np.zeros(len(aid), bool); ishp[hp] = True
    m = ishp[t_all] & (a_all != aid[t_all]); key = np.unique(t_all[m] << 32 | a_all[m]); hn_t, hn_a = key >> 32, key & 0xffffffff
    sel = rng.permutation(len(hn_t))[:250000]; hn_t, hn_a = hn_t[sel], hn_a[sel]
    tid = np.r_[hp, hn_t, hd]; s1 = np.r_[aid[hp], hn_a, aid[hd]]
    lab = np.r_[np.ones(len(hp)), np.zeros(len(hn_t)), np.zeros(len(hd))].astype(np.int8)
    tid, s1, lab, src, trow = pack(tid, s1, lab); r1 = np.load('ce/ce_train_v2_pairs.npz')
    out = {'s1_row': np.r_[r1['s1_row'], s1].astype(np.int32), 'src': np.r_[r1['src'], src].astype(np.int8), 't_row': np.r_[r1['t_row'], trow].astype(np.int32),
           'label': np.r_[r1['label'], lab].astype(np.int8), 'tid': np.r_[r1['tid'], tid].astype(np.int32), 'split': np.r_[r1['split'], np.ones(len(lab))].astype(np.int8)}
    k = (out['split'].astype(np.int64) << 60) | (out['tid'].astype(np.int64) << 28) | out['s1_row'].astype(np.int64)
    _, first = np.unique(k, return_index=True); out = {kk: v[np.sort(first)] for kk, v in out.items()}
    perm = rng.permutation(len(out['label'])); return {kk: v[perm] for kk, v in out.items()}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('round', choices=['round1', 'round2'])
    ap.add_argument('--decisions', default='artifacts_v3/stage2_v6/test_decisions.npz'); a = ap.parse_args()
    if a.round == 'round1': np.savez_compressed('ce/ce_train_v2_pairs.npz', **round1(np.random.default_rng(11)))
    else: np.savez_compressed('ce/ce_train_v3_pairs.npz', **round2(np.random.default_rng(21), a.decisions))
    print('written', a.round)
