"""Where does validation F0.5 go? Per-entity loss split into miss/false-link causes (folds 90-99).

usage: error_budget.py DECISIONS.npz REPORT.json   (train_decisions.npz + stage2_report.json from stack_v7)
"""
import sys, glob, json
import numpy as np
from decode import load, decode

SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')])

def main():
    d = np.load(sys.argv[1]); aid, q, mg = d['aid'], d['q'].astype(np.float64), d['margin']
    knobs = json.load(open(sys.argv[2]))['knobs']
    _, owner, _, _, _ = load('artifacts_v3/train_scores', 'artifacts_v3/calibration/probabilities.npy')
    folds = np.load('artifacts/train_anchor_folds.npy'); truth = np.load('artifacts/train_truth_counts.npy').astype(np.int64)
    ctry = np.load('artifacts/train_anchor_countries.npy', allow_pickle=True)
    mask = decode(aid, q, mg, **knobs)
    cand = np.concatenate([np.fromfile(f, dtype=SF) for f in sorted(glob.glob('artifacts_v3/train_scores/candidates-*.bin'))])
    has_true = np.zeros(len(aid), bool)
    ok = owner[cand['tid']] == cand['aid'].astype(np.int64); has_true[cand['tid'][ok]] = True
    n = len(folds); tp = np.zeros(n); fp = np.zeros(n)
    lk = np.flatnonzero(mask); good = owner[lk] == aid[lk]
    np.add.at(tp, aid[lk][good], 1); np.add.at(fp, aid[lk][~good], 1)
    # false-link kinds: target is a decoy (owner<0) vs belongs to another entity
    fp_decoy = np.zeros(n); np.add.at(fp_decoy, aid[lk][~good & (owner[lk] < 0)], 1)
    # missed true links by cause, charged to the true owner
    t = np.flatnonzero(owner >= 0); o = owner[t]; linked_ok = mask[t] & (aid[t] == o)
    miss = t[~linked_ok]
    cause = np.where(~has_true[miss], 0, np.where(aid[miss] != owner[miss], 1, 2))  # 0 blocking, 1 wrong anchor top, 2 right top but not linked
    fn = np.zeros((3, n)); np.add.at(fn, (cause, owner[miss]), 1)
    fnt = fn.sum(0)
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0); R = np.where(truth > 0, tp / np.maximum(truth, 1), 1.0)
    F = np.where((truth == 0) & (tp + fp == 0), 1.0, np.where(tp == 0, 0.0, 1.25 * P * R / np.maximum(.25 * P + R, 1e-12)))
    v = folds >= 90
    print(f'val F0.5 {F[v].mean():.5f}  entities {v.sum()}  loss {1 - F[v].mean():.5f}')
    # attribution: loss if we fixed only FPs / only each FN cause (recompute F)
    def f_with(tp_, fp_):
        P = np.where(tp_ + fp_ > 0, tp_ / np.maximum(tp_ + fp_, 1), 1.0); R = np.where(truth > 0, tp_ / np.maximum(truth, 1), 1.0)
        return np.where((truth == 0) & (tp_ + fp_ == 0), 1.0, np.where(tp_ == 0, 0.0, 1.25 * P * R / np.maximum(.25 * P + R, 1e-12)))
    base = F[v].mean()
    print(f'  gain if no false links from decoys      {f_with(tp, fp - fp_decoy)[v].mean() - base:+.5f}')
    print(f'  gain if no false links from other ents   {f_with(tp, fp_decoy)[v].mean() - base:+.5f}')
    for i, name in enumerate(['blocking miss', 'wrong entity ranked top', 'right top but rejected']):
        print(f'  gain if no misses: {name:24s} {f_with(tp + fn[i], fp)[v].mean() - base:+.5f}   ({int(fn[i][v].sum())} links)')
    print(f'  (false links: decoy {int(fp_decoy[v].sum())}, other entity {int((fp - fp_decoy)[v].sum())}; true links {int(truth[v].sum())})')
    for k in ('US', 'India'):
        w = v & (ctry == k); print(f'  {k:6s} F {F[w].mean():.5f}  singletons wrongly linked {int(((truth == 0) & (tp + fp > 0) & w).sum())} of {int(((truth == 0) & w).sum())}')
    np.savez('reports/error_budget_arrays.npz', tp=tp, fp=fp, fp_decoy=fp_decoy, fn=fn, F=F)

if __name__ == '__main__':
    main()
