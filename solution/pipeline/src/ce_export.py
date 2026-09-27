"""Export cross-encoder pair lists as raw-TSV row indices (no text, no labels leak).

train: pairs whose anchor fold is < --max-fold (default 40) and whose target is
       unowned or owned inside those folds; all positives are sampled at --pos-rate,
       negatives favour the hardest (retrieval rank 0-2, ANN, high name dice).
infer: the top --k candidates per target by stage-1 probability, from a
       score_pairs output directory (candidates-*.bin).

Output .npz: s1_row (S1 raw row), src (2/3), t_row (raw row in that source), label.
Rows are 0-based data rows of the organiser TSVs (header excluded), which is how
records.duckdb assigned rid.
"""
from pathlib import Path
import argparse, json, glob
import numpy as np
from train_model import pair_dtype, feature_count

def split_target(tid, n_s2):
    src = np.where(tid < n_s2, 2, 3).astype(np.int8)
    return src, np.where(tid < n_s2, tid, tid - n_s2).astype(np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'infer'])
    ap.add_argument('--pairs', help='retrieval pair dir (train mode)')
    ap.add_argument('--scores', help='score_pairs output dir (infer mode)')
    ap.add_argument('--n-s2', type=int, required=True, help='row count of source 2 for this split')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-fold', type=int, default=40)
    ap.add_argument('--pos-rate', type=float, default=0.35)
    ap.add_argument('--neg-per-pos', type=float, default=1.0)
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--probabilities', help='infer: calibrated per-target probabilities (with --scores best.bin) to restrict to the uncertain band')
    ap.add_argument('--band', type=float, nargs=2, default=[0.02, 0.995])
    ap.add_argument('--anchor-folds', help='infer (train split): npy of anchor folds; keep targets whose best anchor fold >= --min-fold')
    ap.add_argument('--min-fold', type=int, default=40)
    a = ap.parse_args(); rng = np.random.default_rng(7)
    if a.mode == 'train':
        names = json.loads((Path(a.pairs) / 'features.json').read_text()); fi = {n: i for i, n in enumerate(names)}
        DT = pair_dtype(feature_count(a.pairs)); keep = []
        for f in sorted(glob.glob(f'{a.pairs}/pairs-*.bin')):
            z = np.memmap(f, dtype=DT, mode='r')
            for st in range(0, len(z), 5_000_000):
                c = z[st:st + 5_000_000]
                ok = (c['afold'] < a.max_fold) & ((c['tfold'] < a.max_fold) | (c['tfold'] == -1))
                c = c[ok]; y = (c['aid'].astype(np.int64) == c['owner']).astype(np.int8)
                x = c['x']; hard = (x[:, fi['retrieval_rank']] < 3) | (x[:, fi['dense_route']] > 0) | (x[:, fi['name_trigram_dice']] > .6)
                pos = (y == 1) & (rng.random(len(c)) < a.pos_rate)
                neg = (y == 0) & hard & (rng.random(len(c)) < a.pos_rate * a.neg_per_pos * 3)
                m = pos | neg
                keep.append(np.column_stack([c['aid'][m], c['tid'][m], y[m]]).astype(np.int64))
        k = np.concatenate(keep)
        # rebalance negatives to the requested ratio
        p, n = k[k[:, 2] == 1], k[k[:, 2] == 0]
        n = n[rng.permutation(len(n))[:int(len(p) * a.neg_per_pos)]]
        k = np.concatenate([p, n]); k = k[rng.permutation(len(k))]
        s1, tid, label = k[:, 0], k[:, 1], k[:, 2]
    else:
        parts = [np.fromfile(f, dtype=np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')])) for f in sorted(glob.glob(f'{a.scores}/candidates-*.bin'))]
        z = np.concatenate(parts); z = z[np.lexsort((-z['p'], z['tid']))]
        start = np.r_[0, np.flatnonzero(np.diff(z['tid'])) + 1]; rank = np.arange(len(z)) - np.repeat(start, np.diff(np.r_[start, len(z)]))
        z = z[rank < a.k]
        if a.probabilities:
            from decode import load
            best_aid, _, _, q, _ = load(a.scores, a.probabilities)
            keep_t = (best_aid >= 0) & (q >= a.band[0]) & (q < a.band[1])
            if a.anchor_folds:
                folds = np.load(a.anchor_folds); keep_t &= folds[np.maximum(best_aid, 0)] >= a.min_fold
            z = z[keep_t[z['tid']]]
        s1, tid, label = z['aid'].astype(np.int64), z['tid'].astype(np.int64), np.full(len(z), -1)
    src, t_row = split_target(tid, a.n_s2)
    np.savez_compressed(a.out, s1_row=s1.astype(np.int32), src=src, t_row=t_row, label=label.astype(np.int8), tid=tid.astype(np.int32))
    print(json.dumps({'mode': a.mode, 'pairs': int(len(s1)), 'positives': int((label == 1).sum()), 'out': a.out}))

if __name__ == '__main__':
    main()
