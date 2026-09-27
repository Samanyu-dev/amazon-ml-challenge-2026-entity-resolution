"""Joint stage-1 CatBoost: US + India real labels plus France pseudo-labels.

Stage 1 has never seen a French pair. This adds French test pairs whose labels come
from the v6 decisions under rules measured on labelled train data:
  positive : v6 q >= 0.999 and the pair is the chosen (target, anchor) -> 99.97% correct
  negative : every other candidate of such a confident target (a target has one owner)
  negative : all candidates of a target with v6 q <= 0.001      -> 99.99% correct
French rows get --pseudo-weight (default 0.5). Real rows use exactly train_model.py's
sampling, folds and weights, so US/India validation stays comparable.
"""
from pathlib import Path
import argparse, glob, json, time
import numpy as np
from catboost import CatBoostClassifier, Pool
from train_model import pair_dtype, feature_count
from embedding_features import EmbeddingLookup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-pairs', default='artifacts_v3/train_pairs'); ap.add_argument('--test-pairs', default='artifacts_v3/test_pairs')
    ap.add_argument('--decisions', default='artifacts_v3/stage2_v6/test_decisions.npz')
    ap.add_argument('--artifacts', default='artifacts'); ap.add_argument('--out', required=True)
    ap.add_argument('--pseudo-weight', type=float, default=0.5); ap.add_argument('--pseudo-rate', type=float, default=0.5)
    ap.add_argument('--iterations', type=int, default=4000); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); art = Path(a.artifacts); t0 = time.time()
    nf = feature_count(a.train_pairs); DT = pair_dtype(nf); rng = np.random.default_rng(3)
    counts = np.load(art / 'train_truth_counts.npy')
    lk_tr, lk_te = EmbeddingLookup(art, 'train'), EmbeddingLookup(art, 'test')
    Xs, ys, ws, Xd, yd = [], [], [], [], []
    # 1) real US/India rows: identical rules to train_model.py
    for f in sorted(glob.glob(f'{a.train_pairs}/pairs-*.bin')):
        z = np.memmap(f, dtype=DT, mode='r')
        for st in range(0, len(z), 2_000_000):
            c = np.asarray(z[st:st + 2_000_000]); y = (c['aid'].astype(np.int32) == c['owner']).astype(np.uint8)
            h = (c['tid'].astype(np.uint64) * 2654435761 + c['aid'].astype(np.uint64) * 2246822519) % 4294967291; rank = c['x'][:, 42]
            keep = (y == 1) | ((rank == 0) & (h % 2 == 0)) | ((rank == 1) & (h % 8 == 0)) | (h % 64 == 0)
            train = (c['tfold'] < 80) & (c['afold'] < 75) & ((c['tfold'] < 75) | (c['tfold'] == -1)) & keep
            dev = (c['afold'] >= 75) & (c['afold'] < 80) & ((c['tfold'] >= 75) | (c['tfold'] == -1)) & (c['tfold'] < 80) & ((y == 1) | (h % 8 == 0))
            if train.any():
                Xs.append(lk_tr.features(c[train])); ys.append(y[train]); ws.append((1 / np.sqrt(np.maximum(1, counts[c['aid'][train]]))).astype(np.float32))
            if dev.any(): Xd.append(lk_tr.features(c[dev])); yd.append(y[dev])
    n_real = sum(len(v) for v in ys); print('real rows', n_real, round(time.time() - t0), 's', flush=True)
    # 2) France pseudo rows from test pairs
    d = np.load(a.decisions); aid, q = d['aid'], d['q']
    ctry = np.load(art / 'test_anchor_countries.npy', allow_pickle=True); fr = (aid >= 0) & (ctry[np.maximum(aid, 0)] == 'France')
    pos_t = fr & (q >= 0.999); neg_t = fr & (q <= 0.001)
    take = rng.random(len(aid)) < a.pseudo_rate
    n_ps = 0
    for f in sorted(glob.glob(f'{a.test_pairs}/pairs-*.bin')):
        z = np.memmap(f, dtype=DT, mode='r')
        for st in range(0, len(z), 2_000_000):
            c = np.asarray(z[st:st + 2_000_000]); t = c['tid'].astype(np.int64)
            m = (pos_t[t] | neg_t[t]) & take[t]
            if not m.any(): continue
            c, t = c[m], t[m]
            y = (pos_t[t] & (aid[t] == c['aid'])).astype(np.uint8)
            # same negative subsampling as the real rows (hard negatives kept, easy ones thinned)
            h = (c['tid'].astype(np.uint64) * 2654435761 + c['aid'].astype(np.uint64) * 2246822519) % 4294967291; rank = c['x'][:, 42]
            k = (y == 1) | ((rank == 0) & (h % 2 == 0)) | ((rank == 1) & (h % 8 == 0)) | (h % 64 == 0)
            c, y = c[k], y[k]
            if not len(y): continue
            Xs.append(lk_te.features(c)); ys.append(y); ws.append(np.full(len(y), a.pseudo_weight, np.float32)); n_ps += len(y)
    print('France pseudo rows', n_ps, 'positives', int(sum(v.sum() for v in ys)) , round(time.time() - t0), 's', flush=True)
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws); del Xs
    model = CatBoostClassifier(iterations=a.iterations, depth=7, learning_rate=.06, l2_leaf_reg=8, loss_function='Logloss', eval_metric='Logloss',
                               random_seed=42, thread_count=a.threads, verbose=250, allow_writing_files=False, border_count=96)
    model.fit(Pool(X, y, weight=w), eval_set=Pool(np.concatenate(Xd), np.concatenate(yd)), early_stopping_rounds=100)
    model.save_model(str(out / 'model.cbm'))
    (out / 'training.json').write_text(json.dumps({'real_rows': int(n_real), 'pseudo_rows': int(n_ps), 'pseudo_weight': a.pseudo_weight,
                                                  'trees': model.tree_count_, 'seconds': time.time() - t0}, indent=2))
    print('TRAIN_COMPLETE', model.tree_count_, round(time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
