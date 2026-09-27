"""Stage 2 (v6): top-4 small-CE features plus an optional second CE (e5-base) on the top-2 pairs.

Derived from stack_ce4.py. Stage 2 over up to 4 candidates per uncertain target (band 0.005-0.999).

Same protocol as stack_ce.py (fit folds 40-74 on CE-unseen targets, dev 75-79, decoder
tuned on 85-89, report on 90-99), with features that compare each candidate with the
best *other* candidate of the same target.  CE scores come from several scored pair
files (the original top-2 run plus the new top-4 run) merged by (target, anchor).
"""
from pathlib import Path
import argparse, glob, json
import numpy as np
from catboost import CatBoostClassifier
from decode import load, decode, evaluate
from stack_ce import redecide

SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')])
NAMES = ['p', 'rank_p', 'ce', 'rank_ce', 'other_p', 'other_ce', 'ce_gap', 'p_gap', 'q_best', 'n_cand', 'ce2', 'other_ce2', 'ce2_gap', 'has_ce2']

def merged_ce(pairs):
    """[(npz, scores.npy), ...] -> (tid, aid, ce) with duplicates removed."""
    t, a, c = [], [], []
    for npz, sc in pairs:
        d = np.load(npz); t.append(d['tid'].astype(np.int64)); a.append(d['s1_row'].astype(np.int64)); c.append(np.load(sc).astype(np.float32))
    t, a, c = map(np.concatenate, (t, a, c)); key = t << 32 | a
    _, first = np.unique(key, return_index=True)
    return t[first], a[first], c[first]

def group_other_max(g_start, v):
    """For each row, max of v over the *other* rows of its group (-1 if none)."""
    idx = np.repeat(np.arange(len(g_start)), np.diff(np.r_[g_start, len(v)]))
    m1 = np.maximum.reduceat(v, g_start)
    is_max = v == m1[idx]
    first_max = np.zeros(len(v), bool)
    fm = np.minimum.reduceat(np.where(is_max, np.arange(len(v)), len(v)), g_start); first_max[fm] = True
    m2 = np.maximum.reduceat(np.where(first_max, -1, v), g_start)
    return np.where(first_max, m2[idx], m1[idx]), idx

def pair_table(scores, ce_pairs, q, ce2_pairs=None):
    tid, aid, ce = merged_ce(ce_pairs)
    ce2 = np.full(len(tid), -1, np.float32)
    if ce2_pairs:
        t2, a2, c2 = merged_ce(ce2_pairs); k2 = t2 << 32 | a2; o2 = np.argsort(k2); k2, c2 = k2[o2], c2[o2]
        k = tid << 32 | aid; pos = np.clip(np.searchsorted(k2, k), 0, len(k2) - 1); hit = k2[pos] == k; ce2[hit] = c2[pos[hit]]
    cand = np.concatenate([np.fromfile(f, dtype=SF) for f in sorted(glob.glob(f'{scores}/candidates-*.bin'))])
    key = cand['tid'].astype(np.int64) << 32 | cand['aid']; o = np.argsort(key); key = key[o]
    pos = np.searchsorted(key, tid << 32 | aid); assert (key[pos] == (tid << 32 | aid)).all()
    p = cand['p'][o][pos].astype(np.float32)
    o = np.lexsort((-p, tid)); tid, aid, p, ce, ce2 = tid[o], aid[o], p[o], ce[o], ce2[o]
    g = np.r_[0, np.flatnonzero(np.diff(tid)) + 1]
    op, idx = group_other_max(g, p); oce, _ = group_other_max(g, ce)
    rank_p = np.arange(len(tid)) - g[idx]
    oc = np.lexsort((-ce, tid)); rank_ce = np.empty(len(tid)); rank_ce[oc] = np.arange(len(tid)) - g[idx][oc]
    n = np.diff(np.r_[g, len(tid)])[idx]
    oce2, _ = group_other_max(g, ce2)
    X = np.column_stack([p, rank_p, ce, rank_ce, op, oce, ce - oce, p - op, q[tid], n, ce2, oce2, np.where(ce2 >= 0, ce2 - np.maximum(oce2, 0), 0), (ce2 >= 0).astype(np.float32)]).astype(np.float32)
    return tid, aid, X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v3', default='artifacts_v3'); ap.add_argument('--artifacts', default='artifacts')
    ap.add_argument('--train-ce', nargs='+', required=True, help='npz=scores.npy pairs for train')
    ap.add_argument('--test-ce', nargs='+', help='npz=scores.npy pairs for test')
    ap.add_argument('--train-ce2', nargs='+'); ap.add_argument('--test-ce2', nargs='+')
    ap.add_argument('--test-dir'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); v3, art, out = Path(a.v3), Path(a.artifacts), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = lambda xs: [tuple(x.split('=')) for x in xs]
    aid0, owner, margin, q0, _ = load(v3 / 'train_scores', v3 / 'calibration/probabilities.npy')
    folds = np.load(art / 'train_anchor_folds.npy'); truth = np.load(art / 'train_truth_counts.npy')
    countries = np.load(art / 'train_anchor_countries.npy', allow_pickle=True)
    tid, aid, X = pair_table(v3 / 'train_scores', split(a.train_ce), q0, split(a.train_ce2) if a.train_ce2 else None)
    y = (aid == owner[tid]).astype(np.int8)
    seen = np.zeros(len(aid0), bool); seen[np.load('ce/ce_train_pairs.npz')['tid']] = True
    af = folds[aid0[tid]]
    fit = (af >= 40) & (af < 75) & ~seen[tid]; dev = (af >= 75) & (af < 80) & ~seen[tid]
    model = CatBoostClassifier(iterations=2000, depth=6, learning_rate=0.05, loss_function='Logloss', verbose=250,
                               random_seed=42, thread_count=8, allow_writing_files=False)
    model.fit(X[fit], y[fit], eval_set=(X[dev], y[dev]), early_stopping_rounds=100)
    model.save_model(str(out / 'stage2.cbm'))
    aid1, q1 = redecide(tid, aid, model.predict_proba(X)[:, 1], aid0, q0)
    m1 = margin.copy(); m1[np.unique(tid)] = 1.0
    grid = []
    for missing in [.1, .25, .4, .6]:
        for empty in [2, 3, 4, 5, 6, 8]:
            k = dict(missing=missing, empty_scale=empty, min_margin=.015, floor=0.0)
            r, _ = evaluate(decode(aid1, q1, m1, **k), aid1, owner, folds, truth, 85, 90); grid.append((r['macro_f05'], k))
    best_f, knobs = max(grid, key=lambda g: g[0])
    base = json.loads((v3 / 'decoder_b/decoder_report.json').read_text())['knobs_tuned_on_folds_85_89']
    r3, f3 = evaluate(decode(aid0, q0, margin, **base), aid0, owner, folds, truth, 90, 100, countries)
    r4, f4 = evaluate(decode(aid1, q1, m1, **knobs), aid1, owner, folds, truth, 90, 100, countries)
    v4 = json.loads(Path('artifacts_v3/stage2_top4/stage2_report.json').read_text())['top4_decoder_90_99']  # v5 reference
    d = f4 - f3
    rep = {'knobs': knobs, 'tuning_85_89': best_f, 'importance': dict(zip(NAMES, map(float, model.get_feature_importance()))),
           'v3_decoder_90_99': r3, 'v5_90_99': v4, 'v6_90_99': r4,
           'paired_gain_vs_v3': {'mean': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d)))}}
    (out / 'stage2_report.json').write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))
    np.save(out / 'val_entity_f.npy', f4)
    if a.test_ce:
        from build_submission import ids
        aidt, _, margint, qt, _ = load(v3 / 'test_scores', v3 / 'test_calibration/probabilities.npy')
        ttid, taid, TX = pair_table(v3 / 'test_scores', split(a.test_ce), qt, split(a.test_ce2) if a.test_ce2 else None)
        aidt1, qt1 = redecide(ttid, taid, model.predict_proba(TX)[:, 1], aidt, qt)
        mt = margint.copy(); mt[np.unique(ttid)] = 1.0
        np.savez(out / 'test_decisions.npz', aid=aidt1, q=qt1, margin=mt)  # for per-country decoder variants
        mask = decode(aidt1, qt1, mt, **knobs)
        s1 = ids(Path(a.test_dir) / 'test_source1.tsv'); tg = ids(Path(a.test_dir) / 'test_source2.tsv') + ids(Path(a.test_dir) / 'test_source3.tsv')
        rows = [[] for _ in s1]
        for t in np.flatnonzero(mask): rows[int(aidt1[t])].append(tg[t])
        (out / 'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\n' + ''.join(e + '\t' + ','.join(v) + '\n' for e, v in zip(s1, rows)))
        print('TEST_EXPORTED links', int(mask.sum()))

if __name__ == '__main__':
    main()
