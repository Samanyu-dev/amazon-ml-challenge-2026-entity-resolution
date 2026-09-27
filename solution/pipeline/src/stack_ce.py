"""Stage 2: re-decide uncertain targets with cross-encoder (CE) scores.

For every target in the stage-1 uncertain band, its top-2 candidate anchors each get
features [stage-1 raw p, rank, CE score, the other candidate's p and CE, stage-1
calibrated best q, stage-1 margin].  A CatBoost model trained on anchor folds 40-74
(targets never seen by CE training) predicts P(pair is the true link); the target then
takes its higher-probability anchor.  Targets outside the band keep stage 1.
The per-anchor decoder is re-tuned on folds 85-89 and reported on 90-99.
"""
from pathlib import Path
import argparse, glob, json
import numpy as np
from catboost import CatBoostClassifier
from decode import load, decode, evaluate, prune

SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')])

def pair_table(scores, infer_npz, ce_scores, q):
    """Rows = CE-scored (tid, anchor) pairs with stage-1 p, rank and partner features."""
    inf = np.load(infer_npz); ce = np.load(ce_scores).astype(np.float32)
    tid = inf['tid'].astype(np.int64); aid = inf['s1_row'].astype(np.int64)
    cand = np.concatenate([np.fromfile(f, dtype=SF) for f in sorted(glob.glob(f'{scores}/candidates-*.bin'))])
    key = cand['tid'].astype(np.int64) << 32 | cand['aid']; order = np.argsort(key); key = key[order]
    pos = np.searchsorted(key, tid << 32 | aid); assert (key[pos] == (tid << 32 | aid)).all(), 'CE pair missing from stage-1 candidates'
    p = cand['p'][order][pos].astype(np.float32)
    o = np.lexsort((-p, tid)); tid, aid, p, ce = tid[o], aid[o], p[o], ce[o]
    first = np.r_[True, tid[1:] != tid[:-1]]; rank = (~first).astype(np.float32)
    # partner = the other candidate of the same target (targets have 1 or 2 rows)
    partner = np.arange(len(tid)); nxt = np.r_[tid[1:] == tid[:-1], False]; prv = np.r_[False, tid[1:] == tid[:-1]]
    partner[nxt] += 1; partner[prv] -= 1; solo = ~(nxt | prv)
    op = np.where(solo, -1, p[partner]); oce = np.where(solo, -1, ce[partner])
    X = np.column_stack([p, rank, ce, op, oce, ce - np.where(solo, 0, oce), q[tid], p - np.where(solo, 0, op)]).astype(np.float32)
    return tid, aid, X

def redecide(tid, aid, prob, base_aid, base_q):
    """Best anchor per target by stage-2 probability; untouched outside the band."""
    new_aid, new_q = base_aid.copy(), base_q.copy()
    o = np.lexsort((-prob, tid)); first = np.r_[True, tid[o][1:] != tid[o][:-1]]; w = o[first]
    new_aid[tid[w]] = aid[w]; new_q[tid[w]] = prob[w]
    return new_aid, new_q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v3', default='artifacts_v3'); ap.add_argument('--artifacts', default='artifacts')
    ap.add_argument('--ce-train', required=True, help='CE scores for ce/ce_infer_train.npz')
    ap.add_argument('--ce-test', help='CE scores for ce/ce_infer_test.npz (export test file)')
    ap.add_argument('--test-dir'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); v3, art, out = Path(a.v3), Path(a.artifacts), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    aid0, owner, margin, q0, _ = load(v3 / 'train_scores', v3 / 'calibration/probabilities.npy')
    folds = np.load(art / 'train_anchor_folds.npy'); truth = np.load(art / 'train_truth_counts.npy')
    tid, aid, X = pair_table(v3 / 'train_scores', 'ce/ce_infer_train.npz', a.ce_train, q0)
    y = (aid == owner[tid]).astype(np.int8)
    seen = np.zeros(len(aid0), bool); seen[np.load('ce/ce_train_pairs.npz')['tid']] = True
    af = folds[aid0[tid]]  # fold of the target's stage-1 best anchor
    fit = (af >= 40) & (af < 75) & ~seen[tid]; dev = (af >= 75) & (af < 80) & ~seen[tid]
    model = CatBoostClassifier(iterations=1500, depth=6, learning_rate=0.05, loss_function='Logloss', verbose=250,
                               random_seed=42, thread_count=8, allow_writing_files=False)
    model.fit(X[fit], y[fit], eval_set=(X[dev], y[dev]), early_stopping_rounds=100)
    model.save_model(str(out / 'stage2.cbm'))
    prob = model.predict_proba(X)[:, 1]
    aid1, q1 = redecide(tid, aid, prob, aid0, q0)
    margin1 = margin.copy(); margin1[np.unique(tid)] = 1.0  # stage 2 already arbitrated between the two anchors
    grid = []
    for missing in [.1, .25, .4, .6]:
        for empty in [2, 3, 4, 5, 6, 8]:
            k = dict(missing=missing, empty_scale=empty, min_margin=.015, floor=0.0)
            r, _ = evaluate(decode(aid1, q1, margin1, **k), aid1, owner, folds, truth, 85, 90); grid.append((r['macro_f05'], k))
    best_f, knobs = max(grid, key=lambda g: g[0])
    base_knobs = json.loads((v3 / 'decoder_b/decoder_report.json').read_text())['knobs_tuned_on_folds_85_89']
    rep = {'knobs': knobs, 'tuning_85_89': best_f, 'importance': dict(zip(['p', 'rank', 'ce', 'other_p', 'other_ce', 'ce_gap', 'q_best', 'p_gap'], map(float, model.get_feature_importance())))}
    r3, f3 = evaluate(decode(aid0, q0, margin, **base_knobs), aid0, owner, folds, truth, 90, 100, np.load(art / 'train_anchor_countries.npy', allow_pickle=True))
    r4, f4 = evaluate(decode(aid1, q1, margin1, **knobs), aid1, owner, folds, truth, 90, 100, np.load(art / 'train_anchor_countries.npy', allow_pickle=True))
    d = f4 - f3
    rep.update({'v3_decoder_90_99': r3, 'stage2_decoder_90_99': r4,
                'paired_gain': {'mean': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d))), 'better': int((d > 0).sum()), 'worse': int((d < 0).sum())},
                'note': 'about 4% of CE-scored eval pairs involve target text seen in CE training (decoy negatives); slightly optimistic'})
    (out / 'stage2_report.json').write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))
    if a.ce_test:
        from build_submission import ids
        aidt, _, margint, qt, meta = load(v3 / 'test_scores', v3 / 'test_calibration/probabilities.npy')
        ttid, taid, TX = pair_table(v3 / 'test_scores', 'ce/ce_infer_test.npz', a.ce_test, qt)
        aidt1, qt1 = redecide(ttid, taid, model.predict_proba(TX)[:, 1], aidt, qt)
        mt = margint.copy(); mt[np.unique(ttid)] = 1.0
        mask = decode(aidt1, qt1, mt, **knobs)
        s1 = ids(Path(a.test_dir) / 'test_source1.tsv'); tg = ids(Path(a.test_dir) / 'test_source2.tsv') + ids(Path(a.test_dir) / 'test_source3.tsv')
        rows = [[] for _ in s1]
        for t in np.flatnonzero(mask): rows[int(aidt1[t])].append(tg[t])
        (out / 'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\n' + ''.join(e + '\t' + ','.join(v) + '\n' for e, v in zip(s1, rows)))
        print('TEST_EXPORTED links', int(mask.sum()))

if __name__ == '__main__':
    main()
