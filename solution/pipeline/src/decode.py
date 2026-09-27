"""Per-anchor expected-F0.5 decoding over the per-target argmax links.

Each target keeps only its best anchor (labels have no shared targets).  For an
anchor with calibrated link probabilities q (sorted desc), predicting the top k
has approximate expected F0.5
    1.25 * sum(q[:k]) / (k + 0.25 * (sum(q) + missing))        k >= 1
and the empty list scores  empty_scale * prod(1 - q)            k == 0,
where `missing` stands for true links retrieval never surfaced.  k is chosen per
anchor; knobs are tuned on folds 85-89 only.
"""
from pathlib import Path
import argparse, json, itertools
import numpy as np
from score_pairs import BD
from calibrate import entity_scores

def decode(aid, q, margin, missing, empty_scale, min_margin=0.0, floor=0.0):
    """Return a boolean mask over targets: link target -> aid[target]."""
    ok = (aid >= 0) & (margin >= min_margin) & (q > floor)
    idx = np.flatnonzero(ok)
    order = np.lexsort((-q[idx], aid[idx])); idx = idx[order]
    a, p = aid[idx], q[idx].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(a)) + 1]; sizes = np.diff(np.r_[starts, len(a)])
    group = np.repeat(np.arange(len(starts)), sizes)
    csum = np.cumsum(p); base = np.r_[0, csum[starts[1:] - 1]]
    within = csum - base[group]; k = np.arange(len(a)) - starts[group] + 1
    total = np.add.reduceat(p, starts)
    score = 1.25 * within / (k + 0.25 * (total[group] + missing))
    best = np.maximum.reduceat(score, starts)
    # first k reaching the group maximum
    kbest = np.minimum.reduceat(np.where(score == best[group], k, np.iinfo(np.int64).max), starts)
    empty = empty_scale * np.exp(np.add.reduceat(np.log1p(-np.minimum(p, 1 - 1e-7)), starts))
    kbest = np.where(empty > best, 0, kbest)
    mask = np.zeros(len(aid), bool); mask[idx[k <= kbest[group]]] = True
    return mask


def prune(aid, q, margin, mask, cost, over, missing, empty_scale, min_margin=0.0, floor=0.0):
    """Drop decoded links until `over` bytes are saved, cheapest expected-F0.5 loss per byte first.

    Removing an anchor's last (lowest-q) chosen link moves it from k to k-1; the
    loss is recomputed for that anchor after every removal (the metric is macro per S1)."""
    import heapq
    ok = (aid >= 0) & (margin >= min_margin) & (q > floor)
    idx = np.flatnonzero(ok); idx = idx[np.lexsort((-q[idx], aid[idx]))]
    starts = np.r_[0, np.flatnonzero(np.diff(aid[idx])) + 1]; ends = np.r_[starts[1:], len(idx)]
    mask = mask.copy(); heap = []; state = {}
    def score(g, k):
        st, en = state[g][0], state[g][1]; p = q[idx[st:en]]
        if k == 0: return empty_scale * np.prod(1 - np.minimum(p, 1 - 1e-7))
        return 1.25 * p[:k].sum() / (k + 0.25 * (p.sum() + missing))
    def push(g):
        k = state[g][2]
        if k == 0: return
        t = idx[state[g][0] + k - 1]
        heapq.heappush(heap, ((score(g, k) - score(g, k - 1)) / (cost[t] + (k > 1)), g))
    for g, (st, en) in enumerate(zip(starts, ends)):
        state[g] = [st, en, int(mask[idx[st:en]].sum())]; push(g)
    saved = 0; removed = 0
    while saved < over and heap:
        _, g = heapq.heappop(heap); k = state[g][2]; t = idx[state[g][0] + k - 1]
        mask[t] = False; saved += cost[t] + (k > 1); removed += 1; state[g][2] = k - 1; push(g)
    return mask, removed, saved

def threshold_rule(aid, q, margin, threshold, min_margin):
    return (aid >= 0) & (q >= threshold) & (margin >= min_margin)

def evaluate(mask, b_aid, owner, folds, truth, lo, hi, countries=None):
    anchors = np.flatnonzero((folds >= lo) & (folds < hi))
    local = np.full(len(folds), -1, np.int64); local[anchors] = np.arange(len(anchors))
    rows = np.flatnonzero(mask & (b_aid >= 0)); rows = rows[local[b_aid[rows]] >= 0]
    t = truth[anchors]; good = b_aid[rows] == owner[rows]
    f, pred, tp = entity_scores(t, local[b_aid[rows]], good)
    out = {'macro_f05': float(f.mean()), 'precision': float(tp.sum() / max(1, pred.sum())),
           'recall': float(tp.sum() / max(1, t.sum())), 'links': int(pred.sum()),
           'singleton_false_merges': int(((t == 0) & (pred > 0)).sum())}
    if countries is not None:
        out['by_country'] = {str(c): float(f[countries[anchors] == c].mean()) for c in np.unique(countries[anchors])}
    return out, f

def load(scores, probabilities):
    meta = json.loads((Path(scores) / 'complete.json').read_text())
    b = np.memmap(Path(scores) / 'best.bin', mode='r', dtype=BD, shape=(meta['target_universe'],))
    return (np.asarray(b['aid']).astype(np.int64), np.asarray(b['owner']).astype(np.int64),
            np.asarray(b['p'] - b['p2']), np.load(probabilities).astype(np.float64), meta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', default='artifacts'); ap.add_argument('--out', required=True)
    ap.add_argument('--train-scores', default='artifacts/train_scores_dense')
    ap.add_argument('--train-probabilities', default='artifacts/calibration_dense/probabilities.npy')
    ap.add_argument('--missing', type=float, nargs='+', default=[0, .1, .25, .5])
    ap.add_argument('--empty-scale', type=float, nargs='+', default=[.5, 1, 2])
    ap.add_argument('--floor', type=float, nargs='+', default=[0, .3, .5, .6, .7])
    a = ap.parse_args(); art = Path(a.artifacts); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    aid, owner, margin, q, _ = load(a.train_scores, a.train_probabilities)
    folds = np.load(art / 'train_anchor_folds.npy'); truth = np.load(art / 'train_truth_counts.npy')
    countries = np.load(art / 'train_anchor_countries.npy', allow_pickle=True)
    grid = []
    for missing, empty_scale, min_margin, floor in itertools.product(
            a.missing, a.empty_scale, [.015], a.floor):
        r, _ = evaluate(decode(aid, q, margin, missing, empty_scale, min_margin, floor), aid, owner, folds, truth, 85, 90)
        grid.append({'missing': missing, 'empty_scale': empty_scale, 'min_margin': min_margin, 'floor': floor, **r})
    grid.sort(key=lambda g: -g['macro_f05']); knobs = {k: grid[0][k] for k in ['missing', 'empty_scale', 'min_margin', 'floor']}
    rules = {'v2_t0.70': threshold_rule(aid, q, margin, .70, .015), 'v2_t0.75': threshold_rule(aid, q, margin, .75, .015),
             'decoder': decode(aid, q, margin, **knobs)}
    report = {'knobs_tuned_on_folds_85_89': knobs, 'top_grid': grid[:8], 'tuning_85_89': {}, 'diagnostic_90_99': {}}
    per = {}
    for name, mask in rules.items():
        report['tuning_85_89'][name] = evaluate(mask, aid, owner, folds, truth, 85, 90)[0]
        report['diagnostic_90_99'][name], per[name] = evaluate(mask, aid, owner, folds, truth, 90, 100, countries)
    for base in ['v2_t0.70', 'v2_t0.75']:
        d = per['decoder'] - per[base]
        report['diagnostic_90_99'][f'decoder_minus_{base}'] = {'mean': float(d.mean()), 'paired_se': float(d.std(ddof=1) / np.sqrt(len(d))),
            'entities_better': int((d > 0).sum()), 'entities_worse': int((d < 0).sum())}
    (out / 'decoder_report.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


def export(scores, probabilities, knobs, test_dir, out, budget):
    """Write matching_results.tsv for the test set; refuse to exceed the byte budget."""
    from build_submission import ids
    aid, _, margin, q, meta = load(scores, probabilities)
    mask = decode(aid, q, margin, **knobs)
    s1 = ids(Path(test_dir) / 'test_source1.tsv')
    targets = ids(Path(test_dir) / 'test_source2.tsv') + ids(Path(test_dir) / 'test_source3.tsv')
    assert len(s1) == meta['anchor_universe'] and len(targets) == len(aid)
    rows = [[] for _ in s1]
    for t in np.flatnonzero(mask): rows[int(aid[t])].append(targets[t])
    text = 'source1_entity_id\tmatched_entity_ids\n' + ''.join(e + '\t' + ','.join(v) + '\n' for e, v in zip(s1, rows))
    size = len(text.encode()); pruned = 0
    if size > budget:
        cost = np.array([len(x) for x in targets])
        mask, pruned, _ = prune(aid, q, margin, mask, cost, size - budget, **knobs)
        rows = [[] for _ in s1]
        for t in np.flatnonzero(mask): rows[int(aid[t])].append(targets[t])
        text = 'source1_entity_id\tmatched_entity_ids\n' + ''.join(e + '\t' + ','.join(v) + '\n' for e, v in zip(s1, rows))
        size = len(text.encode())
    assert size <= budget, size
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'matching_results.tsv').exists(): raise ValueError('Refusing to overwrite an existing submission')
    (out / 'matching_results.tsv').write_text(text)
    info = {'knobs': knobs, 'links': int(mask.sum()), 'bytes': size, 'pruned_links': pruned, 'nonempty': sum(bool(r) for r in rows)}
    (out / 'decoder_export.json').write_text(json.dumps(info, indent=2)); print(json.dumps(info))

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--report', required=True); ap.add_argument('--scores', default='artifacts/test_scores_dense')
        ap.add_argument('--probabilities', default='artifacts/test_calibration_dense/probabilities.npy'); ap.add_argument('--test-dir', required=True); ap.add_argument('--out', required=True); ap.add_argument('--budget', type=int, default=96_900_000)
        a = ap.parse_args(); export(a.scores, a.probabilities, json.loads(Path(a.report).read_text())['knobs_tuned_on_folds_85_89'], a.test_dir, a.out, a.budget)
    else:
        main()
