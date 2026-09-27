"""Multi-anchor decoding: a target may be listed under several Source-1 entities.

The official format allows the same S2/S3 id in several S1 rows (duplicates are only
forbidden inside one list). When stage 2 splits a target between namesakes (e.g. 0.45 /
0.40), the one-anchor decoder usually drops it; here every stage-2 pair with p >= min_p
becomes its own row (aid = that anchor, q = pair probability) and the usual per-anchor
expected-F0.5 decoder picks k per anchor. Knobs tuned on folds 85-89, reported on 90-99.

usage: multi_decode.py STAGE2_DIR [--test-dir DIR --out DIR]
"""
from pathlib import Path
import argparse, json, itertools
import numpy as np
from decode import load, decode, evaluate

def rows(dec, pp, min_p, scale):
    """Base rows (one per target, its best anchor) + secondary stage-2 pairs as extra rows."""
    aid, q, m = dec['aid'], dec['q'].astype(np.float64), dec['margin']
    t, a, p = pp['tid'].astype(np.int64), pp['aid'].astype(np.int64), pp['p'].astype(np.float64)
    sec = (a != aid[t]) & (p >= min_p)
    t, a, p = t[sec], a[sec], p[sec] * scale
    R_t = np.r_[np.arange(len(aid)), t]; R_a = np.r_[aid, a]; R_q = np.r_[q, p]; R_m = np.r_[m, np.ones(len(t))]
    return R_t, R_a, R_q, R_m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage2'); ap.add_argument('--test-dir'); ap.add_argument('--out')
    ap.add_argument('--country-src', help='optional second stage-2 dir whose rows replace this one for --country')
    ap.add_argument('--country', default='France')
    a = ap.parse_args(); S = Path(a.stage2)
    _, owner, _, _, _ = load('artifacts_v3/train_scores', 'artifacts_v3/calibration/probabilities.npy')
    folds = np.load('artifacts/train_anchor_folds.npy'); truth = np.load('artifacts/train_truth_counts.npy')
    ctry = np.load('artifacts/train_anchor_countries.npy', allow_pickle=True)
    dec, pp = np.load(S / 'train_decisions.npz'), np.load(S / 'train_pair_probs.npz')
    base_knobs = json.load(open(S / 'stage2_report.json'))['knobs']
    b = decode(dec['aid'], dec['q'].astype(np.float64), dec['margin'], **base_knobs)
    r0, f0 = evaluate(b, dec['aid'], owner, folds, truth, 90, 100, ctry)
    t0, _ = evaluate(b, dec['aid'], owner, folds, truth, 85, 90)
    print('one-anchor  tuning', round(t0['macro_f05'], 6), ' val', round(r0['macro_f05'], 6))
    grid = []
    for min_p, scale, missing, empty in itertools.product([.05, .1, .2, .3], [1.0, .8, .6], [base_knobs['missing']], [base_knobs['empty_scale']]):
        R_t, R_a, R_q, R_m = rows(dec, pp, min_p, scale)
        k = dict(base_knobs, missing=missing, empty_scale=empty)
        mask = decode(R_a, R_q, R_m, **k)
        r, _ = evaluate(mask, R_a, owner[R_t], folds, truth, 85, 90)
        grid.append((r['macro_f05'], min_p, scale, k))
    grid.sort(key=lambda g: -g[0])
    for g in grid[:5]: print('  tuning', round(g[0], 6), 'min_p', g[1], 'scale', g[2])
    _, min_p, scale, k = grid[0]
    R_t, R_a, R_q, R_m = rows(dec, pp, min_p, scale); mask = decode(R_a, R_q, R_m, **k)
    r1, f1 = evaluate(mask, R_a, owner[R_t], folds, truth, 90, 100, ctry)
    d = f1 - f0
    print('multi-anchor val', round(r1['macro_f05'], 6), r1.get('by_country'), 'precision', round(r1['precision'], 5), 'recall', round(r1['recall'], 5))
    print(f'paired gain {d.mean():+.6f} SE {d.std(ddof=1) / np.sqrt(len(d)):.1e}  better {(d > 0).sum()} worse {(d < 0).sum()}  extra links {int(mask[len(dec["aid"]):].sum())}')
    rep = {'min_p': min_p, 'scale': scale, 'knobs': k, 'tuning': grid[0][0], 'val': r1, 'gain': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d)))}
    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / 'multi_report.json').write_text(json.dumps(rep, indent=2, default=str))
    if a.test_dir:
        from build_submission import ids
        td, tp = np.load(S / 'test_decisions.npz'), np.load(S / 'test_pair_probs.npz')
        ct = np.load('artifacts/test_anchor_countries.npy', allow_pickle=True)
        R_t, R_a, R_q, R_m = rows(td, tp, min_p, scale)
        if a.country_src:  # replace one country's rows by another stage-2 run (e.g. France from v6)
            C = Path(a.country_src); cd, cp = np.load(C / 'test_decisions.npz'), np.load(C / 'test_pair_probs.npz')
            cknobs = json.load(open(C / 'stage2_report.json'))['knobs']
            C_t, C_a, C_q, C_m = rows(cd, cp, min_p, scale)
            keep = ~((R_a >= 0) & (ct[np.maximum(R_a, 0)] == a.country)); use = (C_a >= 0) & (ct[np.maximum(C_a, 0)] == a.country)
            mask = np.r_[decode(R_a, R_q, R_m, **k)[keep], decode(C_a, C_q, C_m, **dict(k, **{kk: cknobs[kk] for kk in ('missing', 'empty_scale')}))[use]]
            R_t, R_a = np.r_[R_t[keep], C_t[use]], np.r_[R_a[keep], C_a[use]]
        else:
            mask = decode(R_a, R_q, R_m, **k)
        s1 = ids(Path(a.test_dir) / 'test_source1.tsv'); tg = ids(Path(a.test_dir) / 'test_source2.tsv') + ids(Path(a.test_dir) / 'test_source3.tsv')
        out_rows = [[] for _ in s1]
        for i in np.flatnonzero(mask & (R_a >= 0)):
            lst = out_rows[int(R_a[i])]; x = tg[int(R_t[i])]
            if x not in lst: lst.append(x)
        (out / 'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\n' + ''.join(e + '\t' + ','.join(v) + '\n' for e, v in zip(s1, out_rows)))
        print('TEST_EXPORTED rows', int(mask.sum()))

if __name__ == '__main__':
    main()
