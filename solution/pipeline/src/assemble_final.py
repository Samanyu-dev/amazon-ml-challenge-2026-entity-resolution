"""Final matching_results.tsv: US/India rows from the best stage-2 stack, France rows from the
v6 stage 2 (round-1 cross-encoders) with French link odds scaled by --france-odds before decoding.

usage: assemble_final.py --usin artifacts_v3/stage2_v7d_all --france artifacts_v3/stage2_v6 \
         --france-odds 0.5 --test-dir DATASET/test --out output/matching_results.tsv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from decode import decode
from build_submission import ids

def rows_for(stage2, test_dir, s1, tg, keep, odds=1.0):
    d = np.load(Path(stage2) / 'test_decisions.npz'); k = json.load(open(Path(stage2) / 'stage2_report.json'))['knobs']
    a, q, m = d['aid'], d['q'].astype(np.float64), d['margin']
    if odds != 1.0: q = q * odds / (q * odds + (1 - q))
    mask = decode(a, q, m, **k) & (a >= 0)
    mask &= keep[np.maximum(a, 0)]
    out = {}
    for t in np.flatnonzero(mask): out.setdefault(int(a[t]), []).append(tg[t])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--usin', required=True); ap.add_argument('--france', required=True); ap.add_argument('--france-odds', type=float, default=0.5); ap.add_argument('--usin-odds', type=float, default=1.0)
    ap.add_argument('--test-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ctry = np.load('artifacts/test_anchor_countries.npy', allow_pickle=True); fr = ctry == 'France'
    s1 = ids(Path(a.test_dir) / 'test_source1.tsv'); tg = ids(Path(a.test_dir) / 'test_source2.tsv') + ids(Path(a.test_dir) / 'test_source3.tsv')
    rows = rows_for(a.usin, a.test_dir, s1, tg, ~fr, a.usin_odds); rows.update(rows_for(a.france, a.test_dir, s1, tg, fr, a.france_odds))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='\n') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for i, e in enumerate(s1): f.write(e + '\t' + ','.join(rows.get(i, [])) + '\n')
    print('written', a.out, 'links', sum(map(len, rows.values())))

if __name__ == '__main__':
    main()
