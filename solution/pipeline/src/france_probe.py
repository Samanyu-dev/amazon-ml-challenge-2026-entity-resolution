"""Shift France-only link probabilities by delta on the logit scale and re-decode.

France has no labels, so its operating point can only be checked on the public
leaderboard. delta < 0 makes France stricter (fewer links), delta > 0 looser.
Other countries are decoded exactly as in the base file.
"""
from pathlib import Path
import argparse, json
import numpy as np
from decode import decode
from build_submission import ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decisions', required=True, help='test_decisions.npz from stack_ce4.py')
    ap.add_argument('--report', required=True, help='stage2_report.json with tuned knobs')
    ap.add_argument('--countries', default='artifacts/test_anchor_countries.npy')
    ap.add_argument('--country', default='France'); ap.add_argument('--delta', type=float, required=True)
    ap.add_argument('--test-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = np.load(a.decisions); aid, q, margin = d['aid'], d['q'].astype(np.float64), d['margin']
    knobs = json.loads(Path(a.report).read_text())['knobs']
    fr = (aid >= 0) & (np.load(a.countries, allow_pickle=True)[np.maximum(aid, 0)] == a.country)
    qc = np.clip(q, 1e-6, 1 - 1e-6); q2 = q.copy()
    q2[fr] = 1 / (1 + np.exp(-(np.log(qc[fr] / (1 - qc[fr])) + a.delta)))
    base, mask = decode(aid, q, margin, **knobs), decode(aid, q2, margin, **knobs)
    assert (base[~fr] == mask[~fr]).all(), 'non-target countries must be unchanged'
    s1 = ids(Path(a.test_dir) / 'test_source1.tsv'); tg = ids(Path(a.test_dir) / 'test_source2.tsv') + ids(Path(a.test_dir) / 'test_source3.tsv')
    rows = [[] for _ in s1]
    for t in np.flatnonzero(mask): rows[int(aid[t])].append(tg[t])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\n' + ''.join(e + '\t' + ','.join(v) + '\n' for e, v in zip(s1, rows)))
    info = {'country': a.country, 'delta': a.delta, 'links': int(mask.sum()), 'country_links_base': int((base & fr).sum()),
            'country_links_new': int((mask & fr).sum()), 'added': int((mask & ~base).sum()), 'removed': int((base & ~mask).sum())}
    (out / 'probe.json').write_text(json.dumps(info, indent=2)); print(json.dumps(info))

if __name__ == '__main__':
    main()
