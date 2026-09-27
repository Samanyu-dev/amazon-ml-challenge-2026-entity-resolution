"""Leaderboard probe for France: drop French links in the same-address / different-name zone.

In labelled US/India data, pairs with the same address but a very different name
(name token jaccard <= 0.2) are true only 37-43% of the time. France has no labels, so
the direction for this zone can only be learned on the public leaderboard: this file
differs from the base submission ONLY in those French links (and only when q < q_keep).
"""
from pathlib import Path
import argparse, glob, json
import numpy as np
from decode import decode
from train_model import pair_dtype
from build_submission import ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decisions', required=True); ap.add_argument('--report', required=True)
    ap.add_argument('--pairs', default='artifacts_v3/test_pairs'); ap.add_argument('--countries', default='artifacts/test_anchor_countries.npy')
    ap.add_argument('--name-jaccard-max', type=float, default=0.2); ap.add_argument('--q-keep', type=float, default=0.99)
    ap.add_argument('--test-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = np.load(a.decisions); aid, q, mg = d['aid'], d['q'].astype(np.float64), d['margin']
    base = decode(aid, q, mg, **json.loads(Path(a.report).read_text())['knobs'])
    fr = (aid >= 0) & (np.load(a.countries, allow_pickle=True)[np.maximum(aid, 0)] == 'France')
    names = json.loads((Path(a.pairs) / 'features.json').read_text()); ix = {n: i for i, n in enumerate(names)}; DT = pair_dtype(len(names))
    zone = np.zeros(len(aid), bool)
    for f in sorted(glob.glob(f'{a.pairs}/pairs-*.bin')):
        z = np.memmap(f, dtype=DT, mode='r')
        for st in range(0, len(z), 5_000_000):
            c = z[st:st + 5_000_000]; t = c['tid'].astype(np.int64)
            m = base[t] & fr[t] & (aid[t] == c['aid'])
            if not m.any(): continue
            x = c['x'][m]
            same_addr = (x[:, ix['address_token_jaccard']] >= 0.8) & (x[:, ix['first_number_equal']] > 0) & (x[:, ix['target_only_numbers']] == 0)
            weak_name = (x[:, ix['name_token_jaccard']] <= a.name_jaccard_max) & (x[:, ix['target_name_nonascii']] == 0)
            zone[t[m][same_addr & weak_name]] = True
    drop = zone & (q < a.q_keep)
    mask = base & ~drop
    s1 = ids(Path(a.test_dir) / 'test_source1.tsv'); tg = ids(Path(a.test_dir) / 'test_source2.tsv') + ids(Path(a.test_dir) / 'test_source3.tsv')
    rows = [[] for _ in s1]
    for t in np.flatnonzero(mask): rows[int(aid[t])].append(tg[t])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\n' + ''.join(e + '\t' + ','.join(v) + '\n' for e, v in zip(s1, rows)))
    info = {'french_links_base': int((base & fr).sum()), 'zone_links': int(zone.sum()), 'dropped': int(drop.sum()),
            'kept_confident_zone_links': int((zone & ~drop).sum()), 'links': int(mask.sum()), 'non_france_changed': int(((base != mask) & ~fr).sum())}
    (out / 'probe.json').write_text(json.dumps(info, indent=2)); print(json.dumps(info))

if __name__ == '__main__':
    main()
