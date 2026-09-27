"""Per-anchor helper arrays used by training, calibration and decoding.

  {split}_anchor_folds.npy     fold (0-99) of each Source 1 row, from prepare.py's stable hash
  {split}_anchor_countries.npy country label of each Source 1 row
  train_truth_counts.npy       number of true Source 2/3 links of each train Source 1 row
Rows are Source 1 rids (file order). Labels are only read for the train split.
"""
from pathlib import Path
import argparse
import duckdb
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--out', help='default: --work')
    a = ap.parse_args(); out = Path(a.out or a.work); out.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(Path(a.work) / 'records.duckdb'), read_only=True)
    for split in ('train', 'test'):
        r = db.execute(f'SELECT fold, country FROM {split}_s1 ORDER BY rid').fetchnumpy()
        np.save(out / f'{split}_anchor_folds.npy', r['fold'].astype(np.int16))
        np.save(out / f'{split}_anchor_countries.npy', np.asarray(r['country']).astype(str))
    n = db.execute('SELECT count(*) FROM train_s1').fetchone()[0]
    c = db.execute('SELECT aid, count(*) FROM links GROUP BY aid').fetchnumpy()
    counts = np.zeros(n, np.int16); counts[c['aid']] = c['count_star()'] if 'count_star()' in c else list(c.values())[1]
    np.save(out / 'train_truth_counts.npy', counts)
    print('wrote', out)

if __name__ == '__main__':
    main()
