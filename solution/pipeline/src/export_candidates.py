"""candidate_pairs.tsv: every (S1, S2/S3) pair the stage-1 model scores on test (the exact
inference set of the first matching model; all later stages only re-rank within it)."""
import argparse, glob
from pathlib import Path
import numpy as np
from build_submission import ids

SF = np.dtype([('tid', '<u4'), ('aid', '<u4'), ('p', '<f4')])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scores', required=True); ap.add_argument('--test-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); T = Path(a.test_dir)
    c = np.concatenate([np.fromfile(f, dtype=SF)[['tid', 'aid']] for f in sorted(glob.glob(f'{a.scores}/candidates-*.bin'))])
    o = np.lexsort((c['tid'], c['aid'])); aid, tid = c['aid'][o].astype(np.int64), c['tid'][o].astype(np.int64)
    s1 = ids(T / 'test_source1.tsv'); tg = np.array(ids(T / 'test_source2.tsv') + ids(T / 'test_source3.tsv'), dtype=object)
    st = np.searchsorted(aid, np.arange(len(s1) + 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='\n') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for i, e in enumerate(s1): f.write(e + '\t' + ','.join(tg[tid[st[i]:st[i + 1]]]) + '\n')
    print('pairs', len(c), 'per S1', round(len(c) / len(s1), 2))

if __name__ == '__main__':
    main()
