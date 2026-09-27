"""Label-free comparison of matching_results.tsv files, per country, against a reference.

usage: compare_submissions.py REF.tsv NAME=FILE.tsv ...
Rows must be in test_source1 order (the validator checks that).
"""
import sys, hashlib
import numpy as np

def read(path):
    with open(path) as f:
        next(f)
        return [set(l.rstrip('\n').split('\t')[1].split(',')) - {''} for l in f]

def main():
    c = np.load('artifacts/test_anchor_countries.npy', allow_pickle=True)
    ref = read(sys.argv[1]); files = [('v6 (ref)', sys.argv[1])] + [a.split('=', 1) for a in sys.argv[2:]]
    for name, path in files:
        rows = ref if path == sys.argv[1] else read(path)
        assert len(rows) == len(c), (path, len(rows))
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:12]
        print(f'\n{name}  {path}  sha {sha}')
        for ctry in ('US', 'India', 'France'):
            ix = np.flatnonzero(c == ctry)
            n = np.array([len(rows[i]) for i in ix]); added = sum(len(rows[i] - ref[i]) for i in ix); removed = sum(len(ref[i] - rows[i]) for i in ix)
            changed = sum(rows[i] != ref[i] for i in ix)
            print(f'  {ctry:6s} links/entity {n.mean():.3f}  empty {np.mean(n == 0):.4f}  vs v6: +{added} -{removed} links, {changed} entities changed')

if __name__ == '__main__':
    main()
