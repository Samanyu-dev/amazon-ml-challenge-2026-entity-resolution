"""v8 = base TSV (assemble_final) + addition lists; also writes the extended candidate_pairs.tsv (old + new blocking pairs).
usage: build_v8.py BASE_TSV OUT_DIR ADD1.npy [ADD2.npy ...]"""
import sys, numpy as np
from pathlib import Path
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src'); from build_submission import ids
FR = '/private/tmp/claude-501/fable_review/'; T = '/Users/apple/Downloads/student_resource/dataset/test'
base, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
s1 = ids(f'{T}/test_source1.tsv'); tg = ids(f'{T}/test_source2.tsv') + ids(f'{T}/test_source3.tsv')
rows, used = {}, set()
with open(base) as f:
    head = next(f)
    for line in f:
        e, mm = line.rstrip('\n').split('\t'); rows[e] = [x for x in mm.split(',') if x]; used.update(rows[e])
for p in sys.argv[3:]:
    A = np.load(p); added = dup = cap = 0
    for t, a in A:
        e, x = s1[a], tg[t]
        if x in used: dup += 1; continue
        cur = rows.setdefault(e, []); n2 = sum(y.startswith('S2') for y in cur)
        if (x.startswith('S2') and n2 >= 5) or (x.startswith('S3') and len(cur) - n2 >= 6): cap += 1; continue
        cur.append(x); used.add(x); added += 1
    print(f'{Path(p).name}: added {added:,} dup-skip {dup} cap-skip {cap}')
with open(out / 'matching_results.tsv', 'w') as f:
    f.write(head)
    for e in s1: f.write(f"{e}\t{','.join(rows.get(e, []))}\n")
# extended candidates: old file + all new blocking pairs (address-number Indic top-3, acronym initials top-10)
new = {}
for z in ('addr_block_test.npz', 'acr_test.npz'):
    Z = np.load(FR + z)
    for t, a in zip(Z['tid'], Z['aid']): new.setdefault(s1[a], set()).add(tg[t])
old = W + '../../outputs/package_candidates/candidate_pairs.tsv'; extra = 0
with open(old) as f, open(out / 'candidate_pairs.tsv', 'w') as g:
    g.write(next(f))
    for line in f:
        e, cc = line.rstrip('\n').split('\t'); c = cc.split(','); s = set(c); add = [x for x in new.get(e, ()) if x not in s]; extra += len(add)
        g.write(f"{e}\t{','.join(c + add)}\n")
print(f'candidate pairs added {extra:,}')
