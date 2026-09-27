"""France additions: unique exact core-name S1, house number differs only by a dropped/added digit, legal form same or
absent in target, target currently unlinked in v8. Output (tid, s1_row)."""
import re, numpy as np, pandas as pd
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z', 'legal']
rdn = lambda f: pd.read_csv(W + f'artifacts_v3/test_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False, usecols=[0, 1, 3, 4, 10])
s1 = rdn('s1'); tg = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
k1 = s1.ctry + '|' + s1.core; vc = k1.value_counts(); u = k1[k1.map(vc) == 1]; look = pd.Series(u.index.values, index=u.values)
j = (tg.ctry + '|' + tg.core).map(look).fillna(-1).astype(np.int64).values; j[tg.core.values == ''] = -1
linked = set()
for l in list(open(W + '../../outputs/final_submission_v8/matching_results.tsv'))[1:]:
    linked.update(x for x in l.rstrip('\n').split('\t')[1].split(',') if x)
def num(x):
    x = re.sub(r'(?<=\d)[-/ ](?=\d)', '', x); m = re.search(r'\d+', x); return m.group(0).lstrip('0') if m else ''
out = []
for t in np.flatnonzero((j >= 0) & (tg.ctry.values == 'France') & (tg.addr.values != '')):
    if tg.id.values[t] in linked: continue
    a, b = num(tg.addr.values[t]), num(s1.addr.values[j[t]])
    if not a or not b or a == b or not (a in b or b in a): continue
    lt, ls = tg.legal.values[t], s1.legal.values[j[t]]
    if lt != '' and lt != ls: continue
    out.append((t, j[t]))
out = np.array(out); np.save('/private/tmp/claude-501/fable_review/fr_digit_add.npy', out); print('France digit-drop additions:', len(out))
