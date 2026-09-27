"""Address-number blocking for Indian-script targets: candidates = India S1 sharing a (rare-ish) house-number token,
ranked by name-skeleton char-trigram cosine + address token overlap. usage: addr_block.py train|test"""
import sys, re, time
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import HashingVectorizer
sp = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
cols = ['id', 'ctry', 'name', 'core', 'addr', 'skel', 'nonascii', 'x', 'y', 'z']
rdn = lambda f: pd.read_csv(W + f'artifacts/{sp}_{f}.norm.tsv', sep='\t', header=None, names=cols, dtype=str, quoting=3, keep_default_na=False)
s1 = rdn('s1'); tg = pd.concat([rdn('s2'), rdn('s3')], ignore_index=True)
LEG = {'prvt', 'pvt', 'lmtd', 'ltd', 'llp', 'lp', 'prv', 'pv', 'l', 'lmt', 'pvtltd', 'prvtlmtd', 'c', 'cmpny', 'c', 'nc', 'crprtn', 'cntr', 'srvcs', 'grp', 'dr', 'sr', 'shr', 'smt', 'm', 's'}
core = lambda s: ' '.join(w for w in s.split() if w not in LEG)
dig = re.compile(r'\b\d+\b')
ind = np.flatnonzero(s1.ctry.values == 'India')
hv = HashingVectorizer(analyzer='char_wb', ngram_range=(2, 3), n_features=2**20, norm='l2', alternate_sign=False)
A = hv.transform([core(x) for x in s1.skel.values[ind]]).tocsr()
aw = [set(x.split()) for x in s1.addr.values[ind]]
idx = defaultdict(list)
for j, a in enumerate(s1.addr.values[ind]):
    for n in set(dig.findall(a)): idx[n].append(j)
idx = {k: np.array(v) for k, v in idx.items()}
raw = pd.concat([pd.read_csv(f'/Users/apple/Downloads/student_resource/dataset/{sp}/{sp}_source{k}.tsv', sep='\t', dtype=str, quoting=3, keep_default_na=False, usecols=['business_address']) for k in (2, 3)], ignore_index=True).business_address
trunc = raw.str.contains(r',\s*(?:MH|KA|DL|TN|UP|GJ|RJ|WB|TS|TG|AP|KL|HR|PB|MP|OD|BR|CG|JH|UK|HP|GA|AS)\s*$', regex=True).values
T = (tg.nonascii.values == '0') & (tg.ctry.values == 'India') & (tg.addr.values != '') & trunc
import json; sys.path.insert(0, W + 'src'); from decode import decode
ST = W + 'artifacts_v3/' + ('exp_owner_split' if sp == 'train' else 'stage2_v7e_usin') + '/'
dd = np.load(ST + f'{sp}_decisions.npz'); kk = json.load(open(ST + 'stage2_report.json'))['knobs']
da = dd['aid'].astype(np.int64); dq = dd['q'].astype(np.float64); dm = decode(da, dq, dd['margin'], **kk)
T &= ~(dm & (dq >= 0.5))
if sp == 'train':
    from decode import load
    _, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
    fo = np.load(W + 'artifacts/train_anchor_folds.npy')
    pass  # full scope: no fold filter (decoys must all be scored)
T = np.flatnonzero(T)
print(f'{sp}: India S1 {len(ind):,}  target scope {len(T):,}', flush=True)
Tv = hv.transform([core(x) for x in tg.skel.values[T]]).tocsr()
out_t, out_a, out_s, out_n, out_ov = [], [], [], [], []
t0 = time.time()
for i, t in enumerate(T):
    nums = set(dig.findall(tg.addr.values[t]))
    c = [idx[n] for n in nums if n in idx and len(idx[n]) <= 20000]
    if not c: continue
    c = np.unique(np.concatenate(c))
    sim = (A[c] @ Tv[i].T).toarray().ravel()
    pre = np.argsort(-sim)[:30]; tw = set(tg.addr.values[t].split())
    ov = np.zeros(len(c)); ov[pre] = [len(tw & aw[j]) for j in c[pre]]
    top = pre[np.argsort(-(sim[pre] + 0.02 * ov[pre]))][:K]
    out_t.append(np.full(len(top), t)); out_a.append(ind[c[top]]); out_s.append(sim[top]); out_ov.append(ov[top]); out_n.append(np.full(len(top), len(c)))
    if i % 10000 == 0: print(i, f'{time.time() - t0:.0f}s', flush=True)
np.savez(f'/private/tmp/claude-501/fable_review/tr_block_{sp}.npz', tid=np.concatenate(out_t), aid=np.concatenate(out_a),
         sim=np.concatenate(out_s), ov=np.concatenate(out_ov), ncand=np.concatenate(out_n))
print('done', f'{time.time() - t0:.0f}s')
