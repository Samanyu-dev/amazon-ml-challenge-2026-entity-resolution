"""Stage 2 (top-4) plus group evidence for each (target, anchor) pair.

Each entity usually appears 1-5 times per source. For a candidate anchor we add:
  conf_all / conf_same_src : number of *confident* records (stage-1 q >= 0.999) whose
                             best anchor is this anchor (overall / from the target's source)
  sibling_name_eq          : one of those confident records has the target's exact core name
  target_no_addr           : the target has no address (63% of the remaining misses)
Confident records have q >= 0.999, above the uncertain band, so a target is never its own evidence.
Everything else (folds, decoder tuning, reporting) is stack_ce4.main unchanged.
"""
import os, sys, importlib
import numpy as np
from decode import load
# STACK=stack_v6 adds group evidence on top of the two-cross-encoder stage 2
stack_ce4 = importlib.import_module(os.environ.get('STACK', 'stack_ce4'))

CONF = 0.999

def core_hashes(norm_file, with_addr=False):
    """Per-row hash of name_core (column 4) and address-missing flag (column 8) of a v3 .norm.tsv
    (optionally also a hash of (country, address_norm))."""
    h, miss, ad = [], [], []
    with open(norm_file, encoding='utf-8') as f:
        for line in f:
            c = line.split('\t'); h.append(hash(c[3])); miss.append(c[7] == '1')
            if with_addr: ad.append(hash((c[1], c[4])) if c[4] else 0)
    out = (np.array(h, np.int64), np.array(miss, bool))
    return out + (np.array(ad, np.int64),) if with_addr else out

def counts_of(keys, query):
    u, n = np.unique(keys, return_counts=True); pos = np.clip(np.searchsorted(u, query), 0, len(u) - 1)
    return np.where(u[pos] == query, n[pos], 0)

def group_features(split, scores, probabilities, n_s2):
    aid, _, _, q, _ = load(scores, probabilities)
    extra = os.environ.get('GROUP_EXTRA') == '1'
    if extra:
        h2, m2, d2 = core_hashes(f'artifacts_v3/{split}_s2.norm.tsv', True); h3, m3, d3 = core_hashes(f'artifacts_v3/{split}_s3.norm.tsv', True)
    else:
        h2, m2 = core_hashes(f'artifacts_v3/{split}_s2.norm.tsv'); h3, m3 = core_hashes(f'artifacts_v3/{split}_s3.norm.tsv')
    th, tmiss = np.r_[h2, h3], np.r_[m2, m3]; src = (np.arange(len(aid)) >= n_s2).astype(np.int8)
    conf = (aid >= 0) & (q >= CONF)
    n_anchor = aid.max() + 1
    conf_all = np.bincount(aid[conf], minlength=n_anchor)
    conf_src = np.stack([np.bincount(aid[conf & (src == s)], minlength=n_anchor) for s in (0, 1)])
    keys = np.unique(aid[conf].astype(np.int64) * 1_000_003 ^ th[conf])  # (anchor, core name) of confident records
    if extra:
        # decoy / graph evidence: twins, address crowding, anchor load, twins confidently pointing to this anchor
        tw_other = np.r_[counts_of(h3, h2), counts_of(h2, h3)]; tw_same = np.r_[counts_of(h2, h2), counts_of(h3, h3)] - 1
        ad = np.r_[d2, d3]; crowd = np.where(ad != 0, counts_of(ad[ad != 0], ad) - 1, -1)
        aload = np.bincount(aid[aid >= 0], minlength=n_anchor)
        ckey = aid[conf].astype(np.int64) * 1_000_003 ^ th[conf]; csrc = src[conf]
        ckeys_by_src = [np.sort(ckey[csrc == s_]) for s_ in (0, 1)]
    addr_s1 = os.environ.get('ADDR_S1') == '1'
    if addr_s1:  # how many Source-1 companies sit at the record's exact address, and is the candidate one of them
        a1h, _, a1d = core_hashes(f'artifacts_v3/{split}_s1.norm.tsv', True)
        tad = np.r_[core_hashes(f'artifacts_v3/{split}_s2.norm.tsv', True)[2], core_hashes(f'artifacts_v3/{split}_s3.norm.tsv', True)[2]]
        s1_at = np.where(tad != 0, counts_of(a1d[a1d != 0], tad), -1)
    def feats(tid, a):
        k = a.astype(np.int64) * 1_000_003 ^ th[tid]
        pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        base = [conf_all[a], conf_src[src[tid], a], keys[pos] == k, tmiss[tid]]
        if extra:
            other = 1 - src[tid]; agree = np.zeros(len(tid), np.float32)
            for s_ in (0, 1):
                m = other == s_; ks = ckeys_by_src[s_]
                if m.any() and len(ks): agree[m] = np.searchsorted(ks, k[m], side='right') - np.searchsorted(ks, k[m], side='left')
            base += [np.log1p(tw_other[tid]), np.log1p(tw_same[tid]), np.log1p(np.maximum(crowd[tid], 0)), crowd[tid] < 0, np.log1p(aload[a]), agree]
        if addr_s1:
            base += [s1_at[tid], (tad[tid] != 0) & (a1d[a] == tad[tid]), a1h[a] == th[tid]]
        return np.column_stack(base).astype(np.float32)
    return feats

def main():
    argv = sys.argv[1:]
    feats = {'train': group_features('train', 'artifacts_v3/train_scores', 'artifacts_v3/calibration/probabilities.npy', 5034616),
             'test': group_features('test', 'artifacts_v3/test_scores', 'artifacts_v3/test_calibration/probabilities.npy', 4887273)}
    base = stack_ce4.pair_table
    def pair_table(scores, ce_pairs, q, *rest):
        tid, aid, X = base(scores, ce_pairs, q, *rest)
        split = 'train' if 'train' in str(scores) else 'test'
        return tid, aid, np.column_stack([X, feats[split](tid, aid)])
    stack_ce4.pair_table = pair_table
    stack_ce4.NAMES = stack_ce4.NAMES + ['conf_all', 'conf_same_src', 'sibling_name_eq', 'target_no_addr']
    if os.environ.get('GROUP_EXTRA') == '1':
        stack_ce4.NAMES += ['twins_other_src', 'twins_same_src', 'addr_crowding', 'addr_missing', 'anchor_load', 'twins_confident_same_anchor']
    if os.environ.get('ADDR_S1') == '1':
        stack_ce4.NAMES += ['s1_companies_at_address', 'candidate_at_exact_address', 'name_core_exact']
    sys.argv = [sys.argv[0]] + argv
    stack_ce4.main()

if __name__ == '__main__':
    main()
