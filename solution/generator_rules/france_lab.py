"""France disagreement laboratory + judge-gated intervention (Tracks 1 and 8).

A = France model in v7e (v6 stack + coverage), B = e5-large stack, judge = stage 1 (never saw French
pseudo-labels). Rule: keep A; switch to B where A != B and the judge backs B with q >= TH.
The same rule is run on labelled US/India (leak-free A/B models) to measure per-slice accuracy and the
paired entity-F gain per switched record; French switch counts per slice give the expected-gain report.
"""
import sys, json, glob
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np, pandas as pd
from decode import load, decode, evaluate
from train_model import pair_dtype

def dec(S, split):
    d = np.load(f'{W}artifacts_v3/{S}/{split}_decisions.npz'); k = json.load(open(f'{W}artifacts_v3/{S}/stage2_report.json'))['knobs']
    a = d['aid'].astype(np.int64); return np.where(decode(a, d['q'].astype(np.float64), d['margin'], **k) & (a >= 0), a, -1)

def pair_feats(split, tids, anchors):
    """name_exact, address-missing, first-number conflict, name jaccard for the given (target, anchor) pairs."""
    names = json.load(open(f'{W}artifacts_v3/{split}_pairs/features.json')); ix = {n: i for i, n in enumerate(names)}; DT = pair_dtype(len(names))
    want = {'name_exact': 0, 'target_address_missing': 0, 'first_number_conflict': 0, 'name_token_jaccard': -1, 'address_token_jaccard': -1}
    out = {k: np.full(len(tids), v, np.float32) for k, v in want.items()}
    key = tids.astype(np.int64) << 32 | anchors.astype(np.int64); o = np.argsort(key); ks = key[o]
    for f in sorted(glob.glob(f'{W}artifacts_v3/{split}_pairs/pairs-*.bin')):
        z = np.memmap(f, dtype=DT, mode='r')
        for st in range(0, len(z), 5_000_000):
            c = z[st:st + 5_000_000]; kk = c['tid'].astype(np.int64) << 32 | c['aid'].astype(np.int64)
            pos = np.clip(np.searchsorted(ks, kk), 0, len(ks) - 1); hit = ks[pos] == kk
            if not hit.any(): continue
            rows = o[pos[hit]]; X = c['x'][hit]
            for n_ in want: out[n_][rows] = X[:, ix[n_]]
    return out

def slices(F):
    return {'address missing': F['target_address_missing'] > 0,
            'exact name': F['name_exact'] > 0,
            'same address, different name': (F['address_token_jaccard'] >= .8) & (F['name_token_jaccard'] <= .2),
            'house-number conflict': F['first_number_conflict'] > 0,
            'other': None}

def main(TH=0.9):
    # ---------- US/India calibration (leak-free models, folds 90-99)
    A, B = dec('exp_owner_v6cov', 'train'), dec('exp_owner_split', 'train')
    a0, own, _, q0, _ = load(f'{W}artifacts_v3/train_scores', f'{W}artifacts_v3/calibration/probabilities.npy')
    J = np.where((a0 >= 0) & (q0 >= .5), a0, -1)
    folds = np.load(f'{W}artifacts/train_anchor_folds.npy'); truth = np.load(f'{W}artifacts/train_truth_counts.npy')
    switch = (A != B) & (J == B) & (q0 >= TH)
    C = np.where(switch, B, A)
    ev = lambda D: evaluate(D >= 0, D, own, folds, truth, 90, 100)
    rA, fA = ev(A); rB, fB = ev(B); rC, fC = ev(C)
    g, gb = fC - fA, fB - fA
    ref = np.where(own >= 0, own, np.maximum(A, B)); val = (ref >= 0) & (folds[np.maximum(ref, 0)] >= 90)
    sw_val = switch & val
    print(f'US/India val: A {rA["macro_f05"]:.6f} | B {rB["macro_f05"]:.6f} ({gb.mean():+.6f}) | judge-gated C {rC["macro_f05"]:.6f} ({g.mean():+.6f} ± {g.std()/np.sqrt(len(g)):.1e})')
    print(f'   switched records {sw_val.sum():,}; switched decision correct {np.mean(B[sw_val] == own[sw_val]):.3f} (A was correct {np.mean(A[sw_val] == own[sw_val]):.3f})')
    per_switch = g.sum() / max(sw_val.sum(), 1)   # entity-F points gained per switched record
    tA = np.flatnonzero(sw_val); Fv = pair_feats('train', tA, np.maximum(B[tA], A[tA]))
    print('   by slice (US/India):')
    for name, s in slices(Fv).items():
        s = s if s is not None else np.ones(len(tA), bool)
        if name == 'other': s = ~np.any([v for k_, v in slices(Fv).items() if v is not None], axis=0)
        if s.sum(): print(f'     {name:30s} {s.sum():6,d}  switched-to decision correct {np.mean(B[tA][s] == own[tA][s]):.3f}  vs A {np.mean(A[tA][s] == own[tA][s]):.3f}')
    # ---------- France
    At, Bt = dec('stage2_v6cov', 'test'), dec('stage2_v7e_usin', 'test')
    a0t, _, _, q0t, _ = load(f'{W}artifacts_v3/test_scores', f'{W}artifacts_v3/test_calibration/probabilities.npy')
    Jt = np.where((a0t >= 0) & (q0t >= .5), a0t, -1)
    ct = np.load(f'{W}artifacts/test_anchor_countries.npy', allow_pickle=True)
    fr = ((At >= 0) & (ct[np.maximum(At, 0)] == 'France')) | ((Bt >= 0) & (ct[np.maximum(Bt, 0)] == 'France'))
    sw = fr & (At != Bt) & (Jt == Bt) & (q0t >= TH)
    tF = np.flatnonzero(sw); Ff = pair_feats('test', tF, np.maximum(Bt[tF], At[tF]))
    n_fr = int((ct == 'France').sum())
    print(f'\nFRANCE: records switched by the judge-gated rule: {sw.sum():,}  (adds link {int((sw & (At < 0)).sum()):,}, removes link {int((sw & (Bt < 0)).sum()):,}, changes company {int((sw & (At >= 0) & (Bt >= 0)).sum()):,})')
    for name, s in slices(Ff).items():
        s = s if s is not None else np.ones(len(tF), bool)
        if name == 'other': s = ~np.any([v for k_, v in slices(Ff).items() if v is not None], axis=0)
        print(f'     {name:30s} {s.sum():6,d}')
    val_entities = int(((folds >= 90) & (folds < 100)).sum())
    gain_fr = per_switch * sw.sum() / n_fr * val_entities / val_entities   # entity-F points per switch, spread over French entities
    est_fr = per_switch * sw.sum() / n_fr
    print(f'\nexpected France macro-F0.5 change (US/India per-switch transfer): {est_fr:+.4f}  -> public at 15% weight: {0.15 * est_fr:+.5f}')
    json.dump({'TH': TH, 'usin_gain': float(g.mean()), 'usin_switches': int(sw_val.sum()), 'fr_switches': int(sw.sum()), 'est_fr': float(est_fr), 'est_public': float(.15 * est_fr)},
              open(f'/private/tmp/claude-501/fable_review/france_lab_{TH}.json', 'w'), indent=2)
    np.save(f'/private/tmp/claude-501/fable_review/france_switch_{TH}.npy', tF)

if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.9)
