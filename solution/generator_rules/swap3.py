"""One-word swap at same address: is the TARGET's swapped word a generator filler word? Train true-rate by that split;
France: how many LINKED swaps are non-filler (sibling-like)."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/fable_review')
exec(open('/private/tmp/claude-501/fable_review/swap2.py').read().split("from collections import Counter")[0])
import pickle
from decoy_sig import fill_list, deacc
FILL, _, _ = fill_list('train'); FT, _, _ = fill_list('test'); FILL |= FT
h, aid, m, c, P = run('train', 'exp_owner_split')
_, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
t = np.flatnonzero(h); isf = np.array([P[x][0] in FILL for x in t]); ok = own[t] == aid[t]
for nm, s in (('target word IS filler', isf), ('target word NOT filler', ~isf)):
    print(f'TRAIN swap, {nm:24s}: n {s.sum():7,d} true {ok[s].mean():.3f} | linked {m[t][s].mean():.3f}, linked-false share {np.mean(~ok[s][m[t][s]]):.4f}')
h2, aid2, m2, c2, P2 = run('test', 'stage2_v6_pp', 0.5)
for cc in ('France',):
    t2 = np.flatnonzero(h2 & (c2 == cc)); isf2 = np.array([P2[x][0] in FILL for x in t2])
    for nm, s in (('target word IS filler', isf2), ('target word NOT filler', ~isf2)):
        print(f'TEST {cc} swap, {nm:24s}: n {s.sum():6,d} linked {m2[t2][s].mean():.3f} -> linked count {int(m2[t2][s].sum()):,}')
    np.save('/private/tmp/claude-501/fable_review/fr_nonfill_linked.npy', np.column_stack([t2[~isf2 & m2[t2]], aid2[t2[~isf2 & m2[t2]]]]))
    from collections import Counter
    print('   linked non-filler word pairs:', Counter(P2[x] for x in t2[~isf2 & m2[t2]]).most_common(25))
