import sys, numpy as np
sp, K = sys.argv[1], 3; FR = '/private/tmp/claude-501/fable_review/'
Z = dict(np.load(FR + f'tr_block_{sp}.npz')); t = Z['tid']
st = np.r_[0, np.flatnonzero(np.diff(t)) + 1]; rank = np.arange(len(t)) - np.repeat(st, np.diff(np.r_[st, len(t)]))
k = rank < K; Z = {a: b[k] for a, b in Z.items()}; np.savez(FR + f'tr_block_{sp}.npz', **Z)
n2 = 5034616 if sp == 'train' else 4887273; t = Z['tid'].astype(np.int64)
np.savez(FR + f'tr_pairs_{sp}.npz', s1_row=Z['aid'].astype(np.int32), src=np.where(t < n2, 2, 3).astype(np.int8), t_row=np.where(t < n2, t, t - n2).astype(np.int32), tid=t.astype(np.int32))
print(sp, len(t))
