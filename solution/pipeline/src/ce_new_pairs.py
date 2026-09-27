"""Top-4 stage-2 pairs not already scored in the top-2 CE files (ce_infer4 minus ce_infer).
The test file is also split at row 2,086,337 into testa/testb (two Kaggle sessions)."""
import numpy as np

TESTA_ROWS = 2086337

for s in ('train', 'test'):
    a, b = np.load(f'ce/ce_infer_{s}.npz'), np.load(f'ce/ce_infer4_{s}.npz')
    ka = set(zip(a['tid'].tolist(), a['s1_row'].tolist()))
    new = np.array([k not in ka for k in zip(b['tid'].tolist(), b['s1_row'].tolist())])
    out = {k: b[k][new] for k in b.files}
    np.savez_compressed(f'ce/ce_infer4new_{s}.npz', **out)
    if s == 'test':
        np.savez_compressed('ce/ce_infer4new_testa.npz', **{k: v[:TESTA_ROWS] for k, v in out.items()})
        np.savez_compressed('ce/ce_infer4new_testb.npz', **{k: v[TESTA_ROWS:] for k, v in out.items()})
    print(s, 'pairs', len(new), 'new', int(new.sum()))
