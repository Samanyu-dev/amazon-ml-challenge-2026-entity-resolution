"""Do raw-text generator fingerprints on a target separate true duplicates (owner>=0) from decoys?"""
import sys, re
W = '/Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/local_submission/'
sys.path.insert(0, W + 'src')
import numpy as np, pandas as pd
from decode import load
R = '/Users/apple/Downloads/student_resource/dataset/'
rd = lambda f: pd.read_csv(R + f, sep='\t', dtype=str, quoting=3, keep_default_na=False)
def flags(tg):
    n, a = tg.business_name, tg.business_address
    F = {
     'addr_empty': a.str.strip() == '',
     'addr_upper': (a.str.upper() == a) & (a.str.contains('[A-Z]', regex=True)),
     'addr_hash': a.str.contains('#', regex=False),
     'addr_null': a.str.contains(r'\bnull\b', regex=True),
     'addr_numletter': a.str.contains(r'\b\d+[A-Z]\b|\b\d+-[A-Z]\b', regex=True),
     'addr_www': a.str.contains('www.', regex=False),
     'addr_nonlatin': a.str.contains(r'[^\x00-ɏ]', regex=True),
     'name_bracket': n.str.contains(r'[\[\]]', regex=True),
     'name_paren': n.str.contains(r'[()]', regex=True),
     'name_dblspace': n.str.contains('  ', regex=False),
     'name_phone': n.str.contains(r'- \d{6,}', regex=True),
     'name_www': n.str.contains('www.', regex=False) | a.str.contains('www.', regex=False),
     'name_upper': (n.str.upper() == n) & n.str.contains('[A-Z]', regex=True),
     'name_lower': (n.str.lower() == n) & n.str.contains('[a-z]', regex=True),
     'name_accent': n.str.contains(r'[À-ɏ]', regex=True),
     'name_nonlatin': n.str.contains(r'[^\x00-ɏ]', regex=True),
     'name_hyphen_glue': n.str.contains(r'\w-\w|&-', regex=True),
     'name_prefix': n.str.contains(r'^(?:Dr|Smt|Inc|The|Shri|Sri|M/s)\b', regex=True),
     'name_suffix_word': n.str.contains(r'\b(?:Center|Services|Partners|Group|Enterprises|Co|Global)\W*$', regex=True),
     'name_one_word': ~n.str.strip().str.contains(' ', regex=False),
     'name_punct_end': n.str.contains(r'[!&.,]\s*$', regex=True),
    }
    return pd.DataFrame(F)
if __name__ == '__main__':
    tg = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
    _, own, _, _, _ = load(W + 'artifacts_v3/train_scores', W + 'artifacts_v3/calibration/probabilities.npy')
    F = flags(tg); y = own >= 0
    np.save('/private/tmp/claude-501/fable_review/train_fp.npy', F.values); print(list(F.columns))
    print(f'base true-duplicate rate {y.mean():.3f}  n={len(y):,}')
    for c in F:
        m = F[c].values
        print(f'{c:18s} share {m.mean():.4f}  dup-rate if set {y[m].mean() if m.any() else 0:.3f}  unset {y[~m].mean():.3f}')
