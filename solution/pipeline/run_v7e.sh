#!/bin/zsh
# v7e: US/India = v7d stack + e5-large on the top-4 pairs (full band coverage); France = v6 stack + full e5-base r1 coverage (no trim).
set -e
cd "${0:A:h}"; P=../submission_env/bin/python
A=/private/tmp/claude-501/aws_env/bin/aws; B=s3://er-ml26-908404959957-aps1/checkpoints/ce-large
until $A s3 ls $B/ce_large4_scores_testb.npy.done >/dev/null 2>&1; do sleep 60; done
for s in testa testb; do $A s3 cp $B/ce_large4_scores_$s.npy ce/ --only-show-errors; done
../submission_env/bin/python - <<'PY'
import numpy as np
for s in ('testa', 'testb'):
    a, b = np.load(f'ce/ce_large4_scores_{s}.npy').astype(np.float32), np.load(f'ce/ce4_scores_{s}.npy').astype(np.float32)
    r = float(np.corrcoef(a, b)[0, 1]); assert a.shape == b.shape and a.std() > 0.1 and r > 0.8, (s, a.std(), r)
    print('INTEGRITY', s, a.shape, 'corr vs small', round(r, 4))
PY
C=(--train-ce ce/ce_infer_train.npz=ce/ce_scores_train.npy ce/ce_infer4new_train.npz=ce/ce4_scores_train.npy --train-extra ce/ce_infer_train.npz=ce/ce_v2_scores_train.npy --train-extra ce/ce_infer_train.npz=ce/ce_v3_scores_train.npy --train-extra ce/ce_infer_train.npz=ce/ce_large_scores_train.npy ce/ce_infer4new_train.npz=ce/ce_large4_scores_train.npy)
T=(--test-ce ce/ce_infer_test.npz=ce/ce_scores_test.npy ce/ce_infer4new_testa.npz=ce/ce4_scores_testa.npy ce/ce_infer4new_testb.npz=ce/ce4_scores_testb.npy --test-extra ce/ce_infer_test.npz=ce/ce_v2_scores_test.npy --test-extra ce/ce_infer_test.npz=ce/ce_v3_scores_test.npy --test-extra ce/ce_infer_test.npz=ce/ce_large_scores_test.npy ce/ce_infer4new_testa.npz=ce/ce_large4_scores_testa.npy ce/ce_infer4new_testb.npz=ce/ce_large4_scores_testb.npy --test-dir /Users/apple/Downloads/student_resource/dataset/test)
STACK=stack_v7 GROUP_EXTRA=1 nice -n 5 $P src/stack_group.py $C $T --reference artifacts_v3/stage2_v7d_all_val/val_entity_f.npy --out artifacts_v3/stage2_v7e_usin > reports/stage2_v7e_usin.log 2>&1
until [ -s artifacts_v3/stage2_v6cov/test_decisions.npz ]; do sleep 30; done
../submission_env/bin/python - <<'PY'
import json, numpy as np
for d, ref in (('stage2_v7e_usin', 'stage2_v7d_all_val'), ('stage2_v6cov', 'stage2_v6_pp')):
    f, r0 = np.load(f'artifacts_v3/{d}/val_entity_f.npy'), np.load(f'artifacts_v3/{ref}/val_entity_f.npy'); g = f - r0
    print(f'{d:16s} val {f.mean():.6f} vs {ref} {g.mean():+.6f} (SE {g.std()/np.sqrt(len(g)):.1e})')
PY
$P src/assemble_final.py --usin artifacts_v3/stage2_v7e_usin --france artifacts_v3/stage2_v6cov --france-odds 1.0 --test-dir /Users/apple/Downloads/student_resource/dataset/test --out ../../outputs/final_submission_v7e/matching_results.tsv
echo V7E_BUILT
