#!/bin/zsh
# v7d: add AWS e5-large CE to stage 2; US/India from the best validated stack, France = v7c (v6 with odds x0.5).
set -e
cd "${0:A:h}"; P=../submission_env/bin/python
A=/private/tmp/claude-501/aws_env/bin/aws; B=s3://er-ml26-908404959957-aps1/checkpoints/ce-large
$A s3 cp $B/ce_large_scores_train.npy ce/ && $A s3 cp $B/ce_large_scores_test.npy ce/
../submission_env/bin/python - <<'EOF'
import numpy as np
for s in ('train', 'test'):
    a, b = np.load(f'ce/ce_large_scores_{s}.npy'), np.load(f'ce/ce_v2_scores_{s}.npy')
    assert a.shape == b.shape, (s, a.shape, b.shape); assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    r = float(np.corrcoef(a, b)[0, 1]); assert a.std() > 0.1 and r > 0.9, ('scores look unfilled or wrong', s, float(a.mean()), r)
    print('INTEGRITY', s, a.shape, 'corr vs base r1', round(float(np.corrcoef(a, b)[0, 1]), 4), 'mean', round(float(a.mean()), 4))
EOF
C=(--train-ce ce/ce_infer_train.npz=ce/ce_scores_train.npy ce/ce_infer4new_train.npz=ce/ce4_scores_train.npy --train-extra ce/ce_infer_train.npz=ce/ce_v2_scores_train.npy)
T=(--test-ce ce/ce_infer_test.npz=ce/ce_scores_test.npy ce/ce_infer4new_testa.npz=ce/ce4_scores_testa.npy ce/ce_infer4new_testb.npz=ce/ce4_scores_testb.npy --test-extra ce/ce_infer_test.npz=ce/ce_v2_scores_test.npy --test-dir /Users/apple/Downloads/student_resource/dataset/test)
R2=(--train-extra ce/ce_infer_train.npz=ce/ce_v3_scores_train.npy); R2T=(--test-extra ce/ce_infer_test.npz=ce/ce_v3_scores_test.npy)
L=(--train-extra ce/ce_infer_train.npz=ce/ce_large_scores_train.npy); LT=(--test-extra ce/ce_infer_test.npz=ce/ce_large_scores_test.npy)
REF=(--reference artifacts_v3/stage2_v7a_pp/val_entity_f.npy)
STACK=stack_v7 GROUP_EXTRA=1 nice -n 5 $P src/stack_group.py $C $R2 $L $T $R2T $LT $REF --out artifacts_v3/stage2_v7d_all > reports/stage2_v7d_all.log 2>&1 &
STACK=stack_v7 GROUP_EXTRA=1 nice -n 5 $P src/stack_group.py $C $L $T $LT $REF --out artifacts_v3/stage2_v7d_nor2 > reports/stage2_v7d_nor2.log 2>&1
wait
../submission_env/bin/python - <<'EOF'
import json, numpy as np
ref = np.load('artifacts_v3/stage2_v7a_pp/val_entity_f.npy')
for d in ('stage2_v7a_pp', 'stage2_v7d_all', 'stage2_v7d_nor2'):
    f = np.load(f'artifacts_v3/{d}/val_entity_f.npy'); r = json.load(open(f'artifacts_v3/{d}/stage2_report.json')); g = f - ref
    print(f'{d:16s} val {f.mean():.5f}  tuning {r["tuning_85_89"]:.5f}  vs v7a {g.mean():+.6f} (SE {g.std()/np.sqrt(len(g)):.1e})  {r["v6_90_99"]["by_country"]}')
EOF
echo V7D_DONE
