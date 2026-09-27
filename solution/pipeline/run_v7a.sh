#!/bin/zsh
# v7a: v6 stage 2 + round-2 e5-base CE (ce_v3) with/without decoy group features.
# Usage: ./run_v7a.sh <dir with ce_v3_scores_train.npy and ce_v3_scores_test.npy>
set -e
cd "${0:A:h}"; P=../submission_env/bin/python; IN=$1
cp $IN/ce_v3_scores_train.npy $IN/ce_v3_scores_test.npy ce/
../submission_env/bin/python - <<'EOF'
import numpy as np
for s in ('train', 'test'):
    a, b = np.load(f'ce/ce_v3_scores_{s}.npy'), np.load(f'ce/ce_v2_scores_{s}.npy')
    assert a.shape == b.shape, (s, a.shape, b.shape); assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    print('INTEGRITY', s, a.shape, 'corr vs round1', round(float(np.corrcoef(a, b)[0, 1]), 4), 'mean', round(float(a.mean()), 4))
EOF
TR=(--train-ce ce/ce_infer_train.npz=ce/ce_scores_train.npy ce/ce_infer4new_train.npz=ce/ce4_scores_train.npy
    --train-extra ce/ce_infer_train.npz=ce/ce_v2_scores_train.npy --train-extra ce/ce_infer_train.npz=ce/ce_v3_scores_train.npy)
TE=(--test-ce ce/ce_infer_test.npz=ce/ce_scores_test.npy ce/ce_infer4new_testa.npz=ce/ce4_scores_testa.npy ce/ce_infer4new_testb.npz=ce/ce4_scores_testb.npy
    --test-extra ce/ce_infer_test.npz=ce/ce_v2_scores_test.npy --test-extra ce/ce_infer_test.npz=ce/ce_v3_scores_test.npy
    --test-dir /Users/apple/Downloads/student_resource/dataset/test)
# ablation: v6 + CE (no decoy features), validation only
STACK=stack_v7 nice -n 5 $P src/stack_group.py $TR --out artifacts_v3/stage2_v7_ce > reports/stage2_v7_ce.log 2>&1 &
# candidate: v6 + decoy/group + CE, with test decisions
STACK=stack_v7 GROUP_EXTRA=1 nice -n 5 $P src/stack_group.py $TR $TE --out artifacts_v3/stage2_v7a > reports/stage2_v7a.log 2>&1
wait
../submission_env/bin/python - <<'EOF'
import json, numpy as np
ref = np.load('artifacts_v3/stage2_v6/val_entity_f.npy')
runs = {'v6': 'stage2_v6', 'v6+decoy': 'stage2_v6_decoy', 'v6+CE': 'stage2_v7_ce', 'v6+decoy+CE': 'stage2_v7a'}
f = {k: np.load(f'artifacts_v3/{d}/val_entity_f.npy') for k, d in runs.items()}
for k, d in runs.items():
    r = json.load(open(f'artifacts_v3/{d}/stage2_report.json')); g = f[k] - ref
    print(f'{k:14s} val {f[k].mean():.5f}  tuning {r.get("tuning_85_89", float("nan")):.5f}  gain vs v6 {g.mean():+.6f} (SE {g.std()/np.sqrt(len(g)):.1e})')
g = f['v6+decoy+CE'] - f['v6+CE']; print(f'decoy on top of CE {g.mean():+.6f} (SE {g.std()/np.sqrt(len(g)):.1e})')
EOF
echo V7A_DONE
