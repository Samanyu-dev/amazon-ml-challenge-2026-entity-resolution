#!/bin/zsh
set -e
cd /Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/syn_run
P=../submission_env/bin/python; D=/private/tmp/claude-501/synth2/data; NS2=590854; NT=1220557; NA=220535
step(){ echo "=== $(date +%H:%M:%S) $*"; }
until grep -q A_DONE reports/run_A.log; do sleep 20; done
step score test
$P src/score_pairs.py --pairs artifacts_v3/test_pairs --model artifacts_v3/model/model.cbm --out artifacts_v3/test_scores --targets $NT --anchors $NA --dense-features --artifacts artifacts --split test --threads 9 > reports/score.log 2>&1
step calibrate
$P src/calibrate.py --scores artifacts_v3/test_scores --artifacts artifacts --out artifacts_v3/test_calibration --apply --calibrator artifacts_v3/calibration/calibrator.pkl > reports/calib.log 2>&1
step ce export
$P src/ce_export.py infer --scores artifacts_v3/test_scores --probabilities artifacts_v3/test_calibration/probabilities.npy --n-s2 $NS2 --out ce/ce_infer_test.npz
$P src/ce_export.py infer --scores artifacts_v3/test_scores --probabilities artifacts_v3/test_calibration/probabilities.npy --band 0.005 0.999 --k 4 --n-s2 $NS2 --out ce/ce_infer4_test.npz
$P - <<'PY'
import numpy as np
a = dict(np.load('ce/ce_infer_test.npz')); b = dict(np.load('ce/ce_infer4_test.npz'))
ka = set((a['tid'].astype(np.int64) << 32 | a['s1_row'].astype(np.int64)).tolist())
kb = b['tid'].astype(np.int64) << 32 | b['s1_row'].astype(np.int64); keep = np.array([x not in ka for x in kb.tolist()])
np.savez_compressed('ce/ce_infer4new_test.npz', **{k: v[keep] for k, v in b.items()}); print('infer', len(a['tid']), 'infer4', len(kb), 'new', keep.sum())
PY
step ce small
(cd src && ../$P ce_gpu.py infer --data-dir $D --pairs ../ce/ce_infer_test.npz --split test --model ../ce/ce_model_kaggle --out ../ce/ce_scores_test.npy --batch 128 > ../reports/ce_small.log 2>&1)
(cd src && ../$P ce_gpu.py infer --data-dir $D --pairs ../ce/ce_infer4new_test.npz --split test --model ../ce/ce_model_kaggle --out ../ce/ce4_scores_test.npy --batch 128 >> ../reports/ce_small.log 2>&1)
step ce base r1
(cd src && ../$P ce_gpu.py infer --data-dir $D --pairs ../ce/ce_infer_test.npz --split test --model ../ce/ce_model_v2 --out ../ce/ce_v2_scores_test.npy --batch 128 --max-len 96 > ../reports/ce_base.log 2>&1)
step stage2 v6 config
TR=(--train-ce ce/ce_infer_train.npz=ce/ce_scores_train.npy ce/ce_infer4new_train.npz=ce/ce4_scores_train.npy --train-extra ce/ce_infer_train.npz=ce/ce_v2_scores_train.npy)
TE=(--test-ce ce/ce_infer_test.npz=ce/ce_scores_test.npy ce/ce_infer4new_test.npz=ce/ce4_scores_test.npy --test-extra ce/ce_infer_test.npz=ce/ce_v2_scores_test.npy --test-dir $D/test)
STACK=stack_v7 $P src/stack_group.py "${TR[@]}" "${TE[@]}" --reference ../local_submission/artifacts_v3/stage2_v6/val_entity_f.npy --out artifacts_v3/stage2_syn_v6 > reports/stage2.log 2>&1
step B_DONE
