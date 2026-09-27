#!/bin/zsh
set -e
cd /Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/syn_run
P=../submission_env/bin/python; D=/private/tmp/claude-501/synth2/data
step(){ echo "=== $(date +%H:%M:%S) $*"; }
until grep -q INFER_DONE reports/ce_small.log; do sleep 15; done
step ce small 4new
(cd src && ../$P ce_gpu.py infer --data-dir $D --pairs ../ce/ce_infer4new_test.npz --split test --model ../ce/ce_model_kaggle --out ../ce/ce4_scores_test.npy --batch 128 > ../reports/ce_small4.log 2>&1)
until grep -q INFER_DONE reports/ce_base.log; do sleep 15; done
step stage2 v6 config
TR=(--train-ce ce/ce_infer_train.npz=ce/ce_scores_train.npy ce/ce_infer4new_train.npz=ce/ce4_scores_train.npy --train-extra ce/ce_infer_train.npz=ce/ce_v2_scores_train.npy)
TE=(--test-ce ce/ce_infer_test.npz=ce/ce_scores_test.npy ce/ce_infer4new_test.npz=ce/ce4_scores_test.npy --test-extra ce/ce_infer_test.npz=ce/ce_v2_scores_test.npy --test-dir $D/test)
STACK=stack_v7 $P src/stack_group.py "${TR[@]}" "${TE[@]}" --reference ../local_submission/artifacts_v3/stage2_v6/val_entity_f.npy --out artifacts_v3/stage2_syn_v6 > reports/stage2.log 2>&1
step B_DONE
