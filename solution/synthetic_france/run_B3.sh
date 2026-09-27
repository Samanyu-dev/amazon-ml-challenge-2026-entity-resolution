#!/bin/zsh
set -e
cd /Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/syn_run
P=../submission_env/bin/python; D=/private/tmp/claude-501/synth2/data
step(){ echo "=== $(date +%H:%M:%S) $*"; }
until grep -q INFER_DONE reports/ce_small.log; do sleep 15; done
step ce small 4new
(cd src && ../$P ce_gpu.py infer --data-dir $D --pairs ../ce/ce_infer4new_test.npz --split test --model ../ce/ce_model_kaggle --out ../ce/ce4_scores_test.npy --batch 128 > ../reports/ce_small4.log 2>&1)
step start e5-base alone + stage2 small-only in parallel
(cd src && ../$P ce_gpu.py infer --data-dir $D --pairs ../ce/ce_infer_test.npz --split test --model ../ce/ce_model_v2 --out ../ce/ce_v2_scores_test.npy --batch 128 --max-len 96 > ../reports/ce_base.log 2>&1) &
TR=(--train-ce ce/ce_infer_train.npz=ce/ce_scores_train.npy ce/ce_infer4new_train.npz=ce/ce4_scores_train.npy)
TE=(--test-ce ce/ce_infer_test.npz=ce/ce_scores_test.npy ce/ce_infer4new_test.npz=ce/ce4_scores_test.npy --test-dir $D/test)
STACK=stack_v7 $P src/stack_group.py "${TR[@]}" "${TE[@]}" --reference ../local_submission/artifacts_v3/stage2_v6/val_entity_f.npy --out artifacts_v3/stage2_syn_small > reports/stage2_small.log 2>&1
step SMALL_DONE
$P score_syn.py artifacts_v3/stage2_syn_small/ > reports/score_small.txt 2>&1
wait
step stage2 v6 config
TR=(--train-ce ce/ce_infer_train.npz=ce/ce_scores_train.npy ce/ce_infer4new_train.npz=ce/ce4_scores_train.npy --train-extra ce/ce_infer_train.npz=ce/ce_v2_scores_train.npy)
TE=(--test-ce ce/ce_infer_test.npz=ce/ce_scores_test.npy ce/ce_infer4new_test.npz=ce/ce4_scores_test.npy --test-extra ce/ce_infer_test.npz=ce/ce_v2_scores_test.npy --test-dir $D/test)
STACK=stack_v7 $P src/stack_group.py "${TR[@]}" "${TE[@]}" --reference ../local_submission/artifacts_v3/stage2_v6/val_entity_f.npy --out artifacts_v3/stage2_syn_v6 > reports/stage2.log 2>&1
$P score_syn.py artifacts_v3/stage2_syn_v6/ > reports/score_v6.txt 2>&1
step B_DONE
