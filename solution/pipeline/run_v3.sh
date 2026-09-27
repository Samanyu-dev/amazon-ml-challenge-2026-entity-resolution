#!/bin/bash
# v3 pipeline: v3 normalisation + lexical top-8 + ANN top-4 blocking, 63 pair features,
# CatBoost (4000 iters, early stopping), calibration, per-anchor decoder.
# Run from work/local_submission after src/normalize_corpus.py --version v3 and src/ann.py.
set -euo pipefail
P=../submission_env/bin/python
V=artifacts_v3
step(){ echo "=== $(date +%H:%M:%S) $*"; }

cat artifacts/train_ann/s2-*.bin artifacts/train_ann/s3-*.bin > $V/train_dense.bin
cat artifacts/test_ann/s2-*.bin artifacts/test_ann/s3-*.bin > $V/test_dense.bin

step retrieve train
[ -f $V/train_pairs/run.json ] || ./retrieve_v3 $V/train_s1.norm.tsv $V/train_targets.norm.tsv $V/train_pairs 9 8 0 0 $V/train_dense.bin 2> reports/retrieval_v3_train.log
step retrieve test
[ -f $V/test_pairs/run.json ] || ./retrieve_v3 $V/test_s1.norm.tsv $V/test_targets.norm.tsv $V/test_pairs 9 8 0 0 $V/test_dense.bin 2> reports/retrieval_v3_test.log

step train model
[ -f $V/model/model.cbm ] || $P src/train_model.py --pairs $V/train_pairs --out $V/model --artifacts artifacts --iterations 4000 --threads 9 --dense-features
step score train
[ -f $V/train_scores/complete.json ] || $P src/score_pairs.py --pairs $V/train_pairs --model $V/model/model.cbm --out $V/train_scores --targets 10320219 --anchors 2206821 --dense-features --artifacts artifacts --split train --threads 9
step calibrate
[ -f $V/calibration/selection.json ] || $P src/calibrate.py --scores $V/train_scores --artifacts artifacts --out $V/calibration
step decoder grid
[ -f $V/decoder/decoder_report.json ] || $P src/decode.py --out $V/decoder --train-scores $V/train_scores --train-probabilities $V/calibration/probabilities.npy \
  --missing .25 .5 .75 1 --empty-scale 1 2 3 4 --floor 0
step score test
[ -f $V/test_scores/complete.json ] || $P src/score_pairs.py --pairs $V/test_pairs --model $V/model/model.cbm --out $V/test_scores --targets 9969589 --anchors 1732544 --dense-features --artifacts artifacts --split test --threads 9
step calibrate test
[ -f $V/test_calibration/probabilities.npy ] || $P src/calibrate.py --scores $V/test_scores --artifacts artifacts --out $V/test_calibration --apply --calibrator $V/calibration/calibrator.pkl
step PIPELINE_DONE
