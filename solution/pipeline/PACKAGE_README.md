# Business Entity Resolution: reproduction guide

This folder regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the
organiser `dataset/` only. No external data, lookups or hosted APIs are used. All models are
MIT or Apache-2.0 licensed and far below 8B parameters:

| Model | Licence | Parameters |
|---|---|---:|
| `intfloat/multilingual-e5-small` | MIT | 118M |
| `intfloat/multilingual-e5-base` | MIT | 278M |
| `intfloat/multilingual-e5-large` | MIT | 560M |
| CatBoost | Apache-2.0 | — |

## Environment

- Python 3.12.13; `pip install -r requirements.txt`.
- clang++ 21 (or g++ ≥ 11) with OpenMP for the C++ retriever.
- Machine used: Apple M5, 10 cores, 24 GB. CPU stages take about 6 h in total.
- The cross-encoders need a CUDA GPU. We used Kaggle (T4 / 2×T4) and AWS SageMaker ml.g5.2xlarge (A10G).
  Any CUDA GPU works with the same commands.

```bash
clang++ -O3 -std=c++17 -Xpreprocessor -fopenmp -lomp src/retrieve_v3.cpp -o retrieve_v3   # Linux: g++ -O3 -fopenmp
export DATA=/path/to/dataset          # contains train/ and test/
```

All commands run from this folder. Intermediate files go to `artifacts/`, `artifacts_v3/` and `ce/`.

## 1. Data, normalisation, embeddings, candidates (CPU)

```bash
python src/prepare.py --data-root $DATA --work artifacts            # DuckDB copy, SHA-256 fingerprint, stable folds 0-99
python src/make_arrays.py --work artifacts                          # per-S1 fold / country / truth-count arrays
python src/normalize_corpus.py --work artifacts --out artifacts_v3 --version v3 --split all --workers 4
python src/download_models.py                                       # multilingual-e5-small (MIT) for dense retrieval
python src/encode_corpus.py --work artifacts --model models/multilingual-e5-small
for s in train test; do python src/ann.py --work artifacts --split $s --build
  for k in 2 3; do python src/ann.py --work artifacts --split $s --source $k --threads 9 --topk 4; done; done
bash run_v3.sh     # lexical top-8 + ANN top-4 blocking (C++), 63 pair features, stage-1 CatBoost, calibration
```

`artifacts_v3/test_pairs` holds every test candidate pair the stage-1 model scores. That is
`candidate_pairs.tsv` (step 5).

## 2. Cross-encoder training data (CPU)

The fold protocol: CE training uses anchor folds 0–39; stage-2 fit uses 40–74; dev 75–79;
calibration 80–84; tuning 85–89; reporting 90–99.

```bash
python src/ce_export.py train --pairs artifacts_v3/train_pairs --n-s2 5034616 --out ce/ce_train_pairs.npz
python src/ce_export.py infer --scores artifacts_v3/train_scores --probabilities artifacts_v3/calibration/probabilities.npy \
    --anchor-folds artifacts/train_anchor_folds.npy --min-fold 40 --n-s2 5034616 --out ce/ce_infer_train.npz
python src/ce_export.py infer --scores artifacts_v3/test_scores --probabilities artifacts_v3/test_calibration/probabilities.npy \
    --n-s2 4887273 --out ce/ce_infer_test.npz
python src/ce_export.py infer --scores artifacts_v3/train_scores --probabilities artifacts_v3/calibration/probabilities.npy \
    --anchor-folds artifacts/train_anchor_folds.npy --min-fold 40 --band 0.005 0.999 --k 4 --n-s2 5034616 --out ce/ce_infer4_train.npz
python src/ce_export.py infer --scores artifacts_v3/test_scores --probabilities artifacts_v3/test_calibration/probabilities.npy \
    --band 0.005 0.999 --k 4 --n-s2 4887273 --out ce/ce_infer4_test.npz
python src/ce_new_pairs.py            # ce_infer4new_{train,testa,testb}.npz: top-4 pairs not in ce_infer_*
python src/pseudo_labels.py round1     # ce/ce_train_v2_pairs.npz : labelled pairs + French pseudo-labels from stage 1
```

## 3. Cross-encoders (GPU)

Each runs `src/ce_gpu.py`. `src/cloud/` holds the exact Kaggle notebooks and the SageMaker entry script.

| CE | Train pairs | Command (train, then `infer` on `ce_infer_*` / `ce_infer4new_*`) | Output scores |
|---|---|---|---|
| e5-small | `ce_train_pairs.npz` | `ce_gpu.py train --model intfloat/multilingual-e5-small --batch 128` | `ce_scores_*`, `ce4_scores_*` |
| e5-base r1 | `ce_train_v2_pairs.npz` | `ce_gpu.py train --model intfloat/multilingual-e5-base --batch 128 --max-len 96 --lr 2e-5 --data-parallel` | `ce_v2_scores_*` |
| e5-base r2 | `ce_train_v3_pairs.npz` | same as r1 | `ce_v3_scores_*` |
| e5-large | `ce_train_v3_pairs.npz` | `ce_gpu.py train --model intfloat/multilingual-e5-large --batch 64 --max-len 96 --lr 1e-5` | `ce_large_scores_*` |

Round 2 needs the v6 stage-2 decisions (step 4a) first:
`python src/pseudo_labels.py round2 --decisions artifacts_v3/stage2_v6/test_decisions.npz`.

## 4. Stage 2 and final assembly (CPU)

For a leak-free stage-2 evaluation, set `S2_OWNER_SPLIT=1`. This requires both
the predicted owner's fold and the true owner's fold to be in the fit/dev range,
so a stage-1 mistake cannot place a reporting-fold target in stage-2 training.
The corrected US/India report is `artifacts_v3/exp_owner_split` (folds 90-99
macro F0.5 = 0.9884091).

```bash
TR=(--train-ce ce/ce_infer_train.npz=ce/ce_scores_train.npy ce/ce_infer4new_train.npz=ce/ce4_scores_train.npy
    --train-extra ce/ce_infer_train.npz=ce/ce_v2_scores_train.npy)
TE=(--test-ce ce/ce_infer_test.npz=ce/ce_scores_test.npy ce/ce_infer4new_testa.npz=ce/ce4_scores_testa.npy ce/ce_infer4new_testb.npz=ce/ce4_scores_testb.npy
    --test-extra ce/ce_infer_test.npz=ce/ce_v2_scores_test.npy --test-dir $DATA/test)
# 4a. v6 stage 2 (small CE + e5-base r1 + group features): used for France
STACK=stack_v7 python src/stack_group.py "${TR[@]}" "${TE[@]}" --out artifacts_v3/stage2_v6
# 4b. full stage 2 for US/India: + decoy/graph features + e5-base r2 + e5-large
STACK=stack_v7 GROUP_EXTRA=1 python src/stack_group.py "${TR[@]}" \
    --train-extra ce/ce_infer_train.npz=ce/ce_v3_scores_train.npy --train-extra ce/ce_infer_train.npz=ce/ce_large_scores_train.npy \
    "${TE[@]}" --test-extra ce/ce_infer_test.npz=ce/ce_v3_scores_test.npy --test-extra ce/ce_infer_test.npz=ce/ce_large_scores_test.npy \
    --out artifacts_v3/stage2_v7d_all
# 4c. final file: US/India from 4b; France from 4a. The v7e build used odds x1.0.
python src/assemble_final.py --usin artifacts_v3/stage2_v7d_all --france artifacts_v3/stage2_v6 --france-odds 1.0 \
    --test-dir $DATA/test --out output/matching_results.tsv
```

## 5. Candidate file and check

```bash
python src/export_candidates.py --scores artifacts_v3/test_scores --test-dir $DATA/test --out output/candidate_pairs.tsv
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir $DATA/test --check-ids
```

## Verified reproductions

- `assemble_final.py` regenerates the submitted `matching_results.tsv` byte-for-byte.
- `pseudo_labels.py round1 / round2` regenerate both pseudo-label files array-for-array.
- Every stage is seeded; CatBoost uses fixed `random_seed`.

## Source map

| File | Role |
|---|---|
| `prepare.py`, `make_arrays.py` | load TSVs into DuckDB, stable hash folds, helper arrays |
| `normalization_v3.py`, `normalize_corpus.py` | name/address normalisation (legal forms, transliteration, FR/IN address forms) |
| `encode_corpus.py`, `ann.py` | e5-small embeddings, FAISS ANN top-4 per target |
| `retrieve_v3.cpp` | inverted-index blocking and 63 pair features |
| `train_model.py`, `score_pairs.py`, `calibrate.py`, `embedding_features.py` | stage-1 CatBoost, scoring, logistic calibration |
| `ce_export.py`, `ce_gpu.py`, `pseudo_labels.py`, `cloud/` | cross-encoder data, training and inference |
| `stack_v7.py`, `stack_group.py`, `stack_ce.py` | stage-2 CatBoost over uncertain targets (CE scores + group/decoy features) |
| `decode.py` | per-entity expected-F0.5 decoder (chooses how many links, including none) |
| `assemble_final.py`, `build_submission.py`, `export_candidates.py` | output files |
| `test_*.py` | unit tests (normalisation, metric, export lineage) |
