# Dataset contract and measured audit

## Location and schema

The inspected resource folder is `student_resource/` (the original chat spelling omitted an `r`). `DATA_ROOT` is its `dataset/` child. The folder is external to Git. Read with an explicit tab separator and preserve empty strings.

| Relative to DATA_ROOT | Rows | Use |
|---|---:|---|
| train/train_source1.tsv | 2,206,821 | Labeled reference businesses; grouped splits and model queries |
| train/train_source2.tsv | 5,034,616 | Training target gallery and split-eligible supervised pairs |
| train/train_source3.tsv | 5,285,603 | Training target gallery and split-eligible supervised pairs |
| train/train_ground_truth.tsv | 2,206,821 | Sole source of supplied training labels |
| test/test_source1.tsv | 1,732,544 | Every row must appear in both final output files |
| test/test_source2.tsv | 4,887,273 | Valid S2 target IDs for test inference |
| test/test_source3.tsv | 5,082,316 | Valid S3 target IDs for test inference |

Source columns: `entity_id`, `business_name`, `business_address`, `country`. Truth columns: `source1_entity_id`, `matched_entity_ids` (comma-separated, empty for singletons). IDs and their numeric suffixes are join keys, not model features. S1 is deduplicated by the problem description. Audit actual target uniqueness before enforcing graph constraints.

## Measured distributions

The complete-file streaming audit found 7,638,365 training links. Ground-truth match cardinality:

| Matches | Reference businesses |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

The observed maximum is not a test-time cap. Singleton share is about 5.58%.

| Country | Train S1 | Test S1 |
|---|---:|---:|
| US | 1,323,633 | 663,106 |
| India | 883,188 | 809,986 |
| France | 0 | 259,452 |

Missing business addresses: train S2 168,967; train S3 175,916; test S2 129,408; test S3 136,098. The initial audit found no empty source names/countries/IDs and no empty S1 addresses. These are observed release properties, not parser assumptions. Full country/source counts are in [dataset-audit.json](dataset-audit.json).

The six sources have 24,229,173 total rows. Test all-pairs matching would require about 17.27 trillion comparisons. A budget of 100 candidates per test anchor gives 173,254,400 candidate pairs. Use bounded-memory data processing and retrieval.

## Initial diagnostics, not trained-model results

A reproducible seed-42 reservoir sample of 10,000 training S1 anchors was checked against the full training S2/S3 gallery. Equality baselines used simple NFKD-to-ASCII, lowercase, punctuation/space normalization and same-country keys; empty keys were not matched. This deliberately limited normalization is NOT the recommended production representation because it can erase non-Latin text.

| Rule | Sample macro-F0.5 |
|---|---:|
| Normalized name equality | 0.3564273752 |
| Normalized address equality | 0.1974971873 |
| Both normalized fields equal | 0.0845335886 |

No learned model or held-out training procedure produced these numbers. Exact name+address equality had 535 true-positive links and no false positives in this sample; this does not establish universal precision. Do not reuse these scores as the new baseline's validation results.

## Required audit not yet completed

The foundation issue must independently fingerprint files and verify unique IDs, full foreign-key integrity, shared targets, cross-country positives, and duplicate record groups. Row counts and sample diagnostics do not substitute for these checks.

## Data boundaries

- Training labels are used only under the immutable group split and training eligibility rules.
- Validation retrieval uses a realistic full target gallery, but supervised fitting cannot use held-out group targets as negatives.
- Test has no ground truth. Permitted uses in this plan are input integrity, unlabeled distribution checks, index construction, and frozen inference.
- No manual labels/pseudo-label training on test or business identity lookup is planned.
- Learn abbreviations and normalization statistics within the declared training partition. Declare any unsupervised gallery-wide fitting separately.
- Keep supplied raw rows, embeddings, private error examples, and large artifacts outside Git. Share them only through private authorized team storage.
