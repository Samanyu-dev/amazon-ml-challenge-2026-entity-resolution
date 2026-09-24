# Proposed architecture and delivery gates

This is a design, not implemented functionality. Detailed issues are the execution contract.

## Components

1. Stream and fingerprint supplied TSVs, validate labels, and create stable IDs and grouped splits.
2. Preserve raw Unicode plus multiple conservative normalized name/address views. Missingness and numeric evidence remain explicit.
3. Retrieve complementary lexical candidates from S2 and S3. Add multilingual embeddings only after an incremental recall/cost benchmark.
4. Audit recall and oracle macro-F0.5 on the exact post-blocking candidate set entering the composite matcher.
5. Score structured features with CatBoost trained on split-safe hard negatives.
6. Optionally verify uncertain pairs with a task-tuned multilingual cross-encoder. Audit confident cheap decisions as well as routed pairs.
7. Calibrate using held-out predictions. Select zero/multiple matches per business for macro-F0.5. Graph corroboration is optional and never unconditional transitive closure.
8. Run immutable, restartable inference shards. Generate and validate both output TSVs.

## Starting model candidates

| Component | Candidate | Provenance |
|---|---|---|
| Feature classifier | CatBoost | [Apache-2.0 repository](https://github.com/catboost/catboost) |
| Multilingual retriever | BAAI/bge-m3 | [MIT model card](https://huggingface.co/BAAI/bge-m3); about 569M parameters |
| Cross-encoder verifier | BAAI/bge-reranker-v2-m3 | [Apache-2.0 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3); about 568M parameters |

Recheck exact revision/license before using any checkpoint. These are candidates to benchmark, not proven winners for this data. Pin revisions and keep final model size/license evidence. Pretrained documentation lookup does not authorize external business-data enrichment.

## Candidate lineage in a cascade

`candidate_pairs.tsv` is the actual final candidate set supplied to the composite matching stage. If CatBoost cheaply rejects a pair before neural verification, it was still considered by the matcher and belongs in the candidate set. Do not publish only neural-reranked pairs or reconstruct candidate lists from accepted matches. If graph expansion proposes new matches, explicitly generate and score those candidates and preserve their lineage before output.

## Expected final package

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

`matching_results.tsv` columns: `source1_entity_id`, `matched_entity_ids`.
`candidate_pairs.tsv` columns: `source1_entity_id`, `candidate_entity_ids`.
Both use tabs between fields and commas between target IDs, with an empty second field for empty sets. One row for every test S1, only valid test S2/S3 target IDs, no duplicate IDs/rows, and matches subset of candidates.

## Operational fallback

A reproducible lexical + CatBoost + calibrated-threshold solution must remain runnable. Optional retriever tuning, neural verification, and graph reasoning require measured benefit and affordable end-to-end cost. Observed training maximum match count does not limit test predictions. Freeze components before paying for full-corpus encoding/inference.
