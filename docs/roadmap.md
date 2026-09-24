# Implementation roadmap

[Project](https://github.com/users/Samanyu-dev/projects/5) · [All issues](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues)

All items are planned, initially unassigned, and linked with native blocked-by relationships. Ready means no implementation prerequisite; Optional means the experiment can close with a documented no-go. No due dates were invented because the deadline is unknown.

## M0 — Reproducible foundations

Exit gate: dataset manifest, correct macro-F0.5 evaluator, leakage-safe splits, reproducible environment, and verified compute-credit ledger. No paid training before these gates.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [[Data] Stream, validate, fingerprint, and index the supplied TSV dataset](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/1) | P0 | CPU | Ready |
| [[Evaluation] Implement the exact per-reference macro-F0.5 scorer and oracle ceiling](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/2) | P0 | CPU | Ready |
| [[Evaluation] Create immutable entity-grouped splits and country-transfer protocols](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/3) | P0 | CPU | #1, #2 |
| [[Engineering] Establish the runnable package, reproducible environment, and small-fixture CI](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/4) | P0 | CPU | Ready |
| [[Operations] Verify credits, quotas, deadlines, and enforce the $670 compute ledger](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/5) | P0 | CPU | Ready |

## M1 — Strong lexical baseline

Exit gate: Unicode-safe representations, multi-route lexical candidates, blocking audit, hard negatives, pair features, and a calibrated CatBoost baseline with a country/slice error report.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [[Features] Implement loss-aware Unicode, name, and address representations](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/6) | P0 | CPU | #1, #3, #4 |
| [[Retrieval] Build multi-route lexical candidates across both target sources](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/7) | P0 | CPU | #6, #3, #4 |
| [[Evaluation] Measure candidate recall, per-business completeness, and macro score ceiling](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/8) | P0 | CPU | #7, #2 |
| [[Training data] Mine realistic hard negatives with split-safe business weighting](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/9) | P0 | CPU | #7, #3, #1 |
| [[Features] Build batched pair evidence and ambiguity features](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/10) | P0 | CPU | #6, #7, #9 |
| [[Model] Train and calibrate the first strong CatBoost baseline](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/11) | P0 | CPU | #10, #2, #3 |
| [[Evaluation] Build a reproducible error taxonomy and experiment comparison report](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/12) | P0 | CPU | #11, #8 |

## M2 — Multilingual retrieval

Exit gate: frozen multilingual retrieval benchmark and a cached, resumable hybrid index. Retriever fine-tuning is conditional on held-out incremental recall per dollar.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [[Retrieval] Benchmark frozen BGE-M3 for multilingual candidate recovery](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/13) | P1 | Free GPU | #12, #5, #8 |
| [[Optional] Fine-tune the retriever with multi-positive contrastive learning](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/14) | P2 | Paid GPU | #13, #9, #5 |
| [[Retrieval] Build versioned embedding caches and production ANN indexes](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/15) | P1 | Mixed | #13, #1, #5 |

## M3 — Neural pair verification

Exit gate: task-tuned multilingual verifier and an evaluated selective cascade, including routing errors and measured full-run cost.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [[Model] Fine-tune a multilingual cross-encoder for business identity](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/16) | P1 | Paid GPU | #9, #12, #5, #4 |
| [[Inference] Route uncertain pairs to the verifier and audit confident routing errors](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/17) | P1 | Mixed | #16, #11, #8 |

## M4 — Decisions and robustness

Exit gate: held-out calibration and macro-F0.5 decisions, country-transfer tests, optional graph ablation, and a frozen model-selection report.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [[Decisions] Calibrate and combine lexical, tree, and neural evidence without leakage](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/18) | P0 | CPU | #11, #2, #3 |
| [[Decisions] Optimize empty-versus-multiple matches for per-business macro-F0.5](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/19) | P0 | CPU | #18, #2 |
| [[Optional] Add cross-source corroboration and audited target-conflict resolution](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/20) | P2 | CPU | #19, #1, #12 |
| [[Robustness] Validate unseen-country behavior and audit France inference coverage](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/21) | P0 | Mixed | #19, #12, #6 |

## M5 — Full-scale reproducible execution

Exit gate: restartable shards, measured resource budgets, frozen full-data models, and complete deterministic test inference with genuine candidate lineage.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [[Operations] Implement resumable multi-account workers and artifact manifests](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/22) | P0 | Mixed | #4, #5, #1 |
| [[Engineering] Profile full-scale memory, storage, and cost before production runs](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/23) | P0 | Mixed | #10, #5, #22 |
| [[Release] Select the final pipeline, refit on permitted data, and freeze artifacts](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/24) | P0 | Mixed | #19, #21, #23, #12 |
| [[Inference] Generate complete test matches and the genuine candidate TSV](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/25) | P0 | Mixed | #24, #22, #23 |

## M6 — Validated submission and handoff

Exit gate: official output validation, clean-room replay, filled methodology, license/provenance record, and versioned private release. Submission deadline is not yet supplied.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [[Submission] Run the official validator and build a clean-room reproducible ZIP](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/27) | P0 | CPU | #25, #4, #26 |
| [[Documentation] Fill the official methodology with evidence, licenses, and limitations](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/26) | P0 | CPU | #24 |
| [[Release] Publish a private reproducible release and hand off submission artifacts](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues/28) | P0 | CPU | #27, #26 |
