# Implementation roadmap

[Project](https://github.com/users/Samanyu-dev/projects/5) · [All issues](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues)

All items are planned, initially unassigned, and linked with native blocked-by relationships. Ready means no implementation prerequisite; Optional means the experiment can close with a documented no-go. No due dates were invented because the deadline is unknown.

## M0 — Reproducible foundations

Exit gate: dataset manifest, correct macro-F0.5 evaluator, leakage-safe splits, reproducible environment, and verified compute-credit ledger. No paid training before these gates.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [Data] Stream, validate, fingerprint, and index the supplied TSV dataset | P0 | CPU | Ready |
| [Evaluation] Implement the exact per-reference macro-F0.5 scorer and oracle ceiling | P0 | CPU | Ready |
| [Evaluation] Create immutable entity-grouped splits and country-transfer protocols | P0 | CPU | data-contract, metric |
| [Engineering] Establish the runnable package, reproducible environment, and small-fixture CI | P0 | CPU | Ready |
| [Operations] Verify credits, quotas, deadlines, and enforce the $670 compute ledger | P0 | CPU | Ready |

## M1 — Strong lexical baseline

Exit gate: Unicode-safe representations, multi-route lexical candidates, blocking audit, hard negatives, pair features, and a calibrated CatBoost baseline with a country/slice error report.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [Features] Implement loss-aware Unicode, name, and address representations | P0 | CPU | data-contract, splits, environment |
| [Retrieval] Build multi-route lexical candidates across both target sources | P0 | CPU | normalization, splits, environment |
| [Evaluation] Measure candidate recall, per-business completeness, and macro score ceiling | P0 | CPU | lexical, metric |
| [Training data] Mine realistic hard negatives with split-safe business weighting | P0 | CPU | lexical, splits, data-contract |
| [Features] Build batched pair evidence and ambiguity features | P0 | CPU | normalization, lexical, negatives |
| [Model] Train and calibrate the first strong CatBoost baseline | P0 | CPU | pair-features, metric, splits |
| [Evaluation] Build a reproducible error taxonomy and experiment comparison report | P0 | CPU | catboost, blocking-audit |

## M2 — Multilingual retrieval

Exit gate: frozen multilingual retrieval benchmark and a cached, resumable hybrid index. Retriever fine-tuning is conditional on held-out incremental recall per dollar.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [Retrieval] Benchmark frozen BGE-M3 for multilingual candidate recovery | P1 | Free GPU | error-analysis, compute, blocking-audit |
| [Optional] Fine-tune the retriever with multi-positive contrastive learning | P2 | Paid GPU | dense-pilot, negatives, compute |
| [Retrieval] Build versioned embedding caches and production ANN indexes | P1 | Mixed | dense-pilot, data-contract, compute |

## M3 — Neural pair verification

Exit gate: task-tuned multilingual verifier and an evaluated selective cascade, including routing errors and measured full-run cost.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [Model] Fine-tune a multilingual cross-encoder for business identity | P1 | Paid GPU | negatives, error-analysis, compute, environment |
| [Inference] Route uncertain pairs to the verifier and audit confident routing errors | P1 | Mixed | verifier, catboost, blocking-audit |

## M4 — Decisions and robustness

Exit gate: held-out calibration and macro-F0.5 decisions, country-transfer tests, optional graph ablation, and a frozen model-selection report.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [Decisions] Calibrate and combine lexical, tree, and neural evidence without leakage | P0 | CPU | catboost, metric, splits |
| [Decisions] Optimize empty-versus-multiple matches for per-business macro-F0.5 | P0 | CPU | calibration, metric |
| [Optional] Add cross-source corroboration and audited target-conflict resolution | P2 | CPU | set-selection, data-contract, error-analysis |
| [Robustness] Validate unseen-country behavior and audit France inference coverage | P0 | Mixed | set-selection, error-analysis, normalization |

## M5 — Full-scale reproducible execution

Exit gate: restartable shards, measured resource budgets, frozen full-data models, and complete deterministic test inference with genuine candidate lineage.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [Operations] Implement resumable multi-account workers and artifact manifests | P0 | Mixed | environment, compute, data-contract |
| [Engineering] Profile full-scale memory, storage, and cost before production runs | P0 | Mixed | pair-features, compute, shards |
| [Release] Select the final pipeline, refit on permitted data, and freeze artifacts | P0 | Mixed | set-selection, france, performance, error-analysis |
| [Inference] Generate complete test matches and the genuine candidate TSV | P0 | Mixed | freeze, shards, performance |

## M6 — Validated submission and handoff

Exit gate: official output validation, clean-room replay, filled methodology, license/provenance record, and versioned private release. Submission deadline is not yet supplied.

| Issue | Priority | Compute | Dependencies |
|---|---|---|---|
| [Submission] Run the official validator and build a clean-room reproducible ZIP | P0 | CPU | full-inference, environment, methodology |
| [Documentation] Fill the official methodology with evidence, licenses, and limitations | P0 | CPU | freeze |
| [Release] Publish a private reproducible release and hand off submission artifacts | P0 | CPU | package, methodology |
