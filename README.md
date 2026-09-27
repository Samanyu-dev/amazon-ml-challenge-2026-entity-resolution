# Amazon ML Challenge 2026: Business Entity Resolution

> **Final result:** best public **0.986382** (v8b). The final upload was v8b_fr14. Start with **[`docs/final/HANDOFF.md`](docs/final/HANDOFF.md)**. The code, models and submissions are in [`solution/`](solution/README.md).

## Our path: day 1 to day 3

Public leaderboard (macro F0.5 per Source-1 entity; France is 15% of test and has no training labels):

```
0.9624  v2 + decoder      ▏
0.9720  v3                ████
0.9814  v4                ████████
0.9845  v6                █████████▌
0.9842  v7a               █████████▍   (France hurt)
0.9849  v7d               █████████▊
0.9856  v7d_acr           ██████████▏  (French acronym rule)
0.9864  v8b               ██████████▌  ← best scored
```

### Day 1: 25 Sep (baseline to a real pipeline)
| Time | Version | What we did | Validation | **Public** |
|---|---|---|---:|---:|
| 13:14 / 14:27 | v2 | Lexical blocking + CatBoost; both uploads **failed** on the portal side, not the file | — | failed |
| 20:06 | v2 + decoder | Per-entity expected-F0.5 decoder (can predict "no match") | 0.9739 | **0.962428** |
| 22:51 | v3 | v3 normalisation (legal forms, transliteration, FR/IN address forms), dense e5-small ANN blocking, 63 C++ features | 0.98092 | **0.972** |

### Day 2: 26 Sep (cross-encoders and France)
| Time | Version | What we did | Validation | **Public** |
|---|---|---|---:|---:|
| 02:11 | v4 | Stage 2: fine-tuned multilingual-e5-small cross-encoder re-reads uncertain targets (Kaggle T4) | 0.98697 | **0.981393** |
| 08:44 | v6 | + e5-base cross-encoder with 600k French pseudo-labels + group/decoy-context features | 0.98797 | **0.984462** |
| 14:40 | v7a | + round-2 *hard* French pseudo-labels + decoy features. US/India up, **France down** | 0.98828 | **0.984178** |
| 19:30 | v7d | + e5-large (560M, AWS A10G) for US/India; France from v6 with its weakest links trimmed (odds ×0.5) | 0.98865 | **0.984931** |
| night | — | Fixed a cross-encoder coverage bug; orphan-density simulation | | |

### Day 3: 27 Sep (leak fix, generator reverse-engineering, final)
| Time | Version | What we did | Validation | **Public** |
|---|---|---|---:|---:|
| morning | v7e_all_loose18 | Looser decoding everywhere (×1.8) | −0.00024 | **0.984709** |
| morning | v7e_fr_judge | + 12,257 French links backed by stage 1 | same | **0.984725** |
| morning | — | Found a stage-2 **leak** (split by predicted anchor, not true owner): honest validation 0.98841. Qwen2.5-7B judge failed its gates. Upload freeze until offline evidence. | | |
| afternoon | **v7d_acr** | **Generator audit.** Clean records are 99.7% correct, so the loss sits in specific synthetic-noise operations. **French acronym rule** (`BC` = **B**ordeaux **C**ollectif at the same address; 99.98% precise on train) | +0.00002 (US/IN) | **0.985608** |
| evening | **v8b** | Acronym rule v2 + **Indian-script address-number rescue** (+0.00043) + leak-free stage 2 + **meta-model v3** with generator fingerprints (+0.00036) | 0.98877+ | **0.986382** |
| 23:5x | v8b_fr14 | v8b with France odds ×1.4, chosen with a **labelled synthetic France** built from the real French S1 and run through the full pipeline | synthetic FR 0.9231 → 0.9299 | final upload |

**Total:** 0.962428 → **0.986382** (+0.024) in three days. The full story, every tested idea with its measured result, and the lessons are in [`docs/final/HANDOFF.md`](docs/final/HANDOFF.md).

# Amazon ML Challenge 2026 — Business Entity Resolution

Private team workspace for resolving noisy Source 2/3 records to deduplicated Source 1 businesses.

**Current status: planning and reference material initialized. The matching pipeline and trained models are not yet implemented.** The detailed GitHub issues define the implementation work, dependencies, expected artifacts, example interfaces, data boundaries, and acceptance evidence.

- [Delivery project](https://github.com/users/Samanyu-dev/projects/5) — table, delivery board, and milestone roadmap
- [Implementation issues](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/issues)
- [Milestones](https://github.com/Samanyu-dev/amazon-ml-challenge-2026-entity-resolution/milestones)
- [Detailed roadmap](docs/roadmap.md)
- [Dataset contract and measured audit](docs/dataset.md)
- [Architecture and decision rationale](docs/architecture.md)
- [Validation and leakage policy](docs/validation.md)
- [Compute budget and run gates](docs/compute-budget.md)
- [Working agreements](CONTRIBUTING.md)
- [Issue ownership](docs/ownership.md)

## Objective

Find **zero, one, or many** S2/S3 matches for every test S1 reference. Optimize **macro F0.5 per reference business**, with both truth and prediction empty scoring 1. Test includes **France**, absent from labeled training. Every country must receive output rows.

## Proposed pipeline

```mermaid
flowchart LR
    A[Raw TSV and audit] --> B[Unicode-preserving views]
    B --> C[Lexical candidate retrieval]
    B --> D[Optional multilingual retrieval]
    C --> E[Candidate union and lineage]
    D --> E
    E --> F[Pair features and CatBoost]
    F --> G[Selective multilingual verifier]
    F --> H[Calibration and entity decisions]
    G --> H
    H --> I[Two TSVs and official validation]
```

Neural components ship only when measured quality/cost justifies them. A valid lexical/tree baseline remains available throughout.

## Dataset access

Obtain the supplied `student_resource/` privately through the team's authorized competition access. Keep it outside this Git checkout.

```bash
export DATA_ROOT="/absolute/path/to/student_resource/dataset"
```

Raw data, credentials, embeddings, model checkpoints, and generated outputs are ignored by Git. The repository contains only aggregate audit statistics, planning documents, templates, and the supplied validator. No cloud job has been launched by repository setup.

## Team

Repository owner: `Samanyu-dev`. Write-access invitations sent to `mrlegendary25`, `pranshustuff`, and `explorer271`; access depends on accepting invitations. The private project has writer access configured for these three users. Issues are allocated by workstream: data/infrastructure to `mrlegendary25`, modeling/decisions to `pranshustuff`, and retrieval/evaluation to `explorer271`. See [ownership](docs/ownership.md) for the exact issue map and pending-invitation status.

## First work to start

Assigned owners should start with the four ready foundation issues: ingestion/data contract, exact scorer, package/environment, and compute ledger. Complete the grouped split issue before supervised modeling. The project **Readiness** field and native blocked-by links expose dependencies; update readiness when prerequisites close.

## Submission artifacts

The final implementation must generate `output/matching_results.tsv` and `output/candidate_pairs.tsv`, plus a reproducible archive as described in [architecture](docs/architecture.md). Both TSVs must contain every test reference exactly once. Candidate output is the real input to the composite matching model, including cheap-stage rejects.

## Constraints and provenance

Use only provided business data for identity resolution; no external business registration lookup, geocoding, commercial ER API, or internet-derived data augmentation. Final models must meet the stated MIT/Apache-2.0 and <=8B-parameter requirement. Pin and audit every selected checkpoint. See [vendor provenance](vendor/student_resource/PROVENANCE.md).
