# Handoff: Amazon ML Challenge 2026, Business Entity Resolution

Competition closed 27 Sep 2026, 23:59 IST. This document is the single place to start.

## 1. Final state

- **Best public score: 0.986382** (v8b).
- **Final upload: v8b_fr14.** This is v8b with French decoding odds ×1.4 instead of ×0.5. Its score was pending at handoff.
- The private ranking uses the best public submission.
- Honest leak-free US/India validation (folds 90–99), current pipeline:

  | Stage | Score |
  |---|---:|
  | Stage 2 alone | 0.988409 |
  | + meta-model v3 | 0.988767 |
  | + Indian-script rescue | ≈ 0.9892 (the gains add) |

- Leaderboard context on 27 Sep: #1 ≈ 0.9915, top 10 ≈ 0.9903, top 50 ≈ 0.9883.
- One public repo from another team (Ekdant.exe, `Ruthwik000/amazonsummerML`) reports 0.989191.

## 2. Leaderboard history (every scored upload)

| Version | What changed | Public |
|---|---|---:|
| v2 + decoder | lexical blocking + CatBoost + expected-F0.5 decoder | 0.962428 |
| v3 | v3 normalisation, dense (ANN) blocking, 63 features | 0.972 |
| v4 | + e5-small cross-encoder, stage 2 | 0.981393 |
| v6 | + e5-base round 1 (600k French pseudo-labels) + group features | 0.984462 |
| v7a | + decoy features + e5-base round 2 (hard French pseudo-labels) | 0.984178 (France hurt) |
| v7d | + e5-large for US/India; France from v6 with odds ×0.5 | 0.984931 |
| v7e_all_loose18 | odds ×1.8 in every country | 0.984709 |
| v7e_fr_judge | + 12,257 French links backed by stage 1 | 0.984725 |
| **v7d_acr** | v7d + **French acronym rule** (+12,847 links) | **0.985608** |
| **v8b** | acronym rule v2 + Indian-script rescue + leak-free stage 2 + meta-model v3 | **0.986382** |
| v8b_fr14 | v8b with France odds ×1.4 | pending |

Files: `solution/submissions/` holds v7d, v7d_acr, v8b and v8b_fr14 (gzipped). The other versions can be rebuilt from `solution/pipeline`.

## 3. Pipeline (v8b)

1. **Normalisation (v3).**
   - Legal forms are split into their own field (US/IN/FR).
   - Indian scripts are transliterated.
   - French and US street abbreviations are canonicalised (`R.`→rue, `Bd`→blvd).
   - Also handles `M/s` and dotted acronyms.
2. **Blocking.**
   - A C++ inverted index (`retrieve_v3.cpp`) takes the lexical top 8. It indexes name tokens, character trigrams, address tokens and 4-grams, phonetic skeletons and house numbers.
   - multilingual-e5-small with FAISS takes the dense top 4.
   - Both run per country.
   - Result: 117M candidate pairs, 98.48% recall.
3. **Stage 1.** CatBoost on 63 pair features, then logistic calibration.
4. **Stage 2** re-decides uncertain targets (q in 0.005–0.999) using:
   - 4 cross-encoders: e5-small; e5-base rounds 1 and 2; e5-large (560M, AWS);
   - group and decoy features.

   Two separate stacks:

   | Stack | Used for | Split | Config |
   |---|---|---|---|
   | v7e_usin_owner | US/India | leak-free, owner-fold | `S2_OWNER_SPLIT=1` |
   | v6_pp | France | original | e5-small + e5-base round 1 only |
5. **Meta-model v3** (US/India). It re-scores the stage-2 probability using:
   - raw-text generator fingerprints;
   - candidate competition;
   - S2/S3 source;
   - generator-operation flags;
   - S1 name/address ambiguity counts.

   Fitted on folds 80–89, gated on 90–99: +0.000358 ± 0.000055.
6. **Decoder.** Per-entity expected F0.5, which can return an empty list. France uses odds ×0.5 (×1.4 in fr14).
7. **Generator rules**, in `solution/generator_rules/`:
   - **Acronym rule (`acr.py`, `acr2.py`).**
     - A 2–4 capital-letter target name equals the initials of an S1 name at the same address, with a house-number tie-break.
     - Train: 99.6% of such records are true copies, and the initials match in 100% of cases.
     - The rule is 99.98% precise, and 99.87% on records the model left unlinked.
     - Public gain: +0.000677.
   - **Indian-script rescue (`addr_block.py`, `rescue.py`).** Native-script names with truncated addresses (`No 559, Bangalore, KA`) are blocked by house number + city, judged by e5-small, then decided by a CatBoost rescue model. Validation +0.000431 (14 SE).
8. **Build (`build_v8.py`).** Adds rule links without creating duplicates, respects the per-S1 caps (5 in S2, 6 in S3), and extends `candidate_pairs.tsv` with the new blocking pairs.

Reproduce: `solution/pipeline/PACKAGE_README.md` (sections 1–5), then section 6 of `solution/pipeline/README` / `COMPETITION_DOCUMENTATION.md`.

## 4. What the data is (see DATASET_STATISTICS.md)

- **Train (US, India):**
  - 2.2M S1 and 10.3M S2/S3 records.
  - 7.64M true links: 3.46 per S1, 5.58% singletons, at most 5 copies in S2 and 6 in S3.
  - 26% of S2/S3 records are decoys.
- **Test:**
  - 1.73M S1 (France is 15% and has no labels anywhere).
  - 5.5–5.8 records per S1, which means **about 40% decoys**. That estimate is independent of our model: blank-address and domain markers give 41–43%.
- **The data is synthetic.**
  - Records with no detectable generator operation are linked correctly 99.7% of the time. All remaining loss sits in specific generator operations.
  - Copy-only operations have ~97–99.7% true-copy rates: blank address, made-up syllable names, acronyms, domains, aliases (`dba`/`t/a`, almost all in S3).
  - The decoy recipe is: same name, legal form changed, nearby house number (2.2% true).
- **France:**
  - 10× the acronym rate.
  - Twice the rate of S1 companies sharing an address (11.1%).
  - None of the US/India address noise (`##`, `null`, CITY/CDP, id suffixes).
- **S2 vs S3 use different generator settings:**
  - UPPERCASE names: 20% in S2 vs 3–6% in S3.
  - Indian-script names: 24% vs 13%.
  - Aliases: S3 only.

## 5. Where the remaining loss is (leak-free validation, current pipeline)

Total loss 0.01123:

| Class | Misses | False links | Fix-all gain |
|---|---:|---:|---:|
| Owner has same-core-name namesakes | 18,517 | 478 | 0.0079 + 0.0006 |
| Blank address | 15,601 | 378 | 0.0064 + 0.0004 |
| Blocking miss | 11,668 | — | 0.0051 |
| Right owner ranked first but rejected | 5,263 | — | 0.0025 |
| Wrong owner ranked first | 6,487 | — | 0.0027 |
| False link to decoy / other entity | — | 662 / 407 | 0.0008 / 0.0004 |

**Blank-address namesakes are unresolvable from the data:**
- Row order, IDs, copy-count balance and source hints were all tested; the best tie-break equals chance (0.188 vs 0.193).
- An oracle on true copy counts reaches only 0.279.
- A learned transformation ranker picks the owner in 50.8% of held-out groups (random 26.4%). But the groups it's confident on are ones we already link.

## 6. Tested and rejected (measured)

**US/India (validation)**

| Idea | Result |
|---|---|
| Seed ensembles | +0.000008 |
| Deeper retrieval (top 30) | owner uniquely best in 2 of 1,625 recovered links |
| Multi-anchor, per-country knobs, deeper stage 2 | ≈0 |
| Density-matched training | −0.000009 |
| Mutual nearest neighbours | −0.0004 |
| Graph closure | −0.0011 |
| Structural pruning | negative |
| Qwen2.5-7B QLoRA judge | AUC 0.69 vs 0.86; effect 0 |
| Query-relative competition features (meta4) | −0.000035 |
| Decoder re-tune on meta probabilities | +0.000058 ± 0.000054 (not significant) |
| Same-address kill-switch | −0.0117 (destroys 256k true links) |
| Rare-token anchoring | 2–3% precise |
| Made-up-name rescue | +0.00002 |
| Blank-address typo rescue | +0.000002 |
| Truncated-address rescue (Latin names) | +0.000024 |
| Domain-name rule | 63% precise on unlinked records |
| Upward house-number offset filter (Ekdant idea) | −0.0002 to −0.0040; our stage 2 already rejects those decoys |

**France**

| Idea | Result |
|---|---|
| Adding French links by belief | lost every time: v7a, loose18, fr_judge |
| v7e-family French links over v6 | labelled US/India says 87–89% right, but France-specific pseudo-label risk (fr_judge lost) |
| v9 French digit-drop rule | retracted: 98.8% true over all pairs, but **26% on the unlinked residue** (selection effect) |
| French postcodes | only 0.4–0.5% of French records contain one |

## 7. Synthetic France (`solution/synthetic_france/`)

- **What it is.** `gen_fr_v3.py` builds a labelled France-like test set from the real French S1, with copies and decoys that follow the measured generator statistics.
- **Fidelity.**
  - Checked with `synth_compare.py`: operation mix, decoy recipe, density and copy distribution all match.
  - Known gaps: synthetic copies are harder (the model misses 14% of them vs about 2–3% implied on real France), and filler words are too decoy-heavy.
  - The teammate's v1 generator had bigger issues: half its decoys were orphan copies, the decoy recipe was missing, and S3 casing was wrong.
- **Pipeline run.**
  - `run_A.sh`, `run_B*.sh`, `run_C.sh` run the whole pipeline on it in a mirrored folder (`work/syn_run`).
  - Blocking recall on synthetic France: 98.70% (blank-address 79.55%), matching the train profile.
- **Result.** French odds ×0.5 → ×1.0 → ×1.4 → ×2.0 scores 0.9231 → 0.9281 → 0.9299 → 0.9310, and it flattens after that. This motivated fr14.
- **Bias.** Because synthetic copies are harder, the set favours looser settings. So we stopped at ×1.4 and kept the default empty-list weight. With the empty weight lowered (frX), France drops to 5.47% empty lists vs a ~5.8% norm, a risk of false links on singletons.

## 8. Lessons

1. **Measure the generator, not the model.** The big late gains (acronyms, Indian-script rescue) came from finding generator operations where the model failed, not from more model capacity.
2. **Gate every rule on the records the model currently gets wrong.** A rule's precision over all pairs is meaningless when the model has already taken the easy ones (v9: 98.8% overall vs 26% on the residue).
3. **Leak-free splits matter.** Splitting stage 2 by the predicted anchor's fold inflated validation by 0.00034, and hid the meta-model gain (+0.00004 leaky vs +0.00028 clean).
4. **France:**
   - Only train-validated generator rules, checked for agreement against existing French links, transferred.
   - Pseudo-labels helped once (round 1) and hurt once (round 2, hard labels).
   - Aggregate link counts are a guardrail, not proof of correctness.
5. **Decoy density shift (26% → 40%) is real,** but it is mostly absorbed by recalibration (the meta-model). Stricter decoding on top of it added nothing.

## 9. Not in the repo (too large for GitHub; regenerate with the pipeline)

- Organiser data: `student_resource/dataset/`.
- Cross-encoder weights: e5-small 465 MB, e5-base 1.1 GB, e5-large 2.2 GB. Retrain with `solution/pipeline/src/ce_gpu.py`; the Kaggle/SageMaker scripts are in `src/cloud/`.
- Embeddings: 17 GB.
- Candidate pairs: 117M, 1.5 GB (`export_candidates.py`).
- Stage-1 training caches.
- Intermediate `.npz` arrays: about 17 GB in the analysis folder.
- All other submission versions (v2…v8c): rebuild with the run scripts.

## 10. Security notes

- No keys or tokens are in this repo; a secret scan was run before the push.
- The Kaggle token was exposed in chat on 26 Sep. **Regenerate it.**
- AWS jobs read the token from Secrets Manager (`kaggle-token`), never from code.
- Two teammate Hugging Face datasets were public on 27 Sep (`akshatbakshi/...`, `vaibhav343/...`, the latter with team predictions). Make them private.
