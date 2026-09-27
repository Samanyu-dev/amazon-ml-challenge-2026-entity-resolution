# Amazon ML Challenge 2026: Business Entity Resolution
## Approach, version history, scores and everything tested

Status as of 27 Sep 2026, ~09:15 IST.

- **Best public score:** 0.984931 (v7d).
- **Honest (leak-free) US/India validation:** 0.98841.
- **Leaderboard at the same time:** #1 ≈ 0.99148, top 10 ≈ 0.99028, top 50 ≈ 0.98831.

---

## 1. Problem

Source 1 (S1) is a deduplicated list of reference businesses. Sources 2 and 3 (S2/S3, "targets") hold noisy copies of those businesses plus unrelated decoys. For every S1 entity we list the S2/S3 records that are the same business.

The noise includes:

- typos and digit swaps;
- Indian-script names and back-transliteration (`praaivett limittedd`);
- legal-form changes;
- reordered or missing address parts;
- corrupted house numbers;
- empty addresses.

Train covers US and India. **Test adds France (15% of S1), which has no labels anywhere.**

## 2. Data

Only the organiser files are used: no external data, APIs or gazetteers. All files are loaded into DuckDB with SHA-256 fingerprints.

| File | Rows |
|---|---:|
| train S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 |
| train ground truth | 7,638,365 true links (3.46 per entity; 5.58% of entities have none) |
| test S1 / S2 / S3 | 1,732,544 (US 663,106 · India 809,986 · France 259,452) / 4,887,273 / 5,082,316 |

Label facts:

- A record belongs to at most one S1 entity.
- 100% of true links stay within one country.
- An entity has at most 5 S2 matches and 6 S3 matches.
- Test has 5.75 S2/S3 records per S1 entity, against 4.67 in train.

## 3. Metric and rubric

**Metric:** macro F0.5 per S1 entity, averaged over all entities.

- A no-match entity scores 1 if we predict nothing and 0 otherwise.
- Precision is weighted twice as heavily as recall.
- Our scorer reproduces the leaderboard metric exactly.

**Folds:** a stable hash splits entities into folds 0–99.

| Folds | Use |
|---|---|
| 0–39 | Cross-encoder (CE) training |
| 0–74 | Stage-1 training (40–74 also used for stage 2) |
| 75–79 | Early stopping |
| 80–84 | Calibration |
| 85–89 | Tuning |
| 90–99 | Held-out report (220,938 entities) |

**Rules for shipping a change:**

1. The paired per-entity gain is more than 2 standard errors.
2. The tuning folds show the same direction.
3. Precision holds up.
4. The official validator passes with `--check-ids` and candidate subset checks, with 0 duplicate targets.
5. There is no leakage.

**France rule:** decisions about France can only be tested on the leaderboard, one change per upload.

**Upload freeze (from 27 Sep morning):** no upload until an offline result credibly supports about 0.9885 or more.

## 4. Models

All models are MIT or Apache-licensed and 8B parameters or fewer.

| Component | Model | Role |
|---|---|---|
| Dense retrieval | multilingual-e5-small + FAISS, per country | Candidates for Indian-script and transliterated names; cosine used as a feature |
| CE 1 | e5-small, fine-tuned on 2.1M labelled pairs | Re-reads uncertain pairs |
| CE 2 (base round 1) | e5-base, + 600k French pseudo-labels | First French signal |
| CE 3 (base round 2) | e5-base, + 455k *hard* French pseudo-labels | Helped US/India; hurt France |
| CE 4 | e5-large (560M), 3.16M pairs, AWS A10G | Strongest reader |
| Stage 1 | CatBoost, 63 features + cosine, calibrated | Scores every candidate pair |
| Stage 2 | CatBoost over uncertain targets (top-4 candidates, all CE scores, group/decoy features) | Re-decides uncertain targets |
| Decoder | Per-entity expected F0.5; can output an empty list | Final decisions |
| **Running now** | Qwen2.5-7B-Instruct (Apache-2.0), QLoRA, 24k labelled US/India pairs, no French pseudo-labels | An independent matcher with built-in knowledge of French |

## 5. Pipeline

1. **Normalisation (v3):** legal forms separated out; French and Indian forms; `M/s`; dotted acronyms; French address words.
2. **Blocking:** C++ inverted index (lexical top 8) plus ANN top 4. Recall on true links is 98.48%. There are about 68 candidates per S1 entity; the candidate file has 117M pairs.
3. **Stage 1:** CatBoost, then calibration.
4. **Stage 2:** re-decides records whose stage-1 probability is between 0.005 and 0.999.
5. **Decoder:** as above.
6. **Final assembly:** US/India rows and French rows come from separate stage-2 stacks.

## 6. Version history

"Validation" means US/India, folds 90–99. Before 27 Sep these numbers used the leaky stage-2 split (see §7). "Implied France" is the France score back-calculated from public and validation.

| Version | What changed | Validation | Public |
|---|---|---:|---:|
| v2 + decoder | Lexical blocking + CatBoost + expected-F decoder | 0.97389 | 0.962428 |
| v3 | v3 normalisation, ANN merge, 63 features | 0.98092 | 0.972 |
| v4 | + e5-small CE, stage 2 | 0.98697 | 0.981393 |
| v6 | + e5-base round 1 (French pseudo-labels) + group features | 0.98797 | 0.984462 |
| v7a | + decoy features + e5-base round 2 | 0.98828 | 0.984178 (France hurt) |
| v7d | + e5-large for US/India; France from v6 with its weakest links trimmed (odds × 0.5) | 0.98865 | **0.984931 (best)** |
| v7e_all_loose18 | v7e; looser linking in all countries (odds × 1.8) | −0.00024 | 0.984709 |
| v7e_fr_judge | v7e + 12,257 French links backed by stage 1 | same | 0.984725 |
| v7e / v7g | Built, not uploaded. v7g = v7e US/India + v6 France with odds × 0.35 | 0.98875 (leaky) | — |

**What the leaderboard taught us about France:**

- Every change that **removed** French links helped or held:
  - v5 → v6: about +0.015 on France;
  - the v7d trim.
- Every change that **added** French links lost points:
  - v7a;
  - loose18;
  - fr_judge.
- Stage 1's endorsement (the "independent judge") did not predict French correctness. It backed fr_judge 7.4 : 1, and fr_judge lost.

## 7. Key findings, 26–27 Sep

1. **Coverage bug (fixed).** The large CEs scored only records with confidence 0.02–0.995, while stage 2 used 0.005–0.999. So about 1.8M uncertain test records got only e5-small. Scoring the rest added about +0.0001 on US/India (e5-large) and about +0.0001 on France (e5-base round 1).
2. **Stage-2 leakage (found by an independent review).** The stage-2 fit was split by the predicted anchor's fold, not the true owner's fold.
   - The fix lowers validation by **−0.00034 (7 SE)**, from 0.98875 to **0.98841**.
   - Key gains still hold without the leak: v6 → v7e +0.00074 (12 SE); France coverage +0.00013.
   - The submitted files are reproducible with the original split. The leak-free split is an evaluation flag (`S2_OWNER_SPLIT=1`).
3. **Build bug (fixed).** The fr_judge build left 4 records linked to two companies. It now has 0 duplicates.
4. **Error budget, leak-free validation** (link counts, not shares of F0.5 loss):

   | Missed-link type | Links | Share |
   |---|---:|---:|
   | Absent from candidates | 11,668 | 51.9% |
   | Present, but another entity wins | 6,487 | 28.8% |
   | Correct entity first, but rejected | 4,338 | 19.3% |

   There are also 1,801 false links.
5. **Namesakes.** Many blocking misses are address-less records whose exact name is shared by several S1 entities (median 9).
   - Top-30 retrieval recovers 1,625 true links, but the owner is uniquely the best candidate in only 2 of them.
   - Collective signals point more often to the wrong namesake:
     - same-spelling siblings are wrong 4 times as often as right;
     - rare shared typos: 27 right vs 572 wrong;
     - learned sibling similarity: the owner is uniquely closest only 12.5% of the time, while another namesake is closer 44.8% of the time.
6. **Public vs validation gap is about 0.0035.**
   - The orphan / density simulation (19% of S1 removed) costs only −0.0003.
   - Density-matched training gives −0.000009.
   - Test US/India has *fewer* uncertain final decisions (1.08%) than validation (1.25%).
   - France has **2.7× more** uncertain final decisions (3.3%).
   - **Caveat (review, 27 Sep):** with US/India public ≈ 0.9884, even a perfect France would give only about 0.9902. So teams at 0.9915 must also be stronger on US/India, or the public set is not a 85/15 mix at validation-level US/India accuracy.
7. **France audits (none found a large, fixable pool).**
   - The ANN path is complete: all French records were queried, and 4.0M dense-only pairs reached stage 1.
   - No French S1 entity has zero or one candidate. The median is 51.
   - Postal-code conflicts: 2 of 213,814 uncertain pairs.
   - Legal-suffix conflicts survive in 1 accepted pair.
   - Shared-address pruning: 99.2% of those links are true on labelled data.
   - Naive S2↔S3 graph closure on validation: adds 1,392 links, of which 1,350 are false. F0.5 drops from 0.98841 to 0.98730.
8. **Accent flag collision.** `target_name_nonascii` means "Indian script" in train, but in France it fires on accented names (24%). Its stage-1 importance is moderate (rank 33 of 65). Not fixed; fixing it would mean retraining stage 1.

## 8. Tested and rejected, with measured results

| Idea | Result |
|---|---|
| Seed ensembles | +0.000008 |
| Deeper lexical/ANN retrieval (top 30, French top 24) | Owner uniquely best in 2 of 1,625 recovered links |
| Multi-anchor assignment | 0 gain |
| Learned Indic token dictionary | Recovers 41 of 3,420 misses |
| Per-country decoder knobs | 0 |
| Deeper stage 2 | +0.00005 (1.5 SE) |
| Stage 2 trained on more data | −0.000015 |
| S1 address-ambiguity feature | +0.000027 |
| No-owner gate | ≈ 0 (and not out-of-fold) |
| Density-matched training | −0.000009 |
| Joint US/India/France stage 1 | −2 SE on US/India |
| Tier-A consensus pseudo-labels | 99.933% precision; **0** labels in the uncertain range |
| Structural pruning (name, number, legal conflicts) | −0.002 to −0.078 |
| Sibling and typo clustering | Wrong more often than right |
| Graph closure | −0.0011 |
| GRAPH-01 (sibling disagreement) | +0.00008; kept as a small idea, not shipped |

**Public repositories and notebooks reviewed:** about 95 GitHub repos and about 25 Kaggle notebooks (including sohamcodes54 and ani1611). The best reported validation among them is about 0.978. No top-50 team has published its approach.

## 9. Compute and cost

| Where | What |
|---|---|
| Mac (M5, 24 GB) | All CPU stages |
| Kaggle (T4 / 2×T4) | e5-small and e5-base CEs; e5-base coverage scoring |
| AWS SageMaker (ml.g5.2xlarge, A10G) | e5-large training + scoring (≈ $13); rank-3/4 scoring (≈ $6); Qwen2.5-7B QLoRA (running, ≈ $8–10) |

## 10. Current plan

1. **Qwen2.5-7B.** Training is done (loss 1.45 → 0.056). Scoring: 214k uncertain French pairs, then 53k US/India hold-out pairs, finishing about 10:00 IST.
2. **Offline gates (no upload before these):**
   - **Gate 1, US/India:** fusion with a recalibration-only control. Pass requires validation ≥ 0.9905 with precision ≥ 0.993.
   - **Gate 2, France:** how decisive the model is on uncertain pairs, and how many French decisions it changes.
   - **Gate 3:** matches must be a subset of candidates, with 0 duplicate targets.
3. **A second, larger Qwen run** only if the gates show real signal.
4. **Fallback:** keep the 0.984931 best.
5. **Final package:** code, README, requirements, filled documentation, and `candidate_pairs.tsv` (117M pairs). The package is built and reproduces the submitted files byte for byte.

---

## 11. Addendum, 27 Sep afternoon and evening (final)

- **v7d_acr, 0.985608.** Adds the French acronym rule (a 2–4 capital-letter name is the initials of the owner at the owner's address). Precision on train: 99.98%.
- **v8b, 0.986382 (best).** Four changes over v7d_acr:
  - acronym rule v2;
  - Indian-script address-number rescue (+0.000431 validation);
  - leak-free US/India stage 2;
  - meta-model v3 (+0.000358 validation).
- **v8b_fr14 (final upload, score pending).** France odds ×1.4, chosen from the synthetic-France curve.
- **Retracted:** v9 (French digit-drop rule). It is 26% precise on the unlinked residue.

The full handoff (error table, all tested ideas, synthetic France, lessons) is in `HANDOFF.md`.
