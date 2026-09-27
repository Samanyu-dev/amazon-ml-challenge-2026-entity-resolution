# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The pipeline has five stages:

1. **Blocking:** a C++ inverted index plus multilingual dense retrieval keeps 98.5% of true links at about 68 candidates per Source-1 entity.
2. **Stage 1:** a CatBoost model scores every candidate pair on 63 hand-built features.
3. **Stage 2:** an ensemble of four fine-tuned multilingual cross-encoders (e5 small/base/large, all MIT) re-decides only the uncertain targets.
4. **Group evidence:** stage 2 also uses "decoy/graph" features that describe the rest of each record's candidate neighbourhood.
5. **Decoder:** a per-entity decoder picks the number of links, including none, that maximises the expected F0.5.

France has no labels, so it is handled by self-training on high-precision pseudo-labels. Every rule was measured at ≥ 99.97% correct on labelled data first. Macro F0.5 on held-out US/India entities is **0.9887**.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the training data:

- **Structure.**
  - Every true link stays inside one country.
  - A Source 2/3 record belongs to at most one Source-1 entity.
  - There are 3.46 true links per entity, and 5.6% of entities have none (singletons).
  - About 26% of Source 2/3 records are decoys in train; test has about 40%.
- **Name noise.**
  - Typos and digit-for-letter swaps.
  - Dropped, added or swapped legal forms (`Pvt Ltd` ↔ `Private Limited`, `SARL`, `SAS`, `EURL`).
  - Back-transliteration (`praaivett limittedd`), and Indian names written in 9 native scripts.
  - Dotted acronyms (`L.L.C.`), full acronyms (`MSSM`), and generic-word swaps (`X Club` → `X Services`).
  - Made-up brand names that share only the address.
- **Address noise.**
  - Reordered components and abbreviations (`R.`/`rue`, `Av`, `Bd`, `St`).
  - Corrupted or zero-padded house numbers (`046`) and missing units.
  - City and district swaps, and empty addresses (60% of the links blocking misses).
- **Namesakes.** Many Source-1 entities share an exact name (a mean group of 4.7 in the US and 7.9 in India). An address-less record with such a name cannot be resolved by anyone. This caps achievable recall.
- **France** (15% of test S1) appears only in test. Its true matches are cleaner (exact name 74% vs 51%). Its uncertainty sits in "same address, different name" cases.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier (gradient boosting + cross-encoder re-ranking) + expected-F0.5 decoding.

**Core Innovation:**
1. Uncertainty-targeted cross-encoder re-ranking: only about 20% of targets go to the transformers.
2. Group-level decoy features.
3. An expected-F0.5 decoder that may return an empty list.
4. Label-free adaptation to France with pseudo-labels whose accuracy was measured on labelled data before use.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - A per-country C++ inverted index (country is a hard partition). It indexes:
    - name tokens and character trigrams;
    - address tokens and 4-grams;
    - phonetic consonant skeletons;
    - house-number tokens.
  - Lexical top-8 per target, ranked by IDF-weighted overlap.
  - Plus top-4 by dense retrieval: `multilingual-e5-small` sentence embeddings (MIT), a FAISS IVF-PQ index per country. This recovers native-script and transliterated names.
  - Both views run on v3-normalised text (legal forms stripped into a separate field, transliteration folded, FR/IN address forms canonicalised).
- **Candidate pairs generated:** 117,097,579 on test, 67.6 per Source-1 entity (about 11.7 per target record).
- **How you ensured true matches were not lost:**
  - Recall is measured on labelled train at every change: 98.48% of true links are retrieved.
  - Dense retrieval was added specifically for script and transliteration misses.
  - The misses were analysed. They are mostly address-less records whose exact name is shared by several Source-1 entities (median 9). Deeper retrieval (top-30) does recover some, but then the true entity is uniquely best-scoring in only 2 of 1,625 recovered links, so we kept top-8 + top-4.

---

## 4. Matching Model

**Features used** (63 pair features, computed in C++):
- **Name:** normalised and core-name edit distance and sorted-token edit; token Jaccard/min/max; trigram Dice; IDF overlap; fuzzy token means; first-word edit; exact / compact-exact flags; legal-form presence and conflict; rare unmatched name-token IDF; number-word conflicts; country-relative name frequency.
- **Address:** edit distance, trigram Dice, token Jaccard/IDF; house-number Jaccard; first-number equal/conflict; target-only and anchor-only numbers; smallest distance between unmatched numbers; address-missing flags.
- **Other:** retrieval rank and per-query normalised retrieval scores; dense-route flag and embedding cosine; count of near-identical candidates (ambiguity); name × address interactions.

**Stage 2** (targets with calibrated p between 0.005 and 0.999, top-4 candidates):
- Scores from four cross-encoders, each paired with the best *other* candidate's score and the gap to it.
- Stage-1 p and its rank and gap.
- Group evidence: how many confident links the entity already has (overall and from the same source), whether a sibling has the same name, and whether the target's address is empty.
- Decoy/graph evidence: twins in other and same sources, address crowding, anchor load, and confident twins on the same anchor.

**Model type:**
- **Stage 1:** CatBoost (2,389 trees, early stopping), then logistic calibration.
- **Cross-encoders:** `intfloat/multilingual-e5-small`, `-base` (two self-training rounds) and `-large` (560M), each fine-tuned as a pair classifier on `name | address | country` pairs.
- **Stage 2:** CatBoost over the cross-encoder scores plus the stage-2 features above.

**Threshold selection method:** there is no global threshold. For each Source-1 entity, candidates are sorted by calibrated probability. The decoder picks k (including k = 0) to maximise the expected F0.5, 1.25·Σq[:k] / (k + 0.25·(Σq + missing)). The empty list is scored as `empty_scale`·Π(1−q). Both knobs are tuned on anchor folds 85–89 only.

**France (no labels):** cross-encoder self-training on French pseudo-labels.
- Positives: stage-1 q ≥ 0.999 with margin ≥ 0.3, measured 99.996% correct on labelled train.
- Negatives: q ≤ 0.001 (99.99% correct) and runner-up candidates.
- Round 2 ("hard" French pairs) improved US/India but lowered the public score, so French rows use round-1 models only. The weakest French links are then removed (link odds × 0.5).

**Validation protocol:** stable hash folds 0–99 per Source-1 entity.

| Folds | Use |
|---|---|
| 0–39 | cross-encoder training |
| 0–74 | stage-1 training (40–74 also stage 2) |
| 75–79 | early stopping |
| 80–84 | calibration |
| 85–89 | decoder tuning |
| 90–99 | reporting |

A change ships only if its paired per-entity gain on folds 90–99 exceeds about 2 standard errors and the tuning folds agree.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.98865** on held-out folds 90–99 (US 0.98987, India 0.98681; 220,938 entities, singletons included). Precision 0.9983, recall 0.9694; only 91 of 12,590 singletons wrongly linked.

| Version | Change | Validation F0.5 | Public LB |
|---|---|---:|---:|
| v2 | lexical blocking + CatBoost + decoder | 0.9739 | 0.9624 |
| v3 | normalisation v3 + dense blocking + 63 features | 0.9809 | 0.9720 |
| v4 | + e5-small cross-encoder stage 2 | 0.9870 | 0.9814 |
| v6 | + e5-base (French pseudo-labels) + group features | 0.9880 | 0.9845 |
| v7d | + decoy/graph features + e5-base round 2 + e5-large; France from v6 with trimmed links | **0.98865** | final |

- **Common false positives (wrong merges):**
  - Sibling businesses at the same address with one generic word changed (`Ward Quality Photonics` vs `Quality Wolfe Photonics`).
  - Address-less records whose name matches a namesake.
  - Remaining false links (v7a stack): 943 to decoys and 429 to other entities, over 764,895 true links.
- **Common false negatives (missed matches)**, as shares of the remaining F0.5 loss on folds 90–99 (false links make up the other ~15%):
  - **44% blocking misses** (address-less namesakes).
  - 23% wrong entity ranked first among namesakes.
  - 21% true top candidate rejected because the evidence was too weak (brand-name records, heavy transliteration).

---

## 6. Conclusion

Blocking recall and precision-first decoding matter most. Cross-encoders targeted at uncertain records give the largest model gains (+0.006 over the feature model). The remaining error is dominated by records that are ambiguous by construction (address-less namesakes). For the unlabelled country, self-training helped only while the pseudo-labels were restricted to cases verified as near-certain on labelled data. Pushing it into harder cases lowered the leaderboard score even though validation rose.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains all source in `src/`, plus `README.md` (exact commands in order) and `requirements.txt`. The entry points, in run order, are:

1. `prepare.py`, `make_arrays.py`, `normalize_corpus.py`, `encode_corpus.py`, `ann.py`.
2. `run_v3.sh` (C++ blocking + stage 1).
3. `ce_export.py`, `ce_new_pairs.py`, `pseudo_labels.py`.
4. `ce_gpu.py`, run on a GPU (Kaggle notebooks and the SageMaker script are in `src/cloud/`).
5. `stack_group.py` (stage 2).
6. `assemble_final.py`, which writes `output/matching_results.tsv`.
7. `export_candidates.py`, which writes `output/candidate_pairs.tsv`.

The final assembly and the pseudo-label files were checked to regenerate byte-for-byte.

### B. Additional Results

- Blocking recall by record type: 99.34% for records with an address, 80.5% for address-less records, and 96.8% for native-script names.
- Each validated step, as paired gain on folds 90–99:

| Step | Gain | Standard errors |
|---|---:|---:|
| decoy/graph features | +0.00024 | 5.5 |
| e5-base round 2 | +0.00013 | 3.5 |
| e5-large | +0.00038 | 9 |

- Tested and rejected (measured, no gain):
  - seed ensembles (+0.000008);
  - deeper ANN or lexical retrieval;
  - a joint US/India/France stage 1 (−2 SE on US/India);
  - multi-entity assignment of split records;
  - a learned Indic token dictionary (41 of 3,420 misses recovered);
  - per-country decoder knobs;
  - a deeper stage-2 model (+0.00005, 1.5 SE).


---

## Final version (v8b, public 0.986382): generator-aware extensions

Records with no detectable generator operation are linked correctly 99.7% of the time, so the remaining error sits in specific
synthetic-noise operations. We measured every operation on labelled data (true-copy rate, links lost) and fixed the recoverable ones:

- **Acronym rule (France).** A target named with 2-4 capitals is the initials of its owner's name at the owner's address
  (train: 99.6% true copies, initials match 100%). Rule precision on train 99.98% (99.87% on records the model left unlinked);
  on France it agrees with existing links 99.4%. Public +0.000677.
- **Indian-script rescue.** Native-script names with truncated addresses ("No 559, Bangalore, KA") are blocked by house number + city
  and judged by the e5-small cross-encoder plus a CatBoost rescue model: validation +0.000431 (14 SE), precision unchanged.
- **Leak-free stage 2 + meta-model.** Stage 2 split by the true owner's fold; a meta-model adds raw-text generator fingerprints,
  candidate competition, S2/S3 source, generator-operation flags and S1 name/address ambiguity counts: validation +0.000358 (6.5 SE).
- **Tested and rejected (measured):** same-address kill-switch (-0.0117), rare-token anchoring (2-3% precise), made-up-name and
  domain rescues (ambiguous), namesake tie-breaks (no signal beyond chance), stricter decoding for test decoy density (no gain on
  the recalibrated model), number-conflict rescues (26% precise on the unlinked residue).
