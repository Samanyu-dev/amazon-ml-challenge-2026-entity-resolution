# Independent review, 26 Sep night

## (d) Public-vs-validation gap: orphan hypothesis tested, and it is weak

The gap is about 0.004: validation 0.98875 vs public 0.984931.

**Experiment:** `orphan_sim.py 0.19 0`

1. Remove a random 19% of train S1 anchors (418,886).
2. Their 1.96M S2/S3 copies become decoys.
3. Each copy is re-pointed to its best surviving candidate. It gets the stage-2 pair probability if stage 2 scored that pair; otherwise the stage-1 runner-up score through an isotonic raw-to-calibrated map.
4. Decode, then score the surviving fold 90-99 anchors.

| World | F0.5 | vs base | P | R |
|---|---:|---:|---:|---:|
| base (no orphans) | 0.988702 | — | 0.99838 | 0.96959 |
| orphans, odds ×1.0 | 0.988375 | −0.000327 (SE 2.7e-05) | 0.99790 | 0.96968 |
| orphans, odds ×0.7 (best) | 0.988481 | −0.000221 (SE 3.9e-05) | 0.99839 | 0.96880 |
| orphans, odds ×0.55 | 0.988414 | −0.000288 | 0.99862 | 0.96817 |

- Only 0.4% of re-pointed orphan copies reach q ≥ 0.5 on their new best candidate. A twin with a different address, or an address-less namesake among k, is not trusted.
- Orphans cost about −0.0003, and stricter decoding recovers about +0.0001.
- **Caveat:** the re-pointed probabilities were computed while the true owner was a competitor (`other_p` and `p_gap` features), so this understates somewhat. It cannot plausibly reach 0.004.

**Implication:** with US/India public ≈ validation, the ~0.004 gap is France:
0.15 × (0.988 − F_France) ≈ 0.004, so F_France ≈ 0.96–0.965.
Top teams at 0.9907 most likely have France ≈ 0.985–0.99, plus slightly better US/India.
The strict/loose probes planned for 27 Sep are expected to move the score by < 0.0002.

## (a) Confirmed issues

1. **CE band mismatch** (found 26 Sep, fixed in v7e): big cross-encoders covered stage-1 q 0.02–0.995 while stage 2 used 0.005–0.999. The fix added +0.0001 US/India and +0.0001 on the France config.
2. **Mid-name legal forms not stripped:** `normalization_v3.py` handles the tail only. For example, "COMMUNALE (FRANCE) SARL DÉVELOPPEMENT" keeps its legal form. The cross-encoders read raw text, so the measured impact is low.
3. **"N°" normalises to "ndeg"** (junk token).

## (b) Measured, not shipped

| Change | Paired gain |
|---|---:|
| S1-address-ambiguity features | +0.000027 (SE 2.9e-05) |
| Stage-2 depth 8 | +0.000052 (SE 3.5e-05) |
| Stage-2 + calibration folds | −0.000015 |
| Per-country decoder | 0.000000 |
| Decoder odds ×0.55 on plain validation | −0.000123 |
| Decoder odds ×1.8 on plain validation | −0.000243 |

## (c) Ranked ideas for France (the real lever, ~+0.003 public if France reaches ~0.985)

1. **Leaderboard-steered France probes, one change per slot.** US/India rows stay fixed so the delta is pure France. Candidates:
   - (i) France from v6 without the round-1 French pseudo-label CE (is self-training helping?);
   - (ii) France decoder odds ×1.5 (looser: v7d's ×0.5 trim was neutral to slightly negative, so check the other side);
   - (iii) France using the e5-large stage-2 stack (US/India config) instead of v6.
2. **Understand what France errors look like, label-free.** 11.7% of French S1 share an exact address, 2.4× US/India. Brand-name records at shared addresses are unresolvable. The decoder should treat them as ambiguous, which it already mostly does.
3. **Synthetic French validation:** transplant French name/address style onto US labelled pairs. Rejected earlier because the noise differs. Only revisit with time.
