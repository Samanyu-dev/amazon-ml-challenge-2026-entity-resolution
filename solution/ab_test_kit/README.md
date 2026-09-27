# Is my submission better than the team's best? (hold-out A/B kit)

The leaderboard is the only place a **test** file can be scored, and we get 5 uploads a day.
This kit tells you, **before uploading**, whether your pipeline beats our current best. It
scores both on the same 220,938 labelled train entities that our pipeline never trained on.
The metric is exactly the leaderboard's: macro F0.5 per Source-1 entity, with singletons
included.

## Files

| File | What |
|---|---|
| `holdout_s1_ids.txt` | the 220,938 hold-out Source-1 `entity_id`s (US + India) |
| `our_best_holdout_predictions.tsv` | our current best pipeline's predictions for them (F0.5 0.98865) |
| `score_holdout.py` | the scorer (Python 3.8+, standard library only) |

## Steps

1. **Retrain without the hold-out.** Remove every hold-out entity from your training data, together with every Source 2/3 record that `train_ground_truth.tsv` links to it. That covers the model, the thresholds, and any tuning or dictionary learning. If your model has seen them, the score is meaningless.
2. **Predict for the hold-out entities.** Search against **all** of `train_source2.tsv` + `train_source3.tsv`, so every record is a possible match or distractor, as on the leaderboard. Write a normal `matching_results.tsv`: one row per hold-out ID, `source1_entity_id<TAB>id,id,...`, with an empty list allowed.
3. **Score it.** Add the two `--source` files and `--report` to get the full diagnosis:

```bash
python3 score_holdout.py --pred my_holdout_predictions.tsv \
    --truth  dataset/train/train_ground_truth.tsv --source1 dataset/train/train_source1.tsv \
    --source2 dataset/train/train_source2.tsv --source3 dataset/train/train_source3.tsv \
    --report report.json --examples 10
```

The run takes about 15 s and about 2 GB of RAM.

## What it prints (and what to do with it)

| Section | Meaning | Where it points |
|---|---|---|
| **Headline** | macro F0.5, precision, recall, wrongly linked singletons, per-country F0.5 | — |
| **vs our best + VERDICT** | paired per-entity comparison on the same entities | upload or not |
| **ERROR BUDGET** | F0.5 gained if each error type were fixed, sorted by size | the biggest line is your next job |
| **BY NUMBER OF TRUE LINKS** | F0.5 and gain vs our best for singletons, 1, 2-3, 4-6 and 7+ true links | e.g. low on singletons means an empty list is not predicted often enough |
| **BY RECORD TYPE** | recall and false links for records with or without an address, and with Latin or non-Latin names | blocking or transliteration gaps |
| **EXAMPLES** | real record pairs for each error type (S1 vs record, plus the true owner when the link went to the wrong entity) | read them before changing anything |

**Error types:**

- `missed_links_not_predicted_at_all`: a true record was never linked. Improve blocking recall, or lower your threshold.
- `missed_links_given_to_wrong_entity`: the record went to a lookalike entity. This is a ranking problem between namesakes.
- `false_links_to_decoys`: you linked a record that belongs to nobody (a distractor).
- `false_links_to_other_entities_records`: you linked a record that belongs to another entity.
- `false_links_on_singletons`: the entity has no true match, and any link scores it 0.

**For coding agents:** `report.json` has the same numbers as keys: `macro_f05`, `by_country`, `vs_best.{mean_gain,paired_se,verdict}`, `error_budget.<type>.{f05_gain_if_fixed,links}`, `by_true_link_count`, `by_record_type`, `examples.<type>[]`.

Our best, for reference: the largest loss is `missed_links_not_predicted_at_all` (+0.0098). Most of it is records with no address whose exact name is shared by several Source-1 entities (recall 0.54 on `no_address+latin_name`), which is mostly unresolvable. The rest of the loss is spread across the false-link types (about +0.0015 in total).

## Reading the verdict

- **BETTER** means a paired gain of more than 2 standard errors on the same entities. Send
  the team your test file.
- **no clear difference** means it's within noise, so an upload won't tell us anything.
- **WORSE** means don't upload.

**France is not covered.** It has no labels anywhere. Our best pipeline takes France from
separate models that the leaderboard has already confirmed. A file that is better on
US/India can still lose points on France, so tell us what you changed for France.

For reference, our current best is **0.98865** (US 0.98987, India 0.98681), and it scored
0.984931 on the public leaderboard.
