# Validation and leakage policy

## Exact score

For true set T and prediction S, per-reference F0.5 is `1.25 * |S intersect T| / (|S| + 0.25 * |T|)`. Both sets empty scores 1. Average over every selected Source 1 reference. Micro pair metrics are diagnostics only.

## Splits

Proposed starting design: 80% training, 10% calibration, 10% final validation by stable underlying business group, stratified/audited by country and match count. A target labeled to several anchors forces the connected labeled group to remain together. Positive variants, augmentations, learned aliases, and hard negatives respect group eligibility. Pilot model selection/early stopping uses development data carved from training.

Freeze split manifests before tuning. Calibration fits probability/threshold decisions; use separate calibration subpartitions or cross-fitting for stacking and thresholds. Final validation is an infrequently used locked evaluation, not a tuning loop. IDs and ordering never become predictors.

Validation retrieval uses the realistic full training target gallery. Unlabeled gallery access is different from supervised training eligibility: held-out group targets cannot be mined as training negatives. Record precisely which unsupervised corpus statistics are fitted where.

## Retrieval metrics

Report link recall, mean per-non-singleton recall, complete-entity recall, zero-hit non-singletons, candidate-size distribution, and oracle macro-F0.5. The oracle predicts only true matches present in the actual matcher-input candidate set. Never inject held-out positives to improve reported recall. If training adds a missed positive to teach a matcher, tag it as teacher-forced and exclude it from retrieval success.

## Shift and uncertainty

Train-US/evaluate-India and reverse with fresh country-restricted learned processing. These estimate transfer risks but do not measure France accuracy. France checks are unlabeled coverage, tokenization, candidate distribution, and score-distribution diagnostics. Never force France's match rate to equal a training country's rate.

Compare models on the same anchors using paired entity-level bootstrap intervals. Include singleton false merges, missing-address, script-change, common-name, source, and match-count slices. Gate optional complexity on reproducible end-to-end gain and cost.

## Refit

Retain calibration or use proper cross-fitting during full-data refit. Once the final holdout is consumed for fitting, an independent post-refit score no longer exists; disclose that clearly. Freeze thresholds/configurations before test inference. No unlabeled-test pseudo-label training is included in the initial plan.

## Definitions of done

An issue needs code/config links, exact commands, dataset partition, input/model/config hashes, output evidence, and actual compute spend. A planned snippet or successful format validation is not evidence that a trained model achieves a score. A small clean-room replay is not a full replay.
