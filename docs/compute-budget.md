# Compute budget and execution gates

Stated total: **$670** across **eight accounts**, plus free compute. Planning working allocation: **$532**. Protected reserve: **$138**. Credits are not assumed transferable or eligible for every product. Credit eligibility, expiry, GPU quotas, competition deadline, and actual notebook hosting remain to be confirmed in the operations issue.

| Account alias | Balance stated | Working | Reserve | Intended work |
|---|---:|---:|---:|---|
| aws-1 | $100 | $80 | $20 | Data preparation, indexes, feature processing |
| aws-2 | $100 | $80 | $20 | Embedding/inference shards |
| aws-3 | $100 | $80 | $20 | Embedding/inference shards |
| aws-4 | $100 | $60 | $40 | Final inference, validation, packaging |
| hf-main | $180 | $160 | $20 | Main verifier training and evaluation |
| hf-pilot-1 | $30 | $24 | $6 | Pilot experiments or inference shards |
| hf-pilot-2 | $30 | $24 | $6 | Pilot experiments or inference shards |
| hf-pilot-3 | $30 | $24 | $6 | Pilot experiments or inference shards |
| **Total** | **$670** | **$532** | **$138** | |

The HF main working allocation starts as $120 training + $40 evaluation/inference. Any retriever-training spend must be explicitly reallocated from this pool or the pilot pools; it is not extra money. Optional experiments may be dropped to preserve final inference.

## Rates checked during planning

Hugging Face Jobs listed L4 24GB at $0.80/hour, L40S 48GB at $1.80/hour, and A100 80GB at $2.50/hour. Rates, availability, and credit eligibility must be rechecked immediately before launch: [official Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing). $120 at $2.50/hour represents 48 billable hardware hours, not a prediction of training duration.

AWS depends on instance, region, quota, capacity, storage, and transfer: [official EC2 pricing](https://aws.amazon.com/ec2/pricing/on-demand/). Do not hard-code an unverified AWS hourly rate.

Kaggle quota is whatever the authorized account shows; published documentation describes roughly 30 weekly GPU hours with variation: [GPU usage documentation](https://www.kaggle.com/docs/efficient-gpu-usage). Jupyter itself is an interface; identify its hosting provider and resources. Do not rely on free compute for an uncheckpointed final deadline-critical job.

## Before a paid run

1. Identify account alias, credit coverage, current balance, expiry, GPU quota, and storage location.
2. Benchmark representative sequence lengths/candidate distributions, not only easy tiny examples.
3. Estimate runtime = item count / measured throughput; multiply by actual rate and add setup, retries, storage, transfer, and persistence.
4. Log run config, maximum duration, estimated cost, stop conditions, and artifact/checkpoint paths.
5. Set timeouts and cleanup; budget alerts are notifications and do not guarantee shutdown.
6. Reconcile actual spend, remaining working allocation, and protected reserve after completion.

Independent shards are preferred across accounts/providers. Each job uses stable IDs and a single frozen checkpoint/config. Keep large intermediates near compute, persist results outside ephemeral disks, and never commit credentials. No cloud compute was started by repository setup.
