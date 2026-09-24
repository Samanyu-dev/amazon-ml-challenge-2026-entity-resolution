# Team workflow

## Start assigned work

Accept your repository invitation, open the private delivery project, and select your assigned Ready issue. See [issue ownership](docs/ownership.md) for the allocation and pending-invitation handling. Read native blocking relationships and the issue's dataset scope before coding. The Readiness field is maintained manually when dependencies close; it is not an automatic scheduler.

Use a branch such as `issue-7-lexical-retrieval`, open a PR referencing the issue with `Closes #<number>`, and ask another team member to review correctness and evidence. Branch protection is not assumed configured. Avoid broad refactors in model experiments.

## Data and experiments

Keep raw datasets, credentials, weights, embeddings, and bulk/private reports outside Git. Use `DATA_ROOT` and a private artifact root. Commit code, small configurations, synthetic fixtures, and reviewed aggregate reports only. No business identity lookup, geocoding, or internet-derived augmentation.

Every run records code SHA, configuration, data/split/model checksums, seed, metrics, and actual cost. Label pilot/small-gallery results clearly. Do not change frozen splits to improve a score. User/provider account secrets must never appear in issue bodies, notebooks, logs, or commits.

## Close an issue

Link code and exact command, dataset partition and manifest, expected artifacts, acceptance-check evidence, and actual compute spend. State what was not run. Optional experiments may close with a supported no-go decision; P0 requirements need working evidence. Update project Status and Readiness and unblock dependent work deliberately.

## Paid work

Use account-specific budgets and verified eligibility. Benchmark before full execution, checkpoint, set timeouts, persist outputs, and clean up resources. The $138 reserve is protected; a documented team decision is needed to release it. Creating an issue or assigning a task does not itself launch compute.
