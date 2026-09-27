# Final solution (team repo, post-competition)

The full handoff is in [`../docs/final/HANDOFF.md`](../docs/final/HANDOFF.md).

| Folder | Contents |
|---|---|
| `pipeline/` | Complete pipeline code (`src/`, including the C++ `retrieve_v3.cpp`), run scripts `run_v3.sh`…`run_v7e.sh`, `PACKAGE_README.md` (reproduction), `Documentation_filled.md` |
| `generator_rules/` | Final-day scripts. **Used in v8b:** `acr.py`, `acr2.py`, `addr_block.py`, `ab_mkpairs.py`, `rescue.py`, `rescue_apply.py`, `meta2.py`, `meta3.py`, `fingerprints.py`, `build_v8.py`. **Analysis:** `gen_ops*.py`, `f_attr.py`, `t1_errors.py`, `t2_disagree.py`, `unique_acct.py`, `xform.py`, `namesake_hints*.py`, `numcheck*.py`, `fr_unique.py`, `digit_residue.py`, `killswitch.py`, `offset.py`, `stats.py`, plus the rejected experiments |
| `synthetic_france/` | `gen_fr_v3.py` (labelled synthetic France), `synth_compare.py`-style checks, and the pipeline-on-synthetic scripts (`run_A.sh`, `run_B*.sh`, `run_C.sh`, `score_syn.py`, `sweep_syn.py`, `joint_fr.py`, `err_syn.py`) |
| `models/` | Stage-1 CatBoost + borders; stage-2 CatBoost for `v7e_usin_owner` (US/India), `v6_pp` (France) and `v7d_all_val`; calibrator; Indian-script rescue model |
| `submissions/` | Gzipped `matching_results.tsv` for v7d (0.984931), v7d_acr (0.985608), v8b (**0.986382**) and v8b_fr14 (final upload) |
| `reports/` | Stage and pipeline logs, JSON reports |
| `ab_test_kit/` | Stdlib-only hold-out scorer for comparing submissions on labelled folds |

Paths inside scripts are absolute to the author's machine (`/Users/apple/...`). Change `W` and the dataset path at the top of each file.
Large artefacts are not included: data, cross-encoder weights, embeddings, the candidate file. See HANDOFF §9.
