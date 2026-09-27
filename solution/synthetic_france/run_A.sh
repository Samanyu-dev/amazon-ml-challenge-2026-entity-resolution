#!/bin/zsh
set -e
cd /Users/apple/Documents/Codex/2026-09-25/what-are-the-god-level-solution/work/syn_run
P=../submission_env/bin/python; D=/private/tmp/claude-501/synth2/data
step(){ echo "=== $(date +%H:%M:%S) $*"; }
step drop test tables
$P -c "import duckdb; c=duckdb.connect('artifacts/records.duckdb'); [c.execute(f'DROP TABLE IF EXISTS test_s{k}') for k in (1,2,3)]; c.close()"
step prepare
$P src/prepare.py --data-root $D --work artifacts > reports/prepare.log 2>&1
step make_arrays
$P src/make_arrays.py --work artifacts
step normalize test
$P src/normalize_corpus.py --work artifacts --out artifacts_v3 --version v3 --split test --workers 6 > reports/normalize.log 2>&1
cat artifacts_v3/test_s2.norm.tsv artifacts_v3/test_s3.norm.tsv > artifacts_v3/test_targets.norm.tsv
step encode test
$P src/encode_corpus.py --work artifacts --model models/multilingual-e5-small --tables test_s1 test_s2 test_s3 > reports/encode.log 2>&1
step ann
$P src/ann.py --work artifacts --split test --build > reports/ann.log 2>&1
for k in 2 3; do $P src/ann.py --work artifacts --split test --source $k --threads 9 --topk 4 >> reports/ann.log 2>&1; done
cat artifacts/test_ann/s2-*.bin artifacts/test_ann/s3-*.bin > artifacts_v3/test_dense.bin
step retrieve
./retrieve_v3 artifacts_v3/test_s1.norm.tsv artifacts_v3/test_targets.norm.tsv artifacts_v3/test_pairs 9 8 0 0 artifacts_v3/test_dense.bin 2> reports/retrieval.log
step A_DONE
