from pathlib import Path
import json,time,subprocess
base=Path(__file__).resolve().parents[1];deadline=time.time()+8*3600
if (base/'artifacts/NORMALIZATION_AUDIT_HOLD.json').exists():
 raise RuntimeError('Normalization audit hold: legacy retrieval queue is disabled')
while time.time()<deadline:
 try:
  report=json.loads((base/'artifacts/train_pairs/run.json').read_text())
  if report['queries']==10320219:break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(20)
else:raise RuntimeError('Training retrieval did not finish within the allocated wait window')
print('Training retrieval complete; starting full test retrieval',flush=True)
with (base/'reports/retrieval_test.log').open('w') as f:
 subprocess.run([str(base/'retrieve'),str(base/'artifacts/test_s1.norm.tsv'),str(base/'artifacts/test_targets.norm.tsv'),str(base/'artifacts/test_pairs'),'7','8'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Full test retrieval complete',flush=True)
