"""SageMaker job: QLoRA fine-tune Qwen2.5-7B-Instruct (Apache-2.0, 7.6B) as a pair matcher on labelled US/India
pairs (CE folds 0-39, no French pseudo-labels), then score French top-2 pairs and US/India folds 85-99.
Only organiser data is used (private Kaggle copy, SHA-256 checked). Results land in /opt/ml/checkpoints (synced to S3)."""
import hashlib, json, os, subprocess, sys, boto3
SHA = {'train_source1.tsv': '591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973',
       'train_source2.tsv': '6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed',
       'train_source3.tsv': '67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9',
       'test_source1.tsv': '3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5',
       'test_source2.tsv': '79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd',
       'test_source3.tsv': '850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035'}
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def run(cmd, **kw):
    print('RUN', ' '.join(cmd), flush=True); subprocess.run(cmd, check=True, **kw)

pairs = os.environ.get('SM_CHANNEL_PAIRS', '/opt/ml/input/data/pairs')
out = '/opt/ml/checkpoints'; os.makedirs(out, exist_ok=True)
raw = '/tmp/er_raw'; ds = '/tmp/ds'
# 1) data: private Kaggle dataset, token from Secrets Manager (never logged)
if not os.path.exists(f'{raw}/test_source3.tsv'):
    secret = boto3.client('secretsmanager', region_name='ap-south-1').get_secret_value(SecretId='kaggle-token')['SecretString']
    try: token = json.loads(secret)['token']
    except (ValueError, KeyError, TypeError): token = secret.strip()
    env = dict(os.environ, KAGGLE_API_TOKEN=token)
    run(['kaggle', 'datasets', 'download', 'samanyu1808/amazon-ml-2026-er-data', '-p', raw, '--unzip'], env=env)
for n, h in SHA.items():
    got = sha256(f'{raw}/{n}'); assert got == h, f'{n}: sha256 mismatch'; print('verified', n, flush=True)
for sp in ('train', 'test'):
    os.makedirs(f'{ds}/{sp}', exist_ok=True)
    for k in (1, 2, 3):
        dst = f'{ds}/{sp}/{sp}_source{k}.tsv'
        if not os.path.exists(dst): os.symlink(f'{raw}/{sp}_source{k}.tsv', dst)
run(['nvidia-smi'])
# 2) QLoRA fine-tune Qwen2.5-7B-Instruct (Apache-2.0) on labelled US/India pairs (CE folds 0-39 only)
M = 'Qwen/Qwen2.5-7B-Instruct'; model_dir = f'{out}/llm7b'
if not os.path.exists(f'{model_dir}/adapter/provenance.json'):
    run([sys.executable, 'llm_gpu.py', 'train', '--data-dir', ds, '--pairs', f'{pairs}/llm_train_pairs.npz', '--split', 'train',
         '--out', model_dir, '--model', M, '--batch', '16', '--lr', '1e-4', '--max-len', '160', '--save-every', '500'])
# 3) score: France first (the target domain), then US/India hold-out for validation
for name, split in (('llm_infer_test_fr', 'test'), ('llm_infer_train', 'train')):
    dst = f'{out}/{name}_scores.npy'
    if not os.path.exists(dst + '.done'):
        run([sys.executable, 'llm_gpu.py', 'infer', '--data-dir', ds, '--pairs', f'{pairs}/{name}.npz', '--split', split,
             '--out', dst, '--model', model_dir, '--model-base', M, '--batch', '48', '--max-len', '160'])
        open(dst + '.done', 'w').write('ok')
print('ALL_DONE', flush=True)
