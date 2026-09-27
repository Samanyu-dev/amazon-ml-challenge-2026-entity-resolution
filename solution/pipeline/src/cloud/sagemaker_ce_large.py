"""SageMaker job: fine-tune multilingual-e5-large (MIT, 560M) as a pair cross-encoder on the
organiser data (labelled train pairs + France pseudo-labels), then score the uncertain pairs.
Only organiser data is used; the dataset is pulled from the team's private Kaggle copy and its
SHA-256 is checked against the organiser files. Results land in /opt/ml/checkpoints (synced to S3)."""
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
# 2) train (resumable: checkpoints live in the synced dir)
model_dir = f'{out}/ce_large'
if not os.path.exists(f'{model_dir}/final/provenance.json'):
    run([sys.executable, 'ce_gpu.py', 'train', '--data-dir', ds, '--pairs', f'{pairs}/ce_train_v3_pairs.npz', '--split', 'train',
         '--out', model_dir, '--model', 'intfloat/multilingual-e5-large', '--batch', '64', '--max-len', '96', '--lr', '1e-5', '--save-every', '4000'])
    for f in ('ckpt.pt', 'opt.pt'):
        if os.path.exists(f'{model_dir}/{f}'): os.remove(f'{model_dir}/{f}')
# 3) score the uncertain top-2 pairs
for sp in ('train', 'test'):
    dst = f'{out}/ce_large_scores_{sp}.npy'
    if not os.path.exists(dst + '.done'):
        run([sys.executable, 'ce_gpu.py', 'infer', '--data-dir', ds, '--pairs', f'{pairs}/ce_infer_{sp}.npz', '--split', sp,
             '--model', f'{model_dir}/final', '--out', dst, '--batch', '256', '--max-len', '96'])
        open(dst + '.done', 'w').write('ok')
# 4) score the rank-3/4 candidates of uncertain targets (top-4 pair files)
for sp, split in (('train', 'train'), ('testa', 'test'), ('testb', 'test')):
    dst = f'{out}/ce_large4_scores_{sp}.npy'
    if os.path.exists(f'{pairs}/ce_infer4new_{sp}.npz') and not os.path.exists(dst + '.done'):
        run([sys.executable, 'ce_gpu.py', 'infer', '--data-dir', ds, '--pairs', f'{pairs}/ce_infer4new_{sp}.npz', '--split', split,
             '--model', f'{model_dir}/final', '--out', dst, '--batch', '256', '--max-len', '96'])
        open(dst + '.done', 'w').write('ok')
print('ALL_DONE', flush=True)
