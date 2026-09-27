# Amazon ML Challenge 2026 - cross-encoder fine-tuning on organiser data only.
import glob, hashlib, os, shutil, subprocess, sys
# SHA-256 of the organiser files (local data manifest); a corrupted or partial upload aborts the run.
SHA = {'train_source1.tsv': '591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973',
       'train_source2.tsv': '6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed',
       'train_source3.tsv': '67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9',
       'test_source1.tsv': '3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5',
       'test_source2.tsv': '79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd',
       'test_source3.tsv': '850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035'}
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def find(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    assert hits, f'{name} not found under /kaggle/input'; return hits[0]
bundle = os.path.dirname(find('ce_infer4new_train.npz'))
data = {n: find(n) for n in SHA}
for n, path in data.items():
    got = sha256(path); assert got == SHA[n], f'{n}: sha256 {got} does not match the organiser file'; print('verified', n, flush=True)
ds = '/kaggle/working/ds'
for split in ('train', 'test'):
    os.makedirs(f'{ds}/{split}', exist_ok=True)
    for k in (1, 2, 3):
        dst = f'{ds}/{split}/{split}_source{k}.tsv'
        if not os.path.exists(dst): os.symlink(data[f'{split}_source{k}.tsv'], dst)
subprocess.run(['nvidia-smi'])
model = os.path.dirname(find('provenance.json'))   # trained model from the ce-train notebook output
print('model', model, open(f'{model}/provenance.json').read(), flush=True)
def infer(split, name):
    return [sys.executable, f'{bundle}/ce_gpu.py', 'infer', '--data-dir', ds, '--pairs', f'{bundle}/ce_infer4new_{name}.npz',
            '--split', split, '--model', model, '--out', f'/kaggle/working/ce4_scores_{name}.npy', '--batch', '512']
# GPU0: train then test_a (sequential); GPU1: test_b. Equal pair counts per GPU.
jobs = {0: [infer('train', 'train'), infer('test', 'testa')], 1: [infer('test', 'testb')]}
def chain(gpu, cmds):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    script = ' && '.join(' '.join(f"'{c}'" for c in cmd) for cmd in cmds)
    return subprocess.Popen(['bash', '-c', script], env=env)
procs = [chain(g, c) for g, c in jobs.items()]
codes = [p.wait() for p in procs]
assert codes == [0, 0], f'inference failed: {codes}'
shutil.rmtree(ds)
print('ALL_DONE')
