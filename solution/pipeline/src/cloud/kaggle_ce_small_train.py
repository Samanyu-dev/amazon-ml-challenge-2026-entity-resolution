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
bundle = os.path.dirname(find('ce_gpu.py'))
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
out = '/kaggle/working/ce_model'
subprocess.run([sys.executable, f'{bundle}/ce_gpu.py', 'train', '--data-dir', ds, '--pairs', f'{bundle}/ce_train_pairs.npz',
                '--split', 'train', '--out', out, '--batch', '128', '--save-every', '2000'], check=True)
for f in ('ckpt.pt', 'opt.pt'):
    if os.path.exists(f'{out}/{f}'): os.remove(f'{out}/{f}')
shutil.rmtree(ds)
print('ALL_DONE')
