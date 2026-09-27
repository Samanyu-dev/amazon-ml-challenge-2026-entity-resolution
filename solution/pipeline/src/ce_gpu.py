"""Cross-encoder for business-record pairs: fine-tune and score. Runs on any CUDA/MPS box
(Kaggle, Colab, SageMaker, local). Only organiser data is used; the base model is
intfloat/multilingual-e5-small (MIT, 118M params) unless --model says otherwise.

  python ce_gpu.py train --data-dir DATASET --pairs ce_train_pairs.npz --split train --out ce_model
  python ce_gpu.py infer --data-dir DATASET --pairs ce_infer_test.npz  --split test  --model ce_model --out ce_scores_test.npy

DATASET is the organiser 'dataset' folder (contains train/ and test/ TSVs).
Both commands checkpoint and resume, so a dropped notebook session can simply be rerun.
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

def record_text(line):
    # entity_id, business_name, business_address, country
    r = line.rstrip('\n').split('\t') + ['', '', '']
    return f"{r[1]} | {r[2]} | {r[3]}"

class Texts:
    """Keeps only the rows a pair file references (a full split would need ~5 GB of strings).
    An optional per-pair 'split' field (0=train, 1=test) lets one file mix both splits,
    e.g. train labels plus test pseudo-labels."""
    def __init__(self, data_dir, split, pairs):
        self.split = pairs['split'] if 'split' in pairs else np.full(len(pairs['s1_row']), 0 if split == 'train' else 1, np.int8)
        self.s = {}
        for sp_id, sp in ((0, 'train'), (1, 'test')):
            m = self.split == sp_id
            if not m.any(): continue
            need = {1: set(pairs['s1_row'][m].tolist())}
            for k in (2, 3): need[k] = set(pairs['t_row'][m & (pairs['src'] == k)].tolist())
            for k in (1, 2, 3):
                rows = {}
                with open(Path(data_dir) / sp / f'{sp}_source{k}.tsv', encoding='utf-8') as f:
                    next(f)  # header; data row i is rid i
                    for i, line in enumerate(f):
                        if i in need[k]: rows[i] = record_text(line)
                self.s[sp_id, k] = rows
    def pair(self, i, s1_row, src, t_row):
        sp = int(self.split[i]); return self.s[sp, 1][s1_row], self.s[sp, src][t_row]

def device():
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def batches(texts, pairs, idx, tok, max_len):
    a, b = zip(*(texts.pair(i, int(pairs['s1_row'][i]), int(pairs['src'][i]), int(pairs['t_row'][i])) for i in idx))
    return tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors='pt')

def train(a):
    pairs = dict(np.load(a.pairs)); n = len(pairs['label']); texts = Texts(a.data_dir, a.split, pairs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = device()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
    core = model  # unwrapped module for saving
    if a.data_parallel and dev == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model); print('DataParallel on', torch.cuda.device_count(), 'GPUs', flush=True)
    steps = math.ceil(n / a.batch) * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    state = out / 'state.json'; step = 0
    if state.exists():  # resume
        step = json.loads(state.read_text())['step']
        core.load_state_dict(torch.load(out / 'ckpt.pt', map_location=dev)); opt.load_state_dict(torch.load(out / 'opt.pt', map_location=dev))
        for _ in range(step): sched.step()
        print('resumed at step', step, flush=True)
    scaler = torch.amp.GradScaler('cuda', enabled=dev == 'cuda'); lossf = torch.nn.BCEWithLogitsLoss()
    order = np.concatenate([np.random.default_rng(e).permutation(n) for e in range(a.epochs)])
    y_all = torch.tensor(pairs['label'], dtype=torch.float32); model.train(); t0 = time.time(); run = 0.0
    while step < steps:
        idx = order[step * a.batch:(step + 1) * a.batch]
        enc = batches(texts, pairs, idx, tok, a.max_len).to(dev); y = y_all[idx].to(dev)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            loss = lossf(model(**enc).logits.squeeze(-1).float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        step += 1; run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 200 == 0: print(f'step {step}/{steps} loss {run:.4f} {step * a.batch / (time.time() - t0):.0f} pairs/s', flush=True)
        if step % a.save_every == 0 or step == steps:
            torch.save(core.state_dict(), out / 'ckpt.pt'); torch.save(opt.state_dict(), out / 'opt.pt')
            state.write_text(json.dumps({'step': step, 'steps': steps}))
    core.save_pretrained(out / 'final'); tok.save_pretrained(out / 'final')
    (out / 'final' / 'provenance.json').write_text(json.dumps({'base_model': a.model, 'license': 'MIT', 'pairs': int(n), 'pseudo_pairs': int((texts.split == 1).sum()), 'epochs': a.epochs, 'lr': a.lr, 'max_len': a.max_len}))
    print('TRAIN_DONE', out / 'final', flush=True)

@torch.no_grad()
def infer(a):
    pairs = dict(np.load(a.pairs)); n = len(pairs['s1_row']); texts = Texts(a.data_dir, a.split, pairs); dev = device()
    src = Path(a.model); src = src / 'final' if (src / 'final').exists() else src
    tok = AutoTokenizer.from_pretrained(src); model = AutoModelForSequenceClassification.from_pretrained(src).to(dev).eval()
    if dev == 'cuda': model.half()
    out = Path(a.out); scores = np.lib.format.open_memmap(out, mode='r+' if out.exists() else 'w+', dtype=np.float16, shape=(n,))
    prog = Path(str(out) + '.progress'); done = int(prog.read_text()) if prog.exists() else 0; t0 = time.time(); start = done
    # sort each chunk by text length so padding stays small
    chunk = a.batch * 64
    while done < n:
        ids = np.arange(done, min(n, done + chunk))
        lens = np.array([sum(map(len, texts.pair(i, int(pairs['s1_row'][i]), int(pairs['src'][i]), int(pairs['t_row'][i])))) for i in ids])
        ids = ids[np.argsort(lens)]
        for st in range(0, len(ids), a.batch):
            b = ids[st:st + a.batch]; enc = batches(texts, pairs, b, tok, a.max_len).to(dev)
            scores[b] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy().astype(np.float16)
        done = min(n, done + chunk); scores.flush(); prog.write_text(str(done))
        print(f'{done}/{n} {(done - start) / (time.time() - t0):.0f} pairs/s', flush=True)
    print('INFER_DONE', out, flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'infer']); ap.add_argument('--data-dir', required=True)
    ap.add_argument('--pairs', required=True); ap.add_argument('--split', choices=['train', 'test'], required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--model', default='intfloat/multilingual-e5-small')
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=3e-5); ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--data-parallel', action='store_true', help='split each batch over all visible GPUs')
    a = ap.parse_args(); train(a) if a.cmd == 'train' else infer(a)
