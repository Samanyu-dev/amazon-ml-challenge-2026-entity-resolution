"""7B LLM pair matcher (QLoRA): Qwen2.5-7B-Instruct (Apache-2.0) with a 1-logit classification head.

Same pair files and record text as ce_gpu.py. The LLM brings multilingual world knowledge (French
address conventions, geography, business naming) that our US/India-trained cross-encoders lack.
Trained only on labelled US/India pairs from CE folds 0-39 (no French pseudo-labels).

  python llm_gpu.py train --data-dir DATASET --pairs llm_train_pairs.npz --split train --out llm_model
  python llm_gpu.py infer --data-dir DATASET --pairs llm_infer.npz --split test --model llm_model --out llm_scores.npy
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from ce_gpu import Texts

PROMPT = 'Are these two business records the same real-world business?\nRecord A: {a}\nRecord B: {b}\nAnswer:'

def encode(texts, pairs, idx, tok, max_len):
    txt = [PROMPT.format(a=x, b=y) for x, y in (texts.pair(i, int(pairs['s1_row'][i]), int(pairs['src'][i]), int(pairs['t_row'][i])) for i in idx)]
    return tok(txt, truncation=True, max_length=max_len, padding=True, return_tensors='pt')

def base_model(name, four_bit):
    kw = dict(num_labels=1, torch_dtype=torch.bfloat16)
    if four_bit:
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw['device_map'] = {'': 0}
    m = AutoModelForSequenceClassification.from_pretrained(name, **kw)
    return m

def tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def train(a):
    pairs = dict(np.load(a.pairs)); n = len(pairs['label']); texts = Texts(a.data_dir, a.split, pairs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(a.model); model = base_model(a.model, a.four_bit); model.config.pad_token_id = tok.pad_token_id
    if a.four_bit: model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='SEQ_CLS',
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    model.print_trainable_parameters()
    dev = next(model.parameters()).device
    steps = math.ceil(n / a.batch); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps); lossf = torch.nn.BCEWithLogitsLoss()
    state = out / 'state.json'; step = 0
    if state.exists() and (out / 'adapter').exists():
        step = json.loads(state.read_text())['step']
        model.load_adapter(str(out / 'adapter'), adapter_name='default', is_trainable=True)
        for _ in range(step): sched.step()
        print('resumed at step', step, flush=True)
    order = np.random.default_rng(0).permutation(n); y_all = torch.tensor(pairs['label'], dtype=torch.float32)
    model.train(); t0 = time.time(); run = None; s0 = step
    while step < steps:
        idx = order[step * a.batch:(step + 1) * a.batch]
        enc = encode(texts, pairs, idx, tok, a.max_len).to(dev); y = y_all[idx].to(dev)
        with torch.autocast(device_type=dev.type if dev.type == 'cuda' else 'cpu', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            loss = lossf(model(**enc).logits.squeeze(-1).float(), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); sched.step()
        step += 1; run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 50 == 0: print(f'step {step}/{steps} loss {run:.4f} {(step - s0) * a.batch / (time.time() - t0):.1f} pairs/s', flush=True)
        if step % a.save_every == 0 or step == steps:
            model.save_pretrained(out / 'adapter'); state.write_text(json.dumps({'step': step, 'steps': steps}))
    (out / 'adapter' / 'provenance.json').write_text(json.dumps({'base_model': a.model, 'license': 'Apache-2.0', 'method': 'QLoRA r16', 'pairs': int(n), 'lr': a.lr, 'max_len': a.max_len}))
    print('TRAIN_DONE', out / 'adapter', flush=True)

@torch.no_grad()
def infer(a):
    pairs = dict(np.load(a.pairs)); n = len(pairs['s1_row']); texts = Texts(a.data_dir, a.split, pairs)
    tok = tokenizer(a.model_base); model = base_model(a.model_base, a.four_bit); model.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(model, str(Path(a.model) / 'adapter')).eval(); dev = next(model.parameters()).device
    out = Path(a.out); scores = np.lib.format.open_memmap(out, mode='r+' if out.exists() else 'w+', dtype=np.float32, shape=(n,))
    prog = Path(str(out) + '.progress'); done = int(prog.read_text()) if prog.exists() else 0; t0 = time.time(); start = done
    chunk = a.batch * 32
    while done < n:
        ids = np.arange(done, min(n, done + chunk))
        lens = np.array([sum(map(len, texts.pair(i, int(pairs['s1_row'][i]), int(pairs['src'][i]), int(pairs['t_row'][i])))) for i in ids]); ids = ids[np.argsort(lens)]
        for st in range(0, len(ids), a.batch):
            b = ids[st:st + a.batch]; enc = encode(texts, pairs, b, tok, a.max_len).to(dev)
            with torch.autocast(device_type=dev.type if dev.type == 'cuda' else 'cpu', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                scores[b] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        done = min(n, done + chunk); scores.flush(); prog.write_text(str(done))
        print(f'{done}/{n} {(done - start) / (time.time() - t0):.0f} pairs/s', flush=True)
    print('INFER_DONE', out, flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'infer']); ap.add_argument('--data-dir', required=True)
    ap.add_argument('--pairs', required=True); ap.add_argument('--split', choices=['train', 'test'], required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--model-base', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max-len', type=int, default=160); ap.add_argument('--save-every', type=int, default=500)
    ap.add_argument('--no-four-bit', dest='four_bit', action='store_false')
    a = ap.parse_args(); train(a) if a.cmd == 'train' else infer(a)
