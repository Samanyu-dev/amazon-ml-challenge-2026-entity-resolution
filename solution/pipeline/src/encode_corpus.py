"""Checkpointed, full-corpus multilingual encoding on MPS/CUDA/CPU."""
from pathlib import Path
import argparse,os,json,time,gc
base=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(base/'cache/huggingface'));os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch,duckdb
from transformers import AutoTokenizer,AutoModel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work',required=True);ap.add_argument('--model',required=True);ap.add_argument('--batch-size',type=int,default=128);ap.add_argument('--max-length',type=int,default=128);ap.add_argument('--tables',nargs='*',default=['train_s1','test_s1','train_s2','train_s3','test_s2','test_s3']);a=ap.parse_args()
 out=Path(a.work);dest=out/'embeddings';dest.mkdir(exist_ok=True)
 torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
 dtype=torch.float32 if device=='cpu' else torch.float16
 tok=AutoTokenizer.from_pretrained(a.model);model=AutoModel.from_pretrained(a.model,dtype=dtype).to(device).eval();dim=model.config.hidden_size
 c=duckdb.connect(str(out/'records.duckdb'),read_only=True);c.execute("SET memory_limit='1GB'");c.execute('SET threads=2')
 for table in a.tables:
  n=c.execute(f'SELECT count(*) FROM {table}').fetchone()[0];statep=dest/f'{table}.json';vecp=dest/f'{table}.f16'
  state=json.loads(statep.read_text()) if statep.exists() else {'rows':n,'done':0,'dim':dim,'max_length':a.max_length,'dtype':'float16','model_revision':'614241f622f53c4eeff9890bdc4f31cfecc418b3','representation':'query: name; address; country'}
  assert state['rows']==n and state['max_length']==a.max_length
  if state['done']==n:continue
  vec=np.memmap(vecp,dtype=np.float16,mode='r+' if vecp.exists() else 'w+',shape=(n,dim));done=state['done'];start=time.time();start_done=done
  cur=c.execute(f'SELECT business_name,business_address,country FROM {table} WHERE rid>={done} ORDER BY rid')
  with torch.inference_mode():
   while rows:=cur.fetchmany(a.batch_size*100):
    texts=['query: '+(name or '')+'; '+(addr or '')+'; '+country for name,addr,country in rows]
    for i in range(0,len(texts),a.batch_size):
     sub=texts[i:i+a.batch_size];z=tok(sub,padding=True,truncation=True,max_length=a.max_length,return_tensors='pt').to(device)
     h=model(**z).last_hidden_state.float();mask=z['attention_mask'].unsqueeze(-1);v=(h*mask).sum(1)/mask.sum(1);v=torch.nn.functional.normalize(v,dim=1)
     arr=v.cpu().numpy();assert np.isfinite(arr).all();vec[done:done+len(arr)]=arr;done+=len(arr)
    vec.flush();state['done']=done;state['seconds_this_session']=time.time()-start;tmp=Path(str(statep)+'.tmp');tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,statep)
    if done%(a.batch_size*500)<a.batch_size*100:print(table,done,'/',n,'records_per_second',round((done-start_done)/(time.time()-start),1),flush=True)
  del vec;gc.collect();print('ENCODED',table,done,'seconds',time.time()-start,flush=True)
 c.close()
if __name__=='__main__':main()
