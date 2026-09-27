import os,time,json
from pathlib import Path
base=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(base/'cache/huggingface');os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,duckdb,numpy as np
from transformers import AutoTokenizer,AutoModel
torch.set_num_threads(2)
c=duckdb.connect(str(base/'artifacts/records.duckdb'),read_only=True)
rows=c.execute('SELECT business_name,business_address,country FROM test_s1 LIMIT 1024').fetchall();c.close()
texts=['query: '+n+'; '+a+'; '+co for n,a,co in rows]
p=base/'models/multilingual-e5-small'
tok=AutoTokenizer.from_pretrained(p);model=AutoModel.from_pretrained(p,torch_dtype=torch.float16).to('mps').eval()
results=[]
for batch_size in [32,64,128]:
 start=time.time();vectors=[]
 with torch.inference_mode():
  for i in range(0,len(texts),batch_size):
   z=tok(texts[i:i+batch_size],padding=True,truncation=True,max_length=96,return_tensors='pt').to('mps')
   h=model(**z).last_hidden_state.float();mask=z['attention_mask'].unsqueeze(-1)
   emb=(h*mask).sum(1)/mask.sum(1)
   emb=torch.nn.functional.normalize(emb,dim=1)
   vectors.append(emb.cpu().numpy())
 torch.mps.synchronize();elapsed=time.time()-start
 results.append({'batch_size':batch_size,'seconds':elapsed,'records_per_second':len(texts)/elapsed,'shape':list(np.concatenate(vectors).shape)})
 print(results[-1],flush=True)
(base/'reports/embedding_throughput.json').write_text(json.dumps(results,indent=2))
