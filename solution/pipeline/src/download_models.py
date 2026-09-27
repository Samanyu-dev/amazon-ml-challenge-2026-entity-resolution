import os,json,time
from pathlib import Path
os.environ['HF_HOME']=str(Path(__file__).resolve().parents[1]/'cache/huggingface')
from huggingface_hub import snapshot_download
base=Path(__file__).resolve().parents[1]
models=[('intfloat/multilingual-e5-small','614241f622f53c4eeff9890bdc4f31cfecc418b3','multilingual-e5-small','mit')]
for name,revision,folder,license in models:
    dest=base/'models'/folder
    snapshot_download(name,revision=revision,local_dir=dest,allow_patterns=['*.json','*.safetensors','tokenizer*','sentencepiece*','*.model','modules.json','1_Pooling/*','README.md','LICENSE*'],max_workers=3)
    (dest/'provenance.json').write_text(json.dumps({'repo_id':name,'revision':revision,'license':license},indent=2))
    print('DOWNLOADED',name,flush=True)
