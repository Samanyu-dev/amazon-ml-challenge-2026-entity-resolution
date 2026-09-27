from pathlib import Path
import json
import numpy as np
class EmbeddingLookup:
    def __init__(self,root,split):
        self.root=Path(root)/'embeddings';self.v=[];self.n=[]
        for source in [1,2,3]:
            key=f'{split}_s{source}';m=json.loads((self.root/f'{key}.json').read_text())
            if m['done']!=m['rows']:raise RuntimeError(f'Embeddings incomplete: {key}')
            self.v.append(np.memmap(self.root/f'{key}.f16',dtype=np.float16,mode='r',shape=(m['rows'],m['dim'])))
            self.n.append(m['rows'])
    def features(self,z,dense_route=False):
        n=z['x'].shape[1];result=np.empty((len(z),n+2),dtype=np.float32);result[:,:n]=z['x'];result[:,n+1]=float(dense_route)
        for st in range(0,len(z),20000):
            part=z[st:st+20000];a=np.asarray(self.v[0][part['aid']],dtype=np.float32);t=np.empty_like(a)
            mask=part['tid']<self.n[1]
            t[mask]=self.v[1][part['tid'][mask]]
            t[~mask]=self.v[2][part['tid'][~mask]-self.n[1]]
            result[st:st+len(part),n]=np.einsum('ij,ij->i',a,t)
        assert np.isfinite(result).all()
        return result
