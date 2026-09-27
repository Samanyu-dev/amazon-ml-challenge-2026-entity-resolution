"""Disjoint calibration-fit, threshold-tuning, and final held-out macro scoring."""
from pathlib import Path
import argparse,json,time,pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from score_pairs import BD

def calibration_features(z):
 p=np.clip(z['p'],1e-6,1-1e-6);p2=np.clip(z['p2'],1e-6,1-1e-6)
 return np.column_stack([np.log(p/(1-p)),np.log(p2/(1-p2)),p-p2,z['cos'],z['route'],z['f']]).astype(np.float32)

def entity_scores(t,idx,good):
 pred=np.bincount(idx,minlength=len(t));tp=np.bincount(idx,weights=good,minlength=len(t));den=pred+.25*t
 f=np.ones(len(t),dtype=np.float64);np.divide(1.25*tp,den,out=f,where=den>0)
 return f,pred,tp

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scores',required=True);ap.add_argument('--artifacts',required=True);ap.add_argument('--out',required=True);ap.add_argument('--apply',action='store_true');ap.add_argument('--calibrator');a=ap.parse_args()
 sd=Path(a.scores);art=Path(a.artifacts);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=json.loads((sd/'complete.json').read_text());b=np.memmap(sd/'best.bin',mode='r',dtype=BD,shape=(meta['target_universe'],))
 if a.apply:
  with open(a.calibrator,'rb') as f:cal=pickle.load(f)
 else:
  mask=(b['afold']>=80)&(b['afold']<85)&(b['tfold']<85)&(b['aid']>=0)
  ids=np.flatnonzero(mask);z=b[ids];X=calibration_features(z);y=(z['aid']==z['owner']).astype(np.uint8)
  cal=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=300,random_state=42))
  cal.fit(X,y)
  with (out/'calibrator.pkl').open('wb') as f:pickle.dump(cal,f)
  print('CALIBRATION_FIT',len(y),'positives',int(y.sum()),flush=True)
 prob=np.lib.format.open_memmap(out/'probabilities.npy',mode='w+',dtype=np.float32,shape=(len(b),))
 for st in range(0,len(b),200000):
  z=b[st:st+200000];p=cal.predict_proba(calibration_features(z))[:,1];p[z['aid']<0]=0;prob[st:st+len(z)]=p
 prob.flush()
 if a.apply:print('APPLIED',len(b),flush=True);return
 folds=np.load(art/'train_anchor_folds.npy');truth=np.load(art/'train_truth_counts.npy');countries=np.load(art/'train_anchor_countries.npy')
 def subset(lo,hi):
  mask=(folds>=lo)&(folds<hi);aid=np.flatnonzero(mask);lookup=np.full(len(folds),-1,dtype=np.int32);lookup[aid]=np.arange(len(aid))
  valid=b['aid']>=0;idx=np.flatnonzero(valid);idx=idx[mask[b['aid'][idx]]]
  return aid,idx,lookup[b['aid'][idx]],truth[aid]
 anchors,rows,local,t=subset(85,90);p=prob[rows];margin=b['p'][rows]-b['p2'][rows];good=b['aid'][rows]==b['owner'][rows]
 candidates=[]
 for gap in [0,.015,.03,.06,.1,.15]:
  for threshold in np.r_[np.arange(.1,.91,.025),.92,.94,.96,.97,.98,.985,.99,.995]:
   use=(p>=threshold)&(margin>=gap);f,pred,tp=entity_scores(t,local[use],good[use]);score=float(f.mean());candidates.append((score,float(threshold),gap))
 best=max(candidates,key=lambda x:(round(x[0],5),x[1]));score,threshold,gap=best
 report={'calibration_fit_folds':[80,85],'threshold_tuning_folds':[85,90],'validation_folds':[90,100],'threshold':threshold,'minimum_raw_margin':gap,'tuning_macro_f05':score,'tuning_anchors':len(anchors),'top_thresholds':sorted(candidates,reverse=True)[:15]}
 anchors,rows,local,t=subset(90,100);p=prob[rows];margin=b['p'][rows]-b['p2'][rows];good=b['aid'][rows]==b['owner'][rows];use=(p>=threshold)&(margin>=gap);f,pred,tp=entity_scores(t,local[use],good[use])
 report['validation']={'anchors':len(anchors),'macro_f05':float(f.mean()),'precision':float(tp.sum()/max(1,pred.sum())),'recall':float(tp.sum()/max(1,t.sum())),'true_links':int(t.sum()),'predicted_links':int(pred.sum()),'true_positives':int(tp.sum()),'singletons':int((t==0).sum()),'singleton_false_merges':int(((t==0)&(pred>0)).sum()),'by_country':{str(c):{'anchors':int((countries[anchors]==c).sum()),'macro_f05':float(f[countries[anchors]==c].mean())} for c in np.unique(countries[anchors])}}
 # Standard error across independent reference groups; paired bootstrap is used for later model comparisons.
 report['validation']['mean_standard_error']=float(f.std(ddof=1)/np.sqrt(len(f)))
 (out/'selection.json').write_text(json.dumps(report,indent=2));np.save(out/'validation_entity_scores.npy',f)
 print('VALIDATION',json.dumps(report),flush=True)
if __name__=='__main__':main()
