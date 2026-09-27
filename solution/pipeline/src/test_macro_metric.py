import unittest
import numpy as np
from calibrate import entity_scores

def reference(truth,pred):
    result=[]
    for key,t in truth.items():
        p=pred.get(key,set())
        result.append(1. if not p and not t else 1.25*len(t&p)/(len(p)+.25*len(t)))
    return sum(result)/len(result)

class MetricTests(unittest.TestCase):
    def test_hand_computed(self):
        cases=[({0:set()},{},1.),({0:set()},{0:{1}},0.),({0:{1,2}},{0:{1,2}},1.),
               ({0:{1,2}},{0:{1,2,3}},5/7),({0:{1,2}},{0:{1}},5/6),({0:{1}},{},0.)]
        for t,p,want in cases:self.assertAlmostEqual(reference(t,p),want)
    def test_macro_not_micro(self):
        t={0:{1},1:set(range(10,19))};p={0:{1},1:set()}
        self.assertEqual(reference(t,p),.5)
    def test_vectorized_independent_comparison(self):
        rng=np.random.default_rng(17)
        for _ in range(100):
            truth={i:set(rng.choice(30,size=int(rng.integers(0,10)),replace=False).tolist()) for i in range(12)}
            pred={i:set(rng.choice(30,size=int(rng.integers(0,15)),replace=False).tolist()) for i in range(12)}
            ids=[];good=[]
            for i,p in pred.items():
                for j in p:ids.append(i);good.append(j in truth[i])
            actual,_,_=entity_scores(np.array([len(t) for t in truth.values()]),np.array(ids,dtype=int),np.array(good))
            self.assertAlmostEqual(float(actual.mean()),reference(truth,pred))
    def test_missing_candidates_remain_in_denominator(self):
        f,_,_=entity_scores(np.array([2,0]),np.array([0]),np.array([True]))
        self.assertAlmostEqual(f[0],5/6);self.assertEqual(f[1],1.)

if __name__=='__main__':unittest.main(verbosity=2)
