import unittest
import numpy as np
from build_submission import verify_probabilities
from score_pairs import BD

class FakeCalibrator:
 def predict_proba(self,x):
  p=1/(1+np.exp(-x[:,0]))
  return np.column_stack([1-p,p])

class ExportLineageTests(unittest.TestCase):
 def setUp(self):
  self.b=np.zeros(3,dtype=BD)
  self.b['aid']=[0,1,-1];self.b['p']=[.9,.6,.2];self.b['p2']=.1
 def test_correct_probabilities_and_missing_candidate(self):
  verify_probabilities(self.b,np.array([.9,.6,0],dtype=np.float32),FakeCalibrator(),chunk=2)
 def test_wrong_model_probabilities_rejected(self):
  with self.assertRaisesRegex(ValueError,'mismatch'):
   verify_probabilities(self.b,np.array([.9,.8,0],dtype=np.float32),FakeCalibrator(),chunk=2)
 def test_wrong_length_rejected(self):
  with self.assertRaisesRegex(ValueError,'shape'):
   verify_probabilities(self.b,np.array([.9,.6],dtype=np.float32),FakeCalibrator())
 def test_nan_rejected(self):
  with self.assertRaises(ValueError):
   verify_probabilities(self.b,np.array([.9,np.nan,0],dtype=np.float32),FakeCalibrator())

if __name__=='__main__':unittest.main()
