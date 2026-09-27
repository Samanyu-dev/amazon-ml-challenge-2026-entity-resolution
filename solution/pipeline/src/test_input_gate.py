import unittest,tempfile,pathlib,json
from input_gate import digest,verify_generation

class GateTests(unittest.TestCase):
    def test_detects_mutated_source_and_mixed_generation(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d);m={'status':'APPROVED','generation_id':'a','normalization_code_sha256':'code','normalization_config_sha256':'config','sources':{},'combined_targets':{}}
            for split in ['train','test']:
                for i in [1,2,3]:
                    key=f'{split}_s{i}';f=p/key;f.write_text(key+'\n')
                    m['sources'][key]={'path':key,'sha256':digest(f),'rows':1,'generation_id':'a'}
                f=p/f'{split}_targets';f.write_bytes((p/f'{split}_s2').read_bytes()+(p/f'{split}_s3').read_bytes())
                m['combined_targets'][split]={'path':f.name,'sha256':digest(f),'rows':2,'generation_id':'a','component_sha256':[m['sources'][f'{split}_s{i}']['sha256'] for i in [2,3]]}
            manifest=p/'generation.json';manifest.write_text(json.dumps(m));verify_generation(manifest)
            m['sources']['train_s1']['generation_id']='old';manifest.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Mixed'):verify_generation(manifest)
            m['sources']['train_s1']['generation_id']='a';manifest.write_text(json.dumps(m));(p/'train_s1').write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'fingerprint'):verify_generation(manifest)
    def test_unapproved_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'m.json';p.write_text('{"status":"AUDIT_ONLY"}')
            with self.assertRaisesRegex(ValueError,'audit gate'):verify_generation(p)

if __name__=='__main__':unittest.main(verbosity=2)
