import unittest
from normalization_v3 import views, normalized

core = lambda n: views(n, '', '')['name_core']
norm = lambda n: views(n, '', '')['name_norm']
addr = lambda a: views('', a, '')['address_norm']

class V3Tests(unittest.TestCase):
    def test_dotted_legal_forms_join(self):
        self.assertEqual(norm('Acme Holdings, L.L.C.'), 'acme holdings llc')
        self.assertEqual(core('Acme Holdings, L.L.C.'), 'acme holdings')
        self.assertEqual(core('TENNIS GAVROCHES LYCEE E.U.R.L.'), 'tennis gavroches lycee')
        self.assertEqual(core('QHC Culture [EURL]'), 'qhc culture')
    def test_french_legal_forms(self):
        self.assertEqual(core('Marina Ecole France Sarl'), 'marina ecole france')
        self.assertEqual(core('Unite (France) Sante SASU'), 'unite france sante')
        self.assertEqual(core('SCI Ptit Àmicale'), 'ptit amicale')
    def test_ms_firm_is_not_honorific(self):
        self.assertEqual(norm('M/s Sharma Traders Pvt. Ltd.'), 'messrs sharma traders private limited')
        self.assertEqual(core('M/s Sharma Traders Pvt. Ltd.'), 'sharma traders')
        self.assertEqual(norm('Ms Sharma'), 'ms sharma')
    def test_legal_form_kept_separately(self):
        a, b = views('Hyderabad Technology Pvt Ltd', '', ''), views('Hyderabad Technology Public Limited', '', '')
        self.assertEqual(a['name_core'], b['name_core'])
        self.assertEqual(a['legal_form'], 'private limited'); self.assertEqual(b['legal_form'], 'public limited')
    def test_public_only_before_limited(self):self.assertEqual(core('Public School Society'), 'public school society')
    def test_interior_legal_words_kept(self):self.assertEqual(core('The Company Store'), 'the company store')
    def test_short_name_not_deleted(self):self.assertEqual(core('Co'), 'co')
    def test_honorific_prefix(self):self.assertEqual(core('Dr Deccan Traders Pvt.Ltd.'), 'deccan traders')
    def test_repeated_tokens(self):
        self.assertEqual(norm('Littlejohn Inc. Inc.'), 'littlejohn inc')
        self.assertEqual(norm('Better Defense Defense Studios'), 'better defense studios')
    def test_initials_joined_consistently(self):self.assertEqual(norm('A.B.C. Traders'), norm('ABC Traders'))
    def test_french_address(self):
        self.assertEqual(addr('63 R. DE DIEPPE'), addr('63 RUE DE DIEPPE'))
        self.assertEqual(addr('77 AV LEON JOUHAUX'), addr('77 AVENUE LEON JOUHAUX'))
        self.assertEqual(addr('5 BD VOLTAIRE'), addr('5 BOULEVARD VOLTAIRE'))
    def test_address_letters_not_joined(self):self.assertEqual(addr('Unit 108 B C'), 'unit 108 b c')
    def test_transliterated_legal_forms(self):
        self.assertEqual(core('raam maarketti g praaivett limittedd'), 'raam maarketti g')
        self.assertEqual(views('raam maarketti g praaivett limittedd', '', '')['legal_form'], 'private limited')
        self.assertEqual(core('Nirman piraiveett limttidd'), 'nirman')
    def test_parvati_is_a_name(self):
        self.assertEqual(core('Shree Parvati'), 'shree parvati')
        self.assertEqual(core('Parvati Traders Limited'), 'parvati traders')
    def test_quarantine(self):
        with self.assertRaises(ValueError):normalized('Care', 'Utah', {'address': {'utah': 'unit'}})

if __name__ == '__main__':unittest.main(verbosity=2)
