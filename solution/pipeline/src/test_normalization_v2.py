import unittest
from normalization_v2 import basic,views,normalized,country_key,NAME_ALIASES,ADDRESS_ALIASES

class NormalizationTests(unittest.TestCase):
    def test_business_abbreviations(self):
        self.assertEqual(views('ABC Pvt Ltd','','')['name_norm'],'abc private limited')
        self.assertEqual(views('ABC Incorporated','','')['name_norm'],'abc inc')
    def test_legal_difference_retained(self):
        a=views('ABC Private Limited','','');b=views('ABC Limited','','')
        self.assertNotEqual(a['name_norm'],b['name_norm'])
        self.assertEqual(a['name_core'],b['name_core']) # auxiliary collision is intentional
        self.assertNotEqual(a['legal_suffix'],b['legal_suffix'])
    def test_interior_words_retained(self):
        self.assertEqual(views('The Company Store','','')['name_core'],'the company store')
    def test_short_name_not_deleted(self):
        self.assertEqual(views('Co','','')['name_core'],'co')
    def test_unsafe_aliases_rejected(self):
        with self.assertRaises(ValueError):normalized('Care','Utah',{'address':{'utah':'unit'}})
    def test_semantic_names_not_rewritten(self):
        for word in ['keyr','phyuucr','iistt','stee','sttaar','pr','corporate']:
            self.assertEqual(views(word,'','')['name_norm'],word)
    def test_near_names_distinct(self):
        self.assertNotEqual(views('Balaji Medical','','')['name_norm'],views('Balaji Traders','','')['name_norm'])
    def test_whitespace(self):self.assertEqual(basic('  ABC\t  Sons\n'),'abc sons')
    def test_ampersand(self):self.assertEqual(basic('ABC & Sons'),basic('ABC and Sons'))
    def test_punctuation_boundaries(self):self.assertEqual(basic('A-B/C'),'a b c')
    def test_apostrophe_preserved_unicode(self):self.assertEqual(views("O’Connor",'','')['name_unicode'],"o’connor")
    def test_diacritics(self):
        v=views('Café Étoile','','');self.assertEqual(v['name_norm'],'cafe etoile');self.assertEqual(v['name_unicode'],'café étoile')
    def test_canonical_unicode(self):self.assertEqual(basic('Cafe\u0301'),basic('Café'))
    def test_fullwidth(self):self.assertEqual(basic('ＡＢＣ １２３'),'abc 123')
    def test_multilingual_raw(self):
        for n in ['श्री बालाजी','ஸ்ரீ பாலாஜி','శ్రీ బాలాజీ','শ্রী বালাজী']:
            v=views(n,'','India');self.assertEqual(v['name_raw'],n);self.assertTrue(v['name_norm']);self.assertEqual(v['non_ascii_name'],1)
    def test_no_translation_promise(self):self.assertNotEqual(basic('श्री'), '')
    def test_indic_digits(self):self.assertEqual(views('','दुकान १२७','India')['numbers'],['127'])
    def test_address_number_preservation(self):
        v=views('','Shop 00127, Road 36, Unit 108-B / 115-119','')
        self.assertEqual(v['numbers'],['00127','36','108','115','119'])
        self.assertEqual(v['numbers_unpadded'],['127','36','108','115','119'])
    def test_number_conflict_not_erased(self):
        self.assertNotEqual(views('','Shop 127 Road 36','')['address_norm'],views('','Shop 128 Road 36','')['address_norm'])
    def test_address_abbreviations(self):self.assertEqual(views('','12 Main Road','')['address_norm'],'12 main rd')
    def test_states_not_street_terms(self):
        for word in ['arizona','carolina','utah','california','district','door','new','seventh','tenth','florida','floor']:
            self.assertEqual(views('',word,'')['address_norm'],word)
    def test_saint_not_street(self):self.assertNotEqual(views('','Saint Martin','')['address_norm'],views('','Street Martin','')['address_norm'])
    def test_ordinals_preserved(self):self.assertEqual(views('','49th Avenue','')['address_norm'],'49th ave')
    def test_country_variants(self):
        for c in ['US','USA','U.S.','United States']:self.assertEqual(country_key(c),'US')
        self.assertEqual(country_key('  FRANCE '),country_key('France'))
        self.assertEqual(country_key('New Zealand'),'new zealand')
        self.assertNotEqual(country_key('India'),country_key('France'))
    def test_missing_and_literal_na(self):
        self.assertEqual(views('',None,'')['address_missing'],1)
        self.assertEqual(views('','NA','')['address_missing'],0)
        self.assertEqual(views('','null','')['address_norm'],'null')
    def test_format_characters(self):self.assertEqual(basic('abc\u200bdef'),'abc def')
    def test_static_maps_idempotent(self):
        for src in NAME_ALIASES:self.assertEqual(normalized(normalized(src,'')[0],'')[0],normalized(src,'')[0])
        for src in ADDRESS_ALIASES:self.assertEqual(normalized('',normalized('',src)[2])[2],normalized('',src)[2])

if __name__=='__main__':unittest.main(verbosity=2)
