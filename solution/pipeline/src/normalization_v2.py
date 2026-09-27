"""Audited conservative comparison views. Never an identity/merge decision.

Raw Unicode is retained. Learned aliases are quarantined, not silently applied.
Transliteration and punctuation removal are explicitly lossy auxiliary views.
"""
import re
import unicodedata
from unidecode import unidecode

VERSION = 'normalization-audit-v2'
NAME_ALIASES = {'pvt':'private','ltd':'limited','incorporated':'inc','corporation':'corp'}
ADDRESS_ALIASES = {'road':'rd','street':'st','avenue':'ave','boulevard':'blvd',
                   'drive':'dr','lane':'ln','highway':'hwy','apartment':'apt',
                   'apartments':'apt','suite':'ste','building':'bldg'}
LEGAL = frozenset('private limited llp llc inc corp company co opc'.split())
COUNTRY_ALIASES = {'us':'US','usa':'US','u s':'US','u s a':'US',
                   'united states':'US','united states of america':'US'}
POLICY = {'version':VERSION,'learned_aliases':'quarantined',
          'name_aliases':NAME_ALIASES,'address_aliases':ADDRESS_ALIASES,
          'legal_suffix_tokens':sorted(LEGAL),'country_aliases':COUNTRY_ALIASES,
          'state_city_mapping':'none','ordinal_mapping':'none',
          'primary_fields':'raw and Unicode whitespace-normalized text',
          'comparison_fields':'ASCII transliteration, punctuation-separated tokens; auxiliary only'}

def unicode_view(value):
    text=unicodedata.normalize('NFKC', value or '').casefold()
    # Format characters are token boundaries, preventing invisible concatenation.
    text=''.join(' ' if unicodedata.category(c)=='Cf' else c for c in text)
    return ' '.join(text.split())

def basic(value):
    return ' '.join(re.findall(r'[a-z0-9]+',unidecode(unicode_view(value)).replace('&',' and ')))

def country_key(value):
    key=basic(value)
    return COUNTRY_ALIASES.get(key,key)

def phonetic(value):
    s=re.sub(r'([a-z])\1+',r'\1',value)
    for a,b in [('ph','f'),('kh','k'),('sh','s'),('ch','c'),('th','t'),('dh','d'),('bh','b'),('gh','g'),('w','v'),('q','k')]:s=s.replace(a,b)
    return ' '.join(re.sub('[aeiou]','',w) or w for w in s.split())

def views(name,address,country):
    n=' '.join(NAME_ALIASES.get(t,t) for t in basic(name).split())
    tokens=n.split();cut=len(tokens)
    # Keep interior words, honorifics and meaningful common words. Strip only
    # a terminal legal-form run in the auxiliary core, retaining full name.
    while cut and tokens[cut-1] in LEGAL:cut-=1
    core=' '.join(tokens[:cut]) or n
    addr=' '.join(ADDRESS_ALIASES.get(t,t) for t in basic(address).split())
    nums=re.findall(r'\d+',basic(address))
    return {'name_raw':name or '', 'address_raw':address or '', 'country_raw':country or '',
            'name_unicode':unicode_view(name),'address_unicode':unicode_view(address),
            'country_key':country_key(country),'name_norm':n,'name_core':core,
            'address_norm':addr,'phonetic':phonetic(core),
            'legal_suffix':' '.join(tokens[cut:]) if cut else '',
            'numbers':nums,'numbers_unpadded':[str(int(x)) for x in nums],
            'non_ascii_name':int(any(ord(c)>127 for c in (name or ''))),
            'address_missing':int(not bool((address or '').strip()))}

def normalized(name,address,maps=None):
    if maps and any(maps.values()):
        raise ValueError('Learned aliases are quarantined; nonempty maps are forbidden in v2')
    v=views(name,address,'')
    return tuple(v[k] for k in ['name_norm','name_core','address_norm','phonetic','non_ascii_name','address_missing'])
