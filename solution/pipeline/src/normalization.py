"""Multiple complementary text representations; learned maps use train labels only."""
import re,unicodedata,json
from unidecode import unidecode
TOKEN=re.compile(r'[a-z0-9]+')
SUFFIX=set('private pvt limited ltd llp llc inc incorporated corporation corp company co proprietorship partnership opc ms mr mrs m s the and of for at by'.split())
NAME_ALIASES={'pvt':'private','ltd':'limited','incorporated':'inc','corporation':'corp','corporate':'corp','co':'company','pr':'private','limittedd':'limited','limittd':'limited','praaivett':'private'}
ADDR_ALIASES={
 'road':'rd','street':'st','avenue':'ave','boulevard':'blvd','drive':'dr','lane':'ln','highway':'hwy','apartment':'apt','apartments':'apt','floor':'fl','suite':'ste','building':'bldg','number':'no','opposite':'opp','opposit':'opp','nearby':'near',
 'maharashtra':'mh','delhi':'dl','dillii':'dl','karnataka':'ka','rajasthan':'rj','gujarat':'gj','bihar':'br','telangana':'ts','haryana':'hr','punjab':'pb','kerala':'kl','tamilnadu':'tn','uttarpradesh':'up','westbengal':'wb',
 'california':'ca','texas':'tx','florida':'fl','utah':'ut','missouri':'mo','wisconsin':'wi','arizona':'az','illinois':'il','ohio':'oh','oregon':'or','georgia':'ga','colorado':'co','nevada':'nv','virginia':'va','pennsylvania':'pa','michigan':'mi','washington':'wa','tennessee':'tn','massachusetts':'ma','alabama':'al','alaska':'ak','arkansas':'ar','connecticut':'ct','delaware':'de','hawaii':'hi','idaho':'id','indiana':'in','iowa':'ia','kansas':'ks','kentucky':'ky','louisiana':'la','maine':'me','maryland':'md','minnesota':'mn','mississippi':'ms','montana':'mt','nebraska':'ne','oklahoma':'ok','vermont':'vt','wyoming':'wy',
 'rue':'rue','avenue':'ave','boulevard':'blvd','chemin':'chem','route':'rte','saint':'st','sainte':'ste',
 'first':'1st','second':'2nd','third':'3rd','fourth':'4th','fifth':'5th','sixth':'6th','seventh':'7th','eighth':'8th','ninth':'9th','tenth':'10th',
}
MULTI=[('new york','ny'),('new jersey','nj'),('new mexico','nm'),('new hampshire','nh'),('north carolina','nc'),('south carolina','sc'),('north dakota','nd'),('south dakota','sd'),('rhode island','ri'),('west virginia','wv'),('uttar pradesh','up'),('madhya pradesh','mp'),('west bengal','wb'),('tamil nadu','tn'),('andhra pradesh','ap'),('himachal pradesh','hp'),('pvt ltd','private limited')]
MULTIMAP=dict(MULTI)
MULTIRE=re.compile(r'\b(?:'+'|'.join(re.escape(a) for a in MULTIMAP)+r')\b')

def basic(text):
    t=unidecode(unicodedata.normalize('NFKC',text or '')).casefold()
    t=t.replace('&',' and ')
    return ' '.join(TOKEN.findall(t))

def phonetic(text):
    text=re.sub(r'([a-z])\1+',r'\1',text)
    text=text.replace('ph','f').replace('kh','k').replace('sh','s').replace('ch','c').replace('th','t').replace('dh','d').replace('bh','b').replace('gh','g').replace('w','v').replace('q','k')
    return ' '.join(re.sub('[aeiou]','',w) or w for w in text.split())

def normalized(name,address,maps=None):
    raw_n=basic(name);raw_a=basic(address)
    maps=maps or {}
    nt=[NAME_ALIASES.get(t,t) for t in raw_n.split()]
    nt=[maps.get('name',{}).get(t,t) for t in nt]
    name_norm=' '.join(nt)
    core=' '.join(t for t in nt if t not in SUFFIX)
    core=core or name_norm
    # Domain-like records retain their compact brand view without external lookup.
    core=re.sub(r'\b(?:www|https|http|com|org|net|in)\b',' ',core) if '.' in (name or '') and ('com' in nt or 'www' in nt) else core
    core=' '.join(core.split()) or name_norm
    raw_a=MULTIRE.sub(lambda m:MULTIMAP[m.group()],raw_a)
    # Learned address aliases are intentionally disabled.  Positive-pair token
    # co-occurrence can confuse a state/city/street token with an unrelated
    # address word (for example ``arizona -> road``).  Only the audited,
    # language-generic table above is applied to addresses.
    at=[ADDR_ALIASES.get(t,t) for t in raw_a.split()]
    addr=' '.join(at)
    return name_norm,core,addr,phonetic(core),int(any(ord(c)>127 for c in (name or ''))),int(not bool((address or '').strip()))
