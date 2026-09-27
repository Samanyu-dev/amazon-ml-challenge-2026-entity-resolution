"""Fidelity check: synthetic France (teammate) vs real French test, on every statistic we measured today."""
import sys, re, unicodedata
sys.path.insert(0, '/private/tmp/claude-501/fable_review')
import numpy as np, pandas as pd
SY = sys.argv[1] if len(sys.argv) > 1 else '/private/tmp/claude-501/teammate'
R = '/Users/apple/Downloads/student_resource/dataset/test/'
rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, quoting=3, keep_default_na=False)
real1 = rd(R + 'test_source1.tsv'); real1 = real1[real1.country == 'France']
realt = pd.concat([rd(R + f'test_source{k}.tsv').assign(src=k) for k in (2, 3)]); realt = realt[realt.country == 'France']
syn1 = rd(f'{SY}/test/test_source1.tsv'); synt = pd.concat([rd(f'{SY}/test/test_source{k}.tsv').assign(src=k) for k in (2, 3)])
gt = rd(f'{SY}/synth_ground_truth.tsv'); owner = {x: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in m.split(',') if x}
synt['owner'] = synt.entity_id.map(owner).fillna('')
SYL = ['onyx','kor','delta','xylo','wex','novi','brix','halo','gild','umbra','faye','nex','kelo','zeph','calo','aria','lum','jax','cira','dova','veo','nyla','tavo','mira','yuma','ecto','riza','lyra','flux','pyra','belo','drex','arc','sol','evo','vera','syn','orbi','iri','avi','vio','vantage','zeta','quo','io','x']
def ops(df):
    n, a = df.business_name, df.business_address; nl = n.str.lower().str.replace('1', 'l').str.replace('0', 'o').str.replace('5', 's').str.strip()
    return {'blank address': a.str.strip() == '', 'made-up syllable name': nl.str.fullmatch('(?:' + '|'.join(SYL) + '){2,}'),
            'acronym name (2-4 caps)': n.str.strip().str.fullmatch(r'[A-Z]{2,4}'), 'domain name (.com)': n.str.contains(r'\.c[o0]m\b', regex=True, flags=re.I),
            'alias (dba/aka/t/a/formerly)': n.str.lower().str.contains(r'\b(?:dba|d/b/a|aka|formerly|doing business as|t/a|f/k/a)\b', regex=True),
            'filler word': n.str.contains(r'\b(?:Groupe|Holding|International|Distribution|Développement|Participations|Services|Associés)\W*$', regex=True),
            'legal form in front': n.str.contains(r'^(?:SARL|SAS|SASU|SA|EURL|SCI|SNC|EI|S\.A\.R\.L\.|S\.A\.S\.)\b', regex=True, flags=re.I),
            'bracketed word': n.str.contains(r'[\[(]', regex=True), 'name UPPERCASE': (n.str.upper() == n) & n.str.contains('[A-Z]', regex=True),
            'name lowercase': (n.str.lower() == n) & n.str.contains('[a-z]', regex=True), 'accent in name': n.str.contains(r'[À-ɏ]', regex=True),
            'injected accent (çlub/àmicale)': n.str.contains(r'[çàâôîïûù][a-z]', regex=True, flags=re.I),
            'OCR digit in word': n.str.contains(r'[a-z][015][a-z]', regex=True, flags=re.I), 'address UPPERCASE': (a.str.upper() == a) & a.str.contains('[A-Z]', regex=True),
            'address N°/No': a.str.contains(r'N°|Nº|\bNo\b', regex=True), 'address (43) parens number': a.str.contains(r'\(\d+\)', regex=True),
            'address abbreviated R./Bd/Av': a.str.contains(r'\b(?:R|Bd|Av|Imp|Rte|Pl|All)\.?\s', regex=True), 'address leading zero': a.str.contains(r'\b0\d+', regex=True),
            'address range 12-14': a.str.contains(r'\b\d+\s*-\s*\d+\b', regex=True), 'address department not region': a.str.contains(r'\b(?:Nord|Gironde|Loire-Atlantique|Pas-de-Calais)\s*$', regex=True),
            'address has no region/dept': ~a.str.contains(r'(?:Hauts-de-France|Nouvelle-Aquitaine|Pays de la Loire|Nord|Gironde|Loire-Atlantique|Pas-de-Calais)', regex=True)}
out = {}
for nm, df in (('REAL France S2', realt[realt.src == 2]), ('SYNTH S2', synt[synt.src == 2]), ('REAL France S3', realt[realt.src == 3]), ('SYNTH S3', synt[synt.src == 3])):
    out[nm] = {k: float(v.fillna(False).mean()) for k, v in ops(df).items()}
T = pd.DataFrame(out); T['S2 ratio synth/real'] = T['SYNTH S2'] / T['REAL France S2'].replace(0, np.nan); T['S3 ratio synth/real'] = T['SYNTH S3'] / T['REAL France S3'].replace(0, np.nan)
pd.set_option('display.width', 200); print('== 1. operation frequencies'); print(T.round(4))
print('\n== 2. size/density'); print(f'real France: S1 {len(real1):,}  targets {len(realt):,}  per S1 {len(realt)/len(real1):.3f} | synth: S1 {len(syn1):,} targets {len(synt):,} per S1 {len(synt)/len(syn1):.3f}; synth decoy share {np.mean(synt.owner == ""):.3f}; labelled S1 {len(gt):,}')
c = gt.matched_entity_ids.map(lambda m: len([x for x in m.split(',') if x])); print('synth copies per labelled S1:', (c.value_counts(normalize=True).sort_index().round(3)).to_dict())
print('\n== 3. true-copy rate per operation in SYNTH (train truth for reference in DATASET_STATISTICS.md §7)')
for k, v in ops(synt).items():
    v = v.fillna(False).values
    if v.any(): print(f'  {k:32s} synth true-copy rate {np.mean(synt.owner.values[v] != ""):.3f}  (n={v.sum():,})')
# 4. decoy recipe: unique-exact-name matches, legal form x house number
def core(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower(); w = re.findall(r'[a-z0-9]+', s)
    return ' '.join(x for x in w if x not in {'sarl', 'sas', 'sasu', 'sa', 'eurl', 'ei', 'sci', 'snc', 'france'})
def legal(s):
    m = re.search(r'\b(SARL|SAS|SASU|SA|EURL|EI|SCI|SNC|S\.A\.R\.L\.|S\.A\.S\.)\b', s, flags=re.I); return m.group(1).upper().replace('.', '') if m else ''
def num(x):
    m = re.search(r'\d+', x); return m.group(0).lstrip('0') if m else ''
def recipe(s1, t, lab=None):
    k = s1.business_name.map(core); vc = k.value_counts(); u = s1[k.map(vc) == 1]; look = dict(zip(u.business_name.map(core), zip(u.entity_id, u.business_name, u.business_address)))
    rows = []
    for tid, n, a in zip(t.entity_id, t.business_name, t.business_address):
        h = look.get(core(n))
        if not h or not a: continue
        lt, ls = legal(n), legal(h[1]); nt, ns = num(a), num(h[2])
        L = 'no legal' if not lt else ('legal same' if lt == ls else 'legal DIFF')
        if not nt or not ns: N = 'no number'
        elif nt == ns: N = 'same'
        elif nt in ns or ns in nt: N = 'digit drop'
        else:
            try: N = 'nearby' if abs(int(nt) - int(ns)) <= 20 else 'other'
            except ValueError: N = 'other'
        rows.append((L, N, (lab.get(tid, '') == h[0]) if lab is not None else np.nan))
    return pd.DataFrame(rows, columns=['legal', 'number', 'true'])
print('\n== 4. decoy recipe (unique exact-name matches): share of records by legal x number')
rr = recipe(real1, realt); ss = recipe(syn1, synt, owner)
tab = pd.concat([rr.groupby(['legal', 'number']).size().rename('REAL n'), ss.groupby(['legal', 'number']).size().rename('SYNTH n'), ss.groupby(['legal', 'number'])['true'].mean().rename('SYNTH true rate')], axis=1)
tab['REAL share'] = tab['REAL n'] / tab['REAL n'].sum(); tab['SYNTH share'] = tab['SYNTH n'] / tab['SYNTH n'].sum(); print(tab.round(3))
print('   (train truth: legal DIFF+nearby 2.2% true, legal DIFF+same 98.5%, legal same+digit drop 99.6%)')
# 5. shared addresses among S1 and namesake groups
for nm, s1 in (('real', real1), ('synth', syn1)):
    k = s1.business_name.map(core); a = s1.business_address.str.lower()
    print(f'{nm:6s} S1: in namesake group {np.mean(k.map(k.value_counts()) > 1):.3f}; address shared {np.mean(a.map(a.value_counts()) > 1):.3f}')
