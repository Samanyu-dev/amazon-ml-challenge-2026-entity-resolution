"""Synthetic France v2: real French test S1 as donors; copies and decoys follow generator patterns measured today
(real French test frequencies + train true-copy rates). Writes <out>/test/test_source{1,2,3}.tsv + synth_ground_truth.tsv.
Differences vs v1: real decoy recipe (legal changed + nearby number; same-address siblings; filler + number shift),
made-up names, S3-only aliases, source-specific casing, legal change on true copies only with the same number."""
import sys, os, re, random, unicodedata
import pandas as pd
R = random.Random(11); OUT = sys.argv[1]
T = '/Users/apple/Downloads/student_resource/dataset/test/'
rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, quoting=3, keep_default_na=False)
s1 = rd(T + 'test_source1.tsv'); s1 = s1[s1.country == 'France'].reset_index(drop=True)
LEG = ['SARL', 'SAS', 'SASU', 'EURL', 'SA', 'SCI', 'EI', 'SNC']
LEG_ALT = {'SARL': ['Sarl', 'S.A.R.L.', 'Sàrl'], 'SAS': ['S.A.S.', 'Sas', 'S.A.S'], 'SA': ['S.A.']}
FILL = ['Groupe', 'Holding', 'International', 'Distribution', 'Développement', 'Participations', 'Services', 'Associés']
CAT = ['Club', 'Amicale', 'Comite', 'Ecole', 'Sportive', 'Amis', 'Union', 'Centre', 'Maison', 'Pharmacie', 'Lycee', 'College', 'Fetes', 'Loisirs', 'Societe', 'Federation']
SYL = ['onyx', 'kor', 'delta', 'xylo', 'wex', 'novi', 'brix', 'halo', 'gild', 'umbra', 'faye', 'nex', 'kelo', 'zeph', 'calo', 'aria', 'lum', 'jax', 'cira', 'dova', 'veo', 'nyla', 'tavo', 'mira', 'yuma', 'ecto', 'riza', 'lyra', 'flux', 'pyra', 'belo', 'drex', 'evo', 'vera', 'syn', 'orbi', 'iri', 'avi', 'vio', 'vantage', 'zeta', 'quo']
STREET = [('Rue', ['R.', 'R', 'RUE']), ('Avenue', ['Av.', 'Ave', 'AV']), ('Boulevard', ['Bd', 'Bd.', 'BLVD']), ('Impasse', ['Imp', 'Imp.']), ('Allée', ['All', 'Allee']), ('Place', ['Pl', 'Pl.']), ('Route', ['Rte']), ('Chemin', ['Ch', 'Chem'])]
DEPT = {'Hauts-de-France': ['Nord', 'Pas-de-Calais'], 'Nouvelle-Aquitaine': ['Gironde'], 'Pays de la Loire': ['Loire-Atlantique']}
ACC = {'c': 'ç', 'a': 'à', 'o': 'ô', 'e': 'è', 'i': 'ï', 'u': 'û'}
N_COPIES = [(0, .056), (1, .054), (2, .170), (3, .240), (4, .219), (5, .146), (6, .074), (7, .029), (8, .012)]
def split_legal(name):
    w = name.split(); leg = [x for x in w if x.upper().replace('.', '') in LEG]; core = [x for x in w if x not in leg]
    return (core or w), (leg[0] if leg else '')
def num_edit(a):   # true-copy house-number noise: digit drop / leading zero / bis
    m = re.search(r'\d+', a)
    if not m: return a
    n = m.group(); r = R.random()
    rep = n[:-1] if (r < .35 and len(n) >= 2) else ('0' + n if r < .6 else (n + R.choice([' bis', 'B', ' ter']) if r < .8 else f'N° {n}'))
    return a[:m.start()] + rep + a[m.end():]
def num_shift(a):  # decoy: nearby number
    m = re.search(r'\d+', a)
    if not m: return a
    v = int(m.group()) + R.choice([-1, 1]) * R.randint(1, 12); return a[:m.start()] + str(max(1, v)) + a[m.end():]
def typo(w):
    if len(w) < 4: return w
    i = R.randrange(1, len(w) - 1); k = R.random()
    return w[:i] + w[i + 1:] if k < .4 else (w[:i] + w[i + 1] + w[i] + w[i + 2:] if k < .75 else w[:i] + w[i] + w[i:])
def addr_noise(a, src):
    parts = [p.strip() for p in a.split(',') if p.strip()]
    s = ', '.join(parts)
    for full, abbr in STREET:
        if re.search(rf'\b{full}\b', s, flags=re.I) and R.random() < .45: s = re.sub(rf'\b{full}\b', R.choice(abbr), s, count=1, flags=re.I)
    if R.random() < .06: s = num_edit(s)
    elif R.random() < .06: s = re.sub(r'^\s*\d+\s*,?\s*', '', s, count=1)
    elif R.random() < .09: s = re.sub(r'\b(\d+)\b', lambda m: R.choice(['N° ', 'Nº ', 'No ']) + m.group(1), s, count=1)
    elif R.random() < .025: s = re.sub(r'\b(\d+)\b', lambda m: '(' + m.group(1) + ')', s, count=1)
    if R.random() < .05:
        ws = s.split(); j = R.randrange(len(ws))
        if not re.search(r'\d', ws[j]): ws[j] = typo(ws[j]); s = ' '.join(ws)
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if parts and parts[-1] in DEPT and R.random() < .35: parts[-1] = R.choice(DEPT[parts[-1]])
    elif parts and parts[-1] in DEPT and R.random() < .25: parts = parts[:-1]
    if len(parts) >= 2 and R.random() < .2: R.shuffle(parts)
    s = ', '.join(parts)
    if R.random() < (.35 if src == 2 else .12): s = s.upper()
    return s
def case(n, src):
    r = R.random()
    return n.upper() if r < (.17 if src == 2 else .042) else (n.lower() if r < (.17 if src == 2 else .042) + .055 else n)
def copy_name(name, src):
    core, leg = split_legal(name); r = R.random()
    if r < .027 and len(core) >= 2: return ''.join(re.sub(r'[^A-Za-zÀ-ÿ]', '', w)[:1] for w in core if w.lower() not in ('de', 'la', 'le', 'du', 'des', '&', 'et')).upper()[:4] or name
    if r < .046: return ''.join(R.choice(SYL) for _ in range(R.choice([2, 2, 3]))).title()
    if r < .091: return (''.join(re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode().lower()) for w in core) + '.com')
    if src == 3 and r < .115: return f"{''.join(R.choice(SYL) for _ in range(2)).title()} {R.choice(['t/a', 'dba', 'formerly', 'aka'])} {name}"
    core = core[:]
    if R.random() < .05 and len(core) >= 2: core.pop(R.randrange(len(core)))
    if R.random() < .10: core.append(R.choice(FILL))
    if R.random() < .10: j = R.randrange(len(core)); core[j] = typo(core[j])
    if R.random() < .12:
        j = R.randrange(len(core)); w = core[j]; k = [i for i, ch in enumerate(w.lower()) if ch in ACC]
        if k: i = R.choice(k); core[j] = w[:i] + ACC[w[i].lower()] + w[i + 1:]
    if leg:
        r2 = R.random()
        leg = '' if r2 < .12 else (R.choice(LEG_ALT.get(leg, [leg])) if r2 < .25 else (R.choice([x for x in LEG if x != leg]) if r2 < .40 else leg))
    elif R.random() < .05: leg = R.choice(LEG)
    if leg and R.random() < .10: leg = f'({leg})' if R.random() < .5 else f'[{leg}]'
    out = ([leg] + core) if (leg and R.random() < .10) else (core + ([leg] if leg else []))
    return case(' '.join(out), src)
rows = {2: [], 3: []}
def emit(n, a, src):
    tid = f'S{src}-{700000000 + len(rows[2]) + len(rows[3]) + 1_000_000 * src:09d}'; rows[src].append((tid, n, a, 'France')); return tid
idx = list(range(len(s1))); R.shuffle(idx); hold = set(idx[:int(.15 * len(idx))])
gt = {}
for i in range(len(s1)):
    if i in hold: continue
    k = R.choices([x for x, _ in N_COPIES], [w for _, w in N_COPIES])[0]; ids = []; n2 = n3 = 0
    for _ in range(k):
        src = 3 if R.random() < .516 else 2
        if (src == 2 and n2 >= 5) or (src == 3 and n3 >= 6): src = 5 - src
        if (src == 2 and n2 >= 5) or (src == 3 and n3 >= 6): break
        nm = copy_name(s1.business_name[i], src); ad = '' if R.random() < .048 else addr_noise(s1.business_address[i], src)
        ids.append(emit(nm, ad, src)); n2 += src == 2; n3 += src == 3
    gt[s1.entity_id[i]] = ','.join(ids)
n_true = len(rows[2]) + len(rows[3]); n_dec = int(.374 / .626 * n_true); made = 0
kept = [i for i in range(len(s1)) if i not in hold]; hl = list(hold); addrs = s1.business_address.tolist()
while made < n_dec:
    r = R.random(); src = 3 if R.random() < .516 else 2
    if r < .05:                                   # orphan copy of a held-out company
        h = R.choice(hl); emit(copy_name(s1.business_name[h], src), '' if R.random() < .048 else addr_noise(s1.business_address[h], src), src)
    elif r < .185:                                # decoy recipe: same name, legal changed, nearby number
        i = R.choice(kept); core, leg = split_legal(s1.business_name[i]); nl = R.choice([x for x in LEG if x != leg])
        emit(case(' '.join(core + [nl]) if R.random() < .85 else ' '.join([nl] + core), src), addr_noise(num_shift(s1.business_address[i]), src), src)
    elif r < .455:                                # same-address sibling: category word swapped / other name at that address
        i = R.choice(kept); core, leg = split_legal(s1.business_name[i]); c2 = core[:]
        pos = [j for j, w in enumerate(c2) if w.title() in CAT]
        if pos: j = R.choice(pos); c2[j] = R.choice([c for c in CAT if c != c2[j].title()])
        else: c2 = split_legal(s1.business_name[R.choice(kept)])[0]
        emit(case(' '.join(c2 + ([R.choice(LEG)] if R.random() < .7 else [])), src), addr_noise(s1.business_address[i], src), src)
    elif r < .675:                                # filler appended + number shifted
        i = R.choice(kept); core, leg = split_legal(s1.business_name[i])
        emit(case(' '.join(core + [R.choice(FILL)] + ([leg] if leg else [])), src), addr_noise(num_shift(s1.business_address[i]), src), src)
    else:                                         # unrelated company-style name at a real address with another number
        emit(case(' '.join(split_legal(s1.business_name[R.choice(kept)])[0] + [R.choice(FILL + LEG)]), src), addr_noise(num_shift(R.choice(addrs)), src), src)
    made += 1
os.makedirs(f'{OUT}/test', exist_ok=True); cols = ['entity_id', 'business_name', 'business_address', 'country']
s1.drop(index=list(hold))[cols].to_csv(f'{OUT}/test/test_source1.tsv', sep='\t', index=False)
for src in (2, 3): pd.DataFrame(rows[src], columns=cols).sample(frac=1, random_state=5).to_csv(f'{OUT}/test/test_source{src}.tsv', sep='\t', index=False)
pd.DataFrame({'source1_entity_id': list(gt), 'matched_entity_ids': list(gt.values())}).to_csv(f'{OUT}/synth_ground_truth.tsv', sep='\t', index=False)
print(f'S1 {len(s1) - len(hold):,}; true copies {n_true:,}; decoys {n_dec:,} ({n_dec / (n_true + n_dec):.3f}); S2 {len(rows[2]):,} S3 {len(rows[3]):,}')
