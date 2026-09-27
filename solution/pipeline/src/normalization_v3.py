"""v3 comparison views: v2 plus dotted-acronym joining, M/s, French legal and
address forms, leading legal/honorific stripping and a separate legal_form.

v2 stays untouched (its hash is part of the v2 generation fingerprint).
"""
import re
from normalization_v2 import unicode_view, basic, country_key, phonetic, NAME_ALIASES, ADDRESS_ALIASES as V2_ADDRESS

VERSION = 'normalization-v3'
LEGAL = frozenset('private public limited llp llc inc corp company co opc sarl sas sasu eurl sa sci snc cie ets etablissements'.split())
HONORIFIC = frozenset('dr shri sri smt mr mrs messrs'.split())
ADDRESS_ALIASES = {**V2_ADDRESS, 'av':'ave', 'bd':'blvd', 'bld':'blvd', 'place':'pl', 'allee':'all',
                   'chemin':'ch', 'impasse':'imp', 'r':'rue'}
MS_FIRM = re.compile(r'(?<![a-z0-9])m\s*/\s*s(?![a-z0-9])')

def join_initials(tokens):
    """a b c -> abc (L.L.C., E.U.R.L.); single letters otherwise kept."""
    out, run = [], []
    for t in tokens + ['']:
        if len(t) == 1 and t.isalpha():
            run.append(t); continue
        out.extend([''.join(run)] if len(run) > 1 else run); run = []
        if t: out.append(t)
    return out

def name_tokens(name):
    # M/s (Messrs, a firm prefix in India) must not become the honorific "ms".
    text = MS_FIRM.sub(' messrs ', unicode_view(name))
    tokens = join_initials(basic(text).split())
    tokens = [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1]]
    return [NAME_ALIASES.get(t, t) for t in tokens]

def is_legal(tokens, i):
    t = tokens[i]
    if t == 'public':  # only "public limited/company", not "Public School"
        return i + 1 < len(tokens) and tokens[i + 1] in ('limited', 'company')
    return t in LEGAL

def canonical_legal(tokens):
    """Back-transliterated legal forms (limittedd, praaivett) at the end of a name.

    Only the trailing run is touched: 'prvt' is also the key of the name Parvati,
    so a private-like token counts only when a limited-like token follows it."""
    out = list(tokens)
    for i in range(len(out) - 1, -1, -1):
        key = phonetic(out[i])
        if key == 'lmtd': out[i] = 'limited'
        elif key == 'prvt' and i + 1 < len(out) and out[i + 1] == 'limited': out[i] = 'private'
        elif out[i] not in LEGAL: break
    return out

def views(name, address, country):
    tokens = canonical_legal(name_tokens(name))
    lo, hi = 0, len(tokens)
    while lo < hi and (tokens[lo] in HONORIFIC or is_legal(tokens, lo)): lo += 1
    while hi > lo and is_legal(tokens, hi - 1): hi -= 1
    n = ' '.join(tokens)
    core = ' '.join(tokens[lo:hi]) or n
    legal = ' '.join(t for i, t in enumerate(tokens) if (i < lo or i >= hi) and is_legal(tokens, i))
    addr = ' '.join(ADDRESS_ALIASES.get(t, t) for t in basic(address).split())
    nums = re.findall(r'\d+', basic(address))
    return {'name_raw': name or '', 'address_raw': address or '', 'country_raw': country or '',
            'name_unicode': unicode_view(name), 'address_unicode': unicode_view(address),
            'country_key': country_key(country), 'name_norm': n, 'name_core': core,
            'address_norm': addr, 'phonetic': phonetic(core), 'legal_form': legal,
            'numbers': nums, 'numbers_unpadded': [str(int(x)) for x in nums],
            'non_ascii_name': int(any(ord(c) > 127 for c in (name or ''))),
            'address_missing': int(not bool((address or '').strip()))}

def normalized(name, address, maps=None):
    if maps and any(maps.values()):
        raise ValueError('Learned aliases are quarantined; nonempty maps are forbidden')
    v = views(name, address, '')
    return tuple(v[k] for k in ['name_norm', 'name_core', 'address_norm', 'phonetic', 'non_ascii_name', 'address_missing'])
