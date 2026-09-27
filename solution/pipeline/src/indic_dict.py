"""Learn transliterated-token -> English-token mappings from labelled native-script pairs.

Native-script names normalise letter by letter ('ddijittaal paaoy aar praaibhett'), so they
never share tokens with their Latin S1 name ('digital power'). For each labelled pair (anchor
fold < 90 only, so folds 90-99 stay clean) the target's tokens are aligned in order to the
S1 tokens; a mapping is kept when seen >= MIN times and it is the token's usual translation.
Only the training labels are used (no external data).
"""
import sys, json, re, collections
import numpy as np, pandas as pd

MIN, SHARE = 3, 0.6
LEGAL = {'pvt', 'ltd', 'private', 'limited', 'llp', 'llc', 'inc', 'co', 'corp', 'company'}

def skel(w):
    w = re.sub(r'(.)\1+', r'\1', w)
    return w[0] + re.sub(r'[aeiouy]', '', w[1:]) if w else w

def sim(a, b):
    """Cheap similarity of two token skeletons (common prefix + shared consonants)."""
    sa, sb = skel(a), skel(b)
    if not sa or not sb: return 0.0
    common = sum((collections.Counter(sa) & collections.Counter(sb)).values())
    return (common / max(len(sa), len(sb))) + (0.3 if sa[0] == sb[0] else 0)

def align(ts, ss):
    """Order-preserving alignment of target tokens to S1 tokens (small DP on similarity)."""
    n, m = len(ts), len(ss)
    if not n or not m or n > 8 or m > 8: return []
    D = np.zeros((n + 1, m + 1)); B = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = [(D[i-1][j-1] + sim(ts[i-1], ss[j-1]), 'd'), (D[i-1][j], 'u'), (D[i][j-1], 'l')]
            D[i][j], B[i, j] = max(opts)
    out, i, j = [], n, m
    while i > 0 and j > 0:
        b = B[i, j]
        if b == 'd':
            if sim(ts[i-1], ss[j-1]) >= 0.5: out.append((ts[i-1], ss[j-1]))
            i, j = i - 1, j - 1
        elif b == 'u': i -= 1
        else: j -= 1
    return out

def learn(pairs):
    """pairs: iterable of (target_core, s1_core) -> {token: english}."""
    cnt = collections.defaultdict(collections.Counter)
    for t, s in pairs:
        for a, b in align(t.split(), s.split()):
            if a != b: cnt[a][b] += 1
    d = {}
    for a, c in cnt.items():
        b, k = c.most_common(1)[0]; tot = sum(c.values())
        if k >= MIN and k / tot >= SHARE: d[a] = b
    return d

def translate(core, d):
    return ' '.join(w for w in (d.get(t, t) for t in core.split()) if w not in LEGAL)

if __name__ == '__main__':
    assert align('ddijittaal paaoy'.split(), 'digital power'.split()) == [('paaoy', 'power'), ('ddijittaal', 'digital')]
    print('self-check ok')
