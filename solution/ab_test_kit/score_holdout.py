#!/usr/bin/env python3
"""Score a matching_results.tsv on the team's labelled hold-out entities, exactly like the
leaderboard metric (macro F0.5 per Source-1 entity, singletons included), compare it with our
current best entity by entity, and break the errors down so you know what to improve.
Python 3.8+, standard library only.

  python3 score_holdout.py --pred my_holdout_predictions.tsv \
      --truth dataset/train/train_ground_truth.tsv --source1 dataset/train/train_source1.tsv \
      [--source2 dataset/train/train_source2.tsv --source3 dataset/train/train_source3.tsv] \
      [--report report.json] [--examples 10]

With --source2/--source3 the breakdown also covers record types (empty address, non-Latin
name) and prints concrete example records for each error type.

FAIR COMPARISON RULES (otherwise the number is meaningless):
  1. Train nothing on the hold-out entities: drop them, and every S2/S3 record linked to them
     in train_ground_truth.tsv, from your training / tuning / threshold data.
  2. Predict for the hold-out entities against ALL train Source 2 + Source 3 records
     (every record is a possible match or distractor, as on the leaderboard).
"""
import argparse, json, math, random, sys
from collections import defaultdict

BEST = {'US': 0.98987, 'India': 0.98681}  # our best per country on these entities

def read_lists(path, keep=None):
    out = {}
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\n')
            if not line: continue
            p = line.split('\t')
            if keep is None or p[0] in keep: out[p[0]] = set(x for x in (p[1] if len(p) > 1 else '').split(',') if x)
    return out

def read_owners(path, need):
    """record -> true S1 entity, for records in `need` only"""
    out = {}
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            for x in (p[1] if len(p) > 1 else '').split(','):
                if x in need: out[x] = p[0]
    return out

def read_records(path, keep=None):
    """entity_id -> (name, address, country); only ids in `keep` if given"""
    out = {}
    with open(path, encoding='utf-8') as f:
        cols = f.readline().rstrip('\n').split('\t'); ix = {c: i for i, c in enumerate(cols)}
        for line in f:
            p = line.rstrip('\n').split('\t')
            g = lambda c: p[ix[c]] if c in ix and ix[c] < len(p) else ''
            e = g('entity_id')
            if keep is None or e in keep: out[e] = (g('business_name'), g('business_address'), g('country'))
    return out

def f05(P, T):
    if not T: return 1.0 if not P else 0.0
    tp = len(P & T)
    if tp == 0: return 0.0
    p, r = tp / len(P), tp / len(T)
    return 1.25 * p * r / (0.25 * p + r)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True); ap.add_argument('--truth', required=True); ap.add_argument('--source1', required=True)
    ap.add_argument('--source2'); ap.add_argument('--source3')
    ap.add_argument('--holdout', default='holdout_s1_ids.txt')
    ap.add_argument('--best', default='our_best_holdout_predictions.tsv', help='our current best on the same entities (paired comparison)')
    ap.add_argument('--report', help='write all numbers and examples as JSON (for coding agents)')
    ap.add_argument('--examples', type=int, default=10, help='example records printed per error type')
    a = ap.parse_args()
    hold = [l.strip() for l in open(a.holdout) if l.strip()]
    H = set(hold); truth, pred, best = read_lists(a.truth, H), read_lists(a.pred, H), read_lists(a.best, H)
    s1 = read_records(a.source1)
    need = set().union(*(pred.get(e, set()) | truth.get(e, set()) for e in hold))  # only records the hold-out touches
    tgt = {}
    for p in (a.source2, a.source3):
        if p: tgt.update(read_records(p, need))
    owner = read_owners(a.truth, need)   # record -> true S1 entity (a record has at most one)
    rep = {'n_entities': len(hold), 'missing_rows': sum(1 for e in hold if e not in pred)}

    # ---- headline -------------------------------------------------------------------
    scores = defaultdict(list); tp = npred = ntrue = sfm = 0
    for e in hold:
        P, T = pred.get(e, set()), truth.get(e, set())
        s = f05(P, T); scores['ALL'].append(s); scores[s1.get(e, ('', '', '?'))[2]].append(s)
        tp += len(P & T); npred += len(P); ntrue += len(T); sfm += (not T and bool(P))
    res = {k: sum(v) / len(v) for k, v in scores.items()}; n = len(hold)
    rep.update(macro_f05=res['ALL'], by_country={k: v for k, v in res.items() if k != 'ALL'},
               precision=tp / max(npred, 1), recall=tp / max(ntrue, 1), links_predicted=npred, links_true=ntrue,
               singletons=sum(1 for e in hold if not truth.get(e)), singletons_wrongly_linked=sfm)
    print(f'hold-out entities {n:,} (missing rows: {rep["missing_rows"]:,})')
    print(f'macro F0.5  {res["ALL"]:.5f}   (precision {rep["precision"]:.5f}, recall {rep["recall"]:.5f}, singletons wrongly linked {sfm} of {rep["singletons"]:,})')
    for k in ('US', 'India'):
        if k in res: print(f'  {k:6s} {res[k]:.5f}   (our best {BEST[k]:.5f})')

    # ---- paired comparison with our best --------------------------------------------
    d = [f05(pred.get(e, set()), truth.get(e, set())) - f05(best.get(e, set()), truth.get(e, set())) for e in hold]
    mean = sum(d) / n; se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
    better, worse = sum(x > 0 for x in d), sum(x < 0 for x in d)
    verdict = 'BETTER' if mean > 2 * se else ('WORSE' if mean < -2 * se else 'NO_CLEAR_DIFFERENCE')
    rep['vs_best'] = {'mean_gain': mean, 'paired_se': se, 'entities_better': better, 'entities_worse': worse, 'verdict': verdict}
    print(f'vs our best: {mean:+.6f} (paired SE {se:.6f}; entities better {better:,}, worse {worse:,})')
    print({'BETTER': 'VERDICT: BETTER than our best on US/India (> 2 SE). France is not covered here: check it before submitting.',
           'WORSE': 'VERDICT: WORSE than our best on US/India. Do not submit.',
           'NO_CLEAR_DIFFERENCE': 'VERDICT: no clear difference from our best. Not worth a submission on its own.'}[verdict])

    # ---- error budget: how much F0.5 each error type costs --------------------------
    fp_decoy, fp_other, fp_single = {}, {}, {}      # entity -> wrong links, by kind
    fn_elsewhere, fn_nowhere = {}, {}               # missed links: given to another hold-out entity / not predicted anywhere
    linked_to = defaultdict(set)
    for e in hold:
        for x in pred.get(e, set()): linked_to[x].add(e)
    for e in hold:
        P, T = pred.get(e, set()), truth.get(e, set())
        for x in P - T:
            kind = fp_single if not T else (fp_decoy if x not in owner else fp_other)
            kind.setdefault(e, set()).add(x)
        for x in T - P:
            (fn_elsewhere if linked_to.get(x) else fn_nowhere).setdefault(e, set()).add(x)
    base = res['ALL']
    def gain(remove=None, add=None):
        tot = 0.0
        for e in hold:
            P = set(pred.get(e, set()))
            if remove and e in remove: P -= remove[e]
            if add and e in add: P |= add[e]
            tot += f05(P, truth.get(e, set()))
        return tot / n - base
    budget = {
        'false_links_on_singletons': (gain(remove=fp_single), sum(map(len, fp_single.values()))),
        'false_links_to_decoys': (gain(remove=fp_decoy), sum(map(len, fp_decoy.values()))),
        'false_links_to_other_entities_records': (gain(remove=fp_other), sum(map(len, fp_other.values()))),
        'missed_links_given_to_wrong_entity': (gain(add=fn_elsewhere), sum(map(len, fn_elsewhere.values()))),
        'missed_links_not_predicted_at_all': (gain(add=fn_nowhere), sum(map(len, fn_nowhere.values()))),
    }
    rep['error_budget'] = {k: {'f05_gain_if_fixed': g, 'links': c} for k, (g, c) in budget.items()}
    print(f'\nERROR BUDGET (F0.5 gained if each error type were fixed; total loss {1 - base:.5f})')
    for k, (g, c) in sorted(budget.items(), key=lambda kv: -kv[1][0]):
        print(f'  {k:40s} +{g:.5f}   ({c:,} links)')

    # ---- by entity size ---------------------------------------------------------------
    buckets = defaultdict(list)
    for e, x in zip(hold, d):
        k = len(truth.get(e, set())); b = '0 (singleton)' if k == 0 else ('1' if k == 1 else ('2-3' if k <= 3 else ('4-6' if k <= 6 else '7+')))
        buckets[b].append((f05(pred.get(e, set()), truth.get(e, set())), x))
    rep['by_true_link_count'] = {}
    print('\nBY NUMBER OF TRUE LINKS (entities, your F0.5, vs our best)')
    for b in ('0 (singleton)', '1', '2-3', '4-6', '7+'):
        v = buckets.get(b, [])
        if v:
            f = sum(s for s, _ in v) / len(v); g = sum(x for _, x in v) / len(v)
            rep['by_true_link_count'][b] = {'entities': len(v), 'f05': f, 'vs_best': g}
            print(f'  {b:14s} {len(v):7,d}   {f:.5f}   {g:+.5f}')

    # ---- by record type (needs source2/3) ------------------------------------------------
    if tgt:
        def rtype(x):
            nm, ad, _ = tgt.get(x, ('', '', ''))
            a_ = 'no_address' if not ad.strip() or ad.strip().lower() in ('null', 'none') else 'address'
            return a_ + '+' + ('nonlatin_name' if any(ord(c) > 0x24F for c in nm) else 'latin_name')
        rt = defaultdict(lambda: [0, 0, 0, 0])  # true links, found, missed, false links
        for e in hold:
            P, T = pred.get(e, set()), truth.get(e, set())
            for x in T: t_ = rtype(x); rt[t_][0] += 1; rt[t_][1 if x in P else 2] += 1
            for x in P - T: rt[rtype(x)][3] += 1
        rep['by_record_type'] = {k: {'true_links': v[0], 'recall': v[1] / max(v[0], 1), 'missed': v[2], 'false_links': v[3]} for k, v in rt.items()}
        print('\nBY RECORD TYPE (true links, recall, missed, false links)')
        for k, v in sorted(rt.items(), key=lambda kv: -kv[1][2]):
            print(f'  {k:28s} {v[0]:8,d}   recall {v[1] / max(v[0], 1):.4f}   missed {v[2]:6,d}   false {v[3]:6,d}')

        # ---- examples -----------------------------------------------------------------
        if a.examples:
            rng = random.Random(0); rep['examples'] = {}
            def show(e, x):
                n1, a1, _ = s1.get(e, ('', '', '')); n2, a2, _ = tgt.get(x, ('', '', '')); o = owner.get(x)
                return {'s1': e, 's1_name': n1, 's1_address': a1, 'record': x, 'record_name': n2, 'record_address': a2,
                        'record_true_owner': o, 'true_owner_name': s1.get(o, ('',))[0] if o else None}
            for k, dct in (('false_links_to_decoys', fp_decoy), ('false_links_on_singletons', fp_single),
                           ('false_links_to_other_entities_records', fp_other),
                           ('missed_links_given_to_wrong_entity', fn_elsewhere), ('missed_links_not_predicted_at_all', fn_nowhere)):
                pairs = [(e, x) for e, xs in dct.items() for x in xs]
                ex = [show(e, x) for e, x in rng.sample(pairs, min(a.examples, len(pairs)))]
                rep['examples'][k] = ex
                print(f'\nEXAMPLES: {k} ({len(pairs):,} total)')
                for r in ex:
                    extra = f'   [true owner: {r["true_owner_name"]}]' if r['true_owner_name'] and r['record_true_owner'] != r['s1'] else ''
                    print(f'  S1  {r["s1_name"]} | {r["s1_address"]}\n  rec {r["record_name"]} | {r["record_address"]}{extra}')
    if a.report:
        with open(a.report, 'w', encoding='utf-8') as f: json.dump(rep, f, indent=2, ensure_ascii=False)
        print(f'\nreport written to {a.report}')

if __name__ == '__main__':
    sys.exit(main())
