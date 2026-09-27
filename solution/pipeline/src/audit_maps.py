"""Audit learned aliases before they are used in a reproducible run."""
from pathlib import Path
import argparse, json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--work',required=True); a=ap.parse_args()
    p=Path(a.work)/'normalization_maps.json'; m=json.loads(p.read_text())
    original={k:len(v) for k,v in m.items()}
    # Learned address aliases are not safe without a language-aware address
    # parser. Retain them in the audit report but disable their application.
    disabled_address=dict(m.get('address',{})); m['address']={}
    # These two mappings are visibly semantic substitutions rather than typo or
    # transliteration corrections; they are not used in the final map.
    removed={k:m['name'].pop(k) for k in ('stee','sttaar') if k in m['name']}
    report={'original_counts':original,'effective_counts':{k:len(v) for k,v in m.items()},
            'disabled_address_aliases':disabled_address,'removed_name_aliases':removed,
            'policy':'hand-written address aliases only; learned name aliases except audited semantic outliers'}
    (Path(a.work)/'normalization_maps_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    p.write_text(json.dumps(m,ensure_ascii=False,indent=2))
    print(json.dumps({'original':original,'effective':{k:len(v) for k,v in m.items()},'removed_name':removed},indent=2))
if __name__=='__main__': main()
