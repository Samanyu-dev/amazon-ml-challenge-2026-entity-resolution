"""Create reviewable audit evidence; does not regenerate or promote a corpus."""
from pathlib import Path
import argparse,csv,hashlib,json,collections,importlib.metadata,random
import duckdb
from normalization_v2 import views,POLICY

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifacts',required=True);ap.add_argument('--out',required=True);ap.add_argument('--data-root',required=True);a=ap.parse_args()
    art=Path(a.artifacts);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((art/'normalization_maps_audit.json').read_text());old=json.loads((art/'normalization_maps.json').read_text())
    old['name'].update(prior['removed_name_aliases']);old['address']=prior['disabled_address_aliases']
    bad={'arizona','carolina','district','door','new','seventh','tenth','utah','arcade','indiana','malkajgiri','twenty'}
    with (out/'alias_dispositions.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['field','source','target','decision','reason','cycle'])
        for field,mapping in old.items():
            for src,dst in sorted(mapping.items()):
                cycle=mapping.get(dst)==src
                reason='No saved token-alignment evidence, negatives or independent validation; not approved for substitution'
                if field=='address' and src in bad:reason='Different address concepts or locations; unsafe co-occurrence substitution'
                if cycle:reason='Bidirectional alias cycle; canonical target not uniquely defined'
                if src=='sttaar':reason='Plausible transliteration of star; previous removal was not evidence of incorrectness; still unvalidated'
                w.writerow([field,src,dst,'QUARANTINE',reason,int(cycle)])
    # Samples drawn from each original source with literal TSV parsing. Include
    # non-ASCII, numeric compounds, suspicious-map tokens and quotes separately.
    samples=[];raw_meta={};manifest=json.loads((art/'data_manifest.json').read_text());rng=random.Random(42)
    for split in ['train','test']:
        for source in [1,2,3]:
            table=f'{split}_s{source}';p=Path(a.data_root)/split/f'{split}_source{source}.tsv';reservoir=[];special={};count=0
            with p.open(newline='') as f:
                reader=csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
                assert reader.fieldnames==['entity_id','business_name','business_address','country']
                for rid,row in enumerate(reader):
                    assert None not in row and all(v is not None for v in row.values()),(table,rid)
                    count+=1;item=(table,rid,row)
                    if len(reservoir)<8:reservoir.append(item)
                    else:
                        pos=rng.randrange(count)
                        if pos<8:reservoir[pos]=item
                    tags=[];n=row['business_name'];ad=row['business_address']
                    if any(ord(c)>127 for c in n):tags.append('non_ascii')
                    if not ad.strip():tags.append('empty_address')
                    if any(t in ad.lower().split() for t in bad):tags.append('suspect_address_token')
                    if '"' in n or '"' in ad:tags.append('literal_quote')
                    if '/' in ad or '-' in ad:tags.append('compound_address')
                    for tag in tags:
                        if len(special.setdefault(tag,[]))<2:special[tag].append(item)
            digest=sha(p);assert count==manifest[table]['rows'];assert digest==manifest[table]['sha256']
            raw_meta[table]={'rows':count,'sha256':digest,'country_counts':manifest[table]['countries']}
            samples.extend(reservoir)
            for vals in special.values():samples.extend(vals)
            print('RAW_VERIFIED',table,count,flush=True)
    db=duckdb.connect(str(art/'records.duckdb'),read_only=True);rows=[];mismatches=[]
    for table,rid,r in samples:
        dbrow=db.execute(f'SELECT entity_id,business_name,business_address,country FROM {table} WHERE rid=?',[rid]).fetchone()
        if tuple(r.values())!=dbrow:mismatches.append({'table':table,'rid':rid,'id':r['entity_id'],'file':list(r.values()),'duckdb':list(dbrow)})
        v=views(r['business_name'],r['business_address'],r['country']);v={'table':table,'rid':rid,'entity_id':r['entity_id'],**v}
        v['numbers']='|'.join(v['numbers']);v['numbers_unpadded']='|'.join(v['numbers_unpadded']);rows.append(v)
    db.close()
    with (out/'normalization_samples.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    config=json.dumps(POLICY,sort_keys=True,separators=(',',':'));(out/'normalization_policy.json').write_text(json.dumps(POLICY,indent=2))
    provenance={'status':'AUDIT_ONLY_NOT_CORPUS_RELEASE','original_aliases':{k:len(v) for k,v in old.items()},'approved_learned_aliases':0,
                'raw_sources':raw_meta,'normalization_code_sha256':sha(Path(__file__).with_name('normalization_v2.py')),
                'normalization_config_sha256':hashlib.sha256(config.encode()).hexdigest(),'git_commit':None,
                'git_note':'Local submission directory is not a git repository; immutable source hash used instead',
                'unidecode_version':importlib.metadata.version('Unidecode'),'sample_count':len(rows),'sample_ingestion_mismatches':mismatches,
                'normalized_corpus_hashes':None,'reason':'No full corpus regeneration authorized through the audit gate yet'}
    (out/'audit_provenance.json').write_text(json.dumps(provenance,indent=2))
    print('AUDIT_EVIDENCE_COMPLETE',json.dumps({'samples':len(rows),'ingestion_mismatches':len(mismatches)}),flush=True)
if __name__=='__main__':main()
