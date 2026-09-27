"""Streaming, fingerprinted ingestion of the supplied challenge files."""
from pathlib import Path
import argparse, json, hashlib, time
import duckdb

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--work',required=True)
    a=ap.parse_args();out=Path(a.work);out.mkdir(parents=True,exist_ok=True)
    db=duckdb.connect(str(out/'records.duckdb'));db.execute("SET memory_limit='3GB'");db.execute('SET threads=4')
    db.execute(f"SET temp_directory='{out.resolve()}/duckdb_tmp'")
    manifest={};t0=time.time()
    for split in ['train','test']:
        for src in [1,2,3]:
            path=Path(a.data_root)/split/f'{split}_source{src}.tsv';table=f'{split}_s{src}'
            exists=db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?",[table]).fetchone()[0]
            if not exists:
                # The organizer files contain literal quote characters in some
                # business names. QUOTE_NONE keeps those bytes as data instead
                # of silently treating them as CSV quoting syntax.
                db.execute(f"CREATE TABLE {table} AS SELECT (row_number() OVER ()-1)::INTEGER AS rid, * FROM read_csv(?, delim='\\t', quote='', header=true, all_varchar=true, nullstr='__IMPOSSIBLE_NULL__', parallel=false)",[str(path)])
            count=db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
            if src==1:
                db.execute(f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS fold INTEGER')
                # SHA-based stable group hash, including exact repeated business records.
                db.execute(f"UPDATE {table} SET fold = (('0x'||substr(sha256(lower(business_name)||'|'||lower(business_address)||'|'||country),1,8))::UBIGINT % 100)::INTEGER WHERE fold IS NULL")
            manifest[table]={'rows':count,'sha256':sha256(path),'file':str(path.relative_to(a.data_root)),
                'countries':dict(db.execute(f'SELECT country,count(*) FROM {table} GROUP BY 1').fetchall()),
                'unique_ids':db.execute(f'SELECT count(DISTINCT entity_id) FROM {table}').fetchone()[0]}
            assert manifest[table]['unique_ids']==count,(table,'duplicate IDs')
            print(table,manifest[table],f'{time.time()-t0:.1f}s',flush=True)
            (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    p=Path(a.data_root)/'train/train_ground_truth.tsv'
    db.execute("CREATE TABLE IF NOT EXISTS truth_raw AS SELECT * FROM read_csv(?, delim='\\t', quote='',header=true,all_varchar=true,nullstr='__IMPOSSIBLE_NULL__',parallel=false)",[str(p)])
    db.execute("CREATE TABLE IF NOT EXISTS links AS SELECT a.rid AS aid, a.fold, t.source1_entity_id, unnest(string_split(t.matched_entity_ids,',')) AS target_id FROM truth_raw t JOIN train_s1 a ON a.entity_id=t.source1_entity_id WHERE length(t.matched_entity_ids)>0")
    audit={}
    audit['truth_rows']=db.execute('SELECT count(*) FROM truth_raw').fetchone()[0]
    audit['links']=db.execute('SELECT count(*) FROM links').fetchone()[0]
    audit['duplicate_links']=db.execute('SELECT count(*) FROM (SELECT aid,target_id,count(*) c FROM links GROUP BY 1,2 HAVING c>1)').fetchone()[0]
    audit['shared_targets']=db.execute('SELECT count(*) FROM (SELECT target_id,count(DISTINCT aid) c FROM links GROUP BY 1 HAVING c>1)').fetchone()[0]
    assert audit['shared_targets']==0,'Shared targets require connected-component splitting before training'
    assert audit['duplicate_links']==0
    audit['missing_references']=db.execute('SELECT count(*) FROM train_s1 a ANTI JOIN truth_raw t ON a.entity_id=t.source1_entity_id').fetchone()[0]
    audit['unknown_references']=db.execute('SELECT count(*) FROM truth_raw t ANTI JOIN train_s1 a ON a.entity_id=t.source1_entity_id').fetchone()[0]
    audit['unknown_targets']=db.execute('SELECT count(*) FROM links l ANTI JOIN (SELECT entity_id FROM train_s2 UNION ALL SELECT entity_id FROM train_s3) t ON l.target_id=t.entity_id').fetchone()[0]
    audit['country_mismatches']=db.execute('SELECT count(*) FROM links l JOIN train_s1 a ON a.rid=l.aid JOIN (SELECT entity_id,country FROM train_s2 UNION ALL SELECT entity_id,country FROM train_s3) t ON l.target_id=t.entity_id WHERE a.country<>t.country').fetchone()[0]
    assert audit['missing_references']==audit['unknown_references']==audit['unknown_targets']==0
    print('LABEL_AUDIT',audit,flush=True);(out/'label_audit.json').write_text(json.dumps(audit,indent=2))
    for src in [2,3]:
        table=f'train_s{src}'
        db.execute(f'CREATE TABLE IF NOT EXISTS {table}_labels AS SELECT t.rid,coalesce(l.aid,-1)::INTEGER AS aid,coalesce(l.fold,-1)::INTEGER AS fold FROM {table} t LEFT JOIN links l ON t.entity_id=l.target_id')
    db.execute('CHECKPOINT');db.close()
    print('INGEST_COMPLETE',time.time()-t0,flush=True)
if __name__=='__main__':main()
