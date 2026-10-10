#!/usr/bin/env python3
"""Full-cohort local T/c sensitivity with the complete two-sided test."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_current_supplement import HERE,FIELDS,metrics,rank,sha,write_csv,jsonwrite

SETTINGS=[(1.1,.8),(1.2,.7),(1.2,.8),(1.2,.9),(1.3,.8)]

def prepare(output):
    manifest=json.loads((output/'cohort_manifest.json').read_text())
    assert manifest['rows']==132960 and manifest['shards']==7
    assert manifest['protocol_sha256']==sha(HERE/'evidence/LOCAL_CONSTRUCTION_PROTOCOL.json')
    assert manifest['script_sha256']==sha(HERE/'evidence/measure_construction.py')
    fields=FIELDS+['split','cached_upper','cached_lower'];a={k:[] for k in fields};upper=[];audits=[]
    for shard in range(7):
        path=output/('shard_%d.jsonl'%shard);audit=json.loads((output/('shard_%d.audit.json'%shard)).read_text())
        assert audit['status']=='PASS' and audit['sha256']==sha(path)
        assert audit['cohort_sha256']==manifest['cohort_sha256'] and audit['script_sha256']==manifest['script_sha256']
        audits.append(audit)
        with path.open() as f:
            for line in f:
                r=json.loads(line);assert r['positions']==r['token_length']
                for k in fields:a[k].append(r[k])
                upper.append([r['settings'][str(T)+'_'+str(c)]['upper'] for T,c in SETTINGS])
    for k in fields:a[k]=np.array(a[k],dtype=np.int64 if k in ['label','token_length'] else (float if k.startswith('cached_') else str))
    a['upper']=np.array(upper);assert len(a['row_id'])==len(set(a['row_id']))==132960
    dest=HERE/'data/sensitivity/local_construction_measurements.npz';np.savez_compressed(dest,**a)
    jsonwrite(HERE/'data/sensitivity/local_construction_forward_audit.json',{'status':'PASS','manifest':manifest,'shards':audits,'export_sha256':sha(dest),'all_texts_hash_checked':True,'all_token_lengths_matched':True})
    print('Exported local construction measurements',flush=True)

def analyze():
    with np.load(HERE/'data/sensitivity/local_construction_measurements.npz') as z:a={k:z[k] for k in z.files}
    cal=a['split']=='development';test=~cal;assert cal.sum()==800 and test.sum()==132160
    q=pd.DataFrame({k:a[k][test] for k in FIELDS});pl=rank(a['cached_lower'][cal],a['cached_lower'][test],False)
    summaries=[];taskrows=[]
    for j,(T,c) in enumerate(SETTINGS):
        pu=rank(a['upper'][cal,j],a['upper'][test,j],True)
        r,t=metrics(q,pu,pl,'OPT',{'T':T,'c':c,'kappa':1.5})
        summaries.append(r);taskrows.extend(t);print('Analyzed local',T,c,flush=True)
    d=pd.DataFrame(taskrows);domains=d.groupby(['T','c','dataset']).auc.mean().unstack(['T','c'])
    draws=np.random.RandomState(20260913).randint(0,12,size=(10000,12))
    for r in summaries:
        delta=domains[r['T'],r['c']].to_numpy()-domains[1.2,.8].to_numpy()
        r['auc_delta_vs_default']=delta.mean();r['auc_delta_low'],r['auc_delta_high']=np.quantile(delta[draws].mean(axis=1),[.025,.975])
    with np.load(HERE/'data/sensitivity/kappa_measurements.npz') as z:
        origin=pd.Series(z['upper'][:,2],index=z['row_id']).reindex(a['row_id']).to_numpy()
    parity=float(abs(origin-a['upper'][:,2]).max())
    write_csv(HERE/'data/sensitivity/local_construction.csv',summaries);write_csv(HERE/'data/sensitivity/local_construction_tasks.csv',taskrows)
    jsonwrite(HERE/'evidence/checks/LOCAL_CONSTRUCTION_CHECKS.json',{'status':'PASS','settings':SETTINGS,'tasks_per_setting':660,'task_rows':len(taskrows),'calibration_humans':800,'evaluation_humans':2360,'evaluation_machines':129800,'maximum_default_difference_from_kappa_forwards':parity,'main_result_exports_unchanged':True,'protocol_sha256':sha(HERE/'evidence/LOCAL_CONSTRUCTION_PROTOCOL.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);args=p.parse_args()
    if args.prepare:prepare(args.prepare)
    analyze()
