#!/usr/bin/env python3
"""Analyze the fixed six-cap grid using paired full-cohort OPT measurements."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_current_supplement import HERE, FIELDS, metrics, rank, sha, write_csv, jsonwrite, ALPHAS

KAPPAS=[1.,1.25,1.5,1.75,2.,3.]

def prepare(output):
    manifest=json.loads((output/'cohort_manifest.json').read_text())
    assert manifest['rows']==132960 and manifest['shards']==7
    assert manifest['protocol_sha256']==sha(HERE/'evidence/SENSITIVITY_PROTOCOL.json')
    assert manifest['script_sha256']==sha(HERE/'evidence/measure_kappa.py')
    fields=FIELDS+['split','cached_upper','cached_lower']
    arrays={k:[] for k in fields};upper=[];neff=[];caps=[];lower=[];audits=[]
    for shard in range(7):
        path=output/('shard_%d.jsonl'%shard)
        audit=json.loads((output/('shard_%d.audit.json'%shard)).read_text())
        assert audit['sha256']==sha(path) and audit['status']=='PASS'
        assert audit['cohort_sha256']==manifest['cohort_sha256'] and audit['script_sha256']==manifest['script_sha256']
        audits.append(audit)
        with path.open() as f:
            for line in f:
                r=json.loads(line)
                assert r['positions']==r['token_length']
                for k in fields:arrays[k].append(r[k])
                upper.append([r['kappas'][str(k)]['upper'] for k in KAPPAS])
                neff.append([r['kappas'][str(k)]['neff'] for k in KAPPAS])
                caps.append([r['kappas'][str(k)]['cap_fraction'] for k in KAPPAS])
                lower.append(r['recomputed_lower'])
    for k in fields:
        dtype=np.int64 if k in ['label','token_length'] else (float if k.startswith('cached_') else str)
        arrays[k]=np.array(arrays[k],dtype=dtype)
    arrays.update(upper=np.array(upper),neff=np.array(neff),cap_fraction=np.array(caps),recomputed_lower=np.array(lower))
    assert len(arrays['row_id'])==len(set(arrays['row_id']))==132960
    dest=HERE/'data/sensitivity/kappa_measurements.npz';np.savez_compressed(dest,**arrays)
    jsonwrite(HERE/'data/sensitivity/kappa_forward_audit.json',{'status':'PASS','manifest':manifest,'shards':audits,'export_sha256':sha(dest),'all_texts_hash_checked':True,'all_token_lengths_matched':True,'scope':'Full OPT cohort; freshly recomputed upper responses, unchanged cached lower responses.'})
    print('Exported full kappa measurements',flush=True)

def analyze():
    with np.load(HERE/'data/sensitivity/kappa_measurements.npz') as z:a={k:z[k] for k in z.files}
    cal=a['split']=='development';test=~cal
    assert cal.sum()==800 and test.sum()==132160 and not a['label'][cal].any()
    q=pd.DataFrame({k:a[k][test] for k in FIELDS});assert q.label.sum()==129800
    pl=rank(a['cached_lower'][cal],a['cached_lower'][test],False)
    summaries=[];taskrows=[];pvalues={}
    for j,k in enumerate(KAPPAS):
        pu=rank(a['upper'][cal,j],a['upper'][test,j],True);pvalues[k]=pu
        r,t=metrics(q,pu,pl,'OPT',{'kappa':k,'T':1.2,'c':.8})
        r['human_neff_ratio_median']=np.median(a['neff'][test,j][q.label==0]/q.loc[q.label==0,'token_length'])
        r['machine_neff_ratio_median']=np.median(a['neff'][test,j][q.label==1]/q.loc[q.label==1,'token_length'])
        r['human_cap_fraction_median']=np.median(a['cap_fraction'][test,j][q.label==0])
        r['machine_cap_fraction_median']=np.median(a['cap_fraction'][test,j][q.label==1])
        summaries.append(r);taskrows.extend(t)
        print('Analyzed kappa',k,flush=True)
    tasks=pd.DataFrame(taskrows)
    by_domain=tasks.groupby(['kappa','dataset']).auc.mean().unstack(0)
    rng=np.random.RandomState(20260913);draws=rng.randint(0,12,size=(10000,12))
    for r in summaries:
        delta=by_domain[r['kappa']].to_numpy()-by_domain[1.5].to_numpy()
        distribution=delta[draws].mean(axis=1)
        r['auc_delta_vs_default']=delta.mean();r['auc_delta_low'],r['auc_delta_high']=np.quantile(distribution,[.025,.975])
    oldpu=rank(a['cached_upper'][cal],a['cached_upper'][test],True)
    oldsummary,_=metrics(q,oldpu,pl,'OPT',{'kappa':1.5})
    main=pd.read_csv(HERE/'data/current/summary.csv').set_index('score')
    np.testing.assert_allclose(oldsummary['auc'],main.loc['Ours_OPT','auc'],atol=1e-12,rtol=0)
    deploy=pd.read_csv(HERE/'data/current/deployment.csv').set_index(['proxy','alpha'])
    audit={'status':'PASS','kappas':KAPPAS,'tasks_per_setting':660,'task_rows':len(taskrows),'calibration_humans':800,'evaluation_humans':2360,'evaluation_machines':129800,'cached_default_metrics':oldsummary,'new_default_metrics':summaries[2],'maximum_upper_forward_difference':float(np.max(abs(a['upper'][:,2]-a['cached_upper']))),'maximum_lower_forward_difference':float(np.max(abs(a['recomputed_lower']-a['cached_lower']))),'lower_source':'Unchanged cached lower response for every cap, including the re-forward default.','main_result_exports_unchanged':True,'protocol_sha256':sha(HERE/'evidence/SENSITIVITY_PROTOCOL.json')}
    for alpha,tag in zip(ALPHAS,['a01','a05']):
        for key in ['power','fpr']:np.testing.assert_allclose(oldsummary[key+'_'+tag],deploy.loc[('OPT',alpha),key],atol=1e-12,rtol=0)
        old=(oldpu<=alpha/2)|(pl<=alpha/2);new=(pvalues[1.5]<=alpha/2)|(pl<=alpha/2)
        for pop,label in [('human',0),('machine',1)]:audit[pop+'_default_rejection_changes_'+tag]=int((old[q.label==label]!=new[q.label==label]).sum())
    write_csv(HERE/'data/sensitivity/kappa.csv',summaries);write_csv(HERE/'data/sensitivity/kappa_tasks.csv',taskrows)
    jsonwrite(HERE/'evidence/checks/KAPPA_CHECKS.json',audit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);args=p.parse_args()
    if args.prepare:prepare(args.prepare)
    analyze()
