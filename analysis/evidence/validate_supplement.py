#!/usr/bin/env python3
"""Independent aggregation and source checks for the new diagnostic tables."""
from pathlib import Path
import json,subprocess,sys,hashlib
import numpy as np
import pandas as pd
from analyze_current_supplement import rank,wilson

HERE=Path(__file__).resolve().parents[1]
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate():
    d=HERE/'data';tail=pd.read_csv(d/'directional/attribution.csv');tt=pd.read_csv(d/'directional/attribution_tasks.csv');length=pd.read_csv(d/'length_strata/summary.csv');lt=pd.read_csv(d/'length_strata/task_auc.csv')
    assert len(tail)==52 and len(tt)==2640 and len(length)==10
    keys=['proxy','alpha','watermark_method'];assert not tail.duplicated(keys).any()
    events=['upper_only','lower_only','both','neither','union']
    close(tail.upper_only+tail.lower_only+tail.both,tail.union)
    close(tail.union+tail.neither,1)
    for r in tail.to_dict('records'):
        if r['watermark_method']=='Human':
            assert r['n_texts']==2360
            for k in events:close(r[k],r[k+'_count']/2360)
            continue
        block=tt[(tt.proxy==r['proxy'])&(tt.alpha==r['alpha'])]
        if r['watermark_method']!='All machine':block=block[block.watermark_method==r['watermark_method']]
        assert len(block)==r['tasks']==(660 if r['watermark_method']=='All machine' else 60)
        for k in events:
            close(block[k]*block.n_machine,block[k+'_count'])
            close(r[k],block[k].mean());assert r[k+'_count']==block[k+'_count'].sum()
    assert tail[(tail.alpha==.01)].both_count.sum()==0
    for proxy in ['OPT','Falcon']:
        with np.load(d/'sensitivity'/('cached_'+proxy+'.npz')) as z:a={k:z[k] for k in z.files}
        pu=rank(a['cal_upper'][:,2],a['query_upper'][:,2],True);pl=rank(a['cal_lower'],a['query_lower'],False)
        meta=pd.DataFrame({'label':a['label'],'source_id':a['source_id'],'generator':a['generator']})
        hi=meta[meta.label==0].sort_values(['source_id','generator']).drop_duplicates('source_id').index.to_numpy()
        assert len(hi)==2360 and (a['label']==1).sum()==129800
        rows=length[length.proxy==proxy];assert rows.n_human.sum()==2360 and rows.n_machine.sum()==129800
        for r in rows.to_dict('records'):
            mask=(a['token_length']>r['lower_exclusive'])&(a['token_length']<=r['upper_inclusive'])
            h=hi[mask[hi]];m=np.flatnonzero(mask&(a['label']==1))
            assert len(h)==r['n_human'] and len(m)==r['n_machine']
            b=lt[(lt.proxy==proxy)&(lt.lower_exclusive==r['lower_exclusive'])];assert len(b)==r['supported_tasks'];close(b.auc.mean(),r['macro_auc'])
            for alpha,tag in [(.01,'a01'),(.05,'a05')]:
                reject=(pu<=alpha/2)|(pl<=alpha/2);fp=int(reject[h].sum())
                assert fp==r['fp_'+tag];close(r['fpr_'+tag],fp/len(h));close([r['fpr_'+tag+'_low'],r['fpr_'+tag+'_high']],wilson(fp,len(h)))
                close(r['power_'+tag],reject[m].mean());assert reject[m].sum()==r['detected_'+tag]
            close(r['human_neff_ratio_median'],np.median(a['neff'][h]/a['token_length'][h]))
    configurations=0;parameter_rows=0
    for stem,keys in [('temperature_mass',['proxy','T','c']),('kappa',['proxy','kappa']),('local_construction',['proxy','T','c'])]:
        s=pd.read_csv(d/'sensitivity'/(stem+'.csv'));t=pd.read_csv(d/'sensitivity'/(stem+'_tasks.csv'));assert not s.duplicated(keys).any();configurations+=len(s);parameter_rows+=len(t)
        for r in s.to_dict('records'):
            b=t
            for key in keys:b=b[b[key]==r[key]]
            assert len(b)==660 and b.dataset.nunique()==12 and b.generator.nunique()==5 and b.watermark_method.nunique()==11
            assert b.n_machine.sum()==129800 and b.groupby(['dataset','generator']).n_human.first().sum()==11800
            close(r['auc'],b.auc.mean());close(r['cvar10'],np.sort(b.auc)[:66].mean());assert r['flips']==(b.auc<.5).sum()
            close(r['unwm_auc'],b[b.watermark_method=='unwatermarked'].auc.mean())
            for tag in ['a01','a05']:close(r['power_'+tag],b['power_'+tag].mean());close(r['fpr_'+tag],r['fp_'+tag]/2360)
    assert configurations==19 and parameter_rows==12540
    for stem,artifact in [('kappa','kappa_measurements.npz'),('local_construction','local_construction_measurements.npz')]:
        r=json.loads((d/'sensitivity'/(stem+'_forward_audit.json')).read_text());m=r['manifest']
        assert r['status']=='PASS' and r['export_sha256']==sha(d/'sensitivity'/artifact)
        assert m['rows']==132960 and sum(x['rows'] for x in r['shards'])==132960
    for name in ['SUPPLEMENT_CACHE_CHECKS.json','KAPPA_CHECKS.json','LOCAL_CONSTRUCTION_CHECKS.json']:assert json.loads((HERE/'evidence/checks'/name).read_text())['status']=='PASS'
    p=subprocess.run([sys.executable,'evidence/render_supplement_tables.py','--check-only'],cwd=HERE,text=True,capture_output=True);assert p.returncode==0,p.stdout+p.stderr
    return {'status':'PASS','source_integrity':'recorded measurement export hashes','historical_code_parity_checked':False,'parameter_configurations':configurations,'parameter_task_rows':parameter_rows,'tail_partition_rows':len(tail),'tail_task_rows':len(tt),'length_strata':len(length),'tables':p.stdout.strip(),'human_denominator':2360,'machine_denominator':129800,'fresh_forward_documents_per_study':132960}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
