#!/usr/bin/env python3
"""Reproducible directional attribution, construction and length diagnostics."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parents[1]
ENDPOINTS=[(1.,.8),(1.,1.),(1.2,.8),(1.2,1.)]
DEFAULT=2
ALPHAS=[.01,.05]
FIELDS=['row_id','source_id','dataset','generator','watermark_method','label','token_length']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank(bank,x,upper):
    b=np.sort(bank)
    count=len(b)-np.searchsorted(b,x,side='left') if upper else np.searchsorted(b,x,side='right')
    return (count+1.)/(len(b)+1.)
def response(N,V):return np.asarray(N)/np.maximum(np.sqrt(np.maximum(V,0)),1e-9)
def human_indices(q):return q[q.label==0].sort_values(['dataset','source_id','generator']).drop_duplicates(['dataset','source_id']).index.to_numpy()
def tasks(q):
    # A human bank is shared across generators if forwards were deduplicated.
    human=q[q.label==0]
    by_cell={key:h.index.to_numpy() for key,h in human.groupby(['dataset','generator'],sort=True)}
    by_domain={ds:h.sort_values(['source_id','generator']).drop_duplicates('source_id').index.to_numpy() for ds,h in human.groupby('dataset',sort=True)}
    for (ds,gen,wm),m in q[q.label==1].groupby(['dataset','generator','watermark_method'],sort=True):
        hi=by_cell.get((ds,gen),by_domain[ds])
        yield ds,gen,wm,hi,m.index.to_numpy()
def write_csv(path,rows):pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g')
def jsonwrite(path,r):Path(path).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')

def prepare(project,proxy):
    base=project/'results/runs/calibrated_scores'
    sources=[base/('calibration_'+proxy+'.pkl'),base/'test'/proxy/'scoring_frame.pkl']
    cal=pd.read_pickle(sources[0]);cal=cal.loc[human_indices(cal)].reset_index(drop=True)
    q=pd.read_pickle(sources[1]).reset_index(drop=True)
    assert (len(cal),len(q),len(human_indices(q)))==(800,141600,2360)
    cu=np.full((800,4),np.nan);qu=np.full((len(q),4),np.nan);neff=np.full(len(q),np.nan)
    cmap={v:i for i,v in enumerate(cal.row_id)};qmap={v:i for i,v in enumerate(q.row_id)}
    audit={}
    for split,lookup,frame,upper in [('development',cmap,cal,cu),('test',qmap,q,qu)]:
        hashes=frame.text_sha256.to_numpy();lengths=frame.token_length.to_numpy()
        paths=sorted((project/'results/runs/reference_shaping'/split/proxy).glob('*.jsonl'))
        assert len(paths)==(20 if split=='development' else 60)
        for p in paths:
            audit[str(p.relative_to(project))]=sha(p)
            with p.open() as f:
                for line in f:
                    r=json.loads(line);idx=lookup.get(r['row_id'])
                    if idx is None:continue
                    assert r['text_sha256']==hashes[idx]
                    assert r['token_length']==lengths[idx]
                    for j,(T,c) in enumerate(ENDPOINTS):
                        key='T%s_P%s'%(str(T).replace('.','p'),str(c).replace('.','p'))
                        s=r['stats']['entropy']['endpoints'][key]
                        upper[idx,j]=response(s['N'],s['V'])
                    if split=='test':neff[idx]=r['stats']['entropy']['neff']
    # Match the main evaluation's source-level human canonicalization exactly.
    # Repeated forwards can differ slightly with FP16 batch composition.
    canonical=dict(zip(q.loc[human_indices(q),'source_id'],human_indices(q)))
    hh=np.flatnonzero(q.label.to_numpy()==0)
    origin=np.array([canonical[s] for s in q.loc[hh,'source_id']])
    qu[hh]=qu[origin];neff[hh]=neff[origin]
    for f,U in [(cal,cu),(q,qu)]:
        assert np.isfinite(U).all()
        np.testing.assert_allclose(U[:,DEFAULT],response(f.entropy_N1,f.entropy_V1),atol=1e-12,rtol=0)
    assert np.isfinite(neff).all()
    data={k:q[k].to_numpy(dtype=np.int64 if k in ['label','token_length'] else str) for k in FIELDS}
    data.update(cal_upper=cu,query_upper=qu,cal_lower=response(cal.uniform_N0,cal.uniform_V0),query_lower=response(q.uniform_N0,q.uniform_V0),neff=neff)
    dest=HERE/'data/sensitivity'/('cached_'+proxy+'.npz');np.savez_compressed(dest,**data)
    audit.update({str(p.relative_to(project)):sha(p) for p in sources})
    jsonwrite(HERE/'data/sensitivity'/('cache_audit_'+proxy+'.json'),{'source_sha256':audit,'export_sha256':sha(dest),'default_response_parity':'absolute error <= 1e-12','calibration':800,'rows':len(q)})
    print('Prepared',proxy,flush=True)

def metrics(q,pu,pl,proxy,setting):
    score=-np.log(np.minimum(pu/.5,pl/.5));rows=[]
    for ds,gen,wm,hi,mi in tasks(q):
        assert len(hi)==len(mi), (ds,gen,wm,len(hi),len(mi))
        labels=np.r_[np.zeros(len(hi)),np.ones(len(mi))]
        r={'proxy':proxy,**setting,'dataset':ds,'generator':gen,'watermark_method':wm,'n_human':len(hi),'n_machine':len(mi),'auc':roc_auc_score(labels,np.r_[score[hi],score[mi]])}
        for alpha,tag in zip(ALPHAS,['a01','a05']):r['power_'+tag]=float(((pu[mi]<=alpha/2)|(pl[mi]<=alpha/2)).mean())
        rows.append(r)
    d=pd.DataFrame(rows);assert len(d)==660
    hi=human_indices(q);a=d.auc.to_numpy()
    summary={'proxy':proxy,**setting,'auc':a.mean(),'unwm_auc':d[d.watermark_method=='unwatermarked'].auc.mean(),'cvar10':np.sort(a)[:66].mean(),'flips':int((a<.5).sum())}
    for alpha,tag in zip(ALPHAS,['a01','a05']):
        fp=int(((pu[hi]<=alpha/2)|(pl[hi]<=alpha/2)).sum())
        summary.update({'power_'+tag:d['power_'+tag].mean(),'fp_'+tag:fp,'fpr_'+tag:fp/len(hi)})
    return summary,rows

def wilson(k,n):
    if not n:return math.nan,math.nan
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)

def attribution(q,pu,pl,proxy):
    rows=[];summary=[]
    for alpha in ALPHAS:
        U=pu<=alpha/2;L=pl<=alpha/2;events={'upper_only':U&~L,'lower_only':L&~U,'both':U&L,'neither':~U&~L,'union':U|L}
        for ds,gen,wm,hi,mi in tasks(q):
            r={'proxy':proxy,'alpha':alpha,'dataset':ds,'generator':gen,'watermark_method':wm,'n_machine':len(mi)}
            for k,mask in events.items():r[k]=float(mask[mi].mean());r[k+'_count']=int(mask[mi].sum())
            rows.append(r)
        block=pd.DataFrame([r for r in rows if r['alpha']==alpha])
        for wm in ['All machine']+sorted(block.watermark_method.unique()):
            d=block if wm=='All machine' else block[block.watermark_method==wm]
            r={'proxy':proxy,'alpha':alpha,'watermark_method':wm,'denominator':'task-macro','n_texts':int(d.n_machine.sum()),'tasks':len(d)}
            for k in events:r[k]=d[k].mean();r[k+'_count']=int(d[k+'_count'].sum())
            summary.append(r)
        hi=human_indices(q);r={'proxy':proxy,'alpha':alpha,'watermark_method':'Human','denominator':'unique-human','n_texts':len(hi),'tasks':0}
        for k,mask in events.items():r[k]=float(mask[hi].mean());r[k+'_count']=int(mask[hi].sum())
        summary.append(r)
    d=pd.DataFrame(summary)
    np.testing.assert_allclose(d.upper_only+d.lower_only+d.both,d.union,atol=1e-14,rtol=0)
    np.testing.assert_allclose(d.union+d.neither,1,atol=1e-14,rtol=0)
    return summary,rows

def length_strata(q,pu,pl,neff,proxy):
    score=-np.log(np.minimum(pu/.5,pl/.5));rows=[];taskrows=[]
    edges=[0,64,128,256,512,2047];unique=human_indices(q);tasklist=list(tasks(q));n=q.token_length.to_numpy()
    for lo,up in zip(edges[:-1],edges[1:]):
        inside=(n>lo)&(n<=up);hi=unique[inside[unique]];mi=np.flatnonzero(inside&(q.label.to_numpy()==1))
        aucs=[]
        for ds,gen,wm,th,tm in tasklist:
            h=th[inside[th]];m=tm[inside[tm]]
            if not len(h) or not len(m):continue
            auc=roc_auc_score(np.r_[np.zeros(len(h)),np.ones(len(m))],np.r_[score[h],score[m]])
            aucs.append(auc);taskrows.append({'proxy':proxy,'lower_exclusive':lo,'upper_inclusive':up,'dataset':ds,'generator':gen,'watermark_method':wm,'n_human':len(h),'n_machine':len(m),'auc':auc})
        r={'proxy':proxy,'lower_exclusive':lo,'upper_inclusive':up,'n_human':len(hi),'n_machine':len(mi),'supported_tasks':len(aucs),'macro_auc':float(np.mean(aucs)) if aucs else math.nan,'human_neff_ratio_median':float(np.median(neff[hi]/n[hi])) if len(hi) else math.nan,'machine_neff_ratio_median':float(np.median(neff[mi]/n[mi])) if len(mi) else math.nan}
        for alpha,tag in zip(ALPHAS,['a01','a05']):
            reject=(pu<=alpha/2)|(pl<=alpha/2);fp=int(reject[hi].sum());low,high=wilson(fp,len(hi))
            r.update({'fp_'+tag:fp,'fpr_'+tag:fp/len(hi) if len(hi) else math.nan,'fpr_'+tag+'_low':low,'fpr_'+tag+'_high':high,'detected_'+tag:int(reject[mi].sum()),'power_'+tag:float(reject[mi].mean()) if len(mi) else math.nan})
        rows.append(r)
    assert sum(r['n_human'] for r in rows)==2360 and sum(r['n_machine'] for r in rows)==129800
    return rows,taskrows

def analyze():
    params=[];paramtasks=[];tails=[];tailtasks=[];length=[];lengthtasks=[]
    expected=pd.read_csv(HERE/'data/current/summary.csv').set_index('score')
    deploy=pd.read_csv(HERE/'data/current/deployment.csv').set_index(['proxy','alpha'])
    for proxy in ['OPT','Falcon']:
        with np.load(HERE/'data/sensitivity'/('cached_'+proxy+'.npz')) as z:a={k:z[k] for k in z.files}
        q=pd.DataFrame({k:a[k] for k in FIELDS});pl=rank(a['cal_lower'],a['query_lower'],False)
        for j,(T,c) in enumerate(ENDPOINTS):
            pu=rank(a['cal_upper'][:,j],a['query_upper'][:,j],True)
            r,ts=metrics(q,pu,pl,proxy,{'T':T,'c':c,'kappa':1.5});params.append(r);paramtasks.extend(ts)
            if j==DEFAULT:
                np.testing.assert_allclose(r['auc'],expected.loc['Ours_'+proxy,'auc'],atol=1e-12,rtol=0)
                for alpha,tag in zip(ALPHAS,['a01','a05']):
                    for k in ['power','fpr']:np.testing.assert_allclose(r[k+'_'+tag],deploy.loc[(proxy,alpha),k],atol=1e-12,rtol=0)
                s,t=attribution(q,pu,pl,proxy);tails.extend(s);tailtasks.extend(t)
                s,t=length_strata(q,pu,pl,a['neff'],proxy);length.extend(s);lengthtasks.extend(t)
        print('Analyzed',proxy,flush=True)
    for folder,filename,rows in [('sensitivity','temperature_mass',params),('sensitivity','temperature_mass_tasks',paramtasks),('directional','attribution',tails),('directional','attribution_tasks',tailtasks),('length_strata','summary',length),('length_strata','task_auc',lengthtasks)]:write_csv(HERE/'data'/folder/(filename+'.csv'),rows)
    jsonwrite(HERE/'evidence/checks/SUPPLEMENT_CACHE_CHECKS.json',{'status':'PASS','default_main_parity':'<=1e-12','parameter_settings':8,'parameter_task_rows':len(paramtasks),'tail_partition_rows':len(tails),'tail_task_rows':len(tailtasks),'length_strata':len(length),'length_supported_task_rows':len(lengthtasks),'unique_humans_per_proxy':2360,'machines_per_proxy':129800,'protocol_sha256':sha(HERE/'evidence/SENSITIVITY_PROTOCOL.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);args=p.parse_args()
    if args.prepare:
        for proxy in ['OPT','Falcon']:prepare(args.prepare,proxy)
    analyze()
