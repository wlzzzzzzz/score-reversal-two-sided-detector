#!/usr/bin/env python3
"""Fixed-current shaping analysis: reference corrections, calibration, joint ranks."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from analyze_current_supplement import rank,tasks

PAPER=Path(__file__).resolve().parents[1]
ROOT=PAPER.parent
BASE=ROOT/'results/runs/calibrated_scores'
OUT=PAPER/'data/reference_calibration'
POSITIVE=['KGW','SWEET','Unigram','EXP','SynthID','DIP']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name,rows):pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,float_format='%.17g')


def credit(h,m):return (m[:,None]>h[None,:]).astype(float)+.5*(m[:,None]==h[None,:])


def analytic_checks():
    rng=np.random.RandomState(20260916);errors=[]
    for same in [True,False]:
        for _ in range(20):
            z=rng.normal(size=(7,13));ell=np.log(softmax(z if same else rng.normal(size=z.shape),axis=1))
            w=rng.dirichlet(np.ones(7));obs=ell[np.arange(7),rng.randint(13,size=7)]
            T=1.2;q=softmax(z/T,axis=1)
            mu=(q*ell).sum(1);center=ell-mu[:,None];nu=(q*center**2).sum(1)
            zm=(q*z).sum(1)
            cov1=(q*center*(z-zm[:,None])).sum(1)
            cov2=(q*(center**2-nu[:,None])*(z-zm[:,None])).sum(1)
            N=w@(obs-mu);V=w**2@nu;R=N/np.sqrt(V)
            derivative=(w@cov1)/(T*T*np.sqrt(V))+R*(w**2@cov2)/(2*T*T*V)
            def response(temp):
                p=softmax(z/temp,axis=1);m=(p*ell).sum(1);v=(p*(ell-m[:,None])**2).sum(1)
                return (w@(obs-m))/np.sqrt(w**2@v)
            numeric=(response(T+1e-5)-response(T-1e-5))/2e-5
            errors.append(abs(numeric-derivative));assert abs(numeric-derivative)<1e-7
    # Inclusive ties and a common strictly increasing transformation preserve ranks.
    b=np.array([-1.,0.,0.,2.]);x=np.array([-2.,0.,1.,2.,3.])
    np.testing.assert_array_equal(rank(b,x,True),rank(b**3,x**3,True))
    b1=b+np.array([.2,-.1,.3,-.2]);x1=x+np.array([.1,.2,-.2,.3,-.1])
    old=x[:,None]-b;new=x1[:,None]-b1
    overtakes=((old<=0)&(new>0)).sum(1);falls=((old>0)&(new<=0)).sum(1)
    np.testing.assert_allclose(rank(b1,x1,True)-rank(b,x,True),(falls-overtakes)/5.,atol=1e-15,rtol=0)
    # A sufficient boundary condition, including points on the old threshold.
    margin=np.array([-3.,-1.,-.5,0.,.5,1.,3.]);width=1.
    e_m=np.array([-.5,.4,.8,.2,.1,0.,-.5]);e_h=-e_m
    assert np.all(abs(e_m)<=width)
    assert np.all((margin+e_m>0)>=(margin>0))
    assert np.all((margin+e_h>0)<=(margin>0))
    return dict(temperature_derivative_max_error=max(errors),tie_crossings=True,monotone_invariance=True,boundary_sufficient_condition=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'scratch/reference_calibration')
    args = parser.parse_args()
    OUT = args.output.resolve()
    if (ROOT/'scratch').resolve() not in OUT.parents:
        raise ValueError('Output must be a new directory below scratch/')
    OUT.mkdir(parents=True, exist_ok=False)
    source={str((BASE/('calibration_'+p+'.pkl')).relative_to(ROOT)):sha(BASE/('calibration_'+p+'.pkl')) for p in ['OPT','Falcon']}
    source.update({str((BASE/'test'/p/'scoring_frame.pkl').relative_to(ROOT)):sha(BASE/'test'/p/'scoring_frame.pkl') for p in ['OPT','Falcon']})
    protocol=dict(scope='Fixed current upper-reference shaping versus no shaping, entropy and lower branch unchanged. No development candidates, parameter selection, or LM forwards.',
        cohort='20-cell800human calibration;60-cell2360human129800machine660tasks per proxy',
        params=dict(T=1.2,c=.8,kappa=1.5,beta=.5),sources=source,script_sha256=sha(__file__),
        prior_test_exposure=True,analysis_not_parameter_selection=True,checks=analytic_checks())
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];humans=[];thresholds=[];audits=[]
    previous=pd.read_csv(PAPER/'data/reference_calibration/by_task.csv')
    for proxy in ['OPT','Falcon']:
        bank=pd.read_pickle(BASE/('calibration_'+proxy+'.pkl')).sort_values('source_id').reset_index(drop=True)
        full=pd.read_pickle(BASE/'test'/proxy/'scoring_frame.pkl')
        hh=full[full.label==0].sort_values(['dataset','source_id','generator']).drop_duplicates(['dataset','source_id'])
        q=pd.concat([hh,full[full.label==1]],ignore_index=True).sort_values('row_id').reset_index(drop=True)
        hm=q.label.to_numpy()==0
        assert len(bank)==800 and bank.source_id.is_unique
        assert (hm.sum(),(~hm).sum())==(2360,129800)
        assert not set(bank.source_id)&set(q.source_id)
        responses=[]
        for f in [bank,q]:
            s0=np.maximum(np.sqrt(f.entropy_V0.to_numpy()),1e-9);s1=np.maximum(np.sqrt(f.entropy_V1.to_numpy()),1e-9)
            r0=f.entropy_N0.to_numpy()/s0;r1=f.entropy_N1.to_numpy()/s1
            r=(f.entropy_N0-f.entropy_N1).to_numpy()/s0;k=s1/s0
            shift=-r/k+(1/k-1)*r0
            np.testing.assert_allclose(r1-r0,shift,atol=1e-12,rtol=1e-12)
            lower=f.uniform_N0.to_numpy()/np.maximum(np.sqrt(f.uniform_V0.to_numpy()),1e-9)
            responses.append((r0,r1,shift,lower))
        b0,b1,bd,bl=responses[0];r0,r1,shift,ql=responses[1]
        p0,p1,pl=rank(b0,r0,True),rank(b1,r1,True),rank(bl,ql,False)
        # Integer evidence avoids round-off at tied joint ranks; beta=.5.
        u0,u1,l= -p0,-p1,-pl
        j0,j1=np.maximum(u0,l),np.maximum(u1,l)
        protected=(l>=u0)&(l>=u1)
        assert np.array_equal(j0[protected],j1[protected])
        cross_count=0
        for start in range(0,len(q),1024):
            old=r0[start:start+1024,None]-b0[None,:]
            new=r1[start:start+1024,None]-b1[None,:]
            over=((old<=0)&(new>0)).sum(1);fall=((old>0)&(new<=0)).sum(1)
            np.testing.assert_allclose(p1[start:start+1024]-p0[start:start+1024],(fall-over)/801.,atol=1e-15,rtol=0)
            cross_count+=len(over)
        decisions={}
        for alpha in [.01,.05]:
            k=int(np.floor(801*alpha/2));t0,t1=np.sort(b0)[-k],np.sort(b1)[-k]
            m0=r0-t0;adjustment=shift-(t1-t0);m1=r1-t1
            np.testing.assert_allclose(m1,m0+adjustment,atol=1e-12,rtol=1e-12)
            a0,a1=p0<=alpha/2,p1<=alpha/2;low=pl<=alpha/2
            np.testing.assert_array_equal(a0,r0>t0);np.testing.assert_array_equal(a1,r1>t1)
            gained=(~low)&(~a0)&a1;lost=(~low)&a0&(~a1)
            assert np.array_equal((a1|low).astype(int)-(a0|low).astype(int),gained.astype(int)-lost.astype(int))
            thresholds.append(dict(proxy=proxy,alpha=alpha,upper_tail_index=k,threshold_original=t0,threshold_shaped=t1,threshold_change=t1-t0))
            humans.append(dict(proxy=proxy,alpha=alpha,humans=int(hm.sum()),original_fp=int((a0|low)[hm].sum()),shaped_fp=int((a1|low)[hm].sum()),gained=int(gained[hm].sum()),lost=int(lost[hm].sum()),
                original_upper_fp=int(a0[hm].sum()),shaped_upper_fp=int(a1[hm].sum())))
            decisions[alpha]=(gained,lost,a0|low,a1|low)
        for ds,gen,wm,hi,mi in tasks(q):
            c0,c1=credit(j0[hi],j0[mi]),credit(j1[hi],j1[mi]);d=c1-c0
            prot=protected[mi]
            rr=dict(proxy=proxy,dataset=ds,generator=gen,watermark_method=wm,
                upper_raw_original=float(credit(r0[hi],r0[mi]).mean()),upper_raw_shaped=float(credit(r1[hi],r1[mi]).mean()),
                upper_rank_original=float(credit(u0[hi],u0[mi]).mean()),upper_rank_shaped=float(credit(u1[hi],u1[mi]).mean()),
                joint_original=float(c0.mean()),joint_shaped=float(c1.mean()),joint_delta=float(d.mean()),
                repair=float(np.maximum(d,0).mean()),damage=float(np.maximum(-d,0).mean()),
                protected_machine_fraction=float(prot.mean()),protected_contribution=float((d*prot[:,None]).mean()),
                remaining_contribution=float((d*(~prot)[:,None]).mean()),
                shift_machine_mean=float(shift[mi].mean()),shift_human_mean=float(shift[hi].mean()))
            for alpha,tag in [(.01,'a01'),(.05,'a05')]:
                gain,loss,a0,a1=decisions[alpha]
                rr['power_gain_'+tag]=float(gain[mi].mean());rr['power_loss_'+tag]=float(loss[mi].mean())
                rr['power_original_'+tag]=float(a0[mi].mean());rr['power_shaped_'+tag]=float(a1[mi].mean())
            old=previous[(previous.proxy==proxy)&(previous.dataset==ds)&(previous.generator==gen)&(previous.watermark_method==wm)].iloc[0]
            for key in ('original','shaped'):
                for metric in ('joint_','upper_rank_','upper_raw_'):
                    assert abs(old[metric+key]-rr[metric+key])<1e-12
            assert abs(rr['repair']-rr['damage']-rr['joint_delta'])<1e-12
            assert abs(rr['protected_contribution']+rr['remaining_contribution']-rr['joint_delta'])<1e-12
            rows.append(rr)
        audits.append(dict(proxy=proxy,rank_crossing_values=cross_count,cohort_and_baseline_parity=True,threshold_margin_identity=True,lower_invariant=True))
    td=pd.DataFrame(rows);save('by_task',td);save('human_transitions',humans);save('thresholds',thresholds)
    numeric=[c for c in td.columns if c not in ['proxy','dataset','generator','watermark_method']]
    save('by_mechanism',td.groupby(['proxy','watermark_method'])[numeric].mean().reset_index())
    groups=[('all',None),('unwatermarked',['unwatermarked']),('positive6',POSITIVE),('CGW_HeavyWater',['CGW','HeavyWater']),('ITSEdit_SemStamp',['ITSEdit','SemStamp'])]
    summary=[]
    for proxy,g in td.groupby('proxy'):
        for name,wm in groups:
            part=g if wm is None else g[g.watermark_method.isin(wm)]
            r=dict(proxy=proxy,cohort=name,tasks=len(part));r.update({k:float(part[k].mean()) for k in numeric})
            r['upper_rank_delta']=r['upper_rank_shaped']-r['upper_rank_original'];summary.append(r)
    save('summary',summary)
    draws=np.random.RandomState(20260916).randint(0,12,size=(10000,12));intervals=[]
    for proxy,g in td.groupby('proxy'):
        for name,wm in groups:
            part=g if wm is None else g[g.watermark_method.isin(wm)]
            for metric in ['joint_delta','protected_contribution']:
                values=part.groupby('dataset')[metric].mean().sort_index().to_numpy()
                lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
                intervals.append(dict(proxy=proxy,cohort=name,metric=metric,mean=float(values.mean()),ci_low=float(lo),ci_high=float(hi)))
    save('paired_intervals',intervals)
    ss=pd.DataFrame(summary)
    audit=dict(status='PASS',cohorts=audits,analytic_checks=protocol['checks'],
        calibration_humans=800,no_parameter_selection=True,new_lm_forwards=0,script_sha256=sha(__file__))
    (OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(ss[['proxy','cohort','upper_rank_delta','joint_delta','protected_machine_fraction','protected_contribution']].to_string(index=False))
    print(pd.DataFrame(humans).to_string(index=False))


if __name__=='__main__':main()
