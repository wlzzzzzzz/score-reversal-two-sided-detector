"""Deterministic stress cases for conditional moments and epsilon semantics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fastdetect_official_module_core import _weights, _document_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'scratch/iclr_review_supplement'


def s3_only(ref,score,labels,dtype=torch.float32,stable_variance=False,chunk=128,epsilon=1e-9):
    """Only rho=0; no unused path or factorial scores. Returns reference diagnostics."""
    target=[];means=[];variances=[];entropies=[];supports=[];outside=0
    for start in range(0,len(labels),chunk):
        r=ref[start:start+chunk].to(dtype);s=score[start:start+chunk].to(device=ref.device,dtype=dtype)
        labs=labels[start:start+chunk].to(ref.device)
        lp=torch.log_softmax(s,-1);lq=torch.log_softmax(r,-1)
        entropies.append((-(lq.exp()*lq).sum(-1)/np.log(r.shape[-1])).cpu())
        target.append(lp.gather(-1,labs[:,None]).squeeze(-1).cpu())
        q,idx=torch.sort(torch.softmax(r/1.2,-1),descending=True,dim=-1)
        cumulative=q.cumsum(-1);mask=torch.ones_like(q,dtype=torch.bool);mask[:,1:]=~(cumulative[:,:-1]>.8)
        dist=q*mask;dist=dist/dist.sum(-1,keepdim=True)
        values=lp.gather(-1,idx);mu=(dist*values).sum(-1)
        var=(dist*(values-mu[:,None]).square()).sum(-1) if stable_variance else ((dist*values.square()).sum(-1)-mu.square()).clamp_min(0)
        means.append(mu.cpu());variances.append(var.cpu());supports.append(mask.sum(-1).cpu())
        positions=(idx==labs[:,None]).nonzero(as_tuple=False)[:,1]
        outside+=int((~mask[torch.arange(len(labs),device=ref.device),positions]).sum())
    target=torch.cat(target);mu=torch.cat(means);var=torch.cat(variances);h=torch.cat(entropies);support=torch.cat(supports)
    w=_weights(h,'linear_entropy',1.5);sd=float((w.square()*var).sum().sqrt())
    return {'score':_document_score(target,mu,var,w,epsilon),'conditional_std':sd,'epsilon_active':sd<epsilon,
            'min_support':int(support.min()),'single_token_support_fraction':float((support==1).double().mean()),
            'observed_outside_fraction':outside/len(labels),'variance_min':float(var.min()),
            'max_variance_contribution':float((w.square()*var).max()/(w.square()*var).sum()) if sd>0 else None}


def main():
    gen=torch.Generator().manual_seed(20260907);rows=[]
    for n in [1,2,8,64,256]:
        for scale in [1.,5.,20.,80.]:
            for repeat in range(5):
                logits=torch.randn(n,128,generator=gen)*scale
                labels=torch.randint(0,128,(n,),generator=gen)
                if repeat%2==0:labels=logits.argmax(-1)
                cases={}
                for dtype,stable,name in [(torch.float32,False,'fp32_second_moment'),(torch.float32,True,'fp32_central_moment'),(torch.float64,True,'fp64_central_moment')]:
                    cases[name]=s3_only(logits,logits,labels,dtype,stable)
                reference=cases['fp64_central_moment']['score']
                for name,res in cases.items():
                    rows.append({'tokens':n,'logit_scale':scale,'repeat':repeat,'variant':name,**res,
                                 'abs_score_error_vs_fp64_central':abs(res['score']-reference),
                                 'relative_score_error_vs_fp64_central':abs(res['score']-reference)/max(1.,abs(reference))})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'synthetic_numerical_stress.csv',index=False)
    # Explicit self-pair counterexample: conditioning can increase variance.
    p=np.r_[.2,np.repeat(.008,100)];ell=np.log(p);full=np.sum(p*(ell-np.sum(p*ell))**2)
    keep=p[:76]/p[:76].sum();v=np.sum(keep*(ell[:76]-np.sum(keep*ell[:76]))**2)
    assert v>full
    # Exhaustive finite auxiliary law verifies the exact weighted-moment identity.
    import itertools
    probs=[np.array([.2,.3,.5]),np.array([.1,.6,.3])];f=[np.array([-3.,-2.,-.5]),np.array([-5.,-1.,-.2])];w=np.array([.4,.6])
    mus=[np.dot(a,b) for a,b in zip(probs,f)];var=sum(wi**2*np.dot(a,(b-m)**2) for wi,a,b,m in zip(w,probs,f,mus))
    eus=eus2=0.
    for a,b in itertools.product(range(3),repeat=2):
        pr=probs[0][a]*probs[1][b];u=w[0]*(f[0][a]-mus[0])+w[1]*(f[1][b]-mus[1]);eus+=pr*u;eus2+=pr*u*u
    assert abs(eus)<1e-14 and abs(eus2-var)<1e-14
    summary={'status':'COMPLETE','scope':'Synthetic stress, not a real-text error-rate estimate',
             'cases':100,'variants':3,'all_scores_finite':bool(np.isfinite(frame.score).all()),
             'conditioning_variance_counterexample':{'full_variance':full,'top_p_variance':v,'retained_mass':float(p[:76].sum())},
             'exact_auxiliary_moment_check':{'mean_U':eus,'enumerated_variance':eus2,'analytic_variance':var},
             'max_relative_score_error_by_variant':frame.groupby('variant').relative_score_error_vs_fp64_central.max().to_dict()}
    (OUT/'synthetic_numerical_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
