"""Numerical checks of the self-pair center proposition and its two-model limit."""
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]/'scratch/iclr_review_supplement'


def center(lp,t,mass,score_lp=None):
    z=lp/t;q=np.exp(z-z.max());q/=q.sum();k=len(q) if mass==1 else min(len(q),int(np.searchsorted(np.cumsum(q),mass,side='right'))+1)
    conditional=q[:k]/q[:k].sum();values=lp[:k] if score_lp is None else score_lp[:k]
    mu=float(conditional@values);var=float(conditional@((values-mu)**2))
    return mu,var,k


def main():
    rng=np.random.default_rng(20260907);temperature_violations=[];mass_violations=[];derivative_errors=[]
    temperatures=np.linspace(.2,3.,43);masses=np.r_[np.linspace(.1,.9,9),1.]
    for case in range(100):
        z=np.sort(rng.normal(size=31)*(1,5,20)[case%3])[::-1];lp=z-np.log(np.exp(z-z.max()).sum())-z.max()
        values=np.array([[center(lp,t,c)[0] for c in masses] for t in temperatures])
        temperature_violations.append(float(np.diff(values,axis=0).max()));mass_violations.append(float(np.diff(values,axis=1).max()))
        for t in [.7,1.,1.2,1.3,2.]:
            step=1e-5;before=center(lp,t-step,.8);now=center(lp,t,.8);after=center(lp,t+step,.8)
            if before[2]==now[2]==after[2]:derivative_errors.append(abs((after[0]-before[0])/(2*step)+now[1]/t**2))
    assert max(temperature_violations)<1e-10 and max(mass_violations)<1e-10
    assert max(derivative_errors)<1e-5
    lq=np.log([.9,.1]);lp=np.log([.1,.9]);opposite=[center(lq,t,1.,lp)[0] for t in [1.,2.]]
    assert opposite[1]>opposite[0]
    report={'status':'COMPLETE','random_probability_vectors':100,'temperature_grid_size':43,'mass_grid_size':10,
            'max_temperature_monotonicity_violation':max(0.,max(temperature_violations)),
            'max_mass_monotonicity_violation':max(0.,max(mass_violations)),
            'fixed_support_derivative_cases':len(derivative_errors),'max_finite_difference_derivative_error':max(derivative_errors),
            'two_model_counterexample':{'q':[.9,.1],'p':[.1,.9],'mean_logp_at_T1_T2':opposite},
            'scope':'Numerical verification supplements the analytic proof; no claim of monotonic final detection performance.'}
    (OUT/'reference_center_proposition_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
