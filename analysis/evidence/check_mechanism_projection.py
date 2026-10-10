#!/usr/bin/env python3
"""Numerical checks of arbitrary-kernel projections and their limitations."""
import json
import numpy as np
from scipy.special import softmax
rng=np.random.default_rng(2026091313);errs={k:0. for k in ['projection_mean','projection_orthogonality','shift','operator_bridge','information','weight_optimality','uniform_common']}
for trial in range(300):
 n,D=13,17;p=softmax(rng.normal(size=(n,D)),axis=1);G=softmax(rng.normal(size=(n,D)),axis=1)
 ell=np.log(p);mu=(p*ell).sum(1);e=ell-mu[:,None];nu=(p*e**2).sum(1)
 Q=softmax(rng.normal(size=(n,D)),axis=1)
 if trial%3==0:
  Q=np.zeros_like(p);Q[np.arange(n),rng.integers(0,D,n)]=1
 g=Q/p-1;b=-(p*g*e).sum(1)/nu;r=g+b[:,None]*e
 errs['projection_mean']=max(errs['projection_mean'],float(np.max(np.abs((p*r).sum(1)))))
 errs['projection_orthogonality']=max(errs['projection_orthogonality'],float(np.max(np.abs((p*r*e).sum(1)))))
 delta=((Q-G)*ell).sum(1);bridge=-(((G-p)*ell).sum(1)+delta)/nu
 errs['operator_bridge']=max(errs['operator_bridge'],float(np.max(np.abs(b-bridge))))
 theta=rng.uniform();w=rng.uniform(size=n);w/=w.sum();Qt=(1-theta)*p+theta*Q
 actual=-(w*(Qt*e).sum(1)).sum()/np.sqrt((w*w*nu).sum());expected=theta*(w*b*nu).sum()/np.sqrt((w*w*nu).sum())
 errs['shift']=max(errs['shift'],abs(actual-expected))
 errs['information']=max(errs['information'],abs(float((p*g*g).sum()-(b*b*nu).sum()-(p*r*r).sum())))
 # Generate valid downward target laws, then calculate strengths rather than assuming them.
 Qminus=softmax(np.log(p)-rng.uniform(.1,1,size=(n,1))*ell,axis=1);bm=-(Qminus*e).sum(1)/nu
 optimum=np.sqrt((bm*bm*nu).sum());value=(w*bm*nu).sum()/np.sqrt((w*w*nu).sum())
 matched=(bm*bm*nu).sum()/np.sqrt((bm*bm*nu).sum())
 errs['weight_optimality']=max(errs['weight_optimality'],value-optimum,abs(matched-optimum))
 # An exact common projection can be constructed by a small affine downward perturbation.
 common=.25/max(float(e.max()),1.);Qcommon=p*(1-common*e)
 bc=-(Qcommon*e).sum(1)/nu
 errs['uniform_common']=max(errs['uniform_common'],float(np.max(np.abs(bc-common))))
 assert (Qcommon>=0).all()
for k,v in errs.items():assert v<1e-9,(k,v)
# A nonzero watermark departure orthogonal to log probability is invisible to this mean score.
p=np.array([.6,.3,.1]);ell=np.log(p);e=ell-p@ell
r=np.cross(p,p*e);r/=np.max(np.abs(r));Q=p*(1+.2*r)
assert np.max(np.abs(Q-p))>.001 and abs(Q@ell-p@ell)<1e-12
# Exact preservation: no downward projection relative to a matched proxy.
assert np.max(np.abs((p/p-1)*e))==0
result={'status':'PASS','arbitrary_kernel_trials':300,'includes_point_mass_alternatives':True,'maximum_errors':errs,'orthogonal_departure_counterexample':True,'matched_preservation_counterexample':True,'claims':'reference-standardized mean optimality in a fixed auxiliary design; not human-null power optimality'}
print(json.dumps(result,indent=2))
