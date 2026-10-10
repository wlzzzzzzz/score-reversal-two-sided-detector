#!/usr/bin/env python3
import json
import numpy as np
from asymmetric_reference import responses, test
rng=np.random.default_rng(20260913);maximum=0.
for _ in range(100):
    n,D=13,9;z=rng.normal(size=(n,D));obs=rng.integers(0,D,n)
    p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);ell=np.log(p)
    mu=(p*ell).sum(1);v=(p*(ell-mu[:,None])**2).sum(1)
    fd=(ell[np.arange(n),obs]-mu).sum()/np.sqrt(v.sum())
    err=abs(responses(z,obs)[1]-fd);maximum=max(maximum,err);assert err<1e-12
# Constant distributions, degenerate variances, and exact nucleus-boundary ties.
r=responses(np.zeros((3,4)),np.array([0,1,3]),T=1,c=.5)
assert np.isfinite(r).all() and np.max(abs(r))<1e-6
b=np.array([[0.,0.],[0.,0.],[1.,-1.]])
assert not test([0,0],b,alpha=.5)['reject']
assert test([2,0],b,alpha=.5)['reject']
assert test([0,-2],b,alpha=.5)['reject']
assert test([0,0],np.zeros((800,2)))['J']<0
# One calibration point cannot reject at 1% with equal branch allocations.
assert not test([100,-100],np.zeros((1,2)))['reject']
res={'status':'PASS','original_uniform_fd_equivalence_trials':100,'max_absolute_error':maximum,
     'finite_degenerate_responses':True,'inclusive_ties':True,'either_branch_rejects':True,
     'unclipped_negative_joint_score':True,'small_calibration_no_rejection':True}
print(json.dumps(res,indent=2))
