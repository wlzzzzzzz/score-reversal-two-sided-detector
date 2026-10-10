#!/usr/bin/env python3
"""Float64 reference for Algorithm 1 after the frozen proxy forward pass.

Calibration rows must represent unique human sources, with the same proxy,
preprocessing and parameters as the query. This module does not enforce
exchangeability or certify error rates after domain shift.
"""
import numpy as np


def responses(logits, observed, T=1.2, c=.8, kappa=1.5,
              eps_h=1e-6, eps_w=1e-8, eps=1e-9):
    z=np.asarray(logits,dtype=np.float64); x=np.asarray(observed)
    if z.ndim!=2 or z.shape[0]<1 or z.shape[1]<2 or not np.isfinite(z).all():
        raise ValueError('Expected finite [prediction positions, vocabulary] logits')
    n,D=z.shape
    if x.shape!=(n,) or x.dtype.kind not in 'iu' or (x<0).any() or (x>=D).any():
        raise ValueError('Invalid observed next-token indices')
    if not all(np.isfinite(v) and v>0 for v in (T,kappa,eps_h,eps_w,eps)) or not 0<c<=1:
        raise ValueError('Invalid scoring parameters')
    def logsoftmax(a):
        a=a-a.max(axis=1,keepdims=True)
        return a-np.log(np.exp(a).sum(axis=1,keepdims=True))
    ell=logsoftmax(z);p=np.exp(ell);q=np.exp(logsoftmax(z/T))
    if c<1:
        order=np.argsort(-q,axis=1,kind='stable');qs=np.take_along_axis(q,order,axis=1)
        before=np.column_stack([np.zeros(n),np.cumsum(qs,axis=1)[:,:-1]])
        qs=qs*(before<=c);qs/=qs.sum(axis=1,keepdims=True)
        q=np.take_along_axis(qs,np.argsort(order,axis=1),axis=1)
    h=-(p*ell).sum(axis=1)/np.log(D);a=np.minimum(h,max(kappa*h.mean(),eps_h));w=a/max(a.sum(),eps_w)
    observed_lp=ell[np.arange(n),x];result=[]
    for ref,weight in [(q,w),(p,np.full(n,1/n))]:
        mu=(ref*ell).sum(axis=1);var=np.maximum((ref*ell*ell).sum(axis=1)-mu*mu,0)
        N=weight@observed_lp-weight@mu;V=(weight*weight)@var
        result.append(float(N/max(np.sqrt(V),eps)))
    return np.array(result)


def test(query,calibration,alpha=.01,beta=.5):
    r=np.asarray(query,dtype=np.float64);bank=np.asarray(calibration,dtype=np.float64)
    if r.shape!=(2,) or bank.ndim!=2 or bank.shape[1]!=2 or len(bank)<1:
        raise ValueError('Expected two query responses and an [m,2] human bank')
    if not np.isfinite(r).all() or not np.isfinite(bank).all() or not 0<alpha<1 or not 0<beta<1:
        raise ValueError('Invalid test inputs')
    pu=(1+np.count_nonzero(bank[:,0]>=r[0]))/(len(bank)+1)
    pl=(1+np.count_nonzero(bank[:,1]<=r[1]))/(len(bank)+1)
    J=-np.log(min(pu/beta,pl/(1-beta)))
    return dict(J=float(J),reject=bool(pu<=alpha*beta or pl<=alpha*(1-beta)),p_upper=pu,p_lower=pl)
