#!/usr/bin/env python3
"""Numerically check entropy matching, shared-weight losses and power scope."""
import json

import numpy as np
from scipy.special import softmax
from scipy.stats import norm

rng = np.random.default_rng(20260915)
errors = {k: 0.0 for k in ['operator_bridge', 'covariance_criterion',
                         'approximate_alignment', 'shared_losses', 'compromise_attainment',
                         'positive_gap', 'opposed_margins', 'complete_event_identity']}


def check(name, actual, expected):
    errors[name] = max(errors[name], float(np.max(np.abs(np.asarray(actual)-expected))))


def effect(w, s, v):
    return np.sum(w*s*v) / np.sqrt(np.sum(w*w*v))


def efficiency(w, s, v):
    return effect(w, s, v)**2 / np.sum(s*s*v)


for trial in range(300):
    n, D = 17, 23
    p = softmax(rng.normal(size=(n, D)), axis=1)
    ell = np.log(p)
    q1 = softmax(ell/1.2, axis=1)
    # Deliberately sparse reference; Q can generate outside its support.
    mask = ell >= np.sort(ell, axis=1)[:, -8, None]
    q1 *= mask
    q1 /= q1.sum(axis=1, keepdims=True)
    mu0, mu1 = (p*ell).sum(axis=1), (q1*ell).sum(axis=1)
    nu1 = (q1*(ell-mu1[:, None])**2).sum(axis=1)
    G = softmax(rng.normal(size=(n, D)), axis=1)
    Q = softmax(rng.normal(size=(n, D)), axis=1)
    xi = ((Q*ell).sum(axis=1)-mu1)/nu1
    bridge = (((G-p)*ell).sum(axis=1)+((Q-G)*ell).sum(axis=1)+mu0-mu1)/nu1
    check('operator_bridge', xi, bridge)
    h = -(p*ell).sum(axis=1)/np.log(D)
    a = np.minimum(h, 1.5*h.mean())
    pi = nu1/nu1.sum()
    ea, ea2, exi = pi@a, pi@(a*a), pi@xi
    cov = pi@(a*xi)-ea*exi
    gain = effect(a, xi, nu1)-effect(np.ones(n), xi, nu1)
    rhs = np.sqrt(nu1.sum())/np.sqrt(ea2)*(cov-exi*(np.sqrt(ea2)-ea))
    check('covariance_criterion', gain, rhs)
    kp = np.sum(a*xi*nu1)/np.sum(a*a*nu1)
    eta = xi-kp*a
    check('approximate_alignment', efficiency(a, xi, nu1),
          1-np.sum(eta*eta*nu1)/np.sum(xi*xi*nu1))
    # Wide profile variation makes the strict efficiency and power losses visible.
    a = rng.uniform(.05, 1.0, n)
    v0, v1 = rng.uniform(.1, 2.0, (2, n))
    one = np.ones(n)
    for v in [v0, v1]:
        pi = v/v.sum()
        fraction = (pi@a)**2/(pi@(a*a))
        check('shared_losses', efficiency(one, a, v), fraction)
        check('shared_losses', efficiency(a, one, v), fraction)
        assert fraction < 1
    rho = np.sum(a*v1)/np.sqrt(np.sum(a*a*v1)*v1.sum())
    midpoint = a/np.sqrt(np.sum(a*a*v1))+one/np.sqrt(v1.sum())
    bound = (1+rho)/2
    check('compromise_attainment', [efficiency(midpoint, a, v1),
                                     efficiency(midpoint, one, v1)], bound)
    for w in rng.uniform(.01, 2.0, (20, n)):
        assert min(efficiency(w, a, v1), efficiency(w, one, v1)) <= bound+1e-12
    for s, matched, shared in [(a, a, one), (one, one, a)]:
        power = lambda w: norm.sf(norm.isf(.005)-.1*effect(w, s, v1))
        assert power(matched) > power(shared)
    d = rng.normal(size=n)
    w = a/a.sum()
    check('positive_gap', w@d-d.mean(), np.mean((n*w-1)*(d-d.mean())))
    uh, up, um, rh, rp, rm = rng.normal(size=6)
    check('opposed_margins', [(up-rp)-(uh-rh), (uh-rh)-(um-rm)],
          np.array([up-uh+rh-rp, uh-um-(rh-rm)]))
    U, L, Lp = rng.uniform(size=(3, 300)) < .3
    check('complete_event_identity', (U|L).mean()-(U|Lp).mean(),
          ((~U)&L&(~Lp)).mean()-((~U)&Lp&(~L)).mean())

# Common sharpening counterexample: its first-order strength is constant,
# despite distinct entropies. Central differences avoid first-order truncation bias.
p = np.array([[.90,.08,.02],[.45,.35,.20]])
ell = np.log(p); mu=(p*ell).sum(axis=1); nu=(p*(ell-mu[:,None])**2).sum(axis=1)
h = -(p*ell).sum(axis=1)
eps=1e-5
tilt=lambda z: softmax(ell+z*ell,axis=1)
derivative=(((tilt(eps)-tilt(-eps))*ell).sum(axis=1)/(2*eps))/nu
assert np.ptp(h)>.1
assert np.max(np.abs(derivative-1))<1e-8
assert efficiency(h, np.ones(2), nu)<1

# An increasing affine transformation of all bank and query scores preserves ranks.
bank = rng.normal(size=800); query = rng.normal(size=100)
before = np.sum(bank[:,None] <= query[None,:], axis=0)
after = np.sum((bank[:,None]-2)/3 <= (query[None,:]-2)/3, axis=0)
assert np.array_equal(before, after)
for name, error in errors.items():
    assert error < 1e-9, (name, error)
result = dict(status='PASS', random_trials=300, seed=20260915, maximum_errors=errors,
              sparse_shaped_reference_checked=True, common_sharpening_counterexample=True,
              gaussian_branch_power_checked=True, affine_rank_invariance=True,
              scope='Fixed-context effects; Gaussian power only under the stated model; no empirical optimality claim.')
print(json.dumps(result, indent=2))
