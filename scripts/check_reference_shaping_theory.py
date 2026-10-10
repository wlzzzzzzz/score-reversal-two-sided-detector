"""Pure mathematical checks; never opens development or test data."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def auc_closed(b, delta, sigma, c, pi=0.5):
    a = delta / (np.sqrt(2.0) * sigma)
    t = np.sqrt(2.0) * (b - c) / sigma
    k = 2.0 * norm.cdf(a) - 1.0
    plus = 0.5 + k * (norm.cdf(t + a) - 0.5)
    minus = 0.5 - k * (norm.cdf(t - a) - 0.5)
    return float(pi * plus + (1.0 - pi) * minus)


def auc_integral(b, delta, sigma, c, pi=0.5):
    def integrand(z):
        radius = abs(b + sigma * z - c)
        def outside(mu):
            return norm.cdf((c - radius - mu) / sigma) + norm.sf(
                (c + radius - mu) / sigma)
        return norm.pdf(z) * (pi * outside(b + delta)
                             + (1.0 - pi) * outside(b - delta))
    split = (c - b) / sigma
    left = quad(integrand, -np.inf, split, epsabs=1e-11, epsrel=1e-11)[0]
    right = quad(integrand, split, np.inf, epsabs=1e-11, epsrel=1e-11)[0]
    return float(left + right)


def auc_pairs(human, machine):
    margin = np.asarray(machine)[:, None] - np.asarray(human)[None, :]
    return float(np.mean((margin > 0) + 0.5 * (margin == 0)))


def main():
    checks = []
    for b, delta, sigma, c, pi in itertools.product(
            [-0.7, 0.8], [0.4, 1.5], [0.6, 1.2], [-0.3, 1.1], [0.5, 1.0]):
        closed = auc_closed(b, delta, sigma, c, pi)
        integral = auc_integral(b, delta, sigma, c, pi)
        checks.append(dict(b=b, delta=delta, sigma=sigma, c=c, pi=pi,
                           closed=closed, integral=integral,
                           abs_error=abs(closed - integral)))
    max_error = max(row['abs_error'] for row in checks)
    assert max_error < 1e-9, max_error

    signs = 0
    for b, c, delta, sigma in itertools.product(
            [-1.0, -0.3, 0.0, 0.4, 1.0],
            [-1.4, -0.5, 0.0, 0.3, 0.9, 1.8], [0.5, 1.2], [0.8, 1.4]):
        gain = auc_closed(b, delta, sigma, c) - auc_closed(b, delta, sigma, 0)
        bias_reduction = 2 * b * c - c * c
        if abs(bias_reduction) < 1e-12:
            assert abs(gain) < 1e-12
        else:
            assert gain * bias_reduction > 0, (b, c, delta, sigma, gain)
        signs += 1

    counterexample = dict(b=1.0, delta=1.0, sigma=1.0, pi=1.0,
                          baseline_auc=auc_closed(1, 1, 1, 0, 1),
                          human_centered_auc=auc_closed(1, 1, 1, 1, 1),
                          G=1.0)
    assert counterexample['human_centered_auc'] < counterexample['baseline_auc']
    scale_counterexample = dict(
        baseline_auc=auc_pairs([1, 4], [2, 3]),
        human_scaled_auc=auc_pairs(np.array([1, 4]) / 4, [2, 3]),
        machine_scaled_auc=auc_pairs([1, 4], np.array([2, 3]) / 4))
    assert scale_counterexample == dict(baseline_auc=0.5,
                                       human_scaled_auc=1.0, machine_scaled_auc=0.0)

    rng = np.random.default_rng(20260911)
    margin_checks = []
    for eps in [0.01, 0.1, 0.5, 1.0]:
        for _ in range(10):
            h = np.exp(rng.normal(size=50))
            m = np.exp(rng.normal(0.3, 1.0, size=60))
            sh = np.exp(0.7 + rng.uniform(-eps, eps, h.shape))
            sm = np.exp(0.7 + rng.uniform(-eps, eps, m.shape))
            change = abs(auc_pairs(h / sh, m / sm) - auc_pairs(h, m))
            bound = float(np.mean(abs(np.log(m)[:, None] - np.log(h)[None, :])
                                  <= 2 * eps))
            assert change <= bound + 1e-12
            margin_checks.append(dict(eps=eps, abs_auc_change=change, bound=bound))

    root = Path(__file__).resolve().parents[1]
    output = root / 'scratch/reference_shaping_theory'
    output.mkdir(parents=True, exist_ok=True)
    result = dict(
        status='PASS', scope='Pure mathematical verification; no empirical data read',
        note='Numerical checks supplement the written proofs; not model validation.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        quadrature_cases=len(checks), max_quadrature_abs_error=max_error,
        gain_sign_cases=signs, quadrature=checks,
        asymmetric_counterexample=counterexample,
        scale_counterexample=scale_counterexample,
        scale_margin_cases=len(margin_checks), scale_margin_checks=margin_checks)
    path = output / 'MATHEMATICAL_CHECKS.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in [
        'status', 'quadrature_cases', 'max_quadrature_abs_error', 'gain_sign_cases',
        'asymmetric_counterexample', 'scale_margin_cases']}, indent=2))
    print(str(path))


if __name__ == '__main__':
    main()
