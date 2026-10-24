"""Orakel-Tests mit unabhängigem Rechenweg (ORACLE-PLAYBOOK): `scipy.stats.zipf` für das diskrete Potenzgesetz (Verteilungsfunktion, Log-Wahrscheinlichkeit, Ziehen),
ein dichtes Gitter statt des Optimierers für den Exponenten, Brute-Force-Wahl der Untergrenze, `scipy.optimize` für die Vergleichsverteilungen,
`scipy.stats.pareto` für den Hill-Schätzer, `networkx.barabasi_albert_graph` für die Erzeuger."""

import math

import numpy as np
import pytest
from scipy import optimize, special, stats

import pgt_algorithm as A


def zipf_tail_sample(n, alpha, xmin, seed):
    """Unabhängiger Sampler: `scipy.stats.zipf` (Träger ab 1), bedingt auf Werte ≥ xmin, ist genau das Potenzgesetz ab xmin."""
    rng = np.random.default_rng(seed)
    out = np.empty(0, dtype=np.int64)
    while len(out) < n:
        draw = stats.zipf.rvs(alpha, size=4 * n, random_state=rng)
        out = np.concatenate([out, draw[draw >= xmin]])
    return out[:n]


@pytest.mark.parametrize("alpha, xmin", [(2.3, 1), (2.8, 3), (3.4, 6)])
def test_loglik_and_cdf_match_scipy_zipf(alpha, xmin):
    ks = np.array([xmin, xmin + 1, xmin + 4, xmin + 20, xmin + 300])
    # P(K = k | K ≥ xmin) = zipf.pmf(k) / zipf.sf(xmin − 1)
    expected_ll = float((stats.zipf.logpmf(ks, alpha) - math.log(stats.zipf.sf(xmin - 1, alpha))).sum())
    assert A.powerlaw_loglik(ks, alpha, xmin) == pytest.approx(expected_ll, rel=1e-9)
    expected_cdf = (stats.zipf.cdf(ks, alpha) - stats.zipf.cdf(xmin - 1, alpha)) / stats.zipf.sf(xmin - 1, alpha)
    assert A.powerlaw_cdf(ks, alpha, xmin).tolist() == pytest.approx(expected_cdf.tolist(), abs=1e-9)
    assert A.powerlaw_tail_ccdf(ks, alpha, xmin).tolist() == pytest.approx((1 - expected_cdf + stats.zipf.pmf(ks, alpha) / stats.zipf.sf(xmin - 1, alpha)).tolist(), abs=1e-9)


@pytest.mark.parametrize("alpha, xmin, seed", [(2.2, 1, 1), (2.7, 4, 2), (3.3, 2, 3)])
def test_fit_alpha_equals_the_maximum_of_a_dense_grid(alpha, xmin, seed):
    x = zipf_tail_sample(3000, alpha, xmin, seed)
    grid = np.arange(1.2, 6.0, 0.0005)
    ll = [A.powerlaw_loglik(x, a, xmin) for a in grid]
    best = float(grid[int(np.argmax(ll))])
    assert A.fit_alpha(x, xmin) == pytest.approx(best, abs=0.001)
    assert A.fit_alpha(x, xmin) == pytest.approx(alpha, abs=0.12)       # 3 000 Werte: der Schätzer liegt nahe am wahren Exponenten


def test_the_stationarity_condition_holds_at_the_estimate():
    """Am Maximum gilt: Mittelwert von ln k = −ζ'(α, xmin) / ζ(α, xmin) (Ableitung durch zentrale Differenzen)."""
    x = zipf_tail_sample(5000, 2.6, 2, 4)
    a = A.fit_alpha(x, 2)
    h = 1e-5
    dz = (special.zeta(a + h, 2) - special.zeta(a - h, 2)) / (2 * h)
    assert float(np.log(x).mean()) == pytest.approx(-dz / special.zeta(a, 2), abs=1e-5)


def test_fit_alpha_recovers_known_exponents_on_large_samples():
    for alpha in (2.1, 2.5, 3.0):
        x = zipf_tail_sample(30000, alpha, 3, 10)
        assert A.fit_alpha(x, 3) == pytest.approx(alpha, abs=0.05)


def test_own_sampler_matches_the_zipf_sampler_in_distribution():
    own = A.sample_powerlaw(20000, 2.5, 3, np.random.default_rng(5))
    ref = zipf_tail_sample(20000, 2.5, 3, 6)
    assert stats.ks_2samp(own, ref).pvalue > 0.001
    assert (own == 3).mean() == pytest.approx((ref == 3).mean(), abs=0.02)


def test_scan_xmin_equals_a_brute_force_search():
    """Brute-Force: alle Kandidaten, Exponent über ein feines Gitter, KS-Abstand aus der Verteilungsfunktion von scipy."""
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.integers(1, 6, size=300), zipf_tail_sample(1200, 2.6, 4, 8)])
    d, xmin, a = A.scan_xmin(x, 100, 12)
    best = None
    for xm in A.candidate_xmins(x, 100, 12):
        t = np.sort(x[x >= xm])
        grid = np.arange(1.2, 6.0, 0.001)
        al = float(grid[int(np.argmax([A.powerlaw_loglik(t, g, xm) for g in grid]))])
        ks = np.unique(t)
        emp = np.searchsorted(t, ks, side="right") / len(t)
        mod = (stats.zipf.cdf(ks, al) - stats.zipf.cdf(xm - 1, al)) / stats.zipf.sf(xm - 1, al)
        dist = float(np.abs(emp - mod).max())
        if best is None or dist < best[0]:
            best = (dist, xm, al)
    assert xmin == best[1] and d == pytest.approx(best[0], abs=0.002) and a == pytest.approx(best[2], abs=0.002)


def test_exponential_fit_matches_the_geometric_distribution():
    """Die diskrete Exponentialverteilung ab xmin ist die geometrische Verteilung von K − xmin + 1; ihr Maximum-Likelihood-Schätzer ist p = 1 / Mittelwert."""
    rng = np.random.default_rng(9)
    x = 5 + rng.geometric(0.3, size=4000) - 1
    ll, lam = A.exponential_pointwise_loglik(x, 5)
    p = 1.0 / float((x - 5 + 1).mean())
    assert lam == pytest.approx(-math.log(1 - p))
    assert ll.tolist() == pytest.approx(stats.geom.logpmf(x - 5 + 1, p).tolist(), abs=1e-9)
    res = optimize.minimize_scalar(lambda l: -float((math.log1p(-math.exp(-l)) - l * (x - 5)).sum()), bounds=(0.01, 5), method="bounded", options={"xatol": 1e-9})
    assert lam == pytest.approx(float(res.x), abs=1e-5)


def test_lognormal_probabilities_sum_to_one_and_the_fit_is_the_maximum():
    xmin = 4
    rng = np.random.default_rng(11)
    x = np.rint(np.exp(rng.normal(2.0, 0.8, size=30000))).astype(np.int64)      # auf die nächste ganze Zahl gerundet: genau die diskretisierte Log-Normal-Verteilung
    tail = x[x >= xmin]
    ll, mu, sg = A.lognormal_pointwise_loglik(tail, xmin)
    assert mu == pytest.approx(2.0, abs=0.06) and sg == pytest.approx(0.8, abs=0.04)
    # Summe der Wahrscheinlichkeiten über alle k ≥ xmin gleich 1 (Brute Force mit scipy.stats.norm)
    ks = np.arange(xmin, 5_000_000)
    pm = stats.norm.cdf((np.log(ks + 0.5) - mu) / sg) - stats.norm.cdf((np.log(ks - 0.5) - mu) / sg)
    pm = pm / stats.norm.sf((math.log(xmin - 0.5) - mu) / sg)
    assert pm.sum() == pytest.approx(1.0, abs=1e-6)
    assert ll.tolist() == pytest.approx(np.log(pm[tail - xmin]).tolist(), abs=1e-7)
    # kein Gitterpunkt in der Nähe hat eine größere Log-Likelihood
    best = ll.sum()
    for dm in (-0.1, 0.0, 0.1):
        for ds in (-0.1, 0.0, 0.1):
            if dm == 0.0 and ds == 0.0:
                continue
            p2 = stats.norm.cdf((np.log(ks + 0.5) - mu - dm) / (sg + ds)) - stats.norm.cdf((np.log(ks - 0.5) - mu - dm) / (sg + ds))
            p2 = p2 / stats.norm.sf((math.log(xmin - 0.5) - mu - dm) / (sg + ds))
            assert float(np.log(p2[tail - xmin]).sum()) <= best + 1e-6


def test_lognormal_fit_stays_finite_for_a_very_heavy_tail():
    """Bei exakten Potenzgesetz-Daten strebt die Streuung gegen unendlich: die Log-Wahrscheinlichkeiten dürfen dabei weder null noch nicht-endlich werden."""
    x = zipf_tail_sample(4000, 2.5, 3, 12)
    ll, mu, sg = A.lognormal_pointwise_loglik(x, 3)
    assert np.isfinite(ll).all() and (ll < 0).all()
    pl = np.array([A.powerlaw_loglik([k], 2.5, 3) for k in x])
    assert abs(float(ll.sum() - pl.sum())) < 0.05 * abs(float(pl.sum()))       # beide Anpassungen sind fast gleich gut


def test_vuong_matches_a_direct_recomputation_with_scipy():
    rng = np.random.default_rng(13)
    a, b = rng.normal(-3, 1, 500), rng.normal(-3.1, 1, 500)
    r, z, p = A.vuong(a, b)
    d = a - b
    assert r == pytest.approx(d.sum()) and z == pytest.approx(d.sum() / (d.std(ddof=1) * math.sqrt(500)))
    assert p == pytest.approx(2 * stats.norm.sf(abs(z)))
    # unter gleichen Verteilungen ist p etwa gleichverteilt: höchstens 25 % der Wiederholungen liegen unter 0,1 (Erwartung 10 %)
    ps = [A.vuong(rng.normal(0, 1, 200), rng.normal(0, 1, 200))[2] for _ in range(200)]
    assert 0.03 < np.mean(np.array(ps) < 0.1) < 0.25


def test_hill_estimator_recovers_the_exponent_of_a_pareto_sample():
    for alpha in (2.2, 2.8):
        x = stats.pareto.rvs(alpha - 1.0, size=100000, random_state=np.random.default_rng(14))      # Dichte ~ x^(−α)
        assert A.hill_estimate(x, 0.05) == pytest.approx(alpha, abs=0.08)


def test_bootstrap_does_not_reject_an_exact_power_law_and_rejects_poisson():
    ok = 0
    for j in range(24):
        x = zipf_tail_sample(1500, 2.5, 3, 100 + j)
        d, xmin, a = A.scan_xmin(x, 50, 25)
        ok += A.bootstrap_pvalue(x, d, xmin, a, 20, 50, 25, np.random.default_rng(j)) >= 0.1
    assert ok / 24 >= 0.75        # Erwartung rund 90 % (Fehler erster Art 10 %)
    rng = np.random.default_rng(15)
    for _ in range(4):
        x = A.poisson_degrees(3000, 8.0, rng)
        d, xmin, a = A.scan_xmin(x, 50, 25)
        assert A.bootstrap_pvalue(x, d, xmin, a, 20, 50, 25, rng) < 0.1


def test_barabasi_albert_degrees_match_networkx():
    nx = pytest.importorskip("networkx")
    n, m, reps = 3000, 3, 10
    own_max, nx_max, own_a, nx_a = [], [], [], []
    for r in range(reps):
        own = A.ba_degrees(n, m, np.random.default_rng(50 + r))
        ref = np.array([d for _, d in nx.barabasi_albert_graph(n, m, seed=70 + r).degree()])
        own_max.append(own.max())
        nx_max.append(ref.max())
        own_a.append(A.fit_alpha(own[own >= 6], 6))
        nx_a.append(A.fit_alpha(ref[ref >= 6], 6))
    assert abs(np.mean(own_max) - np.mean(nx_max)) / np.mean(nx_max) < 0.25
    assert abs(np.mean(own_a) - np.mean(nx_a)) < 0.1
