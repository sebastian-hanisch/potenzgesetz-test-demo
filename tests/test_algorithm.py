"""Kern: jede Einheit an einer Mini-Instanz von Hand (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest
from scipy import special

import pgt_algorithm as A

ZETA2 = math.pi ** 2 / 6          # ζ(2, 1)


def test_powerlaw_loglik_by_hand():
    # P(1) = 1/ζ(2), P(2) = (1/4)/ζ(2): Summe der Logarithmen
    expected = math.log(1 / ZETA2) + math.log(0.25 / ZETA2)
    assert A.powerlaw_loglik([1, 2], 2.0, 1) == pytest.approx(expected)
    # ab xmin = 2: ζ(2, 2) = ζ(2) − 1
    assert A.powerlaw_loglik([2], 2.0, 2) == pytest.approx(math.log(0.25 / (ZETA2 - 1)))


def test_powerlaw_cdf_by_hand():
    assert A.powerlaw_cdf([1], 2.0, 1)[0] == pytest.approx(1 / ZETA2)
    assert A.powerlaw_cdf([2], 2.0, 1)[0] == pytest.approx((1 + 0.25) / ZETA2)
    assert A.powerlaw_cdf([2], 2.0, 2)[0] == pytest.approx(0.25 / (ZETA2 - 1))     # erster Wert ab xmin
    assert A.powerlaw_cdf([1000000], 2.0, 1)[0] == pytest.approx(1.0, abs=2e-6)


def test_ks_distance_by_hand(four_values):
    # Modell bei α = 2, xmin = 1: F(1) = 1/ζ(2), F(2) = 1,25/ζ(2); Daten: 0,5 und 1,0
    expected = max(abs(0.5 - 1 / ZETA2), abs(1.0 - 1.25 / ZETA2))
    assert A.ks_distance(four_values, 2.0, 1) == pytest.approx(expected)
    assert expected == pytest.approx(1.0 - 1.25 / ZETA2)      # der zweite Abstand ist der größere (0,2401)


def test_candidate_xmins_by_hand(mini_values):
    assert A.candidate_xmins(mini_values, 3, 10) == [1, 2, 3]       # ab 5 bleibt nur ein Wert
    assert A.candidate_xmins(mini_values, 3, 2) == [1, 2]           # nur die kleinsten
    assert A.candidate_xmins(mini_values, 4, 10) == [1, 2]
    assert A.candidate_xmins(mini_values, 100, 10) == [1]           # nichts erreicht die Mindestzahl: der kleinste Wert


def test_fit_alpha_is_nan_for_a_single_value_and_inside_the_bounds():
    assert math.isnan(A.fit_alpha([3], 3))
    a = A.fit_alpha([1, 1, 1, 1, 2], 1)       # sehr steil
    assert A.ALPHA_LO < a < A.ALPHA_HI
    assert A.fit_alpha([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3, 1) < a      # flachere Daten, kleinerer Exponent


def test_scan_xmin_picks_the_smallest_ks_distance(mini_values):
    d, xmin, a = A.scan_xmin(mini_values, 3, 10)
    cands = [(A.ks_distance(mini_values[mini_values >= x], A.fit_alpha(mini_values[mini_values >= x], x), x), x) for x in (1, 2, 3)]
    assert (d, xmin) == min(cands)
    assert a == pytest.approx(A.fit_alpha(mini_values[mini_values >= xmin], xmin))


def test_scan_xmin_raises_for_a_sample_without_a_fit():
    with pytest.raises(ValueError):
        A.scan_xmin([4], 1, 5)


def test_sample_powerlaw_support_and_determinism():
    a = A.sample_powerlaw(2000, 2.5, 3, np.random.default_rng(1))
    b = A.sample_powerlaw(2000, 2.5, 3, np.random.default_rng(1))
    assert (a == b).all() and a.min() >= 3 and a.dtype.kind == "i"
    assert (a == 3).mean() == pytest.approx(3 ** -2.5 / special.zeta(2.5, 3), abs=0.04)


def test_exponential_loglik_by_hand():
    ll, lam = A.exponential_pointwise_loglik([1, 2, 3], 1)       # Mittelwert − xmin = 1, also λ = ln 2
    assert lam == pytest.approx(math.log(2))
    assert ll.tolist() == pytest.approx([-math.log(2) * k for k in (1, 2, 3)])       # ln(1/2) − ln 2 · (k − 1)
    ll2, lam2 = A.exponential_pointwise_loglik([5, 5], 5)         # Mittelwert = xmin: λ wird auf 50 gesetzt
    assert lam2 == 50.0 and ll2.shape == (2,)


def test_lognormal_loglik_returns_probabilities_that_sum_to_at_most_one():
    ll, mu, sg = A.lognormal_pointwise_loglik([2, 2, 3, 3, 3, 4, 5, 8, 13, 20], 2)
    assert (ll < 0).all() and sg > 0.05
    assert ll.shape == (10,)


def test_vuong_by_hand():
    r, z, p = A.vuong([0, 0, 0, 0], [-1, -1, -1, -3])
    # Differenzen 1, 1, 1, 3: R = 6, Standardabweichung 1 (Stichprobe), z = 6 / (1 · 2) = 3
    assert r == 6.0 and z == pytest.approx(3.0) and p == pytest.approx(2 * (1 - 0.998650102), abs=1e-6)
    assert A.vuong([1, 2], [1, 2]) == (0.0, 0.0, 1.0)         # keine Streuung: unentschieden
    assert A.vuong([1.0], [0.0]) == (1.0, 0.0, 1.0)           # ein einzelner Wert


def test_hill_estimate_by_hand():
    x = [8, 4, 2, 1, 1]
    # k = 2 größte Werte 8 und 4 über dem Schwellenwert x_(3) = 2: Mittel von ln 4 und ln 2 = 1,0397; 1 + 1/1,0397
    assert A.hill_estimate(x, 0.4, kmin_count=2) == pytest.approx(1 + 1 / ((math.log(4) + math.log(2)) / 2))
    assert math.isnan(A.hill_estimate([3, 3, 3, 3, 3], 0.4, kmin_count=2))      # Schwellenwert gleich allen größeren Werten
    assert math.isnan(A.hill_estimate([5, 4], 0.5, kmin_count=2))                # k nicht kleiner als die Zahl der Werte


def test_ccdf_helpers_by_hand():
    assert A.powerlaw_tail_ccdf([1, 2], 2.0, 1).tolist() == pytest.approx([1.0, 1 - 1 / ZETA2])
    assert A.exponential_tail_ccdf([3, 5], 3, math.log(2)).tolist() == pytest.approx([1.0, 0.25])
    ln = A.lognormal_tail_ccdf([4, 10], 4, 1.0, 1.0)
    assert ln[0] == pytest.approx(1.0) and 0 < ln[1] < 1


def test_generators_have_the_documented_support():
    rng = np.random.default_rng(2)
    assert A.poisson_degrees(500, 8.0, rng).min() >= 1
    assert A.lognormal_degrees(500, 1.3, 1.1, rng).min() >= 1
    deg = A.ba_degrees(300, 3, rng)
    assert deg.sum() == 2 * (3 * 4 // 2 + (300 - 4) * 3) and deg.min() == 3
    assert len(A.ba_degrees(2, 3, rng)) == 4       # n wird auf m + 1 angehoben


def test_scan_xmin_keeps_the_first_of_two_equal_distances(monkeypatch):
    seen = []

    def fake_dist(t, a, xm):
        seen.append(xm)
        return 0.25

    monkeypatch.setattr(A, "ks_distance", fake_dist)
    d, xmin, a = A.scan_xmin(np.array([1, 1, 2, 2, 3, 3]), 2, 5)
    assert seen == [1, 2, 3] and xmin == 1 and d == 0.25         # bei gleichem Abstand gewinnt die kleinere Untergrenze
