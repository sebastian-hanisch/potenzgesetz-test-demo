"""Randfälle: kleinste Stichprobe, lauter gleiche Werte, jede Quelle an den Reglergrenzen, Zufallsrand der Ziehung."""

import math

import numpy as np
import pytest

import pgt_algorithm as A
import pgt_constants as C
import pgt_evaluation as ev
import pgt_visualization as V


def test_pipeline_runs_for_a_sample_of_exactly_the_minimum_tail():
    x = A.sample_powerlaw(C.NMIN_TAIL, 2.5, 1, np.random.default_rng(1))
    f = ev.fit_sample(x, 5, np.random.default_rng(2))
    assert f["ntail"] >= 1 and math.isfinite(f["alpha"]) and 0.0 <= f["p"] <= 1.0


def test_pipeline_runs_for_a_sample_of_one_repeated_value():
    f = ev.fit_sample(np.full(200, 2), 5, np.random.default_rng(3))
    assert f["xmin"] == 2 and f["ntail"] == 200
    assert math.isnan(f["hill"])           # Schwellenwert gleich allen größeren Werten
    assert all(c.winner == "undecided" for c in f["comparisons"].values())      # keine Streuung in den Differenzen


def test_every_source_runs_at_the_smallest_and_largest_settings():
    for src in C.SOURCES:
        for n, nboot in ((C.N_MIN, C.NBOOT_MIN), (C.N_MAX, C.NBOOT_MIN)):
            a = ev.analyse(ev.Settings(source=src, n=n, alpha=C.ALPHA_MIN, sigma=C.SIGMA_MAX, nboot=nboot, frac=C.FRAC_MAX, seed=1))
            assert a.ntail >= 1 and 0.0 <= a.p <= 1.0 and len(a.hill_curve) >= len(C.HILL_FRACS)
    for src in ("powerlaw", "cutoff", "lognormal"):
        a = ev.analyse(ev.Settings(source=src, alpha=C.ALPHA_MAX, sigma=C.SIGMA_MIN, nboot=C.NBOOT_MIN, frac=C.FRAC_MIN, seed=2))
        assert math.isfinite(a.alpha)


def test_the_hill_curve_contains_the_chosen_fraction():
    a = ev.analyse(ev.Settings(n=1000, nboot=10, frac=7, seed=1))
    assert 0.07 in [f for f, _ in a.hill_curve] and a.hill == pytest.approx(dict(a.hill_curve)[0.07])
    assert [f for f, _ in a.hill_curve] == sorted(f for f, _ in a.hill_curve)


def test_a_draw_at_the_upper_edge_of_the_sampler_is_clamped_to_the_last_value():
    class Edge:
        def random(self, n):
            return np.full(n, 1.0)

    assert A.sample_powerlaw(3, 2.5, 2, Edge(), kmax=10).tolist() == [9, 9, 9]       # letzter Wert des abgeschnittenen Trägers


def test_bootstrap_with_all_values_in_the_tail_needs_no_values_below():
    x = A.sample_powerlaw(400, 2.5, 1, np.random.default_rng(4))
    d, xmin, a = A.scan_xmin(x, 50, 5)
    assert xmin == 1
    p = A.bootstrap_pvalue(x, d, xmin, a, 5, 50, 5, np.random.default_rng(5))
    assert 0.0 <= p <= 1.0


def test_bootstrap_with_nothing_in_the_tail_range_is_still_defined():
    x = np.array([1] * 60 + [2] * 40)
    d, xmin, a = A.scan_xmin(x, 10, 5)
    assert 0.0 <= A.bootstrap_pvalue(x, d, xmin, a, 4, 10, 5, np.random.default_rng(6)) <= 1.0


def test_figures_survive_small_and_degenerate_samples():
    f = ev.fit_sample(np.full(100, 2), 3, np.random.default_rng(7))
    a = ev.Analysis(settings=ev.Settings(), x=np.full(100, 2), **f)
    assert len(V.build_ccdf(a).data) == 4 and len(V.build_hill(a).data) == 2


def test_lognormal_fit_of_very_tight_data_stops_at_the_lower_bound_of_the_spread():
    ll, mu, sg = A.lognormal_pointwise_loglik([5] * 50 + [6] * 2, 5)
    assert sg == pytest.approx(0.05, abs=0.01) and np.isfinite(ll).all()


def test_fit_alpha_needs_exactly_two_values():
    assert math.isfinite(A.fit_alpha([2, 3], 1)) and math.isnan(A.fit_alpha([2], 1))


def test_vuong_with_two_values_uses_their_spread():
    r, z, p = A.vuong([0, 0], [-1, -3])       # Differenzen 1 und 3: R = 4, Standardabweichung √2, z = 4 / (√2 · √2) = 2
    assert r == 4.0 and z == pytest.approx(2.0) and p == pytest.approx(0.04550026, abs=1e-6)


def test_bootstrap_counts_every_repetition_and_keeps_the_sample_size(monkeypatch):
    sizes = []

    def fake_scan(x, nmin, ncand):
        sizes.append(len(x))
        return 0.5, 2, 2.5

    monkeypatch.setattr(A, "scan_xmin", fake_scan)
    x = np.array([1] * 30 + [2] * 40 + [3] * 30)            # 30 Werte unter xmin = 2
    assert A.bootstrap_pvalue(x, 0.5, 2, 2.5, 4, 5, 5, np.random.default_rng(1)) == 1.0       # Gleichstand zählt als Treffer: alle vier Wiederholungen
    assert A.bootstrap_pvalue(x, 0.6, 2, 2.5, 4, 5, 5, np.random.default_rng(1)) == 0.0
    assert sizes == [100] * 8                                # jede synthetische Stichprobe hat genau n Werte
