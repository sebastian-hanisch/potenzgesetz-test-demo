"""Auswertung: Analyse, Messreihen (Struktur und Richtung), feste Pläne."""

import math

import numpy as np
import pytest

import pgt_constants as C
import pgt_evaluation as ev


def test_num_uses_a_decimal_comma():
    assert ev.num(3.14159, 2) == "3,14" and ev.num(10, 0) == "10" and ev.num(2.5, 2) == "2,50"


def test_settings_defaults_are_the_documented_ones():
    s = ev.Settings()
    assert (s.source, s.n, s.alpha, s.sigma, s.nboot, s.frac, s.seed) == ("powerlaw", 3000, 2.5, 1.1, 30, 10, 19)


def test_analyse_is_deterministic_and_consistent():
    s = ev.Settings(n=1500, nboot=10, seed=3)
    a, b = ev.analyse(s), ev.analyse(s)
    assert (a.x == b.x).all() and a.p == b.p and a.alpha == b.alpha and a.xmin == b.xmin
    assert a.ntail == int((a.x >= a.xmin).sum()) and set(a.comparisons) == {"Log-Normal", "Exponential"}
    assert 0.0 <= a.p <= 1.0 and a.ks > 0 and math.isfinite(a.lam) and a.lognormal[1] > 0


def test_fit_sample_without_bootstrap_has_no_p_value():
    f = ev.fit_sample(np.random.default_rng(1).integers(1, 20, size=500), 1, np.random.default_rng(2), with_bootstrap=False)
    assert math.isnan(f["p"])


def test_comparison_winner_follows_the_sign_of_r():
    f = ev.fit_sample(ev.make_sample("poisson", 3000, 2.5, 1.1, 1), 5, np.random.default_rng(1))
    for c in f["comparisons"].values():
        assert c.r < 0 and c.winner == "alternative"       # eine Poisson-Verteilung passt in beiden Vergleichen besser als ein Potenzgesetz


def test_series_types_structure_and_direction():
    rows = {r["source"]: r for r in ev.series_types(n=1500, reps=4, nboot=8, sources=("powerlaw", "poisson"))}
    assert set(rows) == {"powerlaw", "poisson"}
    assert rows["poisson"]["plausible"] == 0.0 and rows["poisson"]["ln_better"] == 1.0 and rows["poisson"]["exp_better"] == 1.0
    assert rows["powerlaw"]["plausible"] >= 0.5 and rows["powerlaw"]["reps"] == 4
    for r in rows.values():
        assert r["ln_better"] + r["ln_undecided"] + r["pl_better_ln"] == pytest.approx(1.0) and r["name"] == C.SOURCES[r["source"]]


def test_series_power_structure_and_direction():
    rows = ev.series_power(ns=(1000, 10000), reps=6, sources=("lognormal",))
    by = {r["n"]: r for r in rows}
    assert set(by) == {1000, 10000} and all(r["source"] == "lognormal" for r in rows)
    assert by[10000]["undecided"] <= by[1000]["undecided"]
    for r in rows:
        assert r["undecided"] + r["ln_better"] + r["pl_better"] == pytest.approx(1.0)


def test_series_hill_structure_and_direction():
    rows = {r["source"]: r for r in ev.series_hill(n=2000, reps=4, sources=("powerlaw", "poisson"))}
    assert [f for f, _ in rows["powerlaw"]["curve"]] == list(C.HILL_FRACS)
    use = [v for _, v in ev.hill_curve_usable(rows["powerlaw"]["curve"], 2000)]
    assert len(use) == len(C.HILL_FRACS) - 1 and rows["powerlaw"]["spread"] == pytest.approx(max(use) - min(use))      # 2 % von 2 000 sind nur 40 Werte
    assert rows["powerlaw"]["spread"] < rows["poisson"]["spread"]


def test_the_fixed_plans_are_the_documented_ones():
    assert C.SERIES_SEED == 100000 and (C.SERIES_N, C.SERIES_REPS, C.SERIES_BOOT) == (5000, 10, 20)
    assert C.POWER_NS == (1000, 3000, 10000, 30000) and C.POWER_REPS == 20
    assert C.HILL_FRACS == (0.02, 0.05, 0.1, 0.2, 0.35, 0.5) and (C.NMIN_TAIL, C.NCAND) == (50, 25)
    assert (C.P_PLAUSIBLE, C.P_VUONG, C.HILL_STABLE_SPREAD, C.HILL_MIN_COUNT) == (0.1, 0.1, 0.3, 100) and set(C.SOURCES) == {"powerlaw", "lognormal", "poisson", "cutoff", "ba"}
