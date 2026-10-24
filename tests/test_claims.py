"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4): über dieselben Auswertungsfunktionen wie in der App, mit den festen Plänen.

Die Zufallszahlen eines numpy-Generators sind über Versionen nicht stabil: die Messwerte gelten deshalb als Bänder um die in der README genannten Zahlen."""

import math

import pytest

import pgt_constants as C
import pgt_evaluation as ev


@pytest.fixture(scope="module")
def presets():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


@pytest.fixture(scope="module")
def types():
    return {r["source"]: r for r in ev.series_types()}


@pytest.fixture(scope="module")
def power():
    return {(r["source"], r["n"]): r for r in ev.series_power()}


@pytest.fixture(scope="module")
def hill():
    return {r["source"]: r for r in ev.series_hill()}


def test_the_series_use_the_documented_fixed_plans():
    assert C.SERIES_SEED == 100000 and (C.SERIES_N, C.SERIES_REPS, C.SERIES_BOOT) == (5000, 10, 20) and C.POWER_NS == (1000, 3000, 10000, 30000) and C.POWER_REPS == 20
    assert (C.N_MIN, C.N_MAX, C.DEFAULT_N, C.DEFAULT_NBOOT, C.DEFAULT_SEED) == (500, 10000, 3000, 30, 19) and (C.PL_XMIN, C.CUTOFF_XMIN, C.CUTOFF_AT, C.BA_M) == (3, 2, 150, 3)


def test_the_exact_power_law_preset(presets):
    a = presets["Exaktes Potenzgesetz (α = 2,5)"]
    assert 3 <= a.xmin <= 6 and a.alpha == pytest.approx(2.53, abs=0.15) and a.p >= 0.1 and a.ntail == pytest.approx(1852, rel=0.3)
    assert a.comparisons["Log-Normal"].winner == "undecided" and a.comparisons["Exponential"].winner == "powerlaw"
    assert a.hill == pytest.approx(2.66, abs=0.2)


def test_the_small_log_normal_preset_is_not_rejected(presets):
    a = presets["Log-Normal, kleine Stichprobe"]
    assert a.p >= 0.1 and a.comparisons["Log-Normal"].winner == "undecided" and 80 <= a.ntail <= 250 and a.alpha == pytest.approx(2.94, abs=0.4)


def test_the_large_log_normal_preset_is_rejected_and_the_log_normal_wins(presets):
    a = presets["Log-Normal, große Stichprobe"]
    assert a.p < 0.1 and a.comparisons["Log-Normal"].winner == "alternative" and a.comparisons["Log-Normal"].r < 0
    assert a.alpha == pytest.approx(2.87, abs=0.35) and a.ntail == pytest.approx(1193, rel=0.35)


def test_the_poisson_preset_is_rejected_by_everything(presets):
    a = presets["Poisson-Grade (Zufallsnetz)"]
    assert a.p < 0.1 and a.alpha >= 7.0 and all(c.winner == "alternative" for c in a.comparisons.values())
    assert a.hill > 10


def test_the_barabasi_albert_preset_is_not_rejected(presets):
    a = presets["Barabási–Albert-Netz"]
    assert a.p >= 0.1 and a.alpha == pytest.approx(2.74, abs=0.2) and a.comparisons["Log-Normal"].winner == "undecided"


def test_the_preset_seed_is_a_typical_outcome_not_the_only_one():
    """README: die Zufalls-Seeds der Schnellstart-Szenarien zeigen den häufigen Ausgang. Mit Zufalls-Seed 35 (früherer Standard) wird dasselbe exakte Potenzgesetz verworfen."""
    a = ev.analyse(ev.Settings(seed=35))
    assert a.p < 0.1


def test_how_often_each_source_is_rejected(types):
    assert types["powerlaw"]["plausible"] >= 0.7 and types["ba"]["plausible"] >= 0.5 and types["cutoff"]["plausible"] >= 0.4
    assert 0.2 <= types["lognormal"]["plausible"] <= 0.7
    assert types["poisson"]["plausible"] == 0.0 and types["poisson"]["ln_better"] == 1.0 and types["poisson"]["exp_better"] == 1.0
    assert types["poisson"]["plausible"] < types["lognormal"]["plausible"] < types["powerlaw"]["plausible"] + 0.01
    assert types["powerlaw"]["alpha"] == pytest.approx(2.50, abs=0.1) and types["powerlaw"]["xmin"] == 3.0 and types["ba"]["alpha"] == pytest.approx(2.76, abs=0.15)
    assert types["lognormal"]["alpha"] == pytest.approx(3.06, abs=0.4)
    assert types["powerlaw"]["ln_better"] <= 0.3 and types["powerlaw"]["pl_better_ln"] == 0.0


def test_log_normal_data_are_hard_to_tell_from_a_power_law_with_few_values(power):
    und = [power[("lognormal", n)]["undecided"] for n in C.POWER_NS]
    assert und[0] >= 0.6 and und[-1] <= 0.15 and und == sorted(und, reverse=True)
    assert und[0] == pytest.approx(0.8, abs=0.2) and und[2] == pytest.approx(0.25, abs=0.2)
    assert all(power[("lognormal", n)]["pl_better"] == 0.0 for n in C.POWER_NS)


def test_exact_power_law_data_are_rarely_called_log_normal(power):
    assert all(power[("powerlaw", n)]["undecided"] >= 0.75 for n in C.POWER_NS)
    assert all(power[("powerlaw", n)]["ln_better"] <= 0.2 for n in C.POWER_NS)


def test_the_hill_curve_is_flat_for_power_laws_and_not_for_log_normal(hill):
    assert hill["powerlaw"]["spread"] == pytest.approx(0.18, abs=0.12) and hill["ba"]["spread"] == pytest.approx(0.17, abs=0.12)
    assert hill["powerlaw"]["spread"] <= C.HILL_STABLE_SPREAD and hill["ba"]["spread"] <= C.HILL_STABLE_SPREAD
    assert hill["lognormal"]["spread"] == pytest.approx(1.06, abs=0.3) and hill["poisson"]["spread"] > 5
    ln = dict(hill["lognormal"]["curve"])
    assert ln[0.02] == pytest.approx(3.53, abs=0.4) and ln[0.5] == pytest.approx(2.47, abs=0.3)
    assert all(math.isfinite(v) for r in hill.values() for _, v in r["curve"])


def test_the_verdict_texts_match_the_series(types, power, hill):
    kind, text = ev.verdict_types(list(types.values()))
    assert "exakte Potenzgesetz in 90 % nicht verworfen" in text and "Poisson in 0 %" in text
    kind, text = ev.verdict_power(list(power.values()))
    assert "bei n = 1000 in 80 % unentschieden" in text and "bei n = 30000 in 0 %" in text
    kind, text = ev.verdict_hill_series(list(hill.values()))
    assert "Log-Normal 1,06" in text and "exaktes Potenzgesetz 0,18" in text
