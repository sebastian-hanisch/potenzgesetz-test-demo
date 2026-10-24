"""Verdikte (reine Funktionen): Art und jede Zahl im Text an handgesetzten Werten, einschließlich der Schwellen und Zweige."""

import numpy as np

import pgt_evaluation as ev
from pgt_evaluation import Analysis, Comparison, Settings


def comp(name, winner, p=0.5, r=1.0):
    return Comparison(name, r, 0.0, p, winner)


def make(p=0.4, ln="undecided", ex="undecided", hill_curve=None, hill=2.57, frac=10, xmin=3, alpha=2.48, ks=0.013, ntail=2950, n=3000):
    curve = hill_curve or [(0.02, 2.5), (0.1, 2.57), (0.5, 2.45)]
    return Analysis(settings=Settings(frac=frac), x=np.zeros(n, dtype=int), xmin=xmin, alpha=alpha, ks=ks, ntail=ntail, p=p,
                    comparisons={"Log-Normal": comp("Log-Normal", ln, 0.36), "Exponential": comp("Exponential", ex, 0.0)}, hill_curve=curve, hill=hill)


def test_decide_boundaries_and_signs():
    assert ev.decide(5.0, 0.0999) == "powerlaw" and ev.decide(-5.0, 0.0999) == "alternative"
    assert ev.decide(5.0, 0.1) == "undecided" and ev.decide(-5.0, 0.5) == "undecided"


def test_verdict_rejected_power_law():
    kind, text = ev.verdict_test(make(p=0.03))
    assert kind == "warning" and "Das Potenzgesetz wird verworfen (p unter 0,1)" in text
    assert "xmin = 3" in text and "α̂ = 2,48" in text and "2950 von 3000 Werten" in text and "KS-Abstand 0,013" in text and "Güte p = 0,03" in text


def test_verdict_threshold_is_p_equals_one_tenth():
    assert ev.verdict_test(make(p=0.0999))[0] == "warning"
    assert ev.verdict_test(make(p=0.1))[0] == "info"


def test_verdict_an_alternative_clearly_better_overrides_a_plausible_fit():
    kind, text = ev.verdict_test(make(p=0.4, ln="alternative"))
    assert kind == "warning" and "nicht verworfen, aber Log-Normal passt klar besser" in text
    kind, text = ev.verdict_test(make(p=0.4, ln="alternative", ex="alternative"))
    assert "Log-Normal und Exponential passt klar besser" in text
    assert "gegen Log-Normal: Alternative klar besser (p = 0,36)" in text


def test_verdict_all_undecided_is_info_and_says_not_proven():
    kind, text = ev.verdict_test(make(p=0.4))
    assert kind == "info" and "alle Vergleiche sind unentschieden" in text and "nicht verworfen heißt nicht bewiesen" in text
    assert "gegen Exponential: unentschieden (p = 0,00)" in text


def test_verdict_success_needs_a_clear_win_and_no_clear_loss():
    kind, text = ev.verdict_test(make(p=0.4, ex="powerlaw"))
    assert kind == "success" and "keine Alternative passt klar besser" in text and "gegen Exponential: Potenzgesetz klar besser" in text
    assert ev.verdict_test(make(p=0.4, ln="undecided", ex="undecided"))[0] == "info"       # ohne klaren Gewinn kein „success“


def test_hill_curve_usable_counts_only_fractions_with_enough_values():
    curve = [(0.02, 3.0), (0.05, 2.6), (0.1, 2.5), (0.5, float("nan"))]
    assert ev.hill_curve_usable(curve, 3000) == [(0.05, 2.6), (0.1, 2.5)]       # 2 % von 3 000 sind 60 Werte, 5 % sind 150; nan fällt weg
    assert ev.hill_curve_usable(curve, 2000) == [(0.05, 2.6), (0.1, 2.5)]       # 5 % von 2 000 sind genau 100 Werte: noch erlaubt
    assert ev.hill_curve_usable(curve, 1999) == [(0.1, 2.5)]                      # 99 Werte: nicht mehr
    assert ev.hill_curve_usable([(0.02, 1.0)], 100) == []                         # mindestens zehn Werte, aber nie unter HILL_MIN_COUNT


def test_hill_verdict_threshold_and_numbers():
    kind, text = ev.verdict_hill(make(hill_curve=[(0.05, 2.5), (0.1, 2.6), (0.5, 2.8)], hill=2.6))
    assert kind == "success" and "Spannweite 0,30" in text and "stabil" in text and "nicht stabil" not in text      # genau an der Schwelle: noch stabil
    assert "Schwanzanteil 10 %: 2,60" in text and "von 5 % bis 50 % (mindestens 100 Werte im Schwanz)" in text and "zwischen 2,50 und 2,80" in text
    kind, text = ev.verdict_hill(make(hill_curve=[(0.05, 2.5), (0.1, 2.6), (0.5, 2.81)], hill=2.6))
    assert kind == "info" and "nicht stabil (Spannweite 0,31)" in text


def test_hill_verdict_ignores_the_noisy_small_fraction_and_missing_values():
    # der Anteil 2 % hat bei n = 3 000 nur 60 Werte und zählt nicht, auch wenn er weit abweicht
    kind, text = ev.verdict_hill(make(hill_curve=[(0.02, 3.5), (0.1, 2.6), (0.5, 2.7)], hill=2.6))
    assert kind == "success" and "zwischen 2,60 und 2,70" in text and "von 10 % bis 50 %" in text
    kind, text = ev.verdict_hill(make(hill_curve=[(0.02, float("nan")), (0.1, 2.6), (0.5, 2.7)], hill=2.6))
    assert kind == "success"


def test_hill_verdict_needs_two_usable_points():
    kind, text = ev.verdict_hill(make(hill_curve=[(0.02, 3.5), (0.1, 2.6)], hill=2.6))
    assert kind == "info" and "Für die Stabilität reichen die Werte nicht" in text and "mindestens 100 Werten" in text and "weniger als zwei" in text


def rows_types():
    mk = lambda src, name, plaus, lnb: dict(source=src, name=name, reps=10, plausible=plaus, ln_better=lnb)
    return [mk("powerlaw", "Potenzgesetz (exakt)", 0.9, 0.1), mk("lognormal", "Log-Normal", 0.4, 0.4), mk("ba", "Barabási–Albert-Netz", 0.75, 0.0), mk("poisson", "Poisson (Erdős–Rényi)", 0.0, 1.0)]


def test_types_verdict_numbers():
    kind, text = ev.verdict_types(rows_types())
    assert kind == "info" and "exakte Potenzgesetz in 90 % nicht verworfen" in text and "Log-Normal in 40 %" in text and "Barabási–Albert-Netz in 75 %" in text and "Poisson in 0 %" in text
    assert "beim exakten Potenzgesetz in 10 %, beim Barabási–Albert-Netz in 0 %" in text and "je 10 Stichproben" in text


def test_power_verdict_uses_the_smallest_and_largest_size():
    rows = [dict(source="lognormal", name="Log-Normal", n=n, undecided=u, ln_better=b) for n, u, b in ((30000, 0.0, 1.0), (1000, 0.8, 0.2), (3000, 0.6, 0.4))]
    rows.append(dict(source="powerlaw", name="Potenzgesetz (exakt)", n=1000, undecided=1.0, ln_better=0.0))
    kind, text = ev.verdict_power(rows)
    assert kind == "info" and "bei n = 1000 in 80 % unentschieden und bei n = 30000 in 0 %" in text and "in 20 % bzw. 100 %" in text


def test_hill_series_verdict():
    rows = [dict(source=s, name=s, spread=v) for s, v in (("powerlaw", 0.13), ("ba", 0.15), ("lognormal", 1.09), ("poisson", 7.27))]
    kind, text = ev.verdict_hill_series(rows)
    assert kind == "info" and "exaktes Potenzgesetz 0,13" in text and "Barabási–Albert 0,15" in text and "Log-Normal 1,09" in text and "Poisson 7,27" in text
    assert "Schwanzanteile 2 % bis 50 %" in text
