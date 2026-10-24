"""Abbildungen: Spuren je Figur, Werte der Kurven, Achsen gesperrt."""

import numpy as np
import pytest

import pgt_algorithm as A
import pgt_constants as C
import pgt_evaluation as ev
import pgt_visualization as V


@pytest.fixture(scope="module")
def analysis():
    return ev.analyse(ev.Settings(n=1500, nboot=8, seed=3))


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_ccdf_points_by_hand():
    ks, ps = V.ccdf_points([1, 1, 2, 4])
    assert ks.tolist() == [1, 2, 4] and ps.tolist() == [1.0, 0.5, 0.25]      # Anteil der Werte ≥ k


def test_ccdf_figure_has_data_and_three_fitted_curves(analysis):
    fig = V.build_ccdf(analysis)
    assert [t.name for t in fig.data] == ["gemessen", "Potenzgesetz", "Log-Normal", "Exponential"] and _locked(fig)
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].x0 == analysis.xmin        # die Linie bei der Untergrenze
    pl = fig.data[1]
    # bei der Untergrenze liegt die angepasste Kurve beim Anteil der Werte im Schwanz
    assert pl.x[0] == analysis.xmin and pl.y[0] == pytest.approx(analysis.ntail / len(analysis.x))
    assert fig.data[2].y[0] == pytest.approx(analysis.ntail / len(analysis.x)) and fig.data[3].y[0] == pytest.approx(analysis.ntail / len(analysis.x))


def test_hill_figure_marks_the_chosen_fraction(analysis):
    fig = V.build_hill(analysis)
    assert [t.name for t in fig.data] == ["Exponent der Anpassung", "Hill-Schätzer"] and _locked(fig)
    assert list(fig.data[0].y) == [analysis.alpha, analysis.alpha]
    assert fig.layout.shapes[0].x0 == analysis.settings.frac and list(fig.data[1].x) == [f * 100 for f, _ in analysis.hill_curve]


def test_types_figure_uses_percent():
    rows = [dict(name="A", plausible=0.9, ln_better=0.1), dict(name="B", plausible=0.0, ln_better=1.0)]
    fig = V.build_types(rows)
    assert [t.name for t in fig.data] == ["Potenzgesetz nicht verworfen (p ≥ 0,1)", "Log-Normal klar besser"] and _locked(fig)
    assert list(fig.data[0].y) == [90.0, 0.0] and list(fig.data[1].y) == [10.0, 100.0]


def test_power_figure_draws_one_line_per_source_sorted_by_size():
    rows = [dict(source="lognormal", name="Log-Normal", n=n, undecided=u) for n, u in ((3000, 0.6), (1000, 0.8))] + [dict(source="powerlaw", name="Potenzgesetz (exakt)", n=1000, undecided=1.0)]
    fig = V.build_power(rows)
    assert [t.name for t in fig.data] == ["Daten: Log-Normal", "Daten: Potenzgesetz (exakt)"] and _locked(fig)
    assert list(fig.data[0].x) == [1000, 3000] and list(fig.data[0].y) == [80.0, 60.0] and fig.layout.xaxis.type == "log"


def test_hill_series_figure_one_line_per_source():
    rows = ev.series_hill(n=1000, reps=2, sources=("powerlaw", "poisson"))
    fig = V.build_hill_series(rows)
    assert [t.name for t in fig.data] == [C.SOURCES["powerlaw"], C.SOURCES["poisson"]] and _locked(fig)
    assert list(fig.data[0].x) == [f * 100 for f in C.HILL_FRACS] and np.allclose(fig.data[0].y, [v for _, v in rows[0]["curve"]])
