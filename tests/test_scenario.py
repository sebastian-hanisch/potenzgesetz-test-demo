"""Szenario: jede Quelle liefert die dokumentierte Stichprobe, deterministisch je Zufalls-Seed."""

import numpy as np
import pytest

import pgt_constants as C
from pgt_scenario import make_sample


def test_every_source_is_deterministic_and_seed_dependent():
    for src in C.SOURCES:
        a, b, c = (make_sample(src, 800, 2.5, 1.1, s) for s in (1, 1, 2))
        assert len(a) == 800 and (a == b).all() and not (a == c).all(), src


def test_support_of_each_source():
    assert make_sample("powerlaw", 3000, 2.5, 1.1, 1).min() >= C.PL_XMIN
    cut = make_sample("cutoff", 3000, 2.0, 1.1, 1)
    assert cut.min() >= C.CUTOFF_XMIN and cut.max() == C.CUTOFF_AT       # bei α = 2 liegen sicher Werte über der Grenze
    assert make_sample("poisson", 3000, 2.5, 1.1, 1).min() >= 1
    assert make_sample("lognormal", 3000, 2.5, 1.1, 1).min() >= 1
    ba = make_sample("ba", 1000, 2.5, 1.1, 1)
    assert ba.min() == C.BA_M and ba.sum() == 2 * (C.BA_M * (C.BA_M + 1) // 2 + (1000 - C.BA_M - 1) * C.BA_M)


def test_the_parameters_reach_the_generators():
    assert make_sample("powerlaw", 4000, 2.0, 1.1, 1).mean() > make_sample("powerlaw", 4000, 3.5, 1.1, 1).mean()
    assert make_sample("lognormal", 4000, 2.5, 1.9, 1).max() > make_sample("lognormal", 4000, 2.5, 0.5, 1).max()
    assert make_sample("poisson", 4000, 2.5, 1.1, 1).mean() == pytest.approx(C.POISSON_MEAN + 1, abs=0.2)


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        make_sample("zipf", 10, 2.5, 1.1, 1)


def test_the_power_law_sample_has_the_documented_head():
    from scipy import special

    x = make_sample("powerlaw", 20000, 2.5, 1.1, 3)
    assert np.mean(x == C.PL_XMIN) == pytest.approx(C.PL_XMIN ** -2.5 / special.zeta(2.5, C.PL_XMIN), abs=0.01)
