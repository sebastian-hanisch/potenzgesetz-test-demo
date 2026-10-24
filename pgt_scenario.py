"""Szenario: die Stichprobe der Grade je Datenquelle."""

import numpy as np

import pgt_algorithm as A
import pgt_constants as C


def make_sample(source: str, n: int, alpha: float, sigma: float, seed: int):
    """n Grade aus der gewählten Quelle (deterministisch je Zufalls-Seed)."""
    rng = np.random.default_rng(seed)
    if source == "powerlaw":
        return A.sample_powerlaw(n, alpha, C.PL_XMIN, rng)
    if source == "cutoff":
        return np.minimum(A.sample_powerlaw(n, alpha, C.CUTOFF_XMIN, rng), C.CUTOFF_AT)
    if source == "lognormal":
        return A.lognormal_degrees(n, C.LOGNORMAL_MU, sigma, rng)
    if source == "poisson":
        return A.poisson_degrees(n, C.POISSON_MEAN, rng)
    if source == "ba":
        return A.ba_degrees(n, C.BA_M, rng)
    raise ValueError(f"unbekannte Quelle: {source}")
