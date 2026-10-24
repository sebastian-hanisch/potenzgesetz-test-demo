"""Kern: diskretes Potenzgesetz nach dem Verfahren von Clauset, Shalizi und Newman (eigene Umsetzung).

Einheiten, jede für sich testbar: Maximum-Likelihood-Schätzer des Exponenten bei gegebener Untergrenze, KS-Abstand, Wahl der Untergrenze (kleinster KS-Abstand),
Ziehen aus dem Potenzgesetz, semiparametrischer Bootstrap für die Güte, Vergleichsverteilungen (Log-Normal, Exponential) und der Vuong-Test, Hill-Schätzer
und die Erzeuger der Gradfolgen (Potenzgesetz, Log-Normal, Poisson, Potenzgesetz mit Abbruch, Barabási–Albert)."""

import math

import numpy as np
from scipy import optimize, special, stats

ALPHA_LO, ALPHA_HI = 1.01, 8.0     # Suchbereich des Exponenten (die Log-Likelihood ist in α konkav, das Maximum ist eindeutig)


# --- Potenzgesetz ----------------------------------------------------------------------------------------------------------------------------


def powerlaw_loglik(tail, alpha: float, xmin: int) -> float:
    """Log-Likelihood der Werte `tail` (alle ≥ xmin) unter P(k) = k^(−α) / ζ(α, xmin)."""
    t = np.asarray(tail, dtype=float)
    return float(-alpha * np.log(t).sum() - len(t) * math.log(special.zeta(alpha, xmin)))


def fit_alpha(tail, xmin: int) -> float:
    """Maximum-Likelihood-Exponent des diskreten Potenzgesetzes ab `xmin` (exakt, nicht nur auf einem Gitter)."""
    t = np.asarray(tail, dtype=float)
    if len(t) < 2:
        return float("nan")
    s = float(np.log(t).sum())
    n = len(t)
    res = optimize.minimize_scalar(lambda a: a * s + n * math.log(special.zeta(a, xmin)), bounds=(ALPHA_LO, ALPHA_HI), method="bounded", options={"xatol": 1e-8})
    return float(res.x)


def powerlaw_cdf(ks, alpha: float, xmin: int):
    """P(K ≤ k) für das diskrete Potenzgesetz ab `xmin`."""
    ks = np.asarray(ks, dtype=float)
    return 1.0 - special.zeta(alpha, ks + 1.0) / special.zeta(alpha, xmin)


def ks_distance(tail, alpha: float, xmin: int) -> float:
    """Größter Abstand zwischen der empirischen und der modellierten Verteilungsfunktion der Werte ≥ xmin (an den vorkommenden Werten)."""
    t = np.sort(np.asarray(tail))
    n = len(t)
    ks = np.unique(t)
    emp = np.searchsorted(t, ks, side="right") / n
    return float(np.max(np.abs(emp - powerlaw_cdf(ks, alpha, xmin))))


def candidate_xmins(x, nmin: int, ncand: int):
    """Kleinste vorkommende Werte, ab denen noch mindestens `nmin` Werte übrig bleiben (höchstens `ncand`); gibt es keinen, den kleinsten Wert."""
    x = np.asarray(x)
    vals = np.unique(x)
    cands = [int(v) for v in vals if (x >= v).sum() >= nmin][:ncand]
    return cands if cands else [int(vals[0])]


def scan_xmin(x, nmin: int, ncand: int):
    """Untergrenze mit dem kleinsten KS-Abstand zwischen den Daten und dem angepassten Potenzgesetz. Rückgabe: (KS-Abstand, xmin, Exponent)."""
    x = np.asarray(x)
    best = None
    for xm in candidate_xmins(x, nmin, ncand):
        t = x[x >= xm]
        a = fit_alpha(t, xm)
        if math.isnan(a):
            continue
        d = ks_distance(t, a, xm)
        if best is None or d < best[0]:
            best = (d, xm, a)
    if best is None:
        raise ValueError("zu wenige Werte für eine Anpassung")
    return best


def sample_powerlaw(n: int, alpha: float, xmin: int, rng: np.random.Generator, kmax: int = 10 ** 6):
    """n Werte aus dem diskreten Potenzgesetz ab xmin (abgeschnitten bei kmax, dort liegt weniger als ein Millionstel der Masse für α ≥ 2)."""
    ks = np.arange(xmin, kmax)
    p = ks.astype(float) ** (-alpha)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    return ks[np.minimum(np.searchsorted(cdf, rng.random(n), side="right"), len(ks) - 1)]


def bootstrap_pvalue(x, d_obs: float, xmin: int, alpha: float, nboot: int, nmin: int, ncand: int, rng: np.random.Generator) -> float:
    """Güte nach Clauset, Shalizi und Newman (semiparametrisch): jede synthetische Stichprobe nimmt die Werte unter xmin aus den Daten
    und zieht die übrigen aus dem angepassten Potenzgesetz; die Untergrenze wird je Stichprobe neu gewählt. Anteil der Stichproben mit mindestens dem beobachteten KS-Abstand."""
    x = np.asarray(x)
    n = len(x)
    below = x[x < xmin]
    ntail = n - len(below)
    hits = 0
    for _ in range(nboot):
        k = int(rng.binomial(n, ntail / n))
        parts = [sample_powerlaw(k, alpha, xmin, rng)]
        if len(below) and n - k:
            parts.append(rng.choice(below, size=n - k))
        d_b, _, _ = scan_xmin(np.concatenate(parts), nmin, ncand)
        hits += d_b >= d_obs
    return hits / nboot


# --- Vergleichsverteilungen und Vuong-Test ---------------------------------------------------------------------------------------------------


def lognormal_pointwise_loglik(tail, xmin: int):
    """Punktweise Log-Likelihood der angepassten diskretisierten Log-Normal-Verteilung auf k ≥ xmin (Wahrscheinlichkeit von k: Fläche der Normalverteilung von ln(k − 1/2) bis ln(k + 1/2),
    auf k ≥ xmin normiert); die Parameter maximieren die Likelihood der Werte (Nelder–Mead, mehrere Startwerte). Rückgabe: (Array der Log-Wahrscheinlichkeiten, μ, σ)."""
    t = np.asarray(tail)
    vals, counts = np.unique(t, return_counts=True)
    lo, hi = np.log(vals - 0.5), np.log(vals + 0.5)
    lo0 = math.log(xmin - 0.5)

    def logpmf(mu, sg):
        # Im Log-Raum gerechnet: bei großer Streuung (σ in den Zehnern) laufen Fläche und Normierung sonst gegen null und der Optimierer fände einen Scheinbestwert
        ls0, ls1 = stats.norm.logsf((lo - mu) / sg), stats.norm.logsf((hi - mu) / sg)
        log_mass = ls0 + np.log(-np.expm1(ls1 - ls0))
        return log_mass - float(stats.norm.logsf((lo0 - mu) / sg))

    def nll(p):
        mu, sg = p
        if sg < 0.05:
            return 1e18
        return -float(np.sum(counts * logpmf(mu, sg)))

    best = None
    for mu0 in (math.log(xmin), math.log(xmin) + 1.0, math.log(xmin) + 2.0):
        for sg0 in (0.7, 1.5):
            r = optimize.minimize(nll, [mu0, sg0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 600})
            if best is None or r.fun < best.fun:
                best = r
    mu, sg = float(best.x[0]), float(best.x[1])
    lp = dict(zip(vals.tolist(), logpmf(mu, sg)))
    return np.array([lp[int(v)] for v in t]), mu, sg


def exponential_pointwise_loglik(tail, xmin: int):
    """Punktweise Log-Likelihood der angepassten diskreten Exponentialverteilung P(k) = (1 − e^(−λ)) e^(−λ(k − xmin)); der Maximum-Likelihood-Wert ist geschlossen
    λ = ln(1 + 1 / (Mittelwert − xmin)). Rückgabe: (Array, λ)."""
    t = np.asarray(tail, dtype=float)
    m = float(t.mean()) - xmin
    lam = math.log1p(1.0 / m) if m > 0 else 50.0
    return math.log1p(-math.exp(-lam)) - lam * (t - xmin), lam


def vuong(ll_a, ll_b):
    """Vuong-Test: Summe R der punktweisen Log-Likelihood-Differenzen, normierte Statistik z und zweiseitiger p-Wert (ohne Streuung der Differenzen: z = 0, p = 1)."""
    d = np.asarray(ll_a, dtype=float) - np.asarray(ll_b, dtype=float)
    n = len(d)
    r = float(d.sum())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd <= 0:
        return r, 0.0, 1.0
    z = r / (sd * math.sqrt(n))
    return r, z, float(2 * stats.norm.sf(abs(z)))


# --- Hill-Schätzer ---------------------------------------------------------------------------------------------------------------------------


def hill_estimate(x, frac: float, kmin_count: int = 10) -> float:
    """Hill-Schätzer des Exponenten aus dem größten Anteil `frac` der Werte: 1 + 1 / Mittelwert von ln(x_i / x_(k)) über die k größten Werte (k mindestens kmin_count);
    nan, wenn der Schwellenwert nicht kleiner ist als alle größeren Werte."""
    s = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(kmin_count, int(frac * len(s)))
    if k >= len(s):
        return float("nan")
    m = float(np.mean(np.log(s[:k] / s[k])))
    return float("nan") if m <= 0 else 1.0 + 1.0 / m


# --- Erzeuger der Gradfolgen -----------------------------------------------------------------------------------------------------------------


def ba_degrees(n: int, m: int, rng: np.random.Generator):
    """Grade eines Barabási–Albert-Netzes (m + 1 Startknoten, jeder neue Knoten wählt m verschiedene Enden aus der Liste der Kantenenden)."""
    n = max(n, m + 1)
    deg = np.zeros(n, dtype=np.int64)
    ends = []
    for u in range(m + 1):
        for v in range(u):
            ends += [u, v]
    deg[: m + 1] = m
    u = rng.random(size=n * m * 4 + 64)
    pos = 0
    for t in range(m + 1, n):
        chosen = []
        while len(chosen) < m:
            if pos >= len(u):
                u = rng.random(size=len(u))
                pos = 0
            v = ends[int(u[pos] * len(ends))]
            pos += 1
            if v not in chosen:
                chosen.append(v)
        for v in chosen:
            deg[v] += 1
            ends.append(v)
            ends.append(t)
        deg[t] = m
    return deg


def lognormal_degrees(n: int, mu: float, sigma: float, rng: np.random.Generator):
    """Ganzzahlige Werte e^(Normal(mu, sigma)) + 1, gerundet (mindestens 1)."""
    return np.round(np.exp(rng.normal(mu, sigma, size=n)) + 1).astype(np.int64)


def poisson_degrees(n: int, mean: float, rng: np.random.Generator):
    """Poisson-Grade des Erdős–Rényi-Netzes, um eins erhöht (mindestens 1)."""
    return rng.poisson(mean, size=n).astype(np.int64) + 1


# --- Verteilungsfunktionen für die Abbildung --------------------------------------------------------------------------------------------------


def powerlaw_tail_ccdf(ks, alpha: float, xmin: int):
    """P(K ≥ k | K ≥ xmin) für das diskrete Potenzgesetz."""
    return special.zeta(alpha, np.asarray(ks, dtype=float)) / special.zeta(alpha, xmin)


def lognormal_tail_ccdf(ks, xmin: int, mu: float, sg: float):
    """P(K ≥ k | K ≥ xmin) für die diskretisierte Log-Normal-Verteilung (dieselbe Wahrscheinlichkeit je Wert wie in `lognormal_pointwise_loglik`)."""
    ks = np.asarray(ks, dtype=float)
    return np.exp(stats.norm.logsf((np.log(ks - 0.5) - mu) / sg) - stats.norm.logsf((math.log(xmin - 0.5) - mu) / sg))


def exponential_tail_ccdf(ks, xmin: int, lam: float):
    """P(K ≥ k | K ≥ xmin) für die diskrete Exponentialverteilung: e^(−λ(k − xmin))."""
    return np.exp(-lam * (np.asarray(ks, dtype=float) - xmin))
