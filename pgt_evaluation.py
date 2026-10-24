"""Kennzahlen, Messreihen und Verdikte (reine Funktionen: die App ruft st.<Art>(Text), die Tests prüfen jede Zahl im Text)."""

import math
from dataclasses import dataclass, field
from statistics import median

import numpy as np

import pgt_algorithm as A
import pgt_constants as C
from pgt_scenario import make_sample


def num(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma (für Texte)."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    source: str = C.DEFAULT_SOURCE
    n: int = C.DEFAULT_N
    alpha: float = C.DEFAULT_ALPHA
    sigma: float = C.DEFAULT_SIGMA
    nboot: int = C.DEFAULT_NBOOT
    frac: int = C.DEFAULT_FRAC     # Prozent
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Comparison:
    """Vuong-Vergleich des Potenzgesetzes mit einer Alternative auf demselben Schwanz."""

    name: str
    r: float          # Summe der Log-Likelihood-Differenzen (Potenzgesetz minus Alternative)
    z: float
    p: float
    winner: str       # "powerlaw", "alternative" oder "undecided"


@dataclass(eq=False)
class Analysis:
    settings: Settings
    x: np.ndarray
    xmin: int
    alpha: float
    ks: float
    ntail: int
    p: float                       # Güte (Bootstrap)
    comparisons: dict = field(default_factory=dict)      # Name -> Comparison
    lognormal: tuple = (float("nan"), float("nan"))      # (μ, σ) der angepassten Log-Normal-Verteilung
    lam: float = float("nan")                            # λ der angepassten Exponentialverteilung
    hill_curve: list = field(default_factory=list)       # [(Anteil, Schätzwert)]
    hill: float = float("nan")                           # Hill-Schätzer beim gewählten Schwanzanteil


def decide(r: float, p: float) -> str:
    """Vorzeichen von R entscheidet nur bei p < P_VUONG (sonst unentschieden)."""
    if p < C.P_VUONG:
        return "powerlaw" if r > 0 else "alternative"
    return "undecided"


def fit_sample(x, nboot: int, rng: np.random.Generator, frac: float = C.DEFAULT_FRAC / 100.0, with_bootstrap: bool = True) -> dict:
    """Die ganze Pipeline für eine Stichprobe: Untergrenze und Exponent, Güte, Vergleiche mit Log-Normal und Exponential, Hill-Kurve."""
    x = np.asarray(x)
    d, xmin, alpha = A.scan_xmin(x, C.NMIN_TAIL, C.NCAND)
    tail = x[x >= xmin]
    ntail = len(tail)
    p = A.bootstrap_pvalue(x, d, xmin, alpha, nboot, C.NMIN_TAIL, C.NCAND, rng) if with_bootstrap else float("nan")
    lpl = -alpha * np.log(tail.astype(float)) - math.log(A.special.zeta(alpha, xmin))
    lln, mu, sg = A.lognormal_pointwise_loglik(tail, xmin)
    lex, lam = A.exponential_pointwise_loglik(tail, xmin)
    comps = {}
    for name, ll in (("Log-Normal", lln), ("Exponential", lex)):
        r, z, pv = A.vuong(lpl, ll)
        comps[name] = Comparison(name, r, z, pv, decide(r, pv))
    fracs = sorted(set(C.HILL_FRACS) | {round(frac, 4)})
    curve = [(f, A.hill_estimate(x, f)) for f in fracs]
    return dict(xmin=int(xmin), alpha=float(alpha), ks=float(d), ntail=int(ntail), p=float(p), comparisons=comps, lognormal=(mu, sg), lam=lam, hill_curve=curve, hill=A.hill_estimate(x, frac))


def analyse(s: Settings) -> Analysis:
    x = make_sample(s.source, s.n, s.alpha, s.sigma, s.seed)
    rng = np.random.default_rng(s.seed + 1)
    return Analysis(settings=s, x=x, **fit_sample(x, s.nboot, rng, s.frac / 100.0))


# --- Messreihen ------------------------------------------------------------------------------------------------------------------------------


def _sample(source: str, n: int, j: int):
    return make_sample(source, n, C.DEFAULT_ALPHA, C.DEFAULT_SIGMA, C.SERIES_SEED + j)


def series_types(n: int = C.SERIES_N, reps: int = C.SERIES_REPS, nboot: int = C.SERIES_BOOT, sources=tuple(C.SOURCES)) -> list[dict]:
    """Je Quelle `reps` feste Stichproben: Anteil mit nicht verworfener Güte, Anteil mit klar besserer Log-Normal- bzw. Exponentialverteilung, Mediane."""
    rows = []
    for src in sources:
        fits = [fit_sample(_sample(src, n, j), nboot, np.random.default_rng(C.SERIES_SEED + 7919 + j)) for j in range(reps)]
        ln = [f["comparisons"]["Log-Normal"].winner for f in fits]
        ex = [f["comparisons"]["Exponential"].winner for f in fits]
        rows.append(dict(
            source=src, name=C.SOURCES[src], reps=reps,
            plausible=sum(f["p"] >= C.P_PLAUSIBLE for f in fits) / reps,
            ln_better=ln.count("alternative") / reps, ln_undecided=ln.count("undecided") / reps, pl_better_ln=ln.count("powerlaw") / reps,
            exp_better=ex.count("alternative") / reps,
            alpha=median(f["alpha"] for f in fits), xmin=median(f["xmin"] for f in fits), ntail=median(f["ntail"] for f in fits),
        ))
    return rows


def series_power(ns=C.POWER_NS, reps: int = C.POWER_REPS, sources=("lognormal", "powerlaw")) -> list[dict]:
    """Trennschärfe des Vergleichs mit der Log-Normal-Verteilung gegen die Stichprobengröße (ohne Bootstrap): Anteil unentschieden und Anteil mit klar besserer Log-Normal-Verteilung."""
    rows = []
    for src in sources:
        for n in ns:
            winners = [fit_sample(_sample(src, n, j), 1, np.random.default_rng(1), with_bootstrap=False)["comparisons"]["Log-Normal"].winner for j in range(reps)]
            rows.append(dict(source=src, name=C.SOURCES[src], n=n, reps=reps, undecided=winners.count("undecided") / reps,
                             ln_better=winners.count("alternative") / reps, pl_better=winners.count("powerlaw") / reps))
    return rows


def series_hill(n: int = C.SERIES_N, reps: int = C.SERIES_REPS, sources=("powerlaw", "lognormal", "ba", "poisson")) -> list[dict]:
    """Median der Hill-Schätzwerte je Schwanzanteil (Stabilitätskurve) und Spannweite der Kurve."""
    rows = []
    for src in sources:
        samples = [_sample(src, n, j) for j in range(reps)]
        curve = [(f, median(A.hill_estimate(x, f) for x in samples)) for f in C.HILL_FRACS]
        vals = [v for _, v in hill_curve_usable(curve, n)]
        rows.append(dict(source=src, name=C.SOURCES[src], curve=curve, spread=max(vals) - min(vals)))
    return rows


# --- Verdikte --------------------------------------------------------------------------------------------------------------------------------

WINNER_TEXT = {"powerlaw": "Potenzgesetz klar besser", "alternative": "Alternative klar besser", "undecided": "unentschieden"}


def verdict_test(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: wird das Potenzgesetz verworfen, und was sagen die Vergleiche mit den Alternativen?"""
    n = len(a.x)
    head = f"Angepasst ab xmin = {a.xmin} mit α̂ = {num(a.alpha, 2)} ({a.ntail} von {n} Werten im Schwanz), KS-Abstand {num(a.ks, 3)}, Güte p = {num(a.p, 2)}."
    comps = "; ".join(f"gegen {c.name}: {WINNER_TEXT[c.winner]} (p = {num(c.p, 2)})" for c in a.comparisons.values())
    better = [c.name for c in a.comparisons.values() if c.winner == "alternative"]
    if a.p < C.P_PLAUSIBLE:
        return "warning", f"{head} Das Potenzgesetz wird verworfen (p unter {num(C.P_PLAUSIBLE, 1)}). Vergleiche: {comps}."
    if better:
        return "warning", f"{head} Das Potenzgesetz wird nicht verworfen, aber {' und '.join(better)} passt klar besser. Vergleiche: {comps}."
    if all(c.winner == "undecided" for c in a.comparisons.values()):
        return "info", (f"{head} Das Potenzgesetz wird nicht verworfen, aber auch nichts anderes: alle Vergleiche sind unentschieden ({comps}). "
                        "Die Daten reichen nicht, um es von den Alternativen zu trennen; nicht verworfen heißt nicht bewiesen.")
    return "success", f"{head} Das Potenzgesetz wird nicht verworfen, und keine Alternative passt klar besser. Vergleiche: {comps}."


def hill_curve_usable(curve, n: int) -> list:
    """Punkte der Hill-Kurve, bei denen mindestens HILL_MIN_COUNT Werte im Schwanz liegen und der Schätzwert definiert ist."""
    return [(f, v) for f, v in curve if max(10, int(f * n)) >= C.HILL_MIN_COUNT and v == v]


def verdict_hill(a: Analysis) -> tuple[str, str]:
    """Hill-Schätzer: Wert beim gewählten Schwanzanteil und Spannweite der Kurve über die Anteile mit genug Werten."""
    use = hill_curve_usable(a.hill_curve, len(a.x))
    head = f"Hill-Schätzer beim Schwanzanteil {a.settings.frac} %: {num(a.hill, 2)}"
    if len(use) < 2:
        return "info", f"{head}. Für die Stabilität reichen die Werte nicht: nur Schwanzanteile mit mindestens {C.HILL_MIN_COUNT} Werten zählen, und davon gibt es hier weniger als zwei."
    vals = [v for _, v in use]
    spread = max(vals) - min(vals)
    head += (f"; über die Anteile von {num(use[0][0] * 100, 0)} % bis {num(use[-1][0] * 100, 0)} % (mindestens {C.HILL_MIN_COUNT} Werte im Schwanz) "
             f"liegt er zwischen {num(min(vals), 2)} und {num(max(vals), 2)}")
    if spread <= C.HILL_STABLE_SPREAD:
        return "success", f"{head}: die Kurve ist stabil (Spannweite {num(spread, 2)}), das ist mit einem Potenzgesetz im Schwanz vereinbar."
    return "info", f"{head}: die Kurve ist nicht stabil (Spannweite {num(spread, 2)}), der Schätzwert hängt davon ab, wie viel vom Schwanz man nimmt."


def verdict_types(rows: list[dict]) -> tuple[str, str]:
    """Messreihe über die Quellen: Anteil nicht verworfener Potenzgesetze je Quelle."""
    by = {r["source"]: r for r in rows}
    first = by["powerlaw"]
    return "info", (
        f"Bei n = {C.SERIES_N} (je {first['reps']} Stichproben) wird das exakte Potenzgesetz in {num(first['plausible'] * 100, 0)} % nicht verworfen, "
        f"Log-Normal in {num(by['lognormal']['plausible'] * 100, 0)} %, das Barabási–Albert-Netz in {num(by['ba']['plausible'] * 100, 0)} % und Poisson in {num(by['poisson']['plausible'] * 100, 0)} %; "
        f"eine klar bessere Log-Normal-Verteilung findet der Vergleich beim exakten Potenzgesetz in {num(first['ln_better'] * 100, 0)} %, beim Barabási–Albert-Netz in {num(by['ba']['ln_better'] * 100, 0)} %."
    )


def verdict_power(rows: list[dict]) -> tuple[str, str]:
    """Messreihe Trennschärfe: Anteil unentschiedener Vergleiche bei Log-Normal-Daten, kleinste und größte Stichprobe."""
    ln = sorted((r for r in rows if r["source"] == "lognormal"), key=lambda r: r["n"])
    lo, hi = ln[0], ln[-1]
    return "info", (
        f"Bei Log-Normal-Daten bleibt der Vergleich mit dem Potenzgesetz bei n = {lo['n']} in {num(lo['undecided'] * 100, 0)} % unentschieden und bei n = {hi['n']} in {num(hi['undecided'] * 100, 0)} %; "
        f"die Log-Normal-Verteilung gewinnt klar in {num(lo['ln_better'] * 100, 0)} % bzw. {num(hi['ln_better'] * 100, 0)} %. Wer wenige Werte hat, kann beide Verteilungen nicht trennen."
    )


def verdict_hill_series(rows: list[dict]) -> tuple[str, str]:
    """Messreihe Hill-Kurve: Spannweite je Quelle."""
    by = {r["source"]: r for r in rows}
    return "info", (
        f"Spannweite der Hill-Kurve (Median über {C.SERIES_REPS} Stichproben, Schwanzanteile {num(C.HILL_FRACS[0] * 100, 0)} % bis {num(C.HILL_FRACS[-1] * 100, 0)} %): "
        f"exaktes Potenzgesetz {num(by['powerlaw']['spread'], 2)}, Barabási–Albert {num(by['ba']['spread'], 2)}, Log-Normal {num(by['lognormal']['spread'], 2)}, Poisson {num(by['poisson']['spread'], 2)}."
    )
