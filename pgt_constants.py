"""Konstanten: Regler-Grenzen, Parameter der Test-Pipeline, feste Pläne der Messreihen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 500, 10000, 3000      # Stichprobe: Zahl der Knoten (Grade)
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA = 2.0, 3.5, 2.5      # Exponent des Erzeugers (Potenzgesetz und Potenzgesetz mit Abbruch)
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 0.5, 2.0, 1.1      # Streuung des Log-Normal-Erzeugers
NBOOT_MIN, NBOOT_MAX, DEFAULT_NBOOT = 10, 100, 30        # Wiederholungen des Bootstraps der Güte
FRAC_MIN, FRAC_MAX, DEFAULT_FRAC = 2, 50, 10             # Schwanzanteil des Hill-Schätzers in Prozent
SEED_MAX, DEFAULT_SEED = 999999, 19   # Standard-Seed 19: zeigt bei allen Schnellstart-Szenarien den häufigen Ausgang (README nennt die Auswahl)

SOURCES = {
    "powerlaw": "Potenzgesetz (exakt)",
    "lognormal": "Log-Normal",
    "poisson": "Poisson (Erdős–Rényi)",
    "cutoff": "Potenzgesetz mit Abbruch",
    "ba": "Barabási–Albert-Netz",
}
DEFAULT_SOURCE = "powerlaw"

# feste Parameter der Erzeuger, die kein Regler sind
PL_XMIN = 3          # Untergrenze des exakten Potenzgesetzes
CUTOFF_XMIN = 2      # Untergrenze des Potenzgesetzes mit Abbruch
CUTOFF_AT = 150      # Werte über dieser Grenze werden auf sie gekappt
LOGNORMAL_MU = 1.3
POISSON_MEAN = 8.0
BA_M = 3

# Test-Pipeline
NMIN_TAIL = 50       # mindestens so viele Werte müssen oberhalb der Untergrenze bleiben
NCAND = 25           # höchstens so viele Kandidaten für die Untergrenze (die kleinsten)
P_PLAUSIBLE = 0.1    # Güte p ≥ 0,1: das Potenzgesetz wird nicht verworfen
P_VUONG = 0.1        # Vorzeichen des Vuong-Tests zählt nur bei p < 0,1
HILL_FRACS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)   # Schwanzanteile der Stabilitätskurve
HILL_STABLE_SPREAD = 0.3    # Spannweite der Hill-Kurve bis dahin „stabil“
HILL_MIN_COUNT = 100        # nur Schwanzanteile mit mindestens so vielen Werten zählen für die Stabilität (bei weniger dominiert der Zufallsfehler)

# feste Pläne der Messreihen (nie ein einzelner Seed)
SERIES_SEED = 100000     # Stichprobe j einer Reihe hat den Seed SERIES_SEED + j
SERIES_N, SERIES_REPS, SERIES_BOOT = 5000, 10, 20
POWER_NS, POWER_REPS = (1000, 3000, 10000, 30000), 20

COLOR_MEASURED = "#2F6B65"
COLOR_THEORY = "#4c78a8"
COLOR_THIRD = "#e8a33d"
COLOR_ALERT = "#d62728"

PRESETS = {
    "Exaktes Potenzgesetz (α = 2,5)": {"source": "powerlaw", "n": DEFAULT_N, "alpha": DEFAULT_ALPHA, "sigma": DEFAULT_SIGMA, "nboot": DEFAULT_NBOOT, "frac": DEFAULT_FRAC, "seed": DEFAULT_SEED},
    "Log-Normal, kleine Stichprobe": {"source": "lognormal", "n": 1000, "alpha": DEFAULT_ALPHA, "sigma": DEFAULT_SIGMA, "nboot": DEFAULT_NBOOT, "frac": DEFAULT_FRAC, "seed": DEFAULT_SEED},
    "Log-Normal, große Stichprobe": {"source": "lognormal", "n": 9000, "alpha": DEFAULT_ALPHA, "sigma": DEFAULT_SIGMA, "nboot": DEFAULT_NBOOT, "frac": DEFAULT_FRAC, "seed": DEFAULT_SEED},
    "Poisson-Grade (Zufallsnetz)": {"source": "poisson", "n": DEFAULT_N, "alpha": DEFAULT_ALPHA, "sigma": DEFAULT_SIGMA, "nboot": DEFAULT_NBOOT, "frac": DEFAULT_FRAC, "seed": DEFAULT_SEED},
    "Barabási–Albert-Netz": {"source": "ba", "n": DEFAULT_N, "alpha": DEFAULT_ALPHA, "sigma": DEFAULT_SIGMA, "nboot": DEFAULT_NBOOT, "frac": DEFAULT_FRAC, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Exaktes Potenzgesetz (α = 2,5)": "3 000 Werte, exakt aus einem Potenzgesetz gezogen: der Test sollte es nicht verwerfen und den Exponenten nahe 2,5 schätzen.",
    "Log-Normal, kleine Stichprobe": "1 000 Log-Normal-Werte: bei so wenigen Werten wird das Potenzgesetz nicht verworfen, und der Vergleich mit der Log-Normal-Verteilung bleibt unentschieden.",
    "Log-Normal, große Stichprobe": "9 000 Log-Normal-Werte: mit mehr Daten wird der Unterschied sichtbar, das Potenzgesetz wird verworfen und die Log-Normal-Verteilung passt klar besser.",
    "Poisson-Grade (Zufallsnetz)": "Grade eines Zufallsnetzes ohne Hubs: der Test verwirft das Potenzgesetz klar.",
    "Barabási–Albert-Netz": "Grade eines gewachsenen Netzes mit bevorzugter Anbindung (3 000 Knoten): das Potenzgesetz wird nicht verworfen; die Messreihe zeigt, wie oft das bei anderen Zufalls-Seeds so ausgeht.",
}
