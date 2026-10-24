"""Einmalige Messung der README-Zahlen (nicht Teil der App): feste Pläne, Ausgabe auf stdout."""

import sys
import time

sys.path.insert(0, ".")
import pgt_constants as C
import pgt_evaluation as ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

t0 = time.time()
print("PRESETS")
for name, p in C.PRESETS.items():
    a = ev.analyse(ev.Settings(**p))
    comps = {k: (c.winner, round(c.p, 3), round(c.r, 2)) for k, c in a.comparisons.items()}
    print(name, "xmin", a.xmin, "ntail", a.ntail, "alpha", round(a.alpha, 3), "ks", round(a.ks, 4), "p", round(a.p, 3), comps, "hill", round(a.hill, 3), [round(v, 2) for _, v in a.hill_curve])
print("TYPES")
for r in ev.series_types():
    print(r)
print("POWER")
for r in ev.series_power():
    print(r)
print("HILL")
for r in ev.series_hill():
    print(r["name"], [round(v, 2) for _, v in r["curve"]], round(r["spread"], 3))
print("sec", round(time.time() - t0, 1))
