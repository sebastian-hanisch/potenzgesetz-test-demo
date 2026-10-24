"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import pgt_algorithm as A
import pgt_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def ccdf_points(x):
    """Empirische Komplementärverteilung P(K ≥ k) an den vorkommenden Werten: (Werte, Anteile)."""
    x = np.sort(np.asarray(x))
    ks = np.unique(x)
    return ks, 1.0 - np.searchsorted(x, ks, side="left") / len(x)


def build_ccdf(a):
    """Komplementäre Verteilungsfunktion der Grade (beide Achsen logarithmisch) mit den drei angepassten Verteilungen ab der gewählten Untergrenze."""
    ks, ps = ccdf_points(a.x)
    n, nt = len(a.x), a.ntail
    share = nt / n
    grid = np.unique(np.round(np.geomspace(a.xmin, max(int(ks.max()), a.xmin + 1), 80)).astype(int))
    mu, sg = a.lognormal
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=ps, mode="markers", marker=dict(size=6, color=C.COLOR_MEASURED), name="gemessen", hovertemplate="Grad ≥ %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=share * A.powerlaw_tail_ccdf(grid, a.alpha, a.xmin), mode="lines", line=dict(color=C.COLOR_THEORY, width=3), name="Potenzgesetz"))
    fig.add_trace(go.Scatter(x=grid, y=share * A.lognormal_tail_ccdf(grid, a.xmin, mu, sg), mode="lines", line=dict(color=C.COLOR_THIRD, width=2, dash="dash"), name="Log-Normal"))
    fig.add_trace(go.Scatter(x=grid, y=share * A.exponential_tail_ccdf(grid, a.xmin, a.lam), mode="lines", line=dict(color=C.COLOR_ALERT, width=2, dash="dot"), name="Exponential"))
    fig.add_vline(x=a.xmin, line=dict(color="rgba(120,120,120,0.7)", width=1.5, dash="dash"))
    fig.update_xaxes(title_text="Grad k", type="log")
    fig.update_yaxes(title_text="Anteil der Knoten mit Grad ≥ k", type="log", range=[np.log10(max(1.0 / n * 0.5, 1e-6)), 0.05])
    return _base(fig)


def build_hill(a):
    """Hill-Schätzer über dem Schwanzanteil (Stabilitätskurve) neben dem Exponenten der Anpassung; der gewählte Anteil ist markiert."""
    fr = [f * 100 for f, _ in a.hill_curve]
    vals = [v for _, v in a.hill_curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[min(fr), max(fr)], y=[a.alpha, a.alpha], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name="Exponent der Anpassung"))
    fig.add_trace(go.Scatter(x=fr, y=vals, mode="lines+markers", line=dict(color=C.COLOR_MEASURED, width=3), name="Hill-Schätzer",
                             hovertemplate="Schwanzanteil %{x:.0f} %: %{y:.2f}<extra></extra>"))
    fig.add_vline(x=a.settings.frac, line=dict(color=C.COLOR_ALERT, width=1.5, dash="dash"))
    fig.update_xaxes(title_text="Schwanzanteil (%)", type="log")
    fig.update_yaxes(title_text="geschätzter Exponent")
    return _base(fig)


def build_types(rows):
    """Anteil der Stichproben je Quelle, bei denen das Potenzgesetz nicht verworfen wird, und Anteil mit klar besserer Log-Normal-Verteilung."""
    names = [r["name"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r["plausible"] * 100 for r in rows], name="Potenzgesetz nicht verworfen (p ≥ 0,1)", marker_color=C.COLOR_MEASURED))
    fig.add_trace(go.Bar(x=names, y=[r["ln_better"] * 100 for r in rows], name="Log-Normal klar besser", marker_color=C.COLOR_THIRD))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Quelle der Stichprobe", tickangle=-20)
    fig.update_yaxes(title_text="Anteil der Stichproben (%)", range=[0, 105])
    return _base(fig, height=420)


def build_power(rows):
    """Anteil der Stichproben je Größe, bei denen der Vergleich mit der Log-Normal-Verteilung unentschieden bleibt (je Quelle eine Linie)."""
    fig = go.Figure()
    colors = {"lognormal": C.COLOR_THIRD, "powerlaw": C.COLOR_THEORY}
    for src in dict.fromkeys(r["source"] for r in rows):
        rs = sorted((r for r in rows if r["source"] == src), key=lambda r: r["n"])
        fig.add_trace(go.Scatter(x=[r["n"] for r in rs], y=[r["undecided"] * 100 for r in rs], mode="lines+markers", line=dict(color=colors.get(src, C.COLOR_MEASURED), width=3),
                                 name=f"Daten: {rs[0]['name']}", hovertemplate="n = %{x}: %{y:.0f} % unentschieden<extra></extra>"))
    fig.update_xaxes(title_text="Stichprobengröße n", type="log")
    fig.update_yaxes(title_text="Vergleich unentschieden (%)", range=[-5, 105])
    return _base(fig)


def build_hill_series(rows):
    """Hill-Kurven (Median über die festen Stichproben) je Quelle über dem Schwanzanteil."""
    colors = {"powerlaw": C.COLOR_THEORY, "lognormal": C.COLOR_THIRD, "ba": C.COLOR_MEASURED, "poisson": C.COLOR_ALERT}
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=[f * 100 for f, _ in r["curve"]], y=[v for _, v in r["curve"]], mode="lines+markers", line=dict(color=colors.get(r["source"], C.COLOR_MEASURED), width=3), name=r["name"],
                                 hovertemplate="Schwanzanteil %{x:.0f} %: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Schwanzanteil (%)", type="log")
    fig.update_yaxes(title_text="geschätzter Exponent (Median)", type="log")
    return _base(fig, height=420)
