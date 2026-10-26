"""Skalenfrei? Der Potenzgesetz-Test - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Linie „Netzmodelle“: Ist die Gradverteilung ein Potenzgesetz, und wie sicher ist das? Maximum-Likelihood-Anpassung mit Wahl der Untergrenze, Güte durch Bootstrap,
Vergleich mit Log-Normal und Exponential (Vuong) und die Stabilitätskurve des Hill-Schätzers. Der Kern liegt in pgt_algorithm.py.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import pgt_constants as C
import pgt_evaluation as ev
from pgt_evaluation import Settings, analyse, num, verdict_hill, verdict_hill_series, verdict_power, verdict_test, verdict_types
from pgt_layout import dunkles_design, kopf_css
from pgt_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from pgt_visualization import build_ccdf, build_hill, build_hill_series, build_power, build_types

st.set_page_config(page_title="Skalenfrei? Der Potenzgesetz-Test – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _series_types():
    return ev.series_types()


@st.cache_data(show_spinner=False)
def _series_power():
    return ev.series_power()


@st.cache_data(show_spinner=False)
def _series_hill():
    return ev.series_hill()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in pgt_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📏 Skalenfrei? Der Potenzgesetz-Test")
st.markdown(
    """
Viele Netze gelten als **skalenfrei**: Ihre Gradverteilung soll einem **Potenzgesetz** folgen. Aber wie prüft man das? Ein gerader Strich im Log-Log-Bild beweist nichts, und der geschätzte Exponent hängt davon ab, ab welchem Grad man anpasst.
Die Demo folgt dem Verfahren von Clauset, Shalizi und Newman: den Exponenten per **Maximum Likelihood** schätzen, die **Untergrenze** über den kleinsten KS-Abstand wählen, die **Güte** durch einen Bootstrap prüfen und
das Potenzgesetz auf demselben Schwanz mit **Log-Normal** und **Exponential** vergleichen (Vuong-Test). Dazu die **Stabilitätskurve des Hill-Schätzers**, die eine zweite Lesart zeigt.
Viertes Stück der Linie „Netzmodelle“; es prüft, ob die Gradverteilungen aus den Wachstumsmodellen und aus Daten wirklich Potenzgesetze sind. Die Demo entscheidet den Streit zwischen den Lesarten nicht, sie zeigt beide nebeneinander.

Weiter unten: 🔬 **Messreihen** (Trefferquote je Quelle, Trennschärfe gegen die Stichprobengröße, Hill-Kurven), 🚧 **Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So prüft der Test", expanded=True):
    st.markdown(
        """
1. **Stichprobe:** die Grade eines Netzes bzw. Werte aus einer gewählten Verteilung.
2. **Anpassung:** für jede Untergrenze xmin den Exponenten per Maximum Likelihood schätzen; xmin ist die Untergrenze mit dem kleinsten **KS-Abstand** zwischen Daten und angepasstem Potenzgesetz.
3. **Güte:** viele synthetische Stichproben aus dem angepassten Potenzgesetz (mit den Daten unter xmin), jeweils neu angepasst; der Anteil mit mindestens dem beobachteten KS-Abstand ist der **p-Wert**. Unter 0,1 wird das Potenzgesetz verworfen.
4. **Vergleich:** Log-Likelihood-Quotient (Vuong) gegen Log-Normal und Exponential auf demselben Schwanz; nur bei p unter 0,1 entscheidet das Vorzeichen, sonst bleibt es **unentschieden**.
5. **Hill-Kurve:** der Exponent aus den größten Werten, aufgetragen über dem Schwanzanteil: bleibt sie flach, ist der Schwanz mit einem Potenzgesetz vereinbar (eine flache Kurve zeigt Stabilität, keinen Beweis). Für die Beurteilung zählen nur Anteile mit mindestens 100 Werten im Schwanz.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Stichprobe**")
    source = st.selectbox("Quelle der Grade", list(C.SOURCES), format_func=C.SOURCES.get, key="source_select",
                          help="Woher die Grade stammen: exakt gezogenes Potenzgesetz, Log-Normal, Poisson-Grade eines Zufallsnetzes, Potenzgesetz mit Abbruch bei 150 oder die Grade eines Barabási–Albert-Netzes.")
    n = st.slider("Zahl der Werte n", *bounds("n_slider"), key="n_slider", step=100, help="Wie viele Grade die Stichprobe hat. Mit mehr Werten trennt der Test schärfer.")
    alpha = st.slider("Exponent α des Erzeugers", *bounds("alpha_slider"), key="alpha_slider", step=0.1, help="Nur für „Potenzgesetz (exakt)“ und „Potenzgesetz mit Abbruch“.")
    sigma = st.slider("Streuung σ der Log-Normal-Quelle", *bounds("sigma_slider"), key="sigma_slider", step=0.1, help="Nur für „Log-Normal“: Streuung von ln(Grad); je größer, desto schwerer der Schwanz.")
    st.markdown("**Test**")
    nboot = st.slider("Wiederholungen des Bootstraps", *bounds("nboot_slider"), key="nboot_slider", step=10, help="Mehr Wiederholungen geben einen genaueren p-Wert, rechnen aber länger.")
    frac = st.slider("Schwanzanteil des Hill-Schätzers (%)", *bounds("frac_slider"), key="frac_slider", help="Welcher Anteil der größten Werte in den Hill-Schätzer eingeht.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"source_select": source, "n_slider": int(n), "alpha_slider": round(float(alpha), 1), "sigma_slider": round(float(sigma), 1), "nboot_slider": int(nboot), "frac_slider": int(frac), "seed_input": int(seed)})

settings = Settings(source=source, n=int(n), alpha=round(float(alpha), 1), sigma=round(float(sigma), 1), nboot=int(nboot), frac=int(frac), seed=int(seed))
with st.spinner(f"Passe an und rechne den Bootstrap ({settings.nboot} Wiederholungen) …"):
    r = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Untergrenze xmin", f"{r.xmin}", delta=f"{r.ntail} von {len(r.x)} Werten im Schwanz", delta_color="off")
m2.metric("Exponent α̂", num(r.alpha, 2), delta=f"KS-Abstand {num(r.ks, 3)}", delta_color="off")
m3.metric("Güte p", num(r.p, 2), delta="nicht verworfen" if r.p >= C.P_PLAUSIBLE else "verworfen (p unter 0,1)", delta_color="off")
m4.metric("Hill-Schätzer", num(r.hill, 2), delta=f"bei {settings.frac} % Schwanzanteil", delta_color="off")
c1, c2 = st.columns(2)
c1.plotly_chart(build_ccdf(r), width="stretch", key="main_ccdf")
c2.plotly_chart(build_hill(r), width="stretch", key="main_hill")
st.caption(
    "Links die Komplementärverteilung der Grade (Anteil der Knoten mit Grad ≥ k, logarithmische Achsen) mit den drei angepassten Verteilungen ab der Untergrenze (gestrichelte Linie); "
    "rechts der Hill-Schätzer über dem Schwanzanteil neben dem Exponenten der Anpassung."
)

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist die Gradverteilung ein Potenzgesetz?")
kind, text = verdict_test(r)
getattr(st, kind)(text)
kind, text = verdict_hill(r)
getattr(st, kind)(text)

st.markdown("---")

# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Messreihen: Trefferquote, Trennschärfe und Hill-Kurven")
st.caption(
    f"Feste Programme, unabhängig von den Reglern: je Quelle {C.SERIES_REPS} Stichproben mit {C.SERIES_N} Werten und {C.SERIES_BOOT} Bootstrap-Wiederholungen; "
    f"Trennschärfe mit {C.POWER_REPS} Stichproben je Größe von {C.POWER_NS[0]} bis {C.POWER_NS[-1]}; Mediane bzw. Anteile. Jede Reihe rechnet auf Klick (je etwa 30 bis 60 Sekunden, auf langsameren Rechnern länger)."
)
tab1, tab2, tab3 = st.tabs(["Wie oft wird verworfen?", "Wann trennt der Test?", "Hill-Kurven"])
with tab1:
    st.markdown("**Ein Potenzgesetz wird nicht immer erkannt, und anderes wird nicht immer verworfen.** Anteil der Stichproben je Quelle, bei denen das Potenzgesetz nicht verworfen wird, neben dem Anteil mit klar besserer Log-Normal-Verteilung.")
    if st.button("Reihe rechnen", key="types_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe (etwa 30 bis 60 Sekunden) …"):
            _series_types()
        st.session_state["types_done"] = time.perf_counter() - t0
    if "types_done" in st.session_state:
        rows = _series_types()
        st.plotly_chart(build_types(rows), width="stretch", key="types_chart")
        kind, text = verdict_types(rows)
        getattr(st, kind)(text)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['types_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")
with tab2:
    st.markdown("**Wer wenige Werte hat, kann Log-Normal und Potenzgesetz oft nicht trennen.** Anteil der Stichproben, bei denen der Vergleich unentschieden bleibt, über der Stichprobengröße.")
    if st.button("Reihe rechnen", key="power_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe (etwa 30 bis 60 Sekunden) …"):
            _series_power()
        st.session_state["power_done"] = time.perf_counter() - t0
    if "power_done" in st.session_state:
        rows = _series_power()
        st.plotly_chart(build_power(rows), width="stretch", key="power_chart")
        kind, text = verdict_power(rows)
        getattr(st, kind)(text)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['power_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")
with tab3:
    st.markdown("**Die Hill-Kurve zeigt, ob der Exponent vom Schwanzanteil abhängt.** Median über die festen Stichproben je Quelle.")
    rows = _series_hill()
    st.plotly_chart(build_hill_series(rows), width="stretch", key="hill_chart")
    kind, text = verdict_hill_series(rows)
    getattr(st, kind)(text)

st.markdown("---")

# --- Zwei Lesarten -------------------------------------------------------------------------------------------------------------------------

st.subheader("⚖️ Zwei Lesarten von „skalenfrei“")
st.markdown(
    """
| | Strenge Lesart | Weite Lesart |
|---|---|---|
| **Frage** | Folgt der Schwanz einem Potenzgesetz, das besser passt als die Alternativen? | Hat der Schwanz eine regelmäßig variierende Form mit Exponent zwischen 2 und 3? |
| **Werkzeug** | Maximum Likelihood, Güte durch Bootstrap, Vergleich mit Alternativen (diese Demo oben) | Hill-Schätzer und seine Stabilitätskurve (diese Demo oben, rechts) |
| **Folge** (nach der Literatur, nicht in dieser Demo gemessen) | Wenige Netze bestehen den Test | Mehr Netze gelten als skalenfrei |
| **Was die Demo zeigt** | Wie oft ein exaktes Potenzgesetz verworfen und eine Log-Normal-Verteilung nicht erkannt wird | Wie stabil die Kurve bei Potenzgesetz und Log-Normal ist |

Die Demo entscheidet nicht, welche Lesart die richtige ist; sie zeigt, was jede der beiden Rechnungen an Stichproben mit **bekanntem** Ursprung liefert. Zuordnung zu den Arbeiten: aus dem Gedächtnis, nicht im Original geprüft (Broido und Clauset 2019 für die strenge Lesart, Voitalov und andere 2019 für die weite).
"""
)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Werte sind unabhängige Ziehungen** | Grade eines Netzes sind voneinander abhängig (sie summieren sich zu 2m); der Test behandelt sie wie eine unabhängige Stichprobe. | Konfigurationsmodell (Vorgänger) |
| **Das Potenzgesetz beginnt bei einer Untergrenze** | Die Wahl über den KS-Abstand ist selbst eine Schätzung; eine andere Untergrenze gibt einen anderen Exponenten und eine andere Güte. | diese Demo (Hill-Kurve) |
| **Der Bootstrap zählt die Wahl der Untergrenze mit** | Mit wenigen Wiederholungen ist der p-Wert grob (bei 30 Wiederholungen und p nahe 0,1 liegt der Zufallsfehler bei etwa 0,05); die Grenze 0,1 ist eine Konvention. | diese Demo (Regler) |
| **Der Vergleich nimmt zwei fremde Verteilungen** | Log-Normal und Potenzgesetz überlappen im Schwanz weitgehend; der Vuong-Test trennt sie erst bei vielen Werten (Messreihe „Wann trennt der Test?“). | diese Demo (Trennschärfe) |
| **Ein Netz ist eine Stichprobe aus einem Modell** | Ein einzelnes, endliches Netz kann ein skalenfreies Modell nicht von einem anderen trennen; für die Modellwahl braucht es mehr als die Gradverteilung. | Modellwahl (Folgestück) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Grade $k_1,\dots,k_n$; gesucht ist, ob ihr Schwanz einem Potenzgesetz folgt.

| Symbol | Bedeutung |
|---|---|
| $x_{\min}$ | Untergrenze, ab der das Potenzgesetz gelten soll |
| $\alpha$ | Exponent |
| $N$ | Zahl der Werte mit $k_i \ge x_{\min}$ (Schwanz) |
| $\zeta(\alpha, x_{\min})$ | Hurwitz-Zeta-Funktion $\sum_{j \ge 0} (x_{\min} + j)^{-\alpha}$ |

**Diskretes Potenzgesetz.** Für $k \ge x_{\min}$ ist

$$P(k) = \frac{k^{-\alpha}}{\zeta(\alpha, x_{\min})}.$$

**Maximum Likelihood.** Die Log-Likelihood der Schwanzwerte ist

$$\ell(\alpha) = -\alpha \sum_{i} \ln k_i \; - \; N \ln \zeta(\alpha, x_{\min}),$$

in $\alpha$ konkav, das Maximum also eindeutig.

**Wahl der Untergrenze.** Für jede Kandidatin $x_{\min}$ wird $\hat\alpha$ geschätzt und der KS-Abstand $D = \max_k |F_{\text{Daten}}(k) - F_{\text{Modell}}(k)|$ über $k \ge x_{\min}$ berechnet; gewählt wird die Untergrenze mit dem kleinsten $D$.

**Güte.** Eine synthetische Stichprobe nimmt die Werte unter $x_{\min}$ aus den Daten und zieht die übrigen aus dem angepassten Potenzgesetz; sie wird samt Wahl der Untergrenze neu angepasst. Der Anteil der Stichproben mit mindestens dem beobachteten $D$ ist der $p$-Wert; unter $0{,}1$ wird das Potenzgesetz verworfen.

**Vuong-Test.** Mit den punktweisen Log-Likelihoods $\ell^{(1)}_i, \ell^{(2)}_i$ zweier Verteilungen auf demselben Schwanz ist

$$R = \sum_i \big(\ell^{(1)}_i - \ell^{(2)}_i\big), \qquad z = \frac{R}{\hat\sigma \sqrt{N}},$$

wobei $\hat\sigma$ die Standardabweichung der Differenzen ist; $p = 2\,(1 - \Phi(|z|))$. Nur bei $p < 0{,}1$ zählt das Vorzeichen von $R$.

**Hill-Schätzer.** Mit den absteigend sortierten Werten $x_{(1)} \ge x_{(2)} \ge \dots$ und $k$ Werten im Schwanz:

$$\hat\alpha_{\text{Hill}} = 1 + \Big( \tfrac{1}{k} \sum_{i=1}^{k} \ln \tfrac{x_{(i)}}{x_{(k+1)}} \Big)^{-1}.$$

**Literatur** (aus dem Gedächtnis, nicht im Original geprüft): Clauset, Shalizi und Newman, *Power-law distributions in empirical data*, SIAM Review 51 (2009); Vuong, *Likelihood ratio tests for model selection and non-nested hypotheses*, Econometrica 57 (1989);
Broido und Clauset, *Scale-free networks are rare*, Nature Communications 10 (2019); Voitalov, van der Hoorn, van der Hofstad und Krioukov, *Scale-free networks well done*, Physical Review Research 1 (2019).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
