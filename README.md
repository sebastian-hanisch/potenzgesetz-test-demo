# Skalenfrei? Der Potenzgesetz-Test – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-potenzgesetz-test-demo.streamlit.app/)** -->

Viertes Stück der **Netzmodelle-Linie** (Graphentheorie) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vorgänger ist
[wachstumsmodelle-demo](https://github.com/sebastian-hanisch/wachstumsmodelle-demo) (woher Hubs kommen können, und dass der geschätzte Exponent von der Untergrenze abhängt); dieses Stück fragt, **wie man prüft**,
ob eine Gradverteilung überhaupt ein Potenzgesetz ist. Die Linie „Netzwerkanalyse“ nennt den Test in [strukturkennzahlen-demo](https://github.com/sebastian-hanisch/strukturkennzahlen-demo) ausdrücklich als nicht gebaut.

Die Demo folgt dem Verfahren von Clauset, Shalizi und Newman (eigene Umsetzung für diskrete Daten): den Exponenten per **Maximum Likelihood** schätzen, die **Untergrenze** über den kleinsten KS-Abstand wählen,
die **Güte** durch einen semiparametrischen **Bootstrap** prüfen, das Potenzgesetz auf demselben Schwanz mit **Log-Normal** und **Exponential** vergleichen (**Vuong-Test**) und daneben die **Stabilitätskurve des Hill-Schätzers** zeigen.
Die Stichproben haben einen bekannten Ursprung (exaktes Potenzgesetz, Log-Normal, Poisson, Potenzgesetz mit Abbruch, Barabási–Albert-Netz), damit sichtbar wird, was der Test tut. `scipy` rechnet im Kern (Hurwitz-Zeta, Optimierer), `networkx` kommt nur als Gegenprobe in die Tests.

## Warum dieses Problem

„Skalenfrei“ ist eine der meistgenannten Eigenschaften von Netzen, und der Beleg ist oft ein gerader Strich im Log-Log-Bild. Der Strich beweist wenig: auch eine Log-Normal-Verteilung sieht über zwei Größenordnungen fast gerade aus, und der geschätzte Exponent hängt von der gewählten Untergrenze ab.
Die Frage der Demo: **Ist die Gradverteilung ein Potenzgesetz, wie sicher ist das, und was kann ein Test bei endlich vielen Werten überhaupt leisten?** Zwei Lesarten von „skalenfrei“ (streng und weit) stehen in der App nebeneinander; die Demo entscheidet den Streit nicht.

## Modell

Diskretes Potenzgesetz ab einer Untergrenze xmin: P(k) = k^(−α) / ζ(α, xmin) für k ≥ xmin (Hurwitz-Zeta). Die Log-Likelihood der Werte im Schwanz ist in α konkav, das Maximum eindeutig; xmin ist die Untergrenze mit dem kleinsten
KS-Abstand zwischen Daten und angepasstem Potenzgesetz. Die Güte p ist der Anteil synthetischer Stichproben (Werte unter xmin aus den Daten, der Rest aus dem angepassten Potenzgesetz, jeweils mit neuer Wahl von xmin) mit mindestens dem beobachteten KS-Abstand; unter 0,1 wird das Potenzgesetz verworfen.
Der Vuong-Test vergleicht die punktweisen Log-Likelihoods mit einer diskretisierten Log-Normal- und einer diskreten Exponentialverteilung auf demselben Schwanz: nur bei p unter 0,1 entscheidet das Vorzeichen, sonst bleibt es **unentschieden**.
Der Hill-Schätzer nimmt den größten Anteil der Werte und rechnet 1 + 1 / Mittelwert von ln(x_i / Schwellenwert). Formeln und Lesehilfe stehen im 📐-Expander der App. Die Literaturangaben dort stammen aus dem Gedächtnis und sind nicht im Original geprüft.

## Methodik

- **Kern** (`pgt_algorithm.py`): Log-Likelihood und Verteilungsfunktion des diskreten Potenzgesetzes, Maximum-Likelihood-Exponent (exakt über einen beschränkten Skalar-Optimierer), KS-Abstand, Kandidaten und Wahl der Untergrenze, Ziehen aus dem Potenzgesetz, Bootstrap der Güte,
  Log-Normal (Nelder–Mead im Log-Raum, mehrere Startwerte) und Exponential (geschlossen) mit punktweiser Log-Likelihood, Vuong-Test, Hill-Schätzer, die Erzeuger der Gradfolgen.
- **Auswertung** (`pgt_evaluation.py`): die ganze Pipeline für eine Stichprobe, drei Messreihen mit festem Programm, alle Verdikte als reine Funktionen.
- **Oberfläche** (`app.py`): fünf Schnellstart-Szenarien, Regler für Quelle, n, α, σ, Bootstrap-Wiederholungen und Schwanzanteil, Komplementärverteilung und Hill-Kurve, Kernfrage, Messreihen auf Klick, die zwei Lesarten, Annahmen und Formeln.

## Befunde (gemessen, keine Behauptungen)

Alle Reihen mit festen Zufalls-Seeds (`SERIES_SEED` 100000 und aufwärts), nie ein einzelner Seed. **Die Schnellstart-Szenarien sind einzelne Läufe mit dem Zufalls-Seed 19**, der so gewählt ist, dass jedes Szenario den häufigen Ausgang zeigt (siehe Ehrliche Grenzen).
**Die Zufallszahlen eines `numpy`-Generators sind nicht über Versionen stabil**: die Tests prüfen deshalb Bänder um diese Zahlen.

| Frage | Befund | Test |
|---|---|---|
| Stimmt der Kern? | Log-Likelihood und Verteilungsfunktion gleich `scipy.stats.zipf` (drei Parameterpaare); der Maximum-Likelihood-Exponent gleich dem Maximum eines dichten Gitters (Abstand höchstens 0,001) und erfüllt die Gleichung des Maximums (Ableitung der Zeta-Funktion); auf 30 000 Werten liegt er bei α = 2,1; 2,5 und 3,0 innerhalb von 0,05; die Untergrenze gleich einer Brute-Force-Suche; Exponential gleich der geometrischen Verteilung; Log-Normal-Wahrscheinlichkeiten summieren sich auf 1, die Anpassung findet μ = 2,0 und σ = 0,8 (innerhalb 0,06 und 0,04) und ist ein lokales Maximum; Hill-Schätzer auf einer Pareto-Stichprobe innerhalb 0,08; Erzeuger gegen `networkx.barabasi_albert_graph`. | `tests/test_oracle_potenzgesetz_test.py` |
| Wird ein exaktes Potenzgesetz erkannt? | Bei n = 5 000 (je 10 feste Stichproben) wird es in **90 %** nicht verworfen (9 von 10), der mediane Exponent ist **2,50** bei Untergrenze 3; in **10 %** findet der Vergleich eine klar bessere Log-Normal-Verteilung. Mit dem Zufalls-Seed 35 statt 19 wird dasselbe Szenario (n = 3 000, α = 2,5) verworfen (p = 0,03): der Test hat einen Fehler erster Art. | `test_claims.py::test_how_often_each_source_is_rejected`, `::test_the_preset_seed_is_a_typical_outcome_not_the_only_one` |
| Wird anderes verworfen? | Poisson-Grade werden in **0 %** der Stichproben nicht verworfen, Log-Normal und Exponential sind in **100 %** klar besser (der Exponent liegt an der Suchgrenze 8). Log-Normal-Daten werden bei n = 5 000 in **50 %** nicht verworfen, Potenzgesetz mit Abbruch bei 150 in **70 %**, das Barabási–Albert-Netz in **80 %** (mit medianem Exponenten **2,76**, Theorie für n → ∞: 3). | `test_claims.py::test_how_often_each_source_is_rejected` |
| Wann trennt der Test Log-Normal und Potenzgesetz? | Bei Log-Normal-Daten bleibt der Vergleich bei n = 1 000 in **80 %**, bei 3 000 in **65 %**, bei 10 000 in **25 %** und bei 30 000 in **0 %** unentschieden; bei exakten Potenzgesetz-Daten bleibt er in **90 bis 100 %** unentschieden. Mit wenigen Werten sind beide Verteilungen nicht zu trennen. | `test_claims.py::test_log_normal_data_are_hard_to_tell_from_a_power_law_with_few_values`, `::test_exact_power_law_data_are_rarely_called_log_normal` |
| Wie stabil ist die Hill-Kurve? | Spannweite der Kurve über die Schwanzanteile 2 % bis 50 % (Median über 10 Stichproben, n = 5 000): exaktes Potenzgesetz **0,18**, Barabási–Albert **0,17**, Log-Normal **1,06**, Poisson über **7**. Der Hill-Wert der Log-Normal-Daten fällt von **3,53** (2 %) auf **2,47** (50 %). | `test_claims.py::test_the_hill_curve_is_flat_for_power_laws_and_not_for_log_normal` |
| Was zeigen die Schnellstart-Szenarien (Zufalls-Seed 19)? | Exaktes Potenzgesetz (n = 3 000): xmin **4**, α̂ **2,53**, Güte **0,80**. Log-Normal mit 1 000 Werten: xmin 13 mit 136 Werten im Schwanz, Güte **0,73**, Vergleich mit Log-Normal **unentschieden**. Log-Normal mit 9 000 Werten: Güte **0,00**, Log-Normal klar besser (p = 0,007). Poisson: Güte 0,00, beide Alternativen klar besser. Barabási–Albert (3 000 Knoten): Güte **0,90**, α̂ **2,74**. | `test_claims.py` (je Szenario ein Test) |

## Ehrliche Grenzen

- **Die Schnellstart-Szenarien sind einzelne Läufe, und der Zufalls-Seed 19 ist gewählt.** Ich habe die Seeds 1 bis 40 durchgesehen und einen genommen, bei dem alle fünf Szenarien den häufigen Ausgang zeigen; bei anderen Seeds weichen sie ab (zum Beispiel wird das exakte Potenzgesetz bei Seed 35 verworfen). Wie oft welcher Ausgang eintritt, zeigen die Messreihen, nicht die Szenarien.
- **Die Messreihen sind klein.** Je Quelle 10 Stichproben (Schritte von 10 Prozentpunkten), für die Trennschärfe 20 je Größe (Schritte von 5), der Bootstrap hat 20 Wiederholungen (der p-Wert ist auf 0,05 genau abgestuft). Die Prozentzahlen sind Größenordnungen, keine genauen Fehlerraten.
- **Nicht verworfen heißt nicht bewiesen.** Mit wenigen Werten bleibt auch ein falsches Potenzgesetz unentdeckt (Log-Normal, n = 1 000: nicht verworfen, Vergleich unentschieden), und umgekehrt verwirft der Test ein echtes Potenzgesetz in etwa jedem zehnten Fall.
- **Der Vuong-Test passt hier nur grob.** Er setzt zwei nicht verschachtelte Modelle voraus; eine Log-Normal-Verteilung mit sehr großer Streuung nähert ein Potenzgesetz an (die Anpassung findet bei exakten Potenzgesetz-Daten Streuungen von etwa 4 bis über 100). Wie weit das die Fehlerquote des Vergleichs verändert, ist nicht untersucht.
- **Die Grenze 0,1, die Hill-Schwelle 0,3 und die Mindestzahl 100 sind Konventionen.** p unter 0,1 verwirft, eine Spannweite der Hill-Kurve bis 0,3 gilt hier als stabil, und nur Schwanzanteile mit mindestens 100 Werten zählen dafür (bei weniger dominiert der Zufallsfehler); alles ist eine eigene Festlegung der Demo, kein Satz. Bei kleinem n bleiben so wenige Punkte der Kurve übrig.
- **Grade sind keine unabhängige Stichprobe.** Sie summieren sich zu 2m und hängen über das Netz zusammen; der Test behandelt sie wie unabhängige Ziehungen.
- **Nur zwei Alternativen.** Log-Normal und Exponential; abgeschnittene Potenzgesetze, gestreckte Exponentialverteilungen und andere Vergleichsverteilungen fehlen. Der Suchbereich des Exponenten endet bei 8; bei Poisson-Daten liegt die Schätzung an dieser Grenze.
- **Keine realen Netze.** Die Demo prüft Stichproben mit bekanntem Ursprung, keine Daten; kein Beleg für oder gegen die Skalenfreiheit eines bestimmten realen Netzes.
- **Eigene Umsetzung.** Das Paket `powerlaw` steht nicht im Test-venv und ist nicht als Gegenprobe benutzt; die Gegenproben sind `scipy` und `networkx`. Die Zuordnung der beiden Lesarten zu den Arbeiten von Broido und Clauset sowie Voitalov und anderen und alle Literaturangaben sind **aus dem Gedächtnis, nicht im Original geprüft**.

## Befunde und Korrekturen gegenüber dem Plan

- **Port:** 9160 statt 9158 (9158 hält `zufallsverfahren-demo`, 9159 `wachstumsmodelle-demo`).
- **Größe der Reihen:** die Vormessung nahm n = 10 000 und 16 Stichproben je Typ mit 40 Wiederholungen; die Demo nimmt n = 5 000 mit 10 Stichproben und 20 Wiederholungen, damit eine Reihe auf Klick in etwa 30 bis 60 Sekunden rechnet (gemessen 33 bis 62 s je nach Last des Rechners).
- **Log-Normal ist besser trennbar als die Vormessung sagte:** die Vormessung fand bei n = 10 000 in 56 % keine klare Entscheidung zwischen Log-Normal-Daten und Potenzgesetz; die Demo findet **25 %**, bei n = 30 000 gar keine unentschiedenen Fälle. Vermutete Ursache (nicht gesondert geprüft): die Anpassung der Log-Normal-Verteilung; die Vormessung brach den Optimierer früh ab und schnitt den Träger bei 200 000 ab, die Demo rechnet im Log-Raum und optimiert streng. Die Aussage „bei n = 10 000 sind es 56 %“ wurde nicht übernommen.
- **Ein Rechenfehler der ersten Fassung:** bei sehr großer Streuung lief die Normierung der Log-Normal-Verteilung numerisch gegen null, der Optimierer fand einen Scheinbestwert (Log-Likelihood 0), und bei exakten Potenzgesetz-Daten schien die Log-Normal-Verteilung in einem Drittel der Fälle „klar besser“. Der Test `test_lognormal_fit_stays_finite_for_a_very_heavy_tail` hält die Korrektur fest.
- **Hill-Kurve im Einzelfall:** die erste Fassung nannte die Kurve des exakten Potenzgesetzes mit 3 000 Werten „nicht stabil“ (Spannweite 0,58), weil der Anteil 2 % nur 60 Werte hat und der Zufallsfehler dort groß ist; seither zählen nur Anteile mit mindestens 100 Werten (`HILL_MIN_COUNT`). Die Mediane der Reihe (n = 5 000, alle Anteile haben mindestens 100 Werte) ändern sich dadurch nicht.
- **Exaktes Potenzgesetz:** die Vormessung erwartete 100 % „plausibel“; gemessen sind **90 %**, wie bei einem Test mit Fehler erster Art von 10 % zu erwarten.
- **Barabási–Albert gegen Log-Normal:** die Vormessung fand 25 % „Log-Normal klar besser“; die Demo findet **10 %**, dazu 20 % verworfene Potenzgesetze (Reihe mit 10 Stichproben, grob).
- **Hill-Kurve bei Log-Normal:** Vormessung 3,46 / 2,98 / 2,45 bei 2 / 10 / 50 %, Demo **3,53 / 2,99 / 2,47**: bestätigt.

## Verifikation

- **Kern:** jede Einheit an einer Mini-Instanz von Hand (Log-Likelihood, Verteilungsfunktion und KS-Abstand mit π²/6, Kandidaten der Untergrenze, Exponential mit λ = ln 2, Vuong mit z = 3, Hill-Schätzer), Randfälle (ein Wert, lauter gleiche Werte, kleinste Stichprobe, Ziehen am oberen Rand).
- **Orakel:** `scipy.stats.zipf`, dichtes Gitter statt Optimierer, Brute-Force-Wahl der Untergrenze, geometrische Verteilung, `scipy.stats.norm` für die Log-Normal-Wahrscheinlichkeiten, `scipy.stats.pareto` für Hill, `networkx.barabasi_albert_graph`, Fehler erster Art des Bootstraps (mindestens 75 % von 24 exakten Potenzgesetzen nicht verworfen).
- **Verdikte:** reine Funktionen, jede Zahl im Text an handgesetzten Werten, einschließlich der Schwellen p = 0,1 und Spannweite 0,3 (`tests/test_verdicts.py`).
- **Presets, Layout, Oberfläche:** wie in allen Demos (Presets vollständig und im Regelbereich, keines auf einem Reglerrand, Schnellstart oben, aktives Preset hervorgehoben, Auswahlfeld in der Layoutprüfung, AppTest jeder Button und jeder Regler an den Rändern).
- **Mutationsprüfung** (`tools/mutate_demo.py`): 103 Stellen im Kern `pgt_algorithm.py`, **92 gefangen** (davon 22 über wortgleiche Zeilen in Nachbar-Demos), 8 als äquivalent festgehalten, 3 Zeitüberschreitungen (Mutanten, die die Rechnung stark verlängern; das Werkzeug zählt sie nicht). Der erste Lauf ließ 15 Mutanten überleben und deckte echte Lücken auf (Zählung und Stichprobengröße des Bootstraps, Gleichstand bei der Wahl der Untergrenze, Vuong-Test mit zwei Werten, Anpassung mit zwei Werten, Log-Normal-Anpassung bei sehr engen Daten), für die es jetzt Tests gibt (`tests/test_edge_cases.py`, `tests/test_algorithm.py`). Ein Mutant (`>=` gegen `>` beim Zählen der Treffer des Bootstraps) steht im Zustand der Datei als überlebend, weil sein Lauf in eine Zeitüberschreitung lief; von Hand in einer Kopie fängt ihn `test_bootstrap_counts_every_repetition_and_keeps_the_sample_size`. Die äquivalenten Stellen: die Größe des Trägers beim Ziehen (zwei Konstanten, die abgeschnittene Masse liegt unter einem Hunderttausendstel), zwei Startwerte der Log-Normal-Anpassung (derselbe Optimierer findet dasselbe Optimum), `<` gegen `<=` beim Vergleich der Anpassungen (nur bei exaktem Gleichstand verschieden), `and` gegen `or` und `n − k` gegen `n + k` bei der Frage, ob Werte unter der Untergrenze zu ziehen sind (in den erreichbaren Fällen wird dann nur eine leere Auswahl gezogen), und `return None` statt `return 1e18` bei Streuung unter 0,05 (im Test liegt das Ergebnis in beiden Fällen an der Untergrenze der Streuung; nicht weiter untersucht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `requirements-dev.txt` | Laufzeit plus pytest und networkx (nur Gegenprobe) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; `scipy` rechnet Zeta-Funktion und Optimierer) |
| `pgt_algorithm.py` | Kern: Maximum Likelihood, KS-Abstand, Untergrenze, Bootstrap, Vergleichsverteilungen, Vuong, Hill, Erzeuger |
| `pgt_constants.py` | Regler-Grenzen, Parameter der Pipeline, feste Pläne der Messreihen, Farben, Presets |
| `pgt_evaluation.py` | Pipeline je Stichprobe, Messreihen, Verdikte |
| `pgt_layout.py` | Fester Kopf (Stylesheet) |
| `pgt_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `pgt_scenario.py` | Stichprobe je Quelle |
| `pgt_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `messen.py` | Einmalige Messung der Zahlen in dieser README (nicht Teil der App) |
| `tests/` | Mini-Instanzen, Kern, Orakel, Verdikte, Randfälle, Presets, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

- **Weitere Vergleichsverteilungen** (abgeschnittenes Potenzgesetz, gestreckte Exponentialverteilung, Weibull): würden die Trennschärfe-Reihe verlängern, ohne den Befund zu ändern; offen für ein Nachtragsstück.
- **Stetige Daten und eigene Daten hochladen:** die Demo bleibt bei ganzzahligen Graden aus bekannten Erzeugern; Dateneingabe wäre ein eigenes Stück samt Datenschutzprüfung.
- **Wahl der Untergrenze durch andere Verfahren** (Bayes, Anpassung am Hill-Plot): nur der KS-Abstand.
- **Entscheidung zwischen den Lesarten:** die App zeigt beide, entscheidet nicht.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
