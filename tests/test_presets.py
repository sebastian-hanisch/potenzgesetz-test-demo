"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keiner auf einem Reglerrand; jedes Preset ändert gegenüber dem Standard nur Quelle und n."""

import pgt_constants as C
import pgt_evaluation as ev
import pgt_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["source"] in C.SOURCES and C.N_MIN <= p["n"] <= C.N_MAX and C.ALPHA_MIN <= p["alpha"] <= C.ALPHA_MAX, name
        assert C.SIGMA_MIN <= p["sigma"] <= C.SIGMA_MAX and C.NBOOT_MIN <= p["nboot"] <= C.NBOOT_MAX and C.FRAC_MIN <= p["frac"] <= C.FRAC_MAX and 0 <= p["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Exaktes Potenzgesetz (α = 2,5)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_preset_sits_on_a_slider_bound():
    """Die Layout-Tests setzen jeden Regler auf seinen Rand: ein Preset dort würde die Hervorhebung des Standard-Presets verfälschen."""
    bounds = {"n": (C.N_MIN, C.N_MAX), "alpha": (C.ALPHA_MIN, C.ALPHA_MAX), "sigma": (C.SIGMA_MIN, C.SIGMA_MAX), "nboot": (C.NBOOT_MIN, C.NBOOT_MAX), "frac": (C.FRAC_MIN, C.FRAC_MAX)}
    for name, preset in C.PRESETS.items():
        for key, (lo, hi) in bounds.items():
            assert preset[key] not in (lo, hi), (name, key)


def test_presets_differ_from_the_default_only_in_source_and_n():
    default = C.PRESETS["Exaktes Potenzgesetz (α = 2,5)"]
    for name, p in C.PRESETS.items():
        assert {k for k in p if p[k] != default[k]} <= {"source", "n"}, name


def test_presets_are_pairwise_different():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_each_source_has_a_preset():
    assert {p["source"] for p in C.PRESETS.values()} == {"powerlaw", "lognormal", "poisson", "ba"}
