"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_values():
    """Sechs Grade 1, 1, 2, 3, 3, 5. Handrechnung: ab Wert 1 bleiben 6, ab 2 bleiben 4, ab 3 bleiben 3, ab 5 bleibt 1 Wert übrig."""
    return np.array([1, 1, 2, 3, 3, 5])


@pytest.fixture
def four_values():
    """Vier Grade 1, 1, 2, 2: die empirische Verteilungsfunktion ist 0,5 bei 1 und 1,0 bei 2."""
    return np.array([1, 1, 2, 2])
