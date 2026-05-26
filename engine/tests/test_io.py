import numpy as np
from sfh_sgp.io.synthetic import random_walk, regime_switch, sine_wave, linear_trend
from sfh_sgp.io.ingestion import ingest_array
from sfh_sgp.io.export import export_json_str


def test_random_walk():
    x = random_walk(256, seed=42)
    assert len(x) == 256
    assert x[0] != 0  # rng no longer starts at 0 for default_rng


def test_regime_switch():
    x = regime_switch(256, seed=42)
    assert len(x) == 256


def test_sine_wave():
    x = sine_wave(100, seed=42)
    assert len(x) == 100


def test_linear_trend():
    x = linear_trend(50, seed=42)
    assert len(x) == 50


def test_ingest_array():
    x = ingest_array([1.0, 2.0, 3.0])
    assert np.allclose(x, [1.0, 2.0, 3.0])


def test_export_json_str():
    data = [{'a': 1, 'b': 2}]
    s = export_json_str(data)
    assert '"a"' in s
