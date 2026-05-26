import numpy as np
from sfh_sgp.core.embedding import (
    m1_signed_ordinal_flow,
    m2_half_correlation,
    m3_signed_compressibility,
    m4_amplitude_transition_asymmetry,
    EmbeddingEngine,
    EmbeddingResult,
    METRICS_V2_079,
)
from sfh_sgp.core.guard import ArchitectureGuard, HardFreezeViolation, FROZEN_HASH
from sfh_sgp.core.transforms import TransformEngine
from sfh_sgp.core.geometry import GeometryAnalyser


def test_m1_monotonic_up():
    x = np.arange(10, dtype=float)
    assert m1_signed_ordinal_flow(x) == 1.0


def test_m1_monotonic_down():
    x = np.arange(10, 0, -1, dtype=float)
    assert m1_signed_ordinal_flow(x) == -1.0


def test_m1_flat():
    x = np.ones(10, dtype=float)
    assert m1_signed_ordinal_flow(x) == 0.0


def test_m2_perfect_corr():
    x = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    assert abs(m2_half_correlation(x) - 1.0) < 1e-9


def test_m3_symmetric():
    x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    val = m3_signed_compressibility(x)
    assert isinstance(val, float)


def test_m4_symmetric():
    x = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
    val = m4_amplitude_transition_asymmetry(x)
    assert isinstance(val, float)


def test_embedding_engine():
    engine = EmbeddingEngine()
    x = np.sin(np.linspace(0, 4 * np.pi, 100))
    result = engine.embed(x, signal_id='test')
    assert isinstance(result, EmbeddingResult)
    assert result.signal_id == 'test'
    assert result.transform == 'base'
    assert result.architecture == 'V2_079'
    for attr in ['m1', 'm2', 'm3', 'm4']:
        assert isinstance(getattr(result, attr), float)


def test_embedding_engine_frozen():
    guard = ArchitectureGuard()
    guard.assert_frozen()


def test_guard_rejects_tampered():
    import json
    tampered = {
        'rewrite_allowed': True,
        'theory_hash': FROZEN_HASH,
        'architecture': 'V2_079',
        'lineage_valid': True,
    }
    try:
        ArchitectureGuard(tampered)
        assert False, 'Should have raised'
    except HardFreezeViolation:
        pass


def test_transforms():
    t = TransformEngine()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert len(t.reverse(x)) == len(x)
    assert len(t.replay(x)) == 2 * len(x)
    assert len(t.swap(x)) == len(x)
    assert len(t.stitch(x)) == len(x)


def test_geometry():
    rng = np.random.default_rng(42)
    embs = rng.normal(0, 1, (100, 4))
    g = GeometryAnalyser()
    result = g.analyse(embs)
    for key in ['pc1', 'dim95', 'curvature', 'neighbor_purity', 'geo_euclidean_corr']:
        assert key in result


def test_embed_many():
    engine = EmbeddingEngine()
    signals = [np.sin(np.linspace(0, 2 * np.pi, 50)) for _ in range(5)]
    results = engine.embed_many(signals)
    assert len(results) == 5


def test_embedding_result_vector():
    r = EmbeddingResult(signal_id='t', m1=0.1, m2=0.2, m3=0.3, m4=0.4)
    vec = r.vector
    assert np.allclose(vec, [0.1, 0.2, 0.3, 0.4])


def test_transform_apply():
    t = TransformEngine()
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(t.apply(x, 'base'), x)
    assert np.allclose(t.apply(x, 'reverse'), x[::-1])


def test_metrics_dict():
    assert set(METRICS_V2_079.keys()) == {'m1', 'm2', 'm3', 'm4'}
