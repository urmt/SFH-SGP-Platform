import numpy as np
from sfh_sgp.core.anomaly import AnomalyDetector, RegimeEvent
from sfh_sgp.core.experiments import ExperimentRunner
from sfh_sgp.core.embedding import EmbeddingEngine
from sfh_sgp.core.transforms import TransformEngine
from sfh_sgp.core.geometry import GeometryAnalyser
from sfh_sgp.core.guard import ArchitectureGuard


def test_anomaly_detector():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 4))
    y = np.array([0] * 25 + [1] * 25)
    detector = AnomalyDetector()
    detector.train(X, y)
    event = detector.score(X[0])
    assert event is None or isinstance(event, RegimeEvent)


def test_experiment_runner_f001():
    guard = ArchitectureGuard()
    engine = EmbeddingEngine(guard)
    t = TransformEngine()
    g = GeometryAnalyser()
    runner = ExperimentRunner(engine, t, g)
    rng = np.random.default_rng(42)
    n = 20
    signals = [np.cumsum(rng.normal(0, 1, 100)) for _ in range(n)]
    labels = [0] * (n // 2) + [1] * (n // 2)
    result = runner.run_f001_embeddings_separable(signals, labels)
    assert 'experiment' in result
    assert 'passed' in result


def test_experiment_runner_f005():
    guard = ArchitectureGuard()
    engine = EmbeddingEngine(guard)
    t = TransformEngine()
    g = GeometryAnalyser()
    runner = ExperimentRunner(engine, t, g)
    rng = np.random.default_rng(42)
    signals = [np.cumsum(rng.normal(0, 1, 100)) for _ in range(10)]
    result = runner.run_f005_manifold_1d(signals)
    assert result['experiment'] == 'F005_manifold_effectively_1D'
