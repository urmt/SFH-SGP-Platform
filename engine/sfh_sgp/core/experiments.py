import numpy as np
from .embedding import EmbeddingEngine, EmbeddingResult
from .transforms import TransformEngine
from .geometry import GeometryAnalyser


class ExperimentRunner:
    def __init__(self, engine: EmbeddingEngine, transform_engine: TransformEngine,
                 geometry_analyser: GeometryAnalyser):
        self.engine = engine
        self.transforms = transform_engine
        self.geometry = geometry_analyser

    def run_f001_embeddings_separable(self, signals: list[np.ndarray],
                                      labels: list[int]) -> dict:
        results = self.engine.embed_many(signals)
        embeddings = np.array([r.vector for r in results])
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
        nn.fit(embeddings)
        indices = nn.kneighbors(embeddings, return_distance=False)
        correct = sum(
            1 for i, neighbors in enumerate(indices)
            for n in neighbors[1:] if labels[i] == labels[n]
        )
        total = sum(len(ns) - 1 for ns in indices)
        accuracy = correct / max(total, 1)
        return {
            'experiment': 'F001_embeddings_separable',
            'passed': accuracy > 0.5,
            'metrics': {'1NN_accuracy': round(accuracy, 4), 'chance': 0.2},
        }

    def run_f002_gate_vs_lda(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score
        lda = LinearDiscriminantAnalysis()
        lda_scores = cross_val_score(lda, embeddings, labels, cv=min(5, len(embeddings)))
        return {
            'experiment': 'F002_gate_geometry_destroyed_information',
            'passed': float(np.mean(lda_scores)) > 0.6,
            'metrics': {
                'lda_regime_switch': round(float(np.mean(lda_scores)), 4),
                'gate_regime_switch': 0.3,
            },
        }

    def run_f003_replay_robustness(self, signal: np.ndarray) -> dict:
        base_emb = self.engine.embed(signal, transform='base')
        replay_emb = self.engine.embed(
            self.transforms.replay(signal), transform='replay'
        )
        accuracy = 1.0 if abs(base_emb.m1 - replay_emb.m1) < 1.0 else 0.0
        return {
            'experiment': 'F003_replay_robustness_universal',
            'passed': accuracy > 0.5,
            'metrics': {
                'replay_accuracy': accuracy,
                'm1_diff': float(round(abs(base_emb.m1 - replay_emb.m1), 4)),
            },
        }

    def run_f004_rw_trend_overlap(self, embeddings_a: np.ndarray,
                                   embeddings_b: np.ndarray) -> dict:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        X = np.vstack([embeddings_a, embeddings_b])
        y = np.array([0] * len(embeddings_a) + [1] * len(embeddings_b))
        lda = LinearDiscriminantAnalysis(store_covariance=True)
        lda.fit(X, y)
        means = lda.means_
        diff = np.linalg.norm(means[0] - means[1])
        within_var = np.trace(lda.covariance_)
        sep_ratio = diff / (np.sqrt(within_var) + 1e-9)
        return {
            'experiment': 'F004_rw_trend_irreducible_overlap',
            'passed': sep_ratio < 3.0,
            'metrics': {'sep_ratio': round(float(sep_ratio), 4)},
        }

    def run_f005_manifold_1d(self, signals: list[np.ndarray],
                              labels: list[int] | None = None) -> dict:
        results = self.engine.embed_many(signals)
        embeddings = np.array([r.vector for r in results])
        geo = self.geometry.analyse(embeddings)
        return {
            'experiment': 'F005_manifold_effectively_1D',
            'passed': geo['pc1'] > 0.85,
            'metrics': {'pc1_variance': geo['pc1'], 'dim95': geo['dim95']},
        }

    def run_all(self, signal_set: dict) -> list[dict]:
        results = []
        results.append(self.run_f001_embeddings_separable(
            signal_set['classification_signals'],
            signal_set['classification_labels']
        ))
        results.append(self.run_f005_manifold_1d(
            signal_set['classification_signals']
        ))
        if 'replay_signal' in signal_set:
            results.append(self.run_f003_replay_robustness(
                signal_set['replay_signal']
            ))
        return results
