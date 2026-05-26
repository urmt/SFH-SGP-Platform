import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

REGIME_THRESHOLD = 0.75


class RegimeEvent:
    def __init__(self, embedding: list[float], probabilities: list[float], regime: str):
        self.embedding = embedding
        self.probabilities = probabilities
        self.regime = regime

    def to_dict(self) -> dict:
        return {
            'embedding': self.embedding,
            'probabilities': self.probabilities,
            'regime': self.regime,
        }


class AnomalyDetector:
    def __init__(self, lda_model: LinearDiscriminantAnalysis | None = None):
        self.lda = lda_model

    def train(self, X: np.ndarray, y: np.ndarray):
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X, y)

    def score(self, embedding: np.ndarray | list[float]) -> RegimeEvent | None:
        if self.lda is None:
            return None
        emb = np.asarray(embedding).reshape(1, -1)
        prob = self.lda.predict_proba(emb)[0]
        if max(prob) > REGIME_THRESHOLD:
            return RegimeEvent(
                embedding=emb.tolist()[0],
                probabilities=prob.tolist(),
                regime=self.lda.classes_[np.argmax(prob)],
            )
        return None

    async def score_stream(self, embedding_stream):
        async for emb in embedding_stream:
            event = self.score(emb)
            if event is not None:
                yield event
