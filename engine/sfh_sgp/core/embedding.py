import numpy as np
import scipy.stats
from .guard import FROZEN_HASH, FROZEN_ARCH, ArchitectureGuard


def m1_signed_ordinal_flow(x: np.ndarray) -> float:
    diffs = np.diff(x)
    n = max(len(diffs), 1)
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / n)


def m2_half_correlation(x: np.ndarray) -> float:
    mid = len(x) // 2
    if mid < 2:
        return 0.0
    a, b = x[:mid], x[mid:2 * mid]
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_centered = a - a_mean
    b_centered = b - b_mean
    denom = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
    if denom < 1e-12:
        return 0.0
    val = np.sum(a_centered * b_centered) / denom
    val = max(-1.0, min(1.0, val))
    return 0.0 if np.isnan(val) else float(val)


def m3_signed_compressibility(x: np.ndarray) -> float:
    mid = len(x) // 2
    h1 = scipy.stats.entropy(np.histogram(x[:mid], bins=20)[0] + 1e-9)
    h2 = scipy.stats.entropy(np.histogram(x[mid:], bins=20)[0] + 1e-9)
    return float(h1 - h2)


def m4_amplitude_transition_asymmetry(x: np.ndarray) -> float:
    abs_diffs = np.abs(np.diff(x))
    up = abs_diffs[np.diff(x) > 0]
    dn = abs_diffs[np.diff(x) < 0]
    denom = np.mean(abs_diffs) + 1e-9
    up_mean = np.mean(up) if len(up) > 0 else 0.0
    dn_mean = np.mean(dn) if len(dn) > 0 else 0.0
    return float((up_mean - dn_mean) / denom)


METRICS_V2_079 = {
    'm1': m1_signed_ordinal_flow,
    'm2': m2_half_correlation,
    'm3': m3_signed_compressibility,
    'm4': m4_amplitude_transition_asymmetry,
}


class EmbeddingResult:
    def __init__(self, signal_id: str, m1: float, m2: float, m3: float, m4: float,
                 transform: str = 'base', architecture: str = FROZEN_ARCH):
        self.signal_id = signal_id
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.transform = transform
        self.architecture = architecture

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.m1, self.m2, self.m3, self.m4])

    def to_dict(self) -> dict:
        return {
            'signal_id': self.signal_id,
            'm1': self.m1,
            'm2': self.m2,
            'm3': self.m3,
            'm4': self.m4,
            'transform': self.transform,
            'architecture': self.architecture,
        }


class EmbeddingEngine:
    def __init__(self, guard: ArchitectureGuard | None = None):
        self.guard = guard or ArchitectureGuard()

    def embed(self, x: np.ndarray, signal_id: str = '', transform: str = 'base') -> EmbeddingResult:
        self.guard.assert_frozen()
        m1 = m1_signed_ordinal_flow(x)
        m2 = m2_half_correlation(x)
        m3 = m3_signed_compressibility(x)
        m4 = m4_amplitude_transition_asymmetry(x)
        return EmbeddingResult(
            signal_id=signal_id, m1=m1, m2=m2, m3=m3, m4=m4, transform=transform
        )

    def embed_many(self, signals: list[np.ndarray], signal_ids: list[str] | None = None,
                   transform: str = 'base') -> list[EmbeddingResult]:
        if signal_ids is None:
            signal_ids = [str(i) for i in range(len(signals))]
        return [self.embed(x, sid, transform) for x, sid in zip(signals, signal_ids)]
