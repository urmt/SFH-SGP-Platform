import numpy as np


def random_walk(n: int = 256, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1, n))


def regime_switch(n: int = 256, switch_point: int | None = None,
                  seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if switch_point is None:
        switch_point = n // 2
    regime1 = np.cumsum(rng.normal(0, 1, switch_point))
    regime2 = np.cumsum(rng.normal(0.5, 1, n - switch_point))
    return np.concatenate([regime1, regime2])


def sine_wave(n: int = 256, freq: float = 0.05, noise: float = 0.1,
              seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * freq * t) + rng.normal(0, noise, n)


def linear_trend(n: int = 256, slope: float = 0.01,
                 noise: float = 0.1, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return slope * t + rng.normal(0, noise, n)
