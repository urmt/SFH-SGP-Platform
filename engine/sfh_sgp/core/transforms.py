import numpy as np


class TransformEngine:
    def reverse(self, x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x[::-1])

    def replay(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([x, x])

    def swap(self, x: np.ndarray) -> np.ndarray:
        mid = len(x) // 2
        return np.concatenate([x[mid:], x[:mid]])

    def stitch(self, x: np.ndarray) -> np.ndarray:
        mid = len(x) // 2
        a, b = x[:mid], x[mid:2 * mid]
        stitched = np.empty(len(a) + len(b), dtype=x.dtype)
        short = min(len(a), len(b))
        stitched[:2 * short] = np.column_stack([a[:short], b[:short]]).ravel()
        if len(a) > short:
            stitched[2 * short:] = a[short:]
        elif len(b) > short:
            stitched[2 * short:] = b[short:]
        return stitched

    def apply(self, x: np.ndarray, transform: str) -> np.ndarray:
        if transform == 'base':
            return x.copy()
        fn = getattr(self, transform, None)
        if fn is None:
            raise ValueError(f'Unknown transform: {transform}')
        return fn(x)

    TRANSFORMS = ['base', 'reverse', 'swap', 'replay', 'stitch']
