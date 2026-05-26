import numpy as np


def ingest_csv(path: str, delimiter: str = ',') -> np.ndarray:
    return np.genfromtxt(path, delimiter=delimiter, skip_header=1)


def ingest_array(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=float)
