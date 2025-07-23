# metrics/gli.py
from __future__ import annotations
import numpy as np
import faiss
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


class GlobalLinearityIndex(Metric):
    """
    GLI = E[ ||x_i - x_j|| / d_geo(i,j) ]   (усреднение по случайным парам)

    • k      — размер соседства при построении графа k-NN
    • m      — сколько случайных пар усреднять (≈ 1-5k от N достаточно)
    • seed   — random_state

    Возвращает одно число ∈ (0,1) — чем ближе к 1, тем «прямее» многообразие.
    """

    def __init__(
        self,
        *a,
        k: int = 10,
        m: int = 5_000,
        seed: int = 42,
        **kw,
    ):
        super().__init__(*a, k=k, **kw)
        self.m = m
        self.rng = np.random.default_rng(seed)
        

    def _geodesic_matrix(self) -> csr_matrix:
        """Возвращает матрицу кратчайших путей на графе k-NN (не взвешенный)."""
        # кэшируем k-NN (N, k)
        knn = self.cache.get_knn(self.layer, self.X, k=self.k)     # (N, k)
        n = len(self.X)

        rows = np.repeat(np.arange(n), self.k)
        cols = knn.ravel()
        A = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n, n))

        D_geo = shortest_path(A, directed=False, unweighted=True, return_predecessors=False)
        return D_geo

    @timecount
    def compute(self) -> float:                     # type: ignore[override]
        print("MAKE GlobalLinearityIndex ...")
        D_geo = self._geodesic_matrix()             # (N, N)
        n = len(self.X)

        pairs = self.rng.integers(n, size=(self.m, 2))
        idx_i, idx_j = pairs[:, 0], pairs[:, 1]

        geo = D_geo[idx_i, idx_j]
        geo[np.isinf(geo)] = self.k + 1

        # евклидово расстояние между парами
        eu = np.linalg.norm(self.X[idx_i] - self.X[idx_j], axis=1) + 1e-9

        print(float(np.mean(eu / geo)))
        print("MAKE GlobalLinearityIndex DONE")
        return float(np.mean(eu / geo))
