from __future__ import annotations
import numpy as np
import scipy.linalg as la
from tqdm import tqdm

from src.core.metric_base import Metric


class Mapc(Metric):
    """
    MAPC – Manifold Alignment Pointwise Curvature.

    • `k`         – число соседей (включая саму точку, первый элемент убираем)
    • `svd_comp`  – индекс сингуляр-вектора, считаемого «нормалью»
    • Возвращает np.ndarray (N,) кривизн, обычно берут .mean().
    """

    def __init__(
        self,
        *a,
        k: int = 11,
        svd_comp: int = 10,
        **kw,
    ):
        super().__init__(*a, k=k, **kw)
        self.svd_comp = svd_comp
        print("MAKE MAPC ...")

    def compute(self) -> np.ndarray:                # type: ignore[override]
        X_use = self.cache.get_x256(self.layer, self.X)
        knn = self.cache.get_knn(self.layer, X_use, k=self.k)  # shape (N, k)
        n_pts, dim = X_use.shape
        mapc = np.empty(n_pts, dtype=np.float32)

        for i, neigh in tqdm(
            enumerate(knn), total=n_pts, desc=f"MAPC layer {self.layer}"
        ):
            # точечная нормаль
            P = X_use[neigh[1:]] - X_use[neigh[1:]].mean(0)
            U, _ = la.svd(P, full_matrices=False)[:2] 
            n_vec = U[:, self.svd_comp]                # (dim,)

            # нормали соседей
            neigh_normals = []
            for j in neigh[1:]:
                Q = X_use[knn[j, 1:]] - X_use[knn[j, 1:]].mean(0)
                Uq, _ = la.svd(Q, full_matrices=False)[:2]
                neigh_normals.append(Uq[:, self.svd_comp])
            neigh_normals = np.stack(neigh_normals)    # (k-1, dim)

            mapc[i] = np.mean(np.linalg.norm(neigh_normals - n_vec, axis=1))
        mapc_mean = mapc.mean()
        print(f"MAPC metric: {mapc_mean}")
        return mapc_mean
