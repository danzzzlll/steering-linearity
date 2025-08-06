# from __future__ import annotations
# import numpy as np
# import scipy.linalg as la
# from tqdm import tqdm

# from src.core.metric_base import Metric
# from src.utils.time_decorator import timecount


# class Mapc(Metric):
#     """
#     MAPC – Manifold Alignment Pointwise Curvature.

#     • `k`         – число соседей (включая саму точку, первый элемент убираем)
#     • `svd_comp`  – индекс сингуляр-вектора, считаемого «нормалью»
#     • Возвращает np.ndarray (N,) кривизн, обычно берут .mean().
#     """

#     def __init__(
#         self,
#         *a,
#         k: int = 11,
#         svd_comp: int = 10,
#         **kw,
#     ):
#         super().__init__(*a, k=k, **kw)
#         self.svd_comp = svd_comp
        
#     @timecount
#     def compute(self) -> np.ndarray:                # type: ignore[override]
#         print("MAKE MAPC ...")
#         X_use = self.cache.get_x256(self.layer, self.X)
#         knn = self.cache.get_knn(self.layer, X_use, k=self.k)  # shape (N, k)
#         n_pts, dim = X_use.shape
#         mapc = np.empty(n_pts, dtype=np.float32)

#         for i, neigh in tqdm(
#             enumerate(knn), total=n_pts, desc=f"MAPC layer {self.layer}"
#         ):
#             # точечная нормаль
#             P = X_use[neigh[1:]] - X_use[neigh[1:]].mean(0)
#             U, _ = la.svd(P, full_matrices=False)[:2] 
#             n_vec = U[:, self.svd_comp]                # (dim,)

#             # нормали соседей
#             neigh_normals = []
#             for j in neigh[1:]:
#                 Q = X_use[knn[j, 1:]] - X_use[knn[j, 1:]].mean(0)
#                 Uq, _ = la.svd(Q, full_matrices=False)[:2]
#                 neigh_normals.append(Uq[:, self.svd_comp])
#             neigh_normals = np.stack(neigh_normals)    # (k-1, dim)

#             mapc[i] = np.mean(np.linalg.norm(neigh_normals - n_vec, axis=1))
#         mapc_mean = mapc.mean()
#         print(f"MAPC metric: {mapc_mean}")
#         print("MAKE MAPC DONE")
#         return mapc_mean


# файл metrics/mapc_better.py
from __future__ import annotations
import numpy as np
from numpy.linalg import norm
import scipy.linalg as la
from tqdm import tqdm

from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


class MapcBetter(Metric):
    """
    Улучшенный MAPC:
      • k         – сколько соседей берём (k-1 реальных + сама точка)
      • d_latent  – сколько главных направлений считать «касательным» пространством
      • angle     – если True, возвращаем угол между нормалями (рад), иначе ‖Δn‖
    """

    def __init__(
        self,
        *a,
        k: int = 11,
        d_latent: int = 8,
        angle: bool = True,
        **kw,
    ):
        super().__init__(*a, k=k, **kw)
        self.d_latent = d_latent      # вместо svd_comp
        self.angle = angle

        # загружаем слой, режем до 256-мер для скорости
        self.X_use = self.cache.get_x256(self.layer, self.X).astype(np.float32)
        self.knn   = self.cache.get_knn(self.layer, self.X_use, k=k)

    # ---------- вспомогательные функции ----------
    def _local_normal(self, neigh_idx: np.ndarray) -> np.ndarray:
        """Нормаль = последний вектор из SVD после среза до d_latent."""
        P = self.X_use[neigh_idx] - self.X_use[neigh_idx].mean(0)
        # оставляем только d_latent компонент
        U, _, _ = la.svd(P, full_matrices=False)
        return U[:, self.d_latent]     # (dim,)

    @staticmethod
    def _angle(n1: np.ndarray, n2: np.ndarray) -> float:
        cos = np.clip(np.dot(n1, n2) / (norm(n1) * norm(n2) + 1e-9), -1.0, 1.0)
        return np.arccos(cos)          # в радианах

    # ---------- основная метрика ----------
    @timecount
    def compute(self) -> float:        # type: ignore[override]
        print("MAKE MapcBetter ...")

        normals = np.empty_like(self.X_use)            # (N, 256)
        for i, neigh in tqdm(
            enumerate(self.knn),
            total=len(self.X_use),
            desc=f"Normals L{self.layer}",
        ):
            normals[i] = self._local_normal(neigh[1:])

        # угол/расстояние до нормалей соседей
        k = self.knn.shape[1]
        dists = np.empty(len(self.X_use), dtype=np.float32)

        for i, neigh in tqdm(
            enumerate(self.knn),
            total=len(self.X_use),
            desc=f"MAPC-Better L{self.layer}",
        ):
            neigh_normals = normals[neigh[1:]]               # (k-1, 256)
            if self.angle:
                d = self._angle(normals[i], neigh_normals.T) # вектор углов
            else:
                d = norm(neigh_normals - normals[i], axis=1) # ‖Δn‖
            dists[i] = d.mean()

        score = float(dists.mean())
        print({"mapc_better": score})
        print("MAKE MapcBetter DONE")
        return score



'''
metric = MapcBetter(
    layer="12",
    X=activations,         # (N, d) numpy or memmap
    cache=shared_cache,    # тот же cache, что и раньше
    k=11,                  # размер локальной окрестности
    d_latent=8,            # размер касательного подпространства
    angle=True,            # сравниваем углы
)
curvature = metric()
print("средний угол (рад):", curvature)
'''