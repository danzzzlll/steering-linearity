# src/metrics/h2_simple.py
import numpy as np
from sklearn.metrics import pairwise_distances
from ripser import ripser

from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


class H2LongCyclesSimple(Metric):
    """
    Простой счётчик «длинных» H2-циклов на небольшом сабсэмпле.
    ↓ меньше = manifold более плоский (меньше камер).

    Идея:
      • случайно берём L точек (через общий кэш сабсэмпла),
      • считаем полную L2-матрицу,
      • ставим порог ε как квантиль расстояний,
      • ripser(maxdim=2, thresh=ε),
      • считаем долю «длинных» 2-циклов: (death - birth) / death ≥ rel_life_thresh.
    """

    def __init__(self, ds, cache, layer, *,
                 k=20, norm="z",
                 n_landmarks: int = 300,
                 eps_quantile: float = 0.80,   # 0.70..0.90 — подстройка плотности
                 rel_life_thresh: float = 0.20 # «длинный» цикл по относительной длительности
                 ):
        super().__init__(ds, cache, layer, k=k, norm=norm)
        self.n_landmarks = int(n_landmarks)
        self.eps_quantile = float(eps_quantile)
        self.rel_life_thresh = float(rel_life_thresh)


    @timecount
    def compute(self) -> dict:
        X = self.X
        N = X.shape[0]
        L = min(self.n_landmarks, N)
        if L < 10:
            return {"h2_long_frac": float("nan"), "h2_count": 0, "h2_num_total": 0,
                    "h2_eps_used": float("nan"), "h2_L": int(L)}

        # один и тот же сабсэмпл для стабильности между запусками
        idx = self.cache.get_sample_idx(N, L)
        Xs = X[idx]

        # полная матрица попарных расстояний (LxL)
        D = pairwise_distances(Xs, metric="euclidean")
        # возьмём ε как квантиль всех расстояний (кроме диагонали)
        tri = D[np.triu_indices(L, k=1)]
        if tri.size == 0:
            return {"h2_long_frac": 0.0, "h2_count": 0, "h2_num_total": 0,
                    "h2_eps_used": 0.0, "h2_L": int(L)}
        eps = float(np.quantile(tri, self.eps_quantile))
        if not np.isfinite(eps) or eps <= 0:
            return {"h2_long_frac": 0.0, "h2_count": 0, "h2_num_total": 0,
                    "h2_eps_used": float(eps), "h2_L": int(L)}

        # вызываем ripser на квадратной матрице с порогом ε
        res = ripser(D, distance_matrix=True, maxdim=2, thresh=eps)
        dgms = res.get("dgms", [])

        # если H2 нет — возвращаем нули
        if len(dgms) < 3 or dgms[2].size == 0:
            return {"h2_long_frac": 0.0, "h2_count": 0, "h2_num_total": 0,
                    "h2_eps_used": float(eps), "h2_L": int(L)}

        H2 = dgms[2]  # shape (m, 2): birth, death
        lifetimes = H2[:, 1] - H2[:, 0]
        rel_life = lifetimes / np.maximum(H2[:, 1], 1e-12)
        mask_long = rel_life >= self.rel_life_thresh

        count_long = int(np.sum(mask_long))
        num_total = int(H2.shape[0])
        frac_long = float(count_long / max(1, num_total))

        return {
            "h2_long_frac": frac_long,     # ↓ лучше
            "h2_count": count_long,        # диагностика
            "h2_num_total": num_total,     # диагностика
            "h2_eps_used": float(eps),
            "h2_L": int(L),
        }
