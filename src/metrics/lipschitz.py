import numpy as np
from src.core.metric_base import Metric


class Lipschitz(Metric):
    """
    Локальный коэффициент Липшица между слоями prev_layer → layer.

    Возвращает словарь:
        {'mean': float, 'p95': float, 'count': int}

    Ускорения:
    • Расчёты идут в PCA-256 пространстве (16× быстрее, RAM ↓).
    • k-NN строится на том же сжатии, поэтому делится между всеми
      «локальными» метриками слоя.
    """

    def __init__(
        self,
        ds,
        cache,
        layer,
        *,
        prev_layer: int | str,
        k: int = 20,
        norm: str = "center",
        **kw,
    ):
        super().__init__(ds, cache, layer, k=k, norm=norm, **kw)

        X_prev = ds[prev_layer].astype(np.float32)
        X_prev -= X_prev.mean(0, keepdims=True)

        # ---------- PCA-256 (общее сжатие) ----------------------------
        self.X_small       = cache.get_x256(str(layer),      self.X)
        self.X_prev_small  = cache.get_x256(str(prev_layer), X_prev)
        print("MAKE Lipschitz")

    def compute(self) -> dict[str, float]:          # type: ignore[override]
        knn = self.cache.get_knn(self.layer, self.X_small, k=self.k)   # (N, k)

        i_idx = np.repeat(np.arange(len(self.X_small)), self.k)
        j_idx = knn.ravel()

        dx = self.X_prev_small[i_idx] - self.X_prev_small[j_idx]
        dh = self.X_small[i_idx]      - self.X_small[j_idx]

        ratios = np.linalg.norm(dh, axis=1) / (np.linalg.norm(dx, axis=1) + 1e-9)

        return {
            "mean":  float(ratios.mean()),
            "p95":   float(np.percentile(ratios, 95)),
            "count": int(ratios.size),
        }
