from typing import Dict

import numpy as np
from src.core.metric_base import Metric


class JlDistortion(Metric):
    """
    Оценивает среднее относительное искажение L2‑дистанций
    при m случайных Johnson–Lindenstrauss проекциях.

    Параметры
    ---------
    d_code        : размерность JL‑проекции
    m_proj        : сколько случайных матриц (default 100)
    n_pairs       : сколько пар точек усреднять
    random_state  : seed
    use_pca       : если True, берём PCA‑d_code; иначе усечение X[:, :d_code]
    """

    def __init__(
        self,
        *a,
        d_code: int = 30,
        m_proj: int = 100,
        n_pairs: int = 1_000,
        random_state: int = 42,
        use_pca: bool = True,
        **kw,
    ):
        super().__init__(*a, **kw)

        self.d_code = d_code
        self.m_proj = m_proj
        rng = np.random.default_rng(random_state)

        if use_pca:
            pca = self.cache.get_pca(
                self.layer,
                self.X,
                n_components=d_code,
                svd_solver="randomized",
                random_state=random_state,
            )
            base = pca.transform(self.X)
        else:
            if self.X.shape[1] < d_code:
                raise ValueError("d_code больше исходной размерности")
            base = self.X[:, :d_code]

        self.base = base.astype(np.float32)

        idx_flat = self.cache.get_sample_idx(len(self.base), n_pairs * 2)
        self.idx1, self.idx2 = idx_flat.reshape(-1, 2).T

        self.orig = np.linalg.norm(
            self.base[self.idx1] - self.base[self.idx2], axis=1
        )

        self.R0 = rng.normal(size=(d_code, d_code)).astype(np.float32) / np.sqrt(d_code)
        self.rng = rng  # сохранённый генератор

    def _random_R(self) -> np.ndarray:
        """R = R0 * diag(±1). Сохраняет распределение JL,
        но не создаёт d_code² новых чисел."""
        signs = self.rng.choice([-1.0, 1.0], size=self.d_code).astype(np.float32)
        return self.R0 * signs  # broadcasting по столбцам

    def compute(self) -> Dict[str, float]:  # type: ignore[override]
        i, j = self.idx1, self.idx2
        rel_err = []

        for _ in range(self.m_proj):
            Y = self.base @ self._random_R()           # (N, d_code)
            proj = np.linalg.norm(Y[i] - Y[j], axis=1)
            rel_err.append(np.mean(np.abs(self.orig - proj) / (self.orig + 1e-9)))

        delta = float(np.mean(rel_err) * 100.0)
        return {"delta_pct": delta, "stable": bool(delta < 10.0)}
