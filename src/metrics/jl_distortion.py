from typing import Dict

import numpy as np
from src.core.metric_base import Metric


class JlDistortion(Metric):
    """
    Измеряет, как меняются L2-дистанции при m случайных JL-проекциях.

    Parameters
    ----------
    d_code       : int   - размерность проекции
    m_proj       : int   - сколько случайных матриц R
    n_pairs      : int   - сколько случайных пар точек усреднять
    random_state : int   - seed
    use_pca      : bool  - если True, берём PCA-30 (как в ноутбуке)
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
        self.n_pairs = n_pairs
        self.rng = np.random.default_rng(random_state)

        # Подготовка точки -> (N, d_code_base)
        if use_pca:
            pca = self.cache.get_pca(
                self.layer,
                self.X,
                n_components=d_code,
                svd_solver="randomized",
                random_state=random_state,
            )
            self.base = pca.transform(self.X)  # (N, d_code)
        else:
            if self.X.shape[1] < d_code:
                raise ValueError("d_code больше исходной размерности")
            self.base = self.X[:, : d_code] 

        N = len(self.base)
        self.pairs = self.rng.integers(N, size=(n_pairs, 2))

        i, j = self.pairs[:, 0], self.pairs[:, 1]
        self.orig = np.linalg.norm(self.base[i] - self.base[j], axis=1)  # (n_pairs,)

    def compute(self) -> Dict[str, float]:  # type: ignore[override]
        rel_errors = []
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d_code = self.d_code

        for _ in range(self.m_proj):
            R = self.rng.normal(size=(d_code, d_code)) / np.sqrt(d_code)
            Y = self.base @ R
            proj = np.linalg.norm(Y[i] - Y[j], axis=1)
            rel_errors.append(np.mean(np.abs(self.orig - proj) / (self.orig + 1e-9)))

        delta = float(np.mean(rel_errors) * 100.0)  # %
        return {
            "delta_pct": delta,
            "stable": bool(delta < 10.0),
        }
