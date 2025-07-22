from typing import Dict, List

import numpy as np
import gudhi as gd
from src.core.metric_base import Metric


class PersistentBetti(Metric):
    """
    Вычисляет числа Бетти β₀, β₁, β₂, … для Rips-комплекса слоя.

    Parameters
    ----------
    max_dim        : int    - до какой размерности строить комплекс
    max_edge       : float  - макс. длина ребра (если  None → авто по 95-перц.)
    use_pca        : bool   - сначала PCA до n_pca компонент
    n_pca          : int    - сколько компонент оставить
    random_state   : int    - seed для PCA
    """

    def __init__(
        self,
        *a,
        max_dim: int = 3,
        max_edge: float | None = None,
        use_pca: bool = True,
        n_pca: int = 30,
        random_state: int = 0,
        **kw,
    ):
        super().__init__(*a, **kw)
        self.max_dim = max_dim
        self.max_edge = max_edge
        self.use_pca = use_pca
        self.n_pca = n_pca
        self.random_state = random_state

        # Проекция PCA (по желанию)
        if use_pca:
            pca = self.cache.get_pca(
                self.layer,
                self.X,
                n_components=n_pca,
                svd_solver="randomized",
                random_state=random_state,
            )
            self.points = pca.transform(self.X).astype(np.float32)
        else:
            self.points = self.X.astype(np.float32)

        if self.max_edge is None:
            d = np.linalg.norm(
                self.points[self.cache.rng.choice(len(self.points), 500, replace=True)]
                - self.points[self.cache.rng.choice(len(self.points), 500, replace=True)],
                axis=1,
            )
            self.max_edge = float(np.percentile(d, 95))


    def compute(self) -> Dict[str, int]:            # type: ignore[override]
        rc = gd.RipsComplex(points=self.points, max_edge_length=self.max_edge)
        st = rc.create_simplex_tree(max_dimension=self.max_dim)
        st.persistence()  # нужен, чтобы betti_numbers() работал

        betti: List[int] = st.betti_numbers()
        return {f"beta_{i}": b for i, b in enumerate(betti)}
