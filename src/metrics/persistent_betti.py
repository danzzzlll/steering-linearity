from typing import Dict, List

import numpy as np
import gudhi as gd
from src.core.metric_base import Metric


class PersistentBetti(Metric):
    """
    Betti‑числа β₀, β₁, … для *Rips‑комплекса* landmark‑подвыборки.

    · landmark‑точки фиксируются через SharedCache (по слою);
    · опционально предварительная PCA‑проекция до n_pca компонент.
    """

    def __init__(
        self,
        *a,
        max_dim: int = 3,
        n_landmark: int = 2_000,
        max_edge: float | None = None,
        use_pca: bool = True,
        n_pca: int = 30,
        random_state: int = 0,
        **kw,
    ):
        super().__init__(*a, **kw)
        self.max_dim = max_dim

        pts = (
            self.cache.get_pca(
                self.layer,
                self.X,
                n_components=n_pca,
                svd_solver="randomized",
                random_state=random_state,
            ).transform(self.X)
            if use_pca
            else self.X
        ).astype(np.float32)

        idx_land = self.cache.get_sample_idx(len(pts), n_landmark)
        self.landmarks = pts[idx_land]

        if max_edge is None:
            d = np.linalg.norm(
                pts[self.cache.rng.choice(len(pts), 500, replace=False)]
                - pts[self.cache.rng.choice(len(pts), 500, replace=False)],
                axis=1,
            )
            max_edge = np.percentile(d, 95)
        self.max_edge = float(max_edge)

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, int]:            # type: ignore[override]
        rc = gd.RipsComplex(points=self.landmarks, max_edge_length=self.max_edge)
        st = rc.create_simplex_tree(max_dimension=self.max_dim)
        st.compute_persistence()
        betti: List[int] = st.betti_numbers()
        return {f"beta_{i}": b for i, b in enumerate(betti)}
