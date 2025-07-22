from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances

from src.core.metric_base import Metric


class LaplacianStressCurve(Metric):
    """
    Laplacian Eigenmaps «stress»-кривая
    (||D_graph − D_embed|| / ||D_graph||).

    • работает на PCA-256 + landmark-подвыборке
    """

    def __init__(
        self,
        ds,
        cache,
        layer,
        *,
        dims: Iterable[int] = range(2, 28, 2),
        k: int = 40,
        n_landmark: int = 3000,
        random_state: int = 0,
        save_dir: str | Path = "plots",
        norm: str = "center",
        **kw,
    ):
        super().__init__(ds, cache, layer, k=k, norm=norm, **kw)
        self.dims = list(dims)
        self.random_state = random_state
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # ---- landmark + PCA-256 -------------------------------------
        idx  = cache.get_sample_idx(len(self.X), n_landmark)
        self.X_sub = cache.get_x256(str(layer), self.X)[idx]

    def _adjacency(self) -> csr_matrix:
        knn = self.cache.get_knn(self.layer, self.X_sub, k=self.k)   # (M, k)
        m = len(self.X_sub)
        rows = np.repeat(np.arange(m), self.k)
        cols = knn.ravel()
        W = lil_matrix((m, m))
        W[rows, cols] = 1
        W[cols, rows] = 1
        return W.tocsr()

    def _stress(self, A: csr_matrix, d: int) -> float:
        embed = SpectralEmbedding(
            n_components=d,
            affinity="precomputed",
            random_state=self.random_state,
        ).fit_transform(A)

        Dg = pairwise_distances(A, metric="euclidean")
        De = pairwise_distances(embed)
        return float(np.linalg.norm(Dg - De) / np.linalg.norm(Dg))

    def compute(self) -> Tuple[np.ndarray, np.ndarray]:   # type: ignore[override]
        A = self._adjacency()
        stress = [
            self._stress(A, d) for d in
            tqdm(self.dims, desc=f"Lap-stress L{self.layer}")
        ]

        fig, ax = plt.subplots()
        ax.plot(self.dims, stress, marker="o")
        ax.set_xlabel("размерность d")
        ax.set_ylabel("stress")
        ax.set_title(f"Laplacian Eigenmaps RV – layer {self.layer}")
        ax.grid(True)
        fig.savefig(self.save_dir / f"lap_stress_L{self.layer}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # return np.asarray(self.dims), np.asarray(stress)
