from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances

from src.core.metric_base import Metric
from src.utils.time_decorator import timecount

class LaplacianStressCurve(Metric):
    """
    Кривая residual‑variance (stress) для Laplacian Eigenmaps.

    • работает на PCA‑256 + landmark‑подвыборке
    • для кэша k‑NN использует отдельный ключ  layer+"_sub",
      чтобы не конфликтовать с k‑NN всего слоя.
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
        self.save_dir = Path(save_dir); self.save_dir.mkdir(exist_ok=True, parents=True)

        idx = cache.get_sample_idx(len(self.X), n_landmark)
        self.X_sub = cache.get_x256(str(layer), self.X)[idx]

        self.layer_sub = f"{layer}_sub"
        

    def _adjacency(self) -> csr_matrix:
        """0/1‑матрица смежности на графе k‑NN landmark‑точек."""
        knn = self.cache.get_knn(self.layer_sub, self.X_sub, k=self.k)
        m = len(self.X_sub)
        rows = np.repeat(np.arange(m), self.k)
        cols = knn.ravel()
        data = np.ones_like(rows, dtype=np.float32)
        A = csr_matrix((data, (rows, cols)), shape=(m, m))
        A = A + A.T                                        # делаем симметричной
        A[A > 1] = 1
        return A


    @staticmethod
    def _stress(A: csr_matrix, embed: np.ndarray) -> float:
        """||D_graph − D_embed|| / ||D_graph||"""
        Dg = pairwise_distances(A, metric="euclidean")
        De = pairwise_distances(embed)
        return float(np.linalg.norm(Dg - De) / np.linalg.norm(Dg))

    @timecount
    def compute(self) -> Tuple[np.ndarray, np.ndarray]:   # type: ignore[override]
        print("MAKE LaplacianStressCurve ...")
        A = self._adjacency()

        stress_vals = []
        for d in tqdm(self.dims, desc=f"Lap-stress L{self.layer}"):
            embed = SpectralEmbedding(
                        n_components=d,
                        affinity="precomputed",
                        random_state=self.random_state,
                    ).fit_transform(A)
            stress_vals.append(self._stress(A, embed))

        fig, ax = plt.subplots()
        ax.plot(self.dims, stress_vals, marker="o")
        ax.set_xlabel("размерность d")
        ax.set_ylabel("stress")
        ax.set_title(f"Laplacian Eigenmaps RV – layer {self.layer}")
        ax.grid(True)
        fig.savefig(self.save_dir / f"lap_stress_L{self.layer}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("MAKE LaplacianStressCurve DONE")
        # return np.asarray(self.dims), np.asarray(stress_vals)
