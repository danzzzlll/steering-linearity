from pathlib import Path
from typing import Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from src.core.metric_base import Metric


class IsomapResidualVariance(Metric):
    """
    Residual-variance (1 − R²) кривaя для Isomap-развёртки.

    • k_neighbors оцениваются на PCA-256 подпространстве
    • Считается только на `n_landmark` точках (общих для всех слоёв)
    """

    def __init__(
        self,
        ds,
        cache,
        layer,
        *,
        dims: Iterable[int] = range(15, 35, 5),
        k: int = 40,
        n_landmark: int = 3000,
        save_dir: str | Path = "plots",
        norm: str = "center",
        **kw,
    ):
        super().__init__(ds, cache, layer, k=k, norm=norm, **kw)
        self.dims = list(dims)
        self.n_landmark = n_landmark
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # ------- фиксированная подвыборка + PCA-256 -------------------
        idx  = cache.get_sample_idx(len(self.X), n_landmark)
        self.X_sub = cache.get_x256(str(layer), self.X)[idx]

    def compute(self) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore[override]
        rv = []
        D_geo = self.cache.get_isomap_geodesic(
            f"{self.layer}_sub", self.X_sub, k=self.k, path_method="auto"
        )

        for d in tqdm(self.dims, desc=f"Isomap RV  L{self.layer}"):
            try:
                Y = Isomap(n_neighbors=self.k, n_components=d,
                           path_method="auto").fit_transform(self.X_sub)
                D_emb = pairwise_distances(Y)
                r2 = np.corrcoef(D_geo.ravel(), D_emb.ravel())[0, 1] ** 2
                rv.append(1 - r2)
            except ValueError as err:
                print(f"[warn] d={d}: {err}")
                rv.append(np.nan)

        fig, ax = plt.subplots()
        ax.plot(self.dims, rv, marker="o")
        ax.set_xlabel("целевое d")
        ax.set_ylabel("residual variance")
        ax.set_title(f"Isomap RV – layer {self.layer}")
        ax.grid(True)
        fig.savefig(self.save_dir / f"isomap_rv_L{self.layer}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        # return np.asarray(self.dims), np.asarray(rv)
