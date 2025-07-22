# metrics/shepard.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression

from src.core.metric_base import Metric


class ShepardRsq(Metric):
    """
    Shepard-plot R² между L2-дистанциями в исходном слое
    и после PCA-проекции.

    Parameters
    ----------
    n_pairs      : сколько случайных пар точек брать для оценки
    n_pca        : сколько компонент держать (по умолчанию 30)
    random_state : RNG seed
    save_dir     : куда положить PNG ('plots')
    """

    def __init__(
        self,
        *a,
        n_pairs: int = 1_000,
        n_pca: int = 30,
        random_state: int = 0,
        save_dir: str | Path = "plots",
        **kw,
    ):
        super().__init__(*a, **kw)
        self.n_pairs = n_pairs
        self.n_pca = n_pca
        self.rng = np.random.default_rng(random_state)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)


    def compute(self) -> float:                     # type: ignore[override]
        n_pairs = self.n_pairs
        flat_idx = self.cache.get_sample_idx(len(self.X), n_pairs * 2)
        pairs = flat_idx.reshape(-1, 2)
        idx1, idx2 = pairs[:, 0], pairs[:, 1]

        pca = self.cache.get_pca(
            self.layer,
            self.X,
            n_components=self.n_pca,
            svd_solver="randomized",
            random_state=0,
        )
        X_pca = pca.transform(self.X)

        # N = len(self.X)
        # pairs = self.rng.integers(N, size=(self.n_pairs, 2))
        # idx1, idx2 = pairs[:, 0], pairs[:, 1]

        d_raw = pairwise_distances(self.X[idx1], self.X[idx2])
        d_lin = pairwise_distances(X_pca[idx1], X_pca[idx2])

        reg = LinearRegression().fit(d_raw, d_lin)
        R2 = float(reg.score(d_raw, d_lin))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(d_raw, d_lin, s=8, alpha=0.5)
        ax.plot(d_raw, reg.predict(d_raw), "r")
        ax.set_xlabel("расстояние в исходном пространстве")
        ax.set_ylabel(f"расстояние в PCA-{self.n_pca}")
        ax.set_title(f"Shepard plot  (R² = {R2:.2f})")
        ax.grid(True)

        fname = self.save_dir / f"shepard_L{self.layer}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"MAKE SHEPARD METRIC")
        return R2
