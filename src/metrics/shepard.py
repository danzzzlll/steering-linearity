
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from src.core.metric_base import Metric


class ShepardRsq(Metric):
    """
    R² Shepard‑плота: сравнивает L2‑дистанции до и после PCA‑проекции.

    Параметры
    ---------
    n_pairs :   сколько случайных пар точек усреднить
    n_pca  :   int ‑ число компонент   | float<1.0 ‑ доля дисперсии
    """

    def __init__(
        self,
        *a,
        n_pairs: int = 1_000,
        n_pca: int | float = 30,
        random_state: int = 42,
        save_dir: str | Path = "plots",
        **kw,
    ):
        super().__init__(*a, **kw)
        self.n_pairs = n_pairs
        self.n_pca   = n_pca
        self.rng     = np.random.default_rng(random_state)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print("MAKE ShepardRsq")

    # ------------------------------------------------------------------
    def compute(self) -> float:                     # type: ignore[override]
        # ---------- фиксированные пары (общие для всех слоёв) ----------
        flat_idx = self.cache.get_sample_idx(len(self.X), self.n_pairs * 2)
        idx1, idx2 = flat_idx.reshape(-1, 2).T

        # ---------- PCA‑проекция --------------------------------------
        svd_solver = (
            "full" if isinstance(self.n_pca, float) and self.n_pca < 1.0
            else "randomized"
        )
        pca = self.cache.get_pca(
            self.layer,
            self.X.astype("float32"),
            n_components=self.n_pca,
            svd_solver=svd_solver,
            random_state=0,
        )
        X_pca = pca.transform(self.X.astype("float32"))

        # ---------- расстояния (векторы длиной n_pairs) ----------------
        d_raw = np.linalg.norm(self.X[idx1]  - self.X[idx2], axis=1).reshape(-1, 1)
        d_lin = np.linalg.norm(X_pca[idx1]   - X_pca[idx2],  axis=1).reshape(-1, 1)

        # ---------- лин. регрессия без интерсепта ----------------------
        reg = LinearRegression(fit_intercept=False).fit(d_raw, d_lin)
        r2  = float(reg.score(d_raw, d_lin))
        slope = float(reg.coef_[0, 0])
        y_pred = reg.predict(d_raw)

        # ---------- график -------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(d_raw, d_lin, s=8, alpha=0.5)
        ax.plot(d_raw, y_pred, "r")                        # линия регрессии
        ax.set_xlabel("distance in original space")
        ax.set_ylabel(f"distance in PCA‑{self.n_pca}")
        ax.set_title(f"Shepard plot (R² = {r2:.2f})")
        ax.grid(True)
        fig.savefig(self.save_dir / f"shepard_L{self.layer}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        return r2
