import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from src.core.metric_base import Metric


class LocalPcaReconError(Metric):
    """Средняя ошибка реконструкции точки локальным PCA."""

    def __init__(
        self,
        *a,
        d_latent: int = 10,
        svd_solver: str = "randomized",
        **kw,
    ):
        super().__init__(*a, **kw)
        self.d_latent = d_latent
        self.svd_solver = svd_solver

        idx = self.cache.get_sample_idx(len(self.X), 5_000)
        self.X_use = self.cache.get_x256(self.layer, self.X)[idx]

        # собственный ключ для k‑NN, чтобы не перезаписать граф всего слоя
        self.layer_sub = f"{self.layer}_sub"
        print("MAKE LocalPcaReconError")

    def compute(self) -> float:             # type: ignore[override]
        knn = self.cache.get_knn(self.layer_sub, self.X_use, k=self.k)  # (M, k)

        errs = []
        for i, neigh in tqdm(
            enumerate(knn),
            total=len(knn),
            desc=f"Local PCA L{self.layer}",
        ):
            Xn   = self.X_use[neigh[1:]]                 # (k‑1, d)
            mu   = Xn.mean(0, keepdims=True)
            Xc   = Xn - mu
            pca  = PCA(
                n_components=self.d_latent,
                svd_solver=self.svd_solver,
                random_state=42,
            ).fit(Xc)

            basis  = pca.components_.T                   # (d, d_latent)
            xi     = self.X_use[i] - mu.squeeze()
            recon  = mu.squeeze() + basis @ (basis.T @ xi)
            errs.append(np.linalg.norm(xi - recon))

        return float(np.mean(errs))
