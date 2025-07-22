import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from src.core.metric_base import Metric


class LocalPcaReconError(Metric):
    """
    Средняя ошибка реконструкции точки через PCA,
    обученный на её k-соседях (без самой точки).

    Parameters
    ----------
    d_latent : int   – размерность локального тангент-пространства
    svd      : str   – "randomized" | "auto"  (передаётся в sklearn.PCA)
    """
    def __init__(
        self,
        *a,
        d_latent: int = 10,
        svd: str = "randomized",
        **kw,
    ):
        super().__init__(*a, **kw)   
        self.d_latent = d_latent
        self.svd = svd


    def compute(self) -> float:             # type: ignore[override]
        idx_sub = self.cache.get_sample_idx(len(self.X), 5000)
        X_use   = self.cache.get_x256(self.layer, self.X)[idx_sub]

        knn = self.cache.get_knn(self.layer, X_use, k=self.k)  # (N, k)

        errs: list[float] = []
        for i, neigh in tqdm(enumerate(knn), total=len(knn), desc=f"Local PCA L{self.layer}"):
            Xn = X_use[neigh[1:]]                       # (k, d)
            mu = Xn.mean(0, keepdims=True)
            Xc = Xn - mu

            pca = PCA(n_components=self.d_latent, svd_solver=self.svd, random_state=42)
            basis = pca.fit(Xc).components_.T            # (d, d_latent)

            xi = X_use[i] - mu.squeeze()
            recon = mu.squeeze() + basis @ (basis.T @ xi)
            errs.append(np.linalg.norm(xi - recon))

        return float(np.mean(errs))
