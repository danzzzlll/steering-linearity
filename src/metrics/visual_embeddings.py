from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
import umap

from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


class UmapIsomapPlot(Metric):
    """
    Строит UMAP‑2D и Isomap‑2D после PCA‑30.

    • тяжёлые шаги (PCA, k‑NN Isomap) берутся из SharedCache;
      для Isomap‑геодезики используется ключ  layer+"_pca30", чтобы
      не конфликтовать с другими метриками слоя.
    • PNG сохраняется в plot_dir / umap_isomap_L{layer}.png
    • Возвращает (XY_umap, XY_isomap)
    """

    def __init__(
        self,
        *a,
        n_pca: int = 30,
        k: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
        plot_dir: str | Path = "plots",
        **kw,
    ):
        super().__init__(*a, k=k, **kw)
        self.n_pca = n_pca
        self.umap_min_dist = umap_min_dist
        self.random_state = random_state
        self.plot_dir = Path(plot_dir); self.plot_dir.mkdir(exist_ok=True, parents=True)
        
    @timecount
    def compute(self) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore[override]
        print("MAKE UmapIsomapPlot ...")
        pca = self.cache.get_pca(
            self.layer,
            self.X,
            n_components=self.n_pca,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        X_pca = pca.transform(self.X).astype(np.float32)

        XY_umap = umap.UMAP(
            n_neighbors=self.k,
            min_dist=self.umap_min_dist,
            n_components=2,
            metric="euclidean",
            random_state=self.random_state,
        ).fit_transform(X_pca)

        layer_key = f"{self.layer}_pca30"
        _ = self.cache.get_isomap_geodesic(layer_key, X_pca, k=self.k, path_method="auto")
        XY_iso = Isomap(n_neighbors=self.k, n_components=2, path_method="auto").fit_transform(X_pca)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(XY_umap[:, 0], XY_umap[:, 1], s=4, alpha=0.6)
        axes[0].set_title("UMAP после PCA‑30"); axes[0].axis("off")

        axes[1].scatter(XY_iso[:, 0], XY_iso[:, 1], s=4, alpha=0.6, c="grey")
        axes[1].set_title("Isomap после PCA‑30"); axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(self.plot_dir / f"umap_isomap_L{self.layer}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("MAKE UmapIsomapPlot DONE")
        # return XY_umap, XY_iso
