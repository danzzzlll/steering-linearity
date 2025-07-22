from src.core.metric_base import Metric

class IsotropyRatio(Metric):
    def __init__(self, *a,
                 svd_solver: str = "randomized",
                 n_components: int | None = None,
                 random_state: int = 42,
                 **kw):
        super().__init__(*a, **kw)
        self.pca_kw = dict(n_components=n_components,
                           svd_solver=svd_solver,
                           random_state=random_state)


    def compute(self) -> float:                        # type: ignore[override]
        pca = self.cache.get_pca(self.layer, self.X, **self.pca_kw)
        var = pca.explained_variance_
        iso_ratio = float(var[0] / var.mean())
        print(f"Isotropy-ratio: {iso_ratio:.2f}")
        return iso_ratio
