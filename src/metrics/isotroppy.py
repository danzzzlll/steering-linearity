from src.core.metric_base import Metric
from src.utils.time_decorator import timecount

class IsotropyRatio(Metric):
    """
    λ₁ / 〈λ〉 — отношение первой собственной дисперсии к средней.

    n_components:
        • None            – берём весь спектр (медленнее, но точнее)
        • int  ≤ d        – считаем по первым `int` компонентам
        • float < 1.0     – доля объяснённой дисперсии (0 … 1)

    Если передана доля, автоматически выбирается svd_solver='full'
    (sklearn не поддерживает вариант float с 'randomized').
    """

    def __init__(
        self,
        *a,
        n_components: int | float | None = None,
        svd_solver: str = "randomized",
        random_state: int = 42,
        **kw,
    ):
        super().__init__(*a, **kw)

        if isinstance(n_components, float) and n_components < 1.0:
            if svd_solver == "randomized":
                svd_solver = "full"     # безопасный вариант для float < 1.0

        self.pca_kw = dict(
            n_components=n_components,
            svd_solver=svd_solver,
            random_state=random_state,
        )
        
    @timecount
    def compute(self) -> float:                     # type: ignore[override]
        print("MAKE IsotropyRatio ...")
        pca = self.cache.get_pca(
            self.layer,
            self.X.astype("float32"),
            **self.pca_kw,
        )
        var = pca.explained_variance_
        iso_ratio = float(var[0] / var.mean())
        print(f"Isotropy‑ratio: {iso_ratio:.2f}")
        print("MAKE IsotropyRatio DONE")
        return iso_ratio
