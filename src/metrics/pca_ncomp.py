from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


class PcaNComponents(Metric):
    """
    Возвращает, сколько главных компонент нужно, чтобы объяснить
    не менее `target_var_ratio` дисперсии (например 0.95 → 95 %).

    Parameters
    ----------
    target_var_ratio : float   – доля дисперсии (0 … 1)
    svd_solver       : str     – 'randomized' | 'auto' | ...
    """

    def __init__(
        self,
        *a,
        target_var_ratio: float = 0.95,
        svd_solver: str = "randomized",
        **kw,
    ):
        super().__init__(*a, **kw)
        self.target_var_ratio = float(target_var_ratio)
        self.svd_solver = svd_solver
        
    @timecount
    def compute(self) -> int:                      # type: ignore[override]
        print("MAKE PcaNComponents ...")
        pca = self.cache.get_pca(
            self.layer,
            self.X.astype("float32"),
            n_components=self.target_var_ratio,    # < 1.0 → доля дисперсии
            svd_solver=self.svd_solver,
        )
        n_comp = int(pca.n_components_)
        print(f"PCA components to reach {self.target_var_ratio:.0%}: {n_comp}")
        print("MAKE PcaNComponents DONE")
        return n_comp
