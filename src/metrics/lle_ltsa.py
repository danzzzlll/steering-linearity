from typing import Literal
from sklearn.manifold import LocallyLinearEmbedding
from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


class _BaseLleError(Metric):
    _method: Literal["ltsa", "standard"] = "standard"

    def __init__(
        self,
        ds,
        cache,
        layer,
        *,
        d_out: int = 20,
        eigen_solver: str = "auto",
        k: int = 20,
        norm: str = "z",
    ):
        self.d_out = d_out
        self.eigen_solver = eigen_solver
        super().__init__(ds, cache, layer, k=k, norm=norm)

    @timecount
    def compute(self) -> float:                      # type: ignore[override]
        lle = LocallyLinearEmbedding(
            n_neighbors=self.k,
            n_components=self.d_out,
            method=self._method,
            eigen_solver=self.eigen_solver,
        )
        _ = lle.fit_transform(self.X)
        return float(lle.reconstruction_error_)


class LtsaError(_BaseLleError):
    _method: Literal["ltsa"] = "ltsa"


class LleError(_BaseLleError):
    _method: Literal["standard"] = "standard"
