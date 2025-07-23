import numpy as np
from sklearn.linear_model import Ridge
from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


class Nlc(Metric):
    """
    Non‑Linear Component между prev_layer → layer
    (secant‑NLC, local‑NLC, MSE линейной аппроксимации).
    """

    def __init__(
        self,
        ds,
        cache,
        layer,
        *,
        prev_layer: int | str | None = None,
        k: int = 20,
        n_pairs: int = 1_000,
        d_latent: int = 256,
        ridge_alpha: float = 1.0,
        norm: str = "center",
        **kw,
    ):
        super().__init__(ds, cache, layer, k=k, norm=norm, **kw)

        if prev_layer is None:
            prev_layer = layer - 1
        if isinstance(prev_layer, int) and prev_layer < 0:
            raise ValueError("NLC не определён для слоя 0")
        self.prev_layer = prev_layer
        self.n_pairs    = n_pairs

        X_prev = ds[prev_layer].astype(np.float32)
        self.Xc = X_prev - X_prev.mean(0, keepdims=True)
        self.Hc = self.X   - self.X.mean(0, keepdims=True)

        self.Xc_small = cache.get_x256(str(prev_layer), self.Xc)[:, :d_latent]
        self.Hc_small = cache.get_x256(str(layer),      self.Hc)[:, :d_latent]

        self.W = (
            Ridge(alpha=ridge_alpha, fit_intercept=False)
            .fit(self.Xc_small, self.Hc_small)
            .coef_.T                                              # (d_latent, d_latent)
        )

        # — собственный ключ для k‑NN, чтобы не делить с другими метриками
        self.layer_sub = f"{layer}_nlc"
        

    def _secant_nlc(self) -> float:
        flat_idx = self.cache.get_sample_idx(len(self.Xc_small), self.n_pairs * 2)
        idx1, idx2 = flat_idx.reshape(-1, 2).T

        dX = self.Xc_small[idx2] - self.Xc_small[idx1]
        dH = self.Hc_small[idx2] - self.Hc_small[idx1]
        num = np.linalg.norm(dH, axis=1)
        den = np.linalg.norm(dX @ self.W, axis=1) + 1e-9
        return float(np.mean(num / den))


    def _local_nlc(self) -> float:
        knn = self.cache.get_knn(self.layer_sub, self.Hc_small, k=self.k)

        i_idx = np.repeat(np.arange(len(self.Hc_small)), self.k)
        j_idx = knn.ravel()

        dX = self.Xc_small[j_idx] - self.Xc_small[i_idx]
        dH = self.Hc_small[j_idx] - self.Hc_small[i_idx]

        num = np.linalg.norm(dH, axis=1)
        den = np.linalg.norm(dX @ self.W, axis=1) + 1e-9
        return float(np.mean(num / den))


    def _mse_linear(self) -> float:
        H_lin = self.Xc_small @ self.W
        return float(np.mean(np.linalg.norm(self.Hc_small - H_lin, axis=1)))

    @timecount
    def compute(self) -> dict[str, float] | None:  # type: ignore[override]
        print("MAKE Nlc ...")
        if self.layer == 0:
            return None
        print({
            "secant": self._secant_nlc(),
            "local":  self._local_nlc(),
            "mse":    self._mse_linear(),
        })
        print("MAKE Nlc DONE")
        return {
            "secant": self._secant_nlc(),
            "local":  self._local_nlc(),
            "mse":    self._mse_linear(),
        }
