import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from src.core.metric_base import Metric
from src.utils.time_decorator import timecount

class H1SystoleLen(Metric):
    def __init__(self, ds, cache, layer, *, k=20, norm="z", n_sample=2000):
        super().__init__(ds, cache, layer, k=k, norm=norm)
        self.n_sample = int(n_sample)

    @timecount
    def compute(self) -> dict:
        # фиксированный сабсэмпл, как и для tp1/delta
        N_total = self.X.shape[0]
        idx = self.cache.get_sample_idx(N_total, self.n_sample)
        X = self.X[idx]

        D = squareform(pdist(X, metric="euclidean"))
        dgms = ripser(D, distance_matrix=True, maxdim=1)["dgms"]
        if len(dgms) < 2 or dgms[1].size == 0:
            return {"h1_systole_len_norm": 0.0}

        pers = dgms[1][:, 1] - dgms[1][:, 0]
        systole = float(np.max(pers)) if pers.size > 0 else 0.0

        scale = np.percentile(D, 95)  # робастная нормировка масштаба
        return {"h1_systole_len_norm": (systole / scale).item() if scale > 0 else 0.0}
