from skdim.id import TwoNN, MLE
from src.core.metric_base import Metric

class TwoNnId(Metric):
    def compute(self) -> float:                      # type: ignore[override]
        d_global = float(TwoNN().fit_transform(self.X))
        print("TwoNN global ID:", float(d_global))
        return d_global

class MleId(Metric):
    def __init__(self, *a, K: int = 20, **kw):
        super().__init__(*a, **kw); self.K = K
    def compute(self) -> float:                      # type: ignore[override]
        d_mle = float(MLE(K=self.K).fit_transform(self.X))
        print("MLE-ID:", float(d_mle))
        return d_mle
