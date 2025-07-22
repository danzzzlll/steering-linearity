# core/metric_base.py
from abc import ABC, abstractmethod
from src.utils.norm import normalize
from src.utils.faiss_helpers import knn_graph

class Metric(ABC):
    def __init__(self, ds, cache, layer, *, k=20, norm="z"):
        self.ds, self.cache, self.layer = ds, cache, layer
        self.k = k
        self.X = normalize(ds[layer], norm)  
        self.knn = self._get_knn()

    def _get_knn(self):
        return self.cache.get_knn(self.layer, self.X, k=self.k)

    @abstractmethod
    def compute(self):
        ...
