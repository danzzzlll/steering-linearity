import hashlib
import ujson as json
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import faiss
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


def _hash_params(params: dict[str, Any]) -> str:
    """Сериализуем словарь"""
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


class SharedCache:
    """
    Хранит ресурсо-ёмкие вычисления, зависящие от ( layer , **params ).
    • PCA           – key = (layer, hash(params))
    • k-NN graph    – key = (layer, hash({'k':k,'metric':…}))
    • Isomap D_geo  – key = (layer, hash({'k':k,'path':…}))
    • Можно добавлять новые словари по той же схеме.
    """

    def __init__(self, random_state: int = 42):
        self.pca: Dict[Tuple[str, str], PCA] = {}
        self.knn: Dict[Tuple[str, str], np.ndarray] = {}
        self.isomap_geo: Dict[Tuple[str, str], np.ndarray] = {}
        self.rng = np.random.default_rng(random_state)
        self.sampling: dict[int, np.ndarray] = {}  # key = n_sample


    def get_x256(self, layer: str, X: np.ndarray) -> np.ndarray:
        key = (layer, "pca256")
        if key not in self.pca:
            self.pca[key] = faiss.PCAMatrix(X.shape[1], 256)  # faiss RandPCA
            self.pca[key].train(X.astype("float32"))
        return self.pca[key].apply_py(X.astype("float32"))


    def get_sample_idx(self, n_total: int, n_sample: int) -> np.ndarray:
        if n_sample not in self.sampling:
            self.sampling[n_sample] = self.rng.choice(
                n_total, n_sample, replace=False
            )
        return self.sampling[n_sample]


    def get_pca(self, layer: str, X: np.ndarray, **pca_kw) -> PCA:
        """
        Возвращает PCA, обученный на `X` (или берёт из кэша).
        Пример параметров: n_components=None, svd_solver='randomized'.
        """
        key = (layer, _hash_params(pca_kw))
        if key not in self.pca:
            self.pca[key] = PCA(**pca_kw).fit(X)
        return self.pca[key]


    def get_knn(
        self,
        layer: str,
        X: np.ndarray,
        *,
        k: int = 20,
        metric: str = "l2",
    ) -> np.ndarray:
        """
        Возвращает индексы k ближайших соседей (без self):
        shape = (N, k)
        """
        key = (layer, _hash_params({"k": k, "metric": metric}))
        if key not in self.knn:
            if metric != "l2":
                raise NotImplementedError("Пока реализован только L2 (faiss.IndexFlatL2)")
            index = faiss.IndexFlatL2(X.shape[1])
            index.add(X.astype(np.float32))
            _, nn = index.search(X.astype(np.float32), k + 1)
            self.knn[key] = nn[:, 1:]                 
        return self.knn[key]


    def get_isomap_geodesic(
        self,
        layer: str,
        X: np.ndarray,
        *,
        k: int = 20,
        path_method: str = "auto",
    ) -> np.ndarray:
        """
        Считает геодезическую (короткий путь по графу k-NN) матрицу Isomap.
        Сама Isomap-эмбеддинг не кэшируется, только D_geo.
        """
        key = (layer, _hash_params({"k": k, "path": path_method}))
        if key not in self.isomap_geo:
            iso = Isomap(n_neighbors=k, n_components=2,
                         path_method=path_method).fit(X)
            self.isomap_geo[key] = iso.dist_matrix_
        return self.isomap_geo[key]
