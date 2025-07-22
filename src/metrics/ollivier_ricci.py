'''
Для windows не работает, может сработает для Linux
'''


from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import faiss
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.core.metric_base import Metric


class OllivierRicciCurvature(Metric):
    """
    Считает кривизну Олливье-Риччи на k-NN графе слоя.

    Parameters
    ----------
    k           : int    – число соседей для графа
    alpha       : float  – параметр Олливье (0 ≤ α ≤ 1)
    use_pca     : bool   – если True → сначала PCA до n_pca компонент
    n_pca       : int    – число компонент PCA (если use_pca=True)
    random_state: int    – random_state для PCA
    """

    def __init__(
        self,
        *a,
        k: int = 15,
        alpha: float = 0.5,
        use_pca: bool = True,
        n_pca: int = 30,
        random_state: int = 0,
        **kw,
    ):
        super().__init__(*a, k=k, **kw)
        self.alpha = alpha
        self.use_pca = use_pca
        self.n_pca = n_pca
        self.random_state = random_state

        if use_pca:
            self._X_proc = self._pca_projection()
        else:
            self._X_proc = self.X.astype(np.float32)


    def _pca_projection(self) -> np.ndarray:
        pca = self.cache.get_pca(
            self.layer,
            self.X,
            n_components=self.n_pca,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        return pca.transform(self.X).astype(np.float32)  # (N, n_pca)


    def _build_graph(self) -> nx.Graph:
        """
        k-NN граф: невзвешенные рёбра weight=1.0 (как в оригинальном примере).
        Можно поменять вес на дистанцию, если нужно.
        """
        knn = self.cache.get_knn(self.layer, self._X_proc, k=self.k)  # (N, k)

        G = nx.Graph()
        for i, neigh in enumerate(knn):
            for j in neigh:          
                G.add_edge(i, j, weight=1.0)
        return G


    def compute(self) -> Dict[str, float]:        # type: ignore[override]
        """
        Возвращает:
            {'mean': ..., 'std': ..., 'edges': <int>}
        """
        G = self._build_graph()

        orc = OllivierRicci(G, alpha=self.alpha, proc=0, verbose="ERROR")
        orc.compute_ricci_curvature()

        curv_vals = np.array(
            [attr["ricciCurvature"] for _, _, attr in G.edges(data=True)],
            dtype=np.float32,
        )
        return {
            "mean": float(curv_vals.mean()),
            "std": float(curv_vals.std(ddof=1)),
            "edges": int(len(curv_vals)),
        }
