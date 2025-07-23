"""
Внимание: Ollivier–Ricci не запускается под Windows из‑за 'fork'.
На Linux работает нормально.
"""
from typing import Dict

import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


class OllivierRicciCurvature(Metric):
    """
    Кривизна Олливье‑Риччи на k‑NN графе слоя
    (можно предварительно сжать слой до n_pca компонент).
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

        self.X_proc = (
            self.cache.get_pca(
                self.layer,
                self.X,
                n_components=n_pca,
                svd_solver="randomized",
                random_state=random_state,
            ).transform(self.X).astype(np.float32)
            if use_pca
            else self.X.astype(np.float32)
        )

        # отдельный ключ для k‑NN графа Ricci
        self.layer_sub = f"{self.layer}_ricci"
        

    def _build_graph(self) -> nx.Graph:
        knn = self.cache.get_knn(self.layer_sub, self.X_proc, k=self.k)

        G = nx.Graph()
        for i, neigh in enumerate(knn):
            for j in neigh:
                G.add_edge(i, j, weight=1.0)
        return G

    @timecount
    def compute(self) -> Dict[str, float]:          # type: ignore[override]
        print("MAKE OllivierRicciCurvature ...")
        G = self._build_graph()
        orc = OllivierRicci(G, alpha=self.alpha, proc=0, verbose="ERROR")
        orc.compute_ricci_curvature()

        vals = np.array(
            [d["ricciCurvature"] for _, _, d in G.edges(data=True)],
            dtype=np.float32,
        )
        print({
            "mean":  float(vals.mean()),
            "std":   float(vals.std(ddof=1)),
            "edges": int(len(vals)),
        })
        print("MAKE OllivierRicciCurvature DONE")
        return {
            "mean":  float(vals.mean()),
            "std":   float(vals.std(ddof=1)),
            "edges": int(len(vals)),
        }
