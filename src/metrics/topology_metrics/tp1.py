# src/metrics/tp1.py
import numpy as np
from src.core.metric_base import Metric
from src.utils.time_decorator import timecount

from sklearn.metrics import pairwise_distances

from ripser import ripser


class TotalPersistenceH1(Metric):
    """
    Total Persistence for H1 (↓ лучше) + Max Lifetime H1 (↓ лучше).
    Оба нормированы на V_sample и медиану длины ребра (k-NN).
    """

    def __init__(self, ds, cache, layer, *, k=20, norm="z", n_sample=3000):
        super().__init__(ds, cache, layer, k=k, norm=norm)
        self.n_sample = int(n_sample)


    @timecount
    def compute(self) -> dict:
        X_full = self.X
        N_total = X_full.shape[0]
        idx = self.cache.get_sample_idx(N_total, self.n_sample)
        X = X_full[idx]

        D = pairwise_distances(X, metric="euclidean")

        res = ripser(D, maxdim=1, distance_matrix=True)
        H1 = res["dgms"][1] if len(res["dgms"]) > 1 else np.empty((0, 2))

        # lifetimes
        if H1.size == 0:
            tp1_raw = 0.0
            maxlife_raw = 0.0
        else:
            lifetimes = H1[:, 1] - H1[:, 0]
            tp1_raw = float(np.sum(lifetimes))
            maxlife_raw = float(np.max(lifetimes))

        # нормировка на размер и типичный масштаб (медиана ребра на том же сабсэмпле)
        edges, w_e = self._get_undirected_edges_with_weights_subset(idx)
        med_edge = float(np.median(w_e)) if w_e.size > 0 else 1.0
        V_sample = int(X.shape[0])
        denom = max(1, V_sample) * max(1e-12, med_edge)

        tp1_norm = tp1_raw / denom
        tp1_maxlife_norm = maxlife_raw / max(1e-12, med_edge)

        return {
            "tp1_norm": tp1_norm,                   # ↓ лучше: общая «масса» циклов
            "tp1_maxlife_norm": tp1_maxlife_norm,   # ↓ лучше: самая «толстая» петля
        }

    def _get_undirected_edges_with_weights_subset(self, idx_subset):
        knn = self.knn
        if isinstance(knn, tuple) and len(knn) == 2:
            nbrs, dists = knn
        elif isinstance(knn, dict) and "indices" in knn:
            nbrs, dists = knn["indices"], knn.get("distances", None)
        else:
            nbrs, dists = knn, None

        idx_set = set(int(i) for i in np.asarray(idx_subset))
        edge_map = {}
        X = self.X

        def pair_dist(i, j):
            v = X[i] - X[j]
            return float(np.sqrt(np.dot(v, v)))

        k = nbrs.shape[1]
        for i in idx_set:
            for jj in range(k):
                j = int(nbrs[i, jj])
                if j == i or j not in idx_set:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in edge_map:
                    continue
                w = float(dists[i, jj]) if dists is not None else pair_dist(a, b)
                edge_map[(a, b)] = w

        if not edge_map:
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

        edges = np.fromiter((x for ab in edge_map.keys() for x in ab),
                            dtype=np.int32).reshape(-1, 2)
        weights = np.fromiter(edge_map.values(), dtype=np.float32)
        return edges, weights
