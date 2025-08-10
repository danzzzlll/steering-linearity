import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


@dataclass
class ECCResult:
    ecc_auc: float
    alpha: np.ndarray     # shape [m]
    chi: np.ndarray       # shape [m]
    V: int
    E: int
    F: int

class EulerCharacteristicCurve(Metric):
    """
    Euler Characteristic Curve (ECC) on a k-NN graph with VR2 triangles.

    χ(τ) = V - E_≤τ + F_≤τ,
    где E_≤τ — число рёбер с весом ≤ τ, F_≤τ — треугольники (2-симплексы) с
    фильтрацией t = max(w_ij, w_jk, w_ik) ≤ τ.

    Мы параметризуем кривую по α = E_≤τ / E_total (доля рёбер),
    считаем χ(α) на равномерной сетке α∈[0,1], затем AUC по α (трапеции).
    Малый ecc_auc ⇒ «плоский» χ-профиль ⇒ ближе к линейному многообразию.
    """

    def __init__(self, ds, cache, layer, *, k=20, norm="z",
                 n_bins: int = 64, triangles: bool = True):
        super().__init__(ds, cache, layer, k=k, norm=norm)
        self.n_bins = int(n_bins)
        self.use_triangles = bool(triangles)


    @timecount
    def compute(self) -> dict:
        """
        Возвращает:
        - ecc_deficit_auc  (↓ лучше): ∫ max(0, 1 - χ(α)/V) dα
        - ecc_centered_auc: ∫ (χ(α)/V - 1) dα
        - служебные поля: ecc_alpha, ecc_chi, ecc_V, ecc_E, ecc_F
        """
        V = int(self.X.shape[0])
        edges, w_e = self._get_undirected_edges_with_weights()
        if edges.size == 0:


            return float("nan")

        order = np.argsort(w_e)
        w_sorted = w_e[order]
        E_total = int(len(w_sorted))

        tri_filtration = (
            self._triangle_filtration(edges, w_e) if self.use_triangles else np.array([], dtype=w_e.dtype)
        )
        if tri_filtration.size > 0:
            tri_filtration.sort()

        m = max(2, int(self.n_bins))
        alpha = np.linspace(0.0, 1.0, num=m)

        chi = np.empty(m, dtype=float)
        for t, a in enumerate(alpha):
            if a <= 0.0:
                tau = -np.inf
                e_count = 0
            elif a >= 1.0:
                tau = np.inf if E_total == 0 else w_sorted[-1]
                e_count = E_total
            else:
                idx = max(0, int(np.floor(a * E_total)) - 1)
                tau = w_sorted[idx]
                e_count = int(np.searchsorted(w_sorted, tau, side="right"))

            if tri_filtration.size > 0:
                f_count = int(np.searchsorted(tri_filtration, tau, side="right"))
            else:
                f_count = 0

            chi[t] = V - e_count + f_count

        V_safe = max(1, V)
        chi_norm = chi / V_safe
        deficit = np.maximum(0.0, 1.0 - chi_norm)           # ниже 1
        ecc_deficit_auc = float(np.trapz(deficit, alpha))   # ↓ лучше
    
        return ecc_deficit_auc


    def _get_undirected_edges_with_weights(self):
        """
        Достаём неориентированные рёбра и веса.
        Поддерживаем несколько возможных форматов self.knn:
          • tuple (indices, distances)
          • dict {'indices': ..., 'distances': ...}
          • ndarray shape (N, k) — только индексы (посчитаем веса сами)
        Возвращает:
          edges: ndarray[int32] shape (M, 2) с i<j
          w:     ndarray[float32] shape (M,)
        """
        knn = self.knn
        if isinstance(knn, tuple) and len(knn) == 2:
            nbrs, dists = knn
        elif isinstance(knn, dict) and "indices" in knn:
            nbrs, dists = knn["indices"], knn.get("distances", None)
        else:
            nbrs, dists = knn, None

        N, k = nbrs.shape[0], nbrs.shape[1]
        # собираем множество неориентированных рёбер
        edge_map = {}
        X = self.X

        # быстрый векторный рассчёт расстояний при необходимости
        def pair_dist(i, j):
            # евклидова метрика (в кэше k-NN построен по L2)
            v = X[i] - X[j]
            return float(np.sqrt(np.dot(v, v)))

        for i in range(N):
            for jj in range(k):
                j = int(nbrs[i, jj])
                if j < 0 or j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in edge_map:
                    continue
                if dists is not None:
                    w = float(dists[i, jj])
                else:
                    w = pair_dist(a, b)
                edge_map[(a, b)] = w

        if not edge_map:
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

        edges = np.fromiter((x for ab in edge_map.keys() for x in ab), dtype=np.int32).reshape(-1, 2)
        weights = np.fromiter(edge_map.values(), dtype=np.float32)
        return edges, weights

    def _triangle_filtration(self, edges: np.ndarray, w_e: np.ndarray) -> np.ndarray:
        """
        Считаем фильтрацию треугольников VR2: t = max(w_ij, w_jk, w_ik).
        Алгоритм O(N * k^2) по k-NN (k~15-20).
        """
        if edges.size == 0:
            return np.array([], dtype=w_e.dtype)

        # строим списки соседей и словарь весов для быстрого запроса (i<j ключ)
        adj = defaultdict(set)
        weight = {}

        for (i, j), w in zip(edges, w_e):
            adj[i].add(j)
            adj[j].add(i)
            a, b = (i, j) if i < j else (j, i)
            weight[(a, b)] = w

        tri_vals = []
        # перебор вершин и пересечение соседей
        for i, Ni in adj.items():
            Ni_list = sorted(Ni)
            L = len(Ni_list)
            for a_idx in range(L):
                j = Ni_list[a_idx]
                # берём пересечение N(i) ∩ N(j), чтобы найти k > j (упорядочим для уникальности)
                common = adj[j].intersection(Ni)
                for k in common:
                    # упорядочивание i < j < k для уникальности
                    a, b, c = sorted((i, j, k))
                    if i != a or j != b or k != c:
                        continue
                    # макс из трёх рёбер
                    w_ij = weight[(a, b)]
                    w_ik = weight[(a, c)] if (a, c) in weight else weight[(c, a)]
                    w_jk = weight[(b, c)]
                    tri_vals.append(max(w_ij, w_ik, w_jk))

        if not tri_vals:
            return np.array([], dtype=w_e.dtype)
        return np.asarray(tri_vals, dtype=w_e.dtype)
