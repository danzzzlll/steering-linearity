import numpy as np
from src.core.metric_base import Metric
from src.utils.time_decorator import timecount

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


class DeltaHyperbolicity(Metric):
    """
    Delta-hyperbolicity (Gromov 4-point condition) on geodesic distances.

    Возвращает:
      - delta_hyperbolicity_norm (↓ лучше): медиана δ, нормированная на диаметр LCC
      - lcc_deficit (↓ лучше): 1 - доля точек в крупнейшей компоненте (0 = граф связен)
      - n_components: число компонент связности (диагностика)

    Примечания:
      • Геодезические расстояния берутся как all-pairs shortest paths в k-NN графе на сабсэмпле.
      • Если граф несвязный, расчёт ведётся на LCC, но lcc_deficit сообщает, сколько точек потеряно.
    """

    def __init__(self, ds, cache, layer, *,
                 k=20, norm="z", n_sample=2000, n_quads=20000, agg="median"):
        super().__init__(ds, cache, layer, k=k, norm=norm)
        self.n_sample = int(n_sample)
        self.n_quads = int(n_quads)
        assert agg in ("mean", "median"), "agg must be 'mean' or 'median'"
        self.agg = agg

    @timecount
    def compute(self) -> dict:
        # 1) Сабсэмпл
        N_total = self.X.shape[0]
        idx = self.cache.get_sample_idx(N_total, self.n_sample)
        idx = np.asarray(idx, dtype=int)
        n = idx.size
        if n < 4:
            return {
                "delta_hyperbolicity_norm": float("nan"),
                "lcc_deficit": float("nan"),
                "n_components": 0,
            }

        # 2) Взвешенный НЕориентированный k-NN граф на сабсэмпле
        if isinstance(self.knn, tuple) and len(self.knn) == 2:
            nbrs, dists = self.knn
        elif isinstance(self.knn, dict) and "indices" in self.knn:
            nbrs, dists = self.knn["indices"], self.knn.get("distances", None)
        else:
            nbrs, dists = self.knn, None

        pos_in_sub = -np.ones(N_total, dtype=int)
        pos_in_sub[idx] = np.arange(n)

        rows, cols, data = [], [], []
        for i_global in idx:
            i_local = pos_in_sub[i_global]
            for jj in range(nbrs.shape[1]):
                j_global = int(nbrs[i_global, jj])
                j_local = pos_in_sub[j_global]
                if j_local < 0 or j_local == i_local:
                    continue
                if dists is not None:
                    w = float(dists[i_global, jj])
                else:
                    v = self.X[i_global] - self.X[j_global]
                    w = float(np.sqrt(np.dot(v, v)))
                # симметризуем
                rows.extend([i_local, j_local])
                cols.extend([j_local, i_local])
                data.extend([w, w])

        if not data:
            return {
                "delta_hyperbolicity_norm": float("nan"),
                "lcc_deficit": float("nan"),
                "n_components": 0,
            }

        G = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        # 3) Компоненты связности и LCC
        n_comp, labels = connected_components(G, directed=False)
        counts = np.bincount(labels)
        lcc_size = int(counts.max())
        lcc_frac = lcc_size / float(labels.size)
        lcc_deficit = 1.0 - lcc_frac  # ↓ лучше

        if n_comp > 1:
            lcc_label = int(np.argmax(counts))
            keep = (labels == lcc_label)
            # Срез до LCC (явно, чтобы сохранить веса)
            mapping = -np.ones(n, dtype=int)
            mapping[np.where(keep)[0]] = np.arange(int(keep.sum()))
            rows_l, cols_l, data_l = [], [], []
            G_coo = G.tocoo()
            for r, c, v in zip(G_coo.row, G_coo.col, G_coo.data):
                if keep[r] and keep[c]:
                    rows_l.append(mapping[r])
                    cols_l.append(mapping[c])
                    data_l.append(v)
            n_lcc = int(keep.sum())
            if n_lcc < 4:
                return {
                    "delta_hyperbolicity_norm": float("nan"),
                    "lcc_deficit": float(lcc_deficit),
                    "n_components": int(n_comp),
                }
            G = coo_matrix((data_l, (rows_l, cols_l)), shape=(n_lcc, n_lcc)).tocsr()
            n = n_lcc  # обновили размер
        # else: граф уже связен

        # 4) Геодезики: all-pairs shortest paths (Dijkstra)
        D_geo = shortest_path(G, directed=False, unweighted=False, method="D")

        d_max = float(np.max(D_geo))  # диаметр LCC
        if not np.isfinite(d_max) or d_max <= 0:
            return {
                "delta_hyperbolicity_norm": float("nan"),
                "lcc_deficit": float(lcc_deficit),
                "n_components": int(n_comp),
            }

        # 5) Оценка δ по случайным 4-кортежам
        rng = np.random.default_rng(42)
        m_quads_cap = n * (n - 1) * (n - 2) * (n - 3) // 24  # C(n,4)
        m_quads = int(min(self.n_quads, m_quads_cap))
        if n < 4 or m_quads <= 0:
            return {
                "delta_hyperbolicity_norm": float("nan"),
                "lcc_deficit": float(lcc_deficit),
                "n_components": int(n_comp),
            }

        deltas = np.empty(m_quads, dtype=float)
        for qi in range(m_quads):
            a, b, c, d = rng.choice(n, size=4, replace=False)
            dab, dcd = D_geo[a, b], D_geo[c, d]
            dac, dbd = D_geo[a, c], D_geo[b, d]
            dad, dbc = D_geo[a, d], D_geo[b, c]
            s1 = dab + dcd
            s2 = dac + dbd
            s3 = dad + dbc
            max_s = max(s1, s2, s3)
            mid_s = sorted((s1, s2, s3))[-2]
            deltas[qi] = 0.5 * (max_s - mid_s)

        if self.agg == "median":
            delta_val = float(np.median(deltas))
        else:
            delta_val = float(np.mean(deltas))

        delta_norm = delta_val / d_max  # ↓ лучше

        return {
            "delta_hyperbolicity_norm": float(delta_norm),
            "lcc_deficit": float(lcc_deficit),   # 0 — связен, меньше — лучше
            "n_components": int(n_comp),
        }
    


class AutoDeltaHyperbolicity(Metric):
    """
    Автоподбор k для DeltaHyperbolicity.

    Идея:
      • Фиксируем один и тот же сабсэмпл (через SharedCache.get_sample_idx).
      • Пробегаем сетку k, считаем delta_hyperbolicity_norm и lcc_deficit.
      • Выбираем минимальный k, где:
          - lcc_deficit ≤ (1 - lcc_thresh)
          - |δ(k+Δk) - δ(k)| / δ(k) ≤ eps_rel   (плато стабильности)
      • Возвращаем метрику при выбранном k*, плюс служебные поля.

    Возвращает:
      - delta_hyperbolicity_norm  (↓ лучше)
      - delta_hyperbolicity_penalized = delta + lcc_deficit  (↓ лучше; учитывает несвязность)
      - lcc_deficit               (↓ лучше; 0 = граф связен)
      - chosen_k                  (выбранный k*)
      - k_scan                    (список результатов по всем k)
    """

    def __init__(
        self,
        ds,
        cache,
        layer,
        *,
        # базовые параметры для внутренних прогонов
        norm: str = "z",
        n_sample: int = 2000,
        n_quads: int = 20000,
        agg: str = "median",
        # сетка по k
        k_grid: tuple[int, ...] = (20, 25, 30, 35, 40, 45, 50, 60),
        # критерии выбора
        lcc_thresh: float = 0.95,   # хотим ≥ 95% точек в LCC
        eps_rel: float = 0.05,      # стабильность δ между соседними k (≤5%)
        # штраф
        use_penalty: bool = True,
    ):
        # В Metric нужен k, но мы будем игнорировать self.k и пробегать k_grid.
        super().__init__(ds, cache, layer, k=k_grid[0], norm=norm)
        self.n_sample = int(n_sample)
        self.n_quads = int(n_quads)
        self.agg = agg
        self.k_grid = tuple(k_grid)
        self.lcc_thresh = float(lcc_thresh)
        self.eps_rel = float(eps_rel)
        self.use_penalty = bool(use_penalty)

    @timecount
    def compute(self) -> dict:
        # 1) Фиксируем сабсэмпл на весь прогон (SharedCache делает это по ключу n_sample)
        N_total = self.X.shape[0]
        _ = self.cache.get_sample_idx(N_total, self.n_sample)  # фиксируем в кэше

        # 2) Скан по k
        scan = []
        for k in self.k_grid:
            m = DeltaHyperbolicity(
                ds=self.ds,
                cache=self.cache,
                layer=self.layer,
                k=k,
                norm="z",
                n_sample=self.n_sample,
                n_quads=self.n_quads,
                agg=self.agg,
            ).compute()
            scan.append({
                "k": int(k),
                "delta": float(m["delta_hyperbolicity_norm"]),
                "lcc_deficit": float(m.get("lcc_deficit", np.nan)),
                "n_components": int(m.get("n_components", 0)),
            })

        # 3) Функции-helpers
        def stable(i: int) -> bool:
            # на последнем k считаем, что стабилен
            if i >= len(scan) - 1:
                return True
            cur = scan[i]["delta"]
            nxt = scan[i + 1]["delta"]
            if not np.isfinite(cur) or cur <= 0:
                return False
            return abs(nxt - cur) / cur <= self.eps_rel

        lcc_deficit_thresh = 1.0 - self.lcc_thresh

        # 4) Выбор k*: минимальный k, где lcc ок и δ стабильно
        chosen_idx = None
        for i in range(len(scan)):
            ok_lcc = np.isfinite(scan[i]["lcc_deficit"]) and (scan[i]["lcc_deficit"] <= lcc_deficit_thresh)
            ok_stb = stable(i)
            if ok_lcc and ok_stb:
                chosen_idx = i
                break

        # Fallback #1: минимальный k, где только lcc ок
        if chosen_idx is None:
            for i in range(len(scan)):
                ok_lcc = np.isfinite(scan[i]["lcc_deficit"]) and (scan[i]["lcc_deficit"] <= lcc_deficit_thresh)
                if ok_lcc:
                    chosen_idx = i
                    break

        # Fallback #2: последний k
        if chosen_idx is None:
            chosen_idx = len(scan) - 1

        chosen = scan[chosen_idx]
        delta = chosen["delta"]
        lcc_deficit = chosen["lcc_deficit"]
        chosen_k = chosen["k"]

        # 5) Штрафованная δ (по желанию)
        if self.use_penalty and np.isfinite(delta) and np.isfinite(lcc_deficit):
            delta_pen = delta + lcc_deficit
        else:
            delta_pen = float(delta)

        return {
            "delta_hyperbolicity_norm": float(delta),
            "delta_hyperbolicity_penalized": float(delta_pen),
            "lcc_deficit": float(lcc_deficit),
            "chosen_k": int(chosen_k),
            "k_scan": scan,  # можно отключить, если не хочешь хранить длинный список
        }

