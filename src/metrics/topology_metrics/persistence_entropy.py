from typing import Any, Callable

import numpy as np
from ripser import ripser

from src.core.metric_base import Metric
from src.core.bootstrap_metrics import BootMetric
from src.utils.time_decorator import timecount


class PersistentEntropy(Metric):
    """
    Персистентная энтропия (H₀) с произвольным семплированием landmark-точек.

    Параметры
    ----------
    n_landmarks      : int
        Сколько точек брать на один запуск метрики.
    rng_seed         : int | None
        Если указан — создаётся локальный np.random.Generator
        и семплирование делается им (не через SharedCache).
    sample_idx_func  : Callable[[int, int], np.ndarray] | None
        Пользовательская функция выбора индексов. Если передана,
        она имеет приоритет над rng_seed и SharedCache.
    """

    def __init__(
        self,
        ds: Any,
        cache: Any,
        layer: str,
        *,
        k: int = 20,
        norm: str = "z",
        n_landmarks: int = 500,
        rng_seed: int | None = None,
        sample_idx_func: Callable[[int, int], np.ndarray] | None = None,
    ) -> None:
        super().__init__(ds, cache, layer, k=k, norm=norm)

        self.n_landmarks = min(n_landmarks, self.X.shape[0])
        self.rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
        self.sample_idx_func = sample_idx_func

    def _sample_indices(self) -> np.ndarray:
        """Возвращает индексы landmark-точек согласно приоритету."""
        if self.sample_idx_func is not None:                   
            return self.sample_idx_func(self.X.shape[0], self.n_landmarks)

        if self.rng is not None:                               
            return self.rng.choice(self.X.shape[0],
                                    self.n_landmarks, replace=False)

        return self.cache.get_sample_idx(self.X.shape[0], self.n_landmarks)


    @timecount
    def compute(self) -> float:  # type: ignore[override]
        idx = self._sample_indices()
        X_sub = self.X[idx]

        dgms = ripser(X_sub, maxdim=0)["dgms"][0]
        finite = dgms[np.isfinite(dgms[:, 1])]
        if finite.size == 0:
            return 0.0

        lifetimes = finite[:, 1] - finite[:, 0]
        p = lifetimes / lifetimes.sum()
        return float(-(p * np.log(p)).sum())


class BootPersistentEntropy(BootMetric):
    def __init__(
        self,
        ds: Any,
        cache: Any,
        layer: str,
        *,
        n_runs: int = 7,
        n_landmarks: int = 800,
        k: int = 20,
        norm: str = "z",
        seed: int | None = 123,
        **kw: Any,
    ):
        super().__init__(
            ds,
            cache,
            layer,
            base_cls=PersistentEntropy,
            n_runs=n_runs,
            seed=seed,
            n_landmarks=n_landmarks,
            k=k,
            norm=norm,
            **kw,
        )