from typing import Any, Callable, Optional

import numpy as np
from ripser import ripser

from src.core.metric_base import Metric
from src.utils.time_decorator import timecount


def _persistent_entropy_from_lifetimes(lifetimes: np.ndarray, eps: float = 1e-12) -> float:
    """
    Персистентная энтропия по векторам длин интервалов.
    lifetimes: (m,) > 0
    """
    lifetimes = lifetimes[(lifetimes > 0) & np.isfinite(lifetimes)]
    if lifetimes.size == 0:
        return 0.0
    p = lifetimes / (lifetimes.sum() + eps)
    return float(-(p * np.log(p + eps)).sum())


class PersistentEntropy(Metric):
    """
    Персистентная энтропия **только H₁** на сабсэмпле landmark-точек.
    Возвращает скаляр pe_h1.
    """

    def __init__(
        self,
        ds: Any,
        cache: Any,
        layer: str,
        *,
        k: int = 20,
        norm: str = "z",
        n_landmarks: int = 400,
        rng_seed: Optional[int] = None,
        sample_idx_func: Optional[Callable[[int, int], np.ndarray]] = None,
    ) -> None:
        super().__init__(ds, cache, layer, k=k, norm=norm)
        self.n_landmarks = int(min(n_landmarks, self.X.shape[0]))
        self.rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
        self.sample_idx_func = sample_idx_func

    def _sample_indices(self) -> np.ndarray:
        N = self.X.shape[0]
        if self.sample_idx_func is not None:
            return np.asarray(self.sample_idx_func(N, self.n_landmarks), dtype=int)
        if self.rng is not None:
            return self.rng.choice(N, self.n_landmarks, replace=False)
        return self.cache.get_sample_idx(N, self.n_landmarks)

    @timecount
    def compute(self) -> float:  # возвращаем именно скаляр pe_h1
        idx = self._sample_indices()
        X_sub = self.X[idx]

        # Ripser на координатах (быстрее, чем distance_matrix), до H1
        dgms = ripser(X_sub, maxdim=1).get("dgms", [])
        H1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

        if H1.size == 0:
            return 0.0

        finite = H1[np.isfinite(H1[:, 1])]
        if finite.size == 0:
            return 0.0

        lifetimes = finite[:, 1] - finite[:, 0]
        return _persistent_entropy_from_lifetimes(lifetimes)


class BootPersistentEntropy(Metric):
    """
    Бутстрап-обёртка над PersistentEntropy (H₁).
    Делает n_runs прогонов с разными seed и возвращает:
      - pe_h1: среднее по прогону,
      - pe_n_runs: число прогонов,
      - pe_n_landmarks: число landmark-точек.
    Никаких лишних полей.
    """

    def __init__(
        self,
        ds: Any,
        cache: Any,
        layer: str,
        *,
        n_runs: int = 7,
        n_landmarks: int = 400,
        k: int = 20,
        norm: str = "z",
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__(ds, cache, layer, k=k, norm=norm)
        self.n_runs = int(n_runs)
        self.n_landmarks = int(min(n_landmarks, self.X.shape[0]))
        self.seed = seed

    @timecount
    def compute(self) -> dict:
        rng = np.random.default_rng(self.seed)
        vals: list[float] = []

        for _ in range(self.n_runs):
            # Новый сид на каждый прогон, чтобы менять сабсэмпл
            run_seed = int(rng.integers(0, 10**9))
            m = PersistentEntropy(
                self.ds, self.cache, self.layer,
                k=self.k, norm="z",
                n_landmarks=self.n_landmarks,
                rng_seed=run_seed,
            )
            vals.append(m.compute())

        pe_h1_mean = float(np.mean(vals)) if len(vals) else 0.0

        return {
            "pe_h1": pe_h1_mean,
            "pe_n_runs": float(self.n_runs),
            "pe_n_landmarks": float(self.n_landmarks),
        }
