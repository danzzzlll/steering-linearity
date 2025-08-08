from typing import Any, Type, Dict, List

import numpy as np
from src.core.metric_base import Metric


class BootMetric:
    """
    Повторяет вычисление любой scalar-метрики `base_cls` n_runs раз
    и возвращает среднее значение; стандартная ошибка доступна как `.sem`.
    """

    def __init__(
        self,
        ds: Any,
        cache: Any,
        layer: str,
        *,
        base_cls: Type[Metric],
        n_runs: int = 7,
        seed: int | None = 123,
        **base_kw: Any,
    ) -> None:
        self.ds, self.cache, self.layer = ds, cache, layer
        self.base_cls = base_cls
        self.n_runs = n_runs
        self.seed0 = 0 if seed is None else seed
        self.base_kw: Dict[str, Any] = base_kw

        self.values: List[float] = []
        self.sem: float | None = None  # стандартная ошибка (σ/√n)

    def compute(self) -> float:
        self.values.clear()

        for i in range(self.n_runs):
            # передаём уникальный сид внутрь метрики
            kw = dict(self.base_kw, rng_seed=self.seed0 + i)
            metric = self.base_cls(self.ds, self.cache, self.layer, **kw)
            self.values.append(metric.compute())

        arr = np.asarray(self.values, dtype=float)
        self.sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return float(arr.mean())



