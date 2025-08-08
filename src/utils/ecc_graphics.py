import numpy as np
import matplotlib.pyplot as plt

def ecc_interpretable_stats(alpha: np.ndarray, chi: np.ndarray, V: int) -> dict:
    """
    Считает интерпретируемые показатели (без отрицательных значений).
    Возвращает dict с:
      - ecc_deficit_auc (↓ лучше)
      - ecc_surplus_auc (диагностика)
      - ecc_centered_auc = ∫(χ/V - 1) dα
      - ecc_linearity_index ∈ (0,1] (↑ лучше)
    """
    V_safe = max(1, int(V))
    chi_norm = chi / V_safe

    deficit = np.maximum(0.0, 1.0 - chi_norm)
    surplus = np.maximum(0.0, chi_norm - 1.0)

    ecc_deficit_auc = float(np.trapz(deficit, alpha))
    ecc_surplus_auc = float(np.trapz(surplus, alpha))
    ecc_centered_auc = float(np.trapz(chi_norm - 1.0, alpha))
    ecc_linearity_index = float(1.0 / (1.0 + ecc_deficit_auc))

    return {
        "ecc_deficit_auc": ecc_deficit_auc,
        "ecc_surplus_auc": ecc_surplus_auc,
        "ecc_centered_auc": ecc_centered_auc,
        "ecc_linearity_index": ecc_linearity_index,
    }


def plot_ecc(alpha: np.ndarray,
             chi: np.ndarray,
             V: int,
             *,
             label: str | None = None,
             fill_deficit: bool = True,
             ax: plt.Axes | None = None) -> dict:
    """
    Рисует χ(α) и (опционально) заливает 'дефицит' ниже уровня χ/V = 1.
    Возвращает тот же dict, что ecc_interpretable_stats(...).

    Пример:
        stats = plot_ecc(alpha, chi, V, label='layer 0')
        plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots()

    stats = ecc_interpretable_stats(alpha, chi, V)
    V_safe = max(1, int(V))
    chi_norm = chi / V_safe

    # линия χ/V
    ax.plot(alpha, chi_norm, label=label if label else "χ(α)/V")
    # базовый уровень 1
    ax.plot(alpha, np.ones_like(alpha), linestyle="--", linewidth=1)

    if fill_deficit:
        # закрашиваем область ниже 1 (дефицит)
        ax.fill_between(alpha, chi_norm, 1.0, where=(chi_norm < 1.0), alpha=0.2)

    ax.set_xlabel("α (доля подключённых рёбер)")
    ax.set_ylabel("χ(α) / V")
    if label:
        ax.legend()

    # компактный сабтайтл со сводкой
    subtitle = (f"deficit_auc={stats['ecc_deficit_auc']:.3f} "
                f"| LI={stats['ecc_linearity_index']:.3f} "
                f"| centered_auc={stats['ecc_centered_auc']:.3f}")
    ax.set_title(subtitle)

    return stats


def plot_ecc_compare(curves: dict[str, tuple[np.ndarray, np.ndarray, int]],
                     *,
                     fill_deficit: bool = False,
                     share_stats: bool = True) -> dict[str, dict]:
    """
    Сравнение нескольких слоёв на одном графике.
    curves: {name: (alpha, chi, V)}
    fill_deficit=False по умолчанию, чтобы не мешался паттерн заливки.

    Возвращает словарь {name: stats_dict}.
    """
    fig, ax = plt.subplots()
    all_stats = {}
    for name, (alpha, chi, V) in curves.items():
        stats = ecc_interpretable_stats(alpha, chi, V)
        all_stats[name] = stats
        V_safe = max(1, int(V))
        chi_norm = chi / V_safe
        ax.plot(alpha, chi_norm, label=name)
        if fill_deficit:
            ax.fill_between(alpha, chi_norm, 1.0, where=(chi_norm < 1.0), alpha=0.15)

    ax.plot(alpha, np.ones_like(alpha), linestyle="--", linewidth=1)
    ax.set_xlabel("α (доля подключённых рёбер)")
    ax.set_ylabel("χ(α) / V")
    ax.legend()

    if share_stats and len(all_stats) > 0:
        # выведем сводку в заголовок
        parts = [f"{k}: LI={v['ecc_linearity_index']:.3f}, def={v['ecc_deficit_auc']:.3f}"
                 for k, v in all_stats.items()]
        ax.set_title(" | ".join(parts))

    return all_stats
