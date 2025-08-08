import numpy as np
import matplotlib.pyplot as plt

def plot_topology_radar(
    results_dict: dict,
    *,
    include_lcc_deficit: bool = False,
    figsize=(10, 10),
    title="Topological Linearity — Radar (higher is better on all axes)",
    save_path: str | None = None,
):
    """
    results_dict: словарь вида
      {
        layer_id: {
          'persistent_entropy': float,
          'EulerCharacteristicCurve_linear_index': float,
          'TotalPersistenceH1': {'tp1_norm': float, ...},
          'DeltaHyperbolicity': {'delta_hyperbolicity_norm': float, 'lcc_deficit': float, ...},
          'H1SystoleLen': {'h1_systole_len_norm': float},
          'LongH2Cycles': {'h2_long_frac': float, ...},
          ...
        },
        ...
      }

    include_lcc_deficit: добавить ли ещё одну ось (↓ лучше)
    """

    # --- какие метрики рисуем и куда они вложены ---
    # spec: (label, extractor, higher_is_better)
    base_specs = [
        ("persistent_entropy", lambda d: d.get("persistent_entropy", np.nan), False),  # ↓ лучше
        ("EulerCharacteristicCurve_linear_index", lambda d: d.get("EulerCharacteristicCurve_linear_index", np.nan), True),  # ↑ лучше
        ("tp1_norm", lambda d: d.get("TotalPersistenceH1", {}).get("tp1_norm", np.nan), False),  # ↓ лучше
        ("delta_hyperbolicity_norm", lambda d: d.get("DeltaHyperbolicity", {}).get("delta_hyperbolicity_norm", np.nan), False),  # ↓ лучше
        ("h1_systole_len_norm", lambda d: d.get("H1SystoleLen", {}).get("h1_systole_len_norm", np.nan), False),  # ↓ лучше
        ("h2_long_frac", lambda d: d.get("LongH2Cycles", {}).get("h2_long_frac", np.nan), False),  # ↓ лучше
    ]
    if include_lcc_deficit:
        base_specs.append(("lcc_deficit", lambda d: d.get("DeltaHyperbolicity", {}).get("lcc_deficit", np.nan), False))  # ↓ лучше

    layers = sorted(results_dict.keys())
    specs = base_specs[:]  # копия

    # --- извлекаем значения: матрица [num_layers, num_metrics] ---
    values = []
    for L in layers:
        ld = results_dict[L]
        row = [extr(ld) for (_, extr, _) in specs]
        values.append(row)
    values = np.array(values, dtype=float)  # shape [L, M]

    # выкидываем метрики, которые полностью NaN по всем слоям
    keep_cols = ~np.all(~np.isfinite(values), axis=0)
    if not np.any(keep_cols):
        raise ValueError("Все выбранные метрики оказались NaN по всем слоям.")
    values = values[:, keep_cols]
    specs = [spec for (spec, keep) in zip(specs, keep_cols) if keep]

    # --- min-max нормализация по слоям для каждой метрики ---
    mins = np.nanmin(values, axis=0)
    maxs = np.nanmax(values, axis=0)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    norm = (values - mins) / denom

    # --- переворачиваем оси, где "меньше — лучше", чтобы "выше = лучше" на графике ---
    higher_is_better = np.array([hib for (_, _, hib) in specs], dtype=bool)
    flip_mask = ~higher_is_better  # где меньше — лучше
    norm[:, flip_mask] = 1.0 - norm[:, flip_mask]

    # --- подписи осей с стрелочками направления ---
    labels = []
    for (name, _, hib) in specs:
        arrow = "↑" if hib else "↓"
        labels.append(f"{name} ({arrow})")

    # --- радарная геометрия ---
    M = norm.shape[1]
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False).tolist()
    angles += angles[:1]  # замкнуть

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    for i, L in enumerate(layers):
        vals = norm[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=f"Layer {L}")
        ax.fill(angles, vals, alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
