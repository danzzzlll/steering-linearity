# metrics/pca_ncomp.py
from src.utils.pca_helpers import pca_fit
from src.core.metric_base import Metric

class PcaNComponents95(Metric):
    """Сколько компонент нужно, чтобы объяснить ≥95 % дисперсии."""
    def compute(self, n_comp = None, svd_solver: str = "randomized") -> int:                        
        pca = pca_fit(self.X, svd_solver=svd_solver, n=n_comp)                        
        pca_comp = pca.n_components_
        print(f'Metric PcaNComponents95: {pca_comp}')
        return pca_comp
