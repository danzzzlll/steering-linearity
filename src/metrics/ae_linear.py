from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.core.metric_base import Metric


class _LinearAE(nn.Module):
    def __init__(self, d_in: int, d_code: int):
        super().__init__()
        self.enc = nn.Linear(d_in, d_code, bias=True)
        self.dec = nn.Linear(d_code, d_in, bias=True)

    def forward(self, x):
        return self.dec(self.enc(x))


class _NonLinearAE(nn.Module):
    def __init__(self, d_in: int, d_code: int, h: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Linear(h, d_code),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_code, h),
            nn.ReLU(),
            nn.Linear(h, d_in),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class AutoencoderLinearity(Metric):
    """
    Сравнивает способность линейного и ReLU-автокодировщика
    восстанавливать слой.  MSE_nonlin / MSE_lin ≈ 1 → слой «почти линейный».

    Параметры
    ---------
    d_code        : размерность скрытого кода (after PCA analogy)
    hidden_dim    : ширина скрытого слоя у нелинейного AE
    epochs        : число эпох обеих тренировок
    batch_size    : размер батча на обучении
    lr            : learning rate Adam
    test_split    : доля теста
    device        : 'cpu' | 'cuda'   (берётся torch.device(device))
    """

    def __init__(
        self,
        *a,
        d_code: int = 30,
        hidden_dim: int = 256,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        test_split: float = 0.2,
        seed: int = 0,
        device: str = "cpu",
        **kw,
    ):
        super().__init__(*a, **kw)
        self.d_code = d_code
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.test_split = test_split
        self.seed = seed
        self.device = torch.device(device)


    def _prepare_data(self):
        torch.manual_seed(self.seed)
        X = torch.tensor(self.X, dtype=torch.float32)  # (N, d)
        N = len(X)
        perm = torch.randperm(N)
        n_train = int((1 - self.test_split) * N)
        train_X = X[perm[:n_train]]
        test_X = X[perm[n_train:]]

        train_loader = DataLoader(
            TensorDataset(train_X), batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(test_X), batch_size=2 * self.batch_size, shuffle=False
        )
        return train_loader, test_loader, X.shape[1]


    @staticmethod
    def _train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device):
        loss_fn = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=lr)
        model.to(device)
        for _ in range(epochs):
            for (x,) in loader:
                x = x.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), x)
                loss.backward()
                opt.step()
        return model


    @staticmethod
    def _eval_mse(model: nn.Module, loader: DataLoader, device) -> float:
        model.eval()
        s, n = 0.0, 0
        with torch.no_grad():
            for (x,) in loader:
                x = x.to(device)
                s += ((model(x) - x) ** 2).sum().item()
                n += x.numel()
        return s / n


    def compute(self) -> Dict[str, float]:  # type: ignore[override]
        train_loader, test_loader, d_in = self._prepare_data()

        lin_ae = _LinearAE(d_in, self.d_code)
        lin_ae = self._train(lin_ae, train_loader, self.epochs, self.lr, self.device)
        mse_lin = self._eval_mse(lin_ae, test_loader, self.device)

        nonlin_ae = _NonLinearAE(d_in, self.d_code, self.hidden_dim)
        nonlin_ae = self._train(
            nonlin_ae, train_loader, self.epochs, self.lr, self.device
        )
        mse_nonlin = self._eval_mse(nonlin_ae, test_loader, self.device)

        return {
            "mse_linear": float(mse_lin),
            "mse_nonlin": float(mse_nonlin),
            "ratio": float(mse_nonlin / mse_lin),
        }
