"""
Training utilities and CLI for `SteerMLP`.

Usage example:
    python mlp_training.py \
        --dataset path/to/dataset.pt \
        --epochs 100 --lr 5e-4 --batch-size 64 \
        --save-path mlp_state_dict.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mlp_nonlin import SteerMLP
from dataset import SteeringDataset, create_train_val_test_loaders, print_split_info


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    lr: float = 1e-3,
    epochs: int = 50,
    device: Union[str, torch.device] = "cpu",
    print_every: int = 10,
):
    """Standard supervised‑regression training loop."""
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X = X.to(device, torch.float32)
            y = y.to(device, torch.float32)
            optimiser.zero_grad()
            preds = model(X)
            loss = F.mse_loss(preds, y)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(epoch_train_loss)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for X, y in val_loader:
                    X = X.to(device, torch.float32)
                    y = y.to(device, torch.float32)
                    preds = model(X)
                    val_loss += F.mse_loss(preds, y).item()
            epoch_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(epoch_val_loss)
        else:
            history["val_loss"].append(float("nan"))

        if epoch % print_every == 0 or epoch in {1, epochs}:
            print(
                f"Epoch {epoch:>3}/{epochs} – "
                f"train_loss: {epoch_train_loss:.6f} "
                f"val_loss: {history['val_loss'][-1]:.6f}"
            )

    return history


def evaluate_model(model: nn.Module, dataloader: DataLoader, *, device: str | torch.device = "cpu"):
    """Return MSE, cosine similarity and L1‑norm aggregated across the loader."""
    model.eval()
    metrics = {"mse": 0.0, "cosine_sim": 0.0, "l1_norm": 0.0}

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device, torch.float32)
            Y_batch = Y_batch.to(device, torch.float32)
            pred = model(X_batch)

            metrics["mse"] += F.mse_loss(pred, Y_batch).item()
            metrics["cosine_sim"] += F.cosine_similarity(pred, Y_batch, dim=1).mean().item()
            metrics["l1_norm"] += F.l1_loss(pred, Y_batch).item()

    for key in metrics:
        metrics[key] /= len(dataloader)
    return metrics



def make_training(data_activations):

    train_loader, val_loader = create_train_val_test_loaders(
        data=data_activations,
        layer_indices=[14],
        batch_size=64,
        target_type='dif',
        split_ratios=(0.8, 0.2),
        split_strategy='by_example',
        random_seed=42
    )

    print_split_info(train_loader, val_loader)

    sample_X, _ = next(iter(train_loader))
    num_features = sample_X.shape[-1]

    model = SteerMLP(num_features)

    train_model(
        model,
        train_loader,
        val_loader,
        lr=1e-3,
        epochs=10,
        device='cpu',
    )
    save_path="mlp.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Finished training – weights stored at {save_path}")

    print(evaluate_model(model, val_loader))
