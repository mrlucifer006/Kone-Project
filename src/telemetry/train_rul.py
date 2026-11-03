"""
ElevatorGuard-AI: BiLSTM RUL Model Training Pipeline
Trains the temporal prognostic model on cyclic degradation time series.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.telemetry.model_bilstm import BiLSTMAttentionRUL
from src.telemetry.dataset_loader import get_telemetry_dataloaders


class MonotonicDegradationLoss(nn.Module):
    """
    Combined MSE loss and Monotonicity penalty.
    Physical prior: Elevator cables monotonically degrade over operational cycles;
    unprompted health recovery is non-physical.
    """

    def __init__(self, lambda_mono: float = 0.15):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_mono = lambda_mono

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        
        # Penalize non-negative slopes in degradation where target is degrading
        if pred.size(0) > 1:
            diffs = pred[1:] - pred[:-1]
            mono_penalty = torch.mean(F_relu := torch.relu(diffs))
        else:
            mono_penalty = torch.tensor(0.0, device=pred.device)

        return mse_loss + (self.lambda_mono * mono_penalty)


def train_bilstm_rul(
    features: np.ndarray,
    health_indices: np.ndarray,
    model_save_path: str = "models/bilstm_rul.pt",
    seq_len: int = 16,
    input_dim: int = 9,
    hidden_dim: int = 64,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.001
) -> Tuple[BiLSTMAttentionRUL, Dict[str, list]]:
    """
    Train BiLSTM-Attention model on sequence data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train BiLSTM] Training on device: {device} (Epochs: {epochs}, Batch: {batch_size})")

    train_loader, val_loader = get_telemetry_dataloaders(
        features=features,
        health_indices=health_indices,
        seq_len=seq_len,
        batch_size=batch_size
    )

    model = BiLSTMAttentionRUL(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        attention_dim=32,
        dropout=0.2
    ).to(device)

    criterion = MonotonicDegradationLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds, _ = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds, _ = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)

    print(f"[Train BiLSTM] Best model saved to {model_save_path} (Val Loss: {best_val_loss:.5f})")
    return model, history
