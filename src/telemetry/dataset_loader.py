"""
ElevatorGuard-AI: Telemetry Dataset Loader
Constructs sliding window sequences for BiLSTM RUL training and evaluation.
"""

from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TelemetrySequenceDataset(Dataset):
    """
    Sliding window dataset for sequential telemetry features mapped to RUL / Health Index.
    """

    def __init__(
        self,
        features: np.ndarray,
        health_indices: np.ndarray,
        seq_len: int = 16,
        step: int = 1
    ):
        """
        Args:
            features: [total_steps, num_features]
            health_indices: [total_steps]
            seq_len: sequence length for LSTM
            step: stride between consecutive sliding windows
        """
        self.seq_len = seq_len
        self.X, self.y = self._create_windows(features, health_indices, seq_len, step)

    def _create_windows(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int,
        step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_list = []
        y_list = []
        n_samples = len(features)

        if n_samples < seq_len:
            # Pad if too short
            pad_len = seq_len - n_samples
            features = np.pad(features, ((pad_len, 0), (0, 0)), mode="edge")
            targets = np.pad(targets, (pad_len, 0), mode="edge")
            n_samples = len(features)

        for i in range(0, n_samples - seq_len + 1, step):
            window_x = features[i: i + seq_len]
            target_y = targets[i + seq_len - 1]
            X_list.append(window_x)
            y_list.append(target_y)

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32).reshape(-1, 1)

        return torch.from_numpy(X_arr), torch.from_numpy(y_arr)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_telemetry_dataloaders(
    features: np.ndarray,
    health_indices: np.ndarray,
    seq_len: int = 16,
    batch_size: int = 32,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """Splits into train and validation DataLoaders."""
    dataset = TelemetrySequenceDataset(features, health_indices, seq_len=seq_len)
    total_len = len(dataset)
    train_len = int(total_len * train_split)
    val_len = total_len - train_len

    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
