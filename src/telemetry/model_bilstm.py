"""
ElevatorGuard-AI: BiLSTM with Self-Attention for Dynamic Health Index (HI_dyn) & RUL Regression
Implements temporal sequence modeling over accelerometer and rope-tension telemetry windows.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau-style additive self-attention over BiLSTM hidden states.
    Allows the model to focus on impulsive vibration shocks and tension spikes.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 32):
        super().__init__()
        self.w_omega = nn.Linear(hidden_dim, attention_dim)
        self.u_omega = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
        Returns:
            context: [batch_size, hidden_dim]
            weights: [batch_size, seq_len]
        """
        # score = tanh(W * h) -> [batch_size, seq_len, attention_dim]
        score = torch.tanh(self.w_omega(hidden_states))
        # attention_logits -> [batch_size, seq_len, 1]
        attention_logits = self.u_omega(score)
        # attention_weights -> [batch_size, seq_len]
        attention_weights = F.softmax(attention_logits.squeeze(-1), dim=-1)
        # weighted context -> [batch_size, hidden_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attention_weights


class BiLSTMAttentionRUL(nn.Module):
    """
    Bidirectional LSTM with Attention for Dynamic Health Index (HI_dyn) Regression.
    Maps input sequences [batch_size, seq_len, input_dim] to continuous HI in [0.0, 1.0].
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        attention_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Normalization layer for raw feature inputs
        self.input_norm = nn.LayerNorm(input_dim)

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        bilstm_out_dim = hidden_dim * 2

        # Attention Layer
        self.attention = BahdanauAttention(hidden_dim=bilstm_out_dim, attention_dim=attention_dim)

        # Regressor Head
        self.fc_layers = nn.Sequential(
            nn.Linear(bilstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid() # Restricts HI_dyn to [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            hi_dyn: [batch_size, 1]
            attention_weights: [batch_size, seq_len]
        """
        x_norm = self.input_norm(x)
        lstm_out, _ = self.bilstm(x_norm)
        context, attention_weights = self.attention(lstm_out)
        hi_dyn = self.fc_layers(context)
        return hi_dyn, attention_weights

    def predict_single_sequence(self, feature_seq: torch.Tensor) -> Tuple[float, list]:
        """
        Convenience inference method for a single sequence [seq_len, input_dim].
        """
        self.eval()
        with torch.no_grad():
            if feature_seq.ndim == 2:
                feature_seq = feature_seq.unsqueeze(0) # [1, seq_len, input_dim]
            hi, att = self.forward(feature_seq)
            return float(hi[0, 0].item()), att[0].tolist()
