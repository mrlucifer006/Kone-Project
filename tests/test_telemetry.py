"""
Unit Tests for Telemetry Feature Extraction & BiLSTM-Attention Model
"""

import numpy as np
import torch
import pytest
from src.telemetry.feature_extraction import TelemetryFeatureExtractor
from src.telemetry.model_bilstm import BiLSTMAttentionRUL


def test_telemetry_feature_extractor():
    extractor = TelemetryFeatureExtractor(sampling_rate_hz=100)
    
    # Balanced signals
    ax = np.random.normal(0, 0.05, 128)
    ay = np.random.normal(0, 0.05, 128)
    az = np.random.normal(9.81, 0.1, 128)
    tensions = np.array([2500.0, 2505.0, 2495.0, 2500.0])

    features = extractor.extract_features(ax, ay, az, tensions)

    assert "acc_rms_z" in features
    assert features["acc_rms_z"] > 9.0
    assert "tension_imbalance_ratio" in features
    assert features["tension_imbalance_ratio"] < 0.05
    assert not features["tension_imbalance_alarm"]


def test_tension_imbalance_alarm():
    extractor = TelemetryFeatureExtractor(sampling_rate_hz=100, tension_imbalance_limit=0.15)
    
    # Heavily unbalanced tensions (Rope 1 has 3200 N, Rope 4 has 1800 N)
    ax = np.zeros(32)
    ay = np.zeros(32)
    az = np.ones(32) * 9.81
    unbalanced_tensions = np.array([3200.0, 2600.0, 2400.0, 1800.0])

    features = extractor.extract_features(ax, ay, az, unbalanced_tensions)
    assert features["tension_imbalance_ratio"] > 0.15
    assert features["tension_imbalance_alarm"] is True


def test_bilstm_attention_forward():
    model = BiLSTMAttentionRUL(input_dim=9, hidden_dim=32, num_layers=2)
    model.eval()

    # Input batch: [batch_size=4, seq_len=16, input_dim=9]
    dummy_input = torch.randn(4, 16, 9)
    with torch.no_grad():
        hi_pred, att_weights = model(dummy_input)

    assert hi_pred.shape == (4, 1)
    # Check that predictions are in [0, 1] due to Sigmoid
    assert torch.all(hi_pred >= 0.0) and torch.all(hi_pred <= 1.0)
    # Check attention weights sum to 1.0
    assert att_weights.shape == (4, 16)
    assert torch.allclose(torch.sum(att_weights, dim=-1), torch.ones(4), atol=1e-4)
