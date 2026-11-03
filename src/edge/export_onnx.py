"""
ElevatorGuard-AI: Edge Model Export Utility
Exports PyTorch BiLSTM-Attention model to ONNX and TorchScript for edge NPUs/micro-servers.
"""

import os
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.model_bilstm import BiLSTMAttentionRUL


def export_bilstm_to_onnx(
    model_checkpoint: str = "models/bilstm_rul.pt",
    onnx_output_path: str = "models/bilstm_rul.onnx",
    seq_len: int = 16,
    input_dim: int = 9
) -> str:
    """
    Exports trained BiLSTM model to ONNX format.
    """
    device = torch.device("cpu")
    model = BiLSTMAttentionRUL(input_dim=input_dim, hidden_dim=64, num_layers=2)

    if os.path.exists(model_checkpoint):
        state_dict = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[ONNX Export] Loaded checkpoint from {model_checkpoint}")
    else:
        print(f"[ONNX Export] Checkpoint {model_checkpoint} not found. Exporting initialized architecture.")

    model.eval()

    # Create dummy input tensor [batch_size=1, seq_len=16, input_dim=9]
    dummy_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    Path(onnx_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Export TorchScript for high-speed edge inference
    try:
        ts_output_path = onnx_output_path.replace(".onnx", ".torchscript.pt")
        traced_cell = torch.jit.trace(model, dummy_input)
        traced_cell.save(ts_output_path)
        print(f"[TorchScript Export] Successfully exported TorchScript to: {ts_output_path}")
    except Exception as e:
        print(f"[TorchScript Export] Warning: TorchScript export skipped: {e}")

    # Export ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["telemetry_sequence"],
            output_names=["dynamic_health_index", "attention_weights"],
            dynamic_axes={
                "telemetry_sequence": {0: "batch_size"},
                "dynamic_health_index": {0: "batch_size"},
                "attention_weights": {0: "batch_size"}
            }
        )
        print(f"[ONNX Export] Successfully exported ONNX model to: {onnx_output_path}")
    except Exception as e:
        print(f"[ONNX Export] Warning: ONNX export skipped: {e}")

    return onnx_output_path


if __name__ == "__main__":
    export_bilstm_to_onnx()
