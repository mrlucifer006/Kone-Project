"""
ElevatorGuard-AI: YOLO Fine-Tuning Pipeline
Trains or fine-tunes YOLOv8-Seg / YOLOv8-Nano on wire rope and guide rail defect datasets.
"""

import os
import yaml
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def create_dataset_yaml(data_dir: str = "data/annotated_yolo", output_path: str = "data/cable_defects.yaml"):
    """Generate YOLO data configuration yaml for wire rope cable inspection."""
    base_path = Path(data_dir).resolve()
    data_config = {
        "path": str(base_path),
        "train": "train/images",
        "val": "val/images",
        "names": {
            0: "broken_wire",
            1: "abrasion_wear",
            2: "corrosion_rust"
        }
    }
    with open(output_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print(f"[YOLO Dataset] Generated cable dataset config at: {output_path}")
    return output_path


def create_rail_dataset_yaml(data_dir: str = "data/annotated_rail_yolo", output_path: str = "data/rail_defects.yaml"):
    """Generate YOLO data configuration yaml for ISO 7465 guide rail inspection."""
    base_path = Path(data_dir).resolve()
    data_config = {
        "path": str(base_path),
        "train": "train/images",
        "val": "val/images",
        "names": {
            0: "rail_surface_wear",
            1: "joint_misalignment",
            2: "pitting_galling",
            3: "rail_corrosion"
        }
    }
    with open(output_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print(f"[YOLO Dataset] Generated rail dataset config at: {output_path}")
    return output_path


def train_yolo(
    data_yaml: str = "data/cable_defects.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 16,
    project_dir: str = "models/yolo_runs",
    name: str = "cable_defect_detector"
):
    """
    Train or fine-tune YOLO model for wire ropes.
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed.")

    print(f"[Train YOLO] Starting cable training with base={base_model}, epochs={epochs}, batch={batch}")
    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_dir,
        name=name,
        verbose=True
    )
    
    # Save best weights to models/yolo_cable_detector.pt
    best_weights = Path(project_dir) / name / "weights" / "best.pt"
    target_weights = Path("models/yolo_cable_detector.pt")
    if best_weights.exists():
        target_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, target_weights)
        print(f"[Train YOLO] Successfully saved best weights to {target_weights}")

    return results


def train_rail_detector(
    data_yaml: str = "data/rail_defects.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 16,
    project_dir: str = "models/yolo_rail_runs",
    name: str = "rail_defect_detector"
):
    """
    Train or fine-tune YOLO model for ISO 7465 elevator guide rails.
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed.")

    print(f"[Train Rail YOLO] Starting rail detector training with base={base_model}, epochs={epochs}, batch={batch}")
    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_dir,
        name=name,
        verbose=True
    )
    
    # Save best weights to models/yolo_rail_detector.pt
    best_weights = Path(project_dir) / name / "weights" / "best.pt"
    target_weights = Path("models/yolo_rail_detector.pt")
    if best_weights.exists():
        target_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, target_weights)
        print(f"[Train Rail YOLO] Successfully saved best weights to {target_weights}")

    return results


if __name__ == "__main__":
    # Check if rail or cable dataset is present
    rail_train_images = list(Path("data/annotated_rail_yolo/train/images").glob("*.jpg"))
    if rail_train_images:
        rail_yaml = create_rail_dataset_yaml()
        train_rail_detector(data_yaml=rail_yaml, epochs=10)
    else:
        cable_train_images = list(Path("data/annotated_yolo/train/images").glob("*.jpg"))
        if cable_train_images:
            cable_yaml = create_dataset_yaml()
            train_yolo(data_yaml=cable_yaml, epochs=10)
        else:
            print("[Train Detector] No training data found in data/annotated_rail_yolo or data/annotated_yolo.")
