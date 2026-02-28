from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_root: Path = project_root / "data" / "raw"
    outputs_root: Path = project_root / "outputs"
    ckpt_dir: Path = outputs_root / "checkpoints"
    plot_dir: Path = outputs_root / "plots"

    # Dataset
    labels_csv: str = "trainLabels.csv"
    images_dir: str = "train"
    img_size: int = 512

    # Split
    seed: int = 42
    val_frac: float = 0.2

    # Training
    batch_size: int = 16
    num_workers: int = 2
    device: str = "cuda"
