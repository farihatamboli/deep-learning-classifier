from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def confusion_from_logits(logits: torch.Tensor, y_onehot: torch.Tensor) -> np.ndarray:
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    true = torch.argmax(y_onehot, dim=1).cpu().numpy()
    tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tp = int(((pred == 1) & (true == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def accuracy_from_logits(logits: torch.Tensor, y_onehot: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    true = torch.argmax(y_onehot, dim=1)
    return (pred == true).float().mean().item()


def plot_curves(history: Dict[str, list], out_path: Path, title: str) -> None:
    plt.figure()
    for k, v in history.items():
        plt.plot(range(1, len(v) + 1), v, label=k)
    plt.xlabel("Epoch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
