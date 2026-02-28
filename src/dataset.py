from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import torch
from torch.utils.data import Dataset


def make_binary_balanced_df(
    csv_path: Path,
    images_dir: Path,
    seed: int = 42
) -> pd.DataFrame:
    """
    Reads Kaggle DR labels CSV and returns a balanced binary dataframe:
    - level == 0 -> y=0
    - level in {1,2,3,4} -> y=1
    - Downsample y=0 to match y=1 count
    """
    df = pd.read_csv(csv_path)

    if "level" not in df.columns:
        raise ValueError(f"Expected 'level' column in {csv_path}, found {df.columns.tolist()}")

    img_col = "image" if "image" in df.columns else df.columns[0]

    df["binary"] = (df["level"] > 0).astype(int)

    def resolve_path(stem: str) -> Optional[str]:
        for ext in [".jpeg", ".jpg", ".png"]:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                return str(p)
        return None

    df["path"] = df[img_col].apply(resolve_path)
    df = df.dropna(subset=["path"]).reset_index(drop=True)

    pos = df[df["binary"] == 1]
    neg = df[df["binary"] == 0]

    neg_sample = neg.sample(n=len(pos), random_state=seed) if len(neg) > len(pos) else neg
    balanced = pd.concat([pos, neg_sample], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def preprocess_image(
    img: Image.Image,
    out_size: int = 512,
    brighten: float = 1.2,
    grayscale: bool = False
) -> Image.Image:
    img = center_crop_to_square(img)
    img = img.resize((out_size, out_size))

    if grayscale:
        img = img.convert("L").convert("RGB")

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brighten)
    return img


class RetinaBinaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int = 512,
        brighten: float = 1.2,
        grayscale: bool = False,
        return_flat: bool = False,
        normalize: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.brighten = brighten
        self.grayscale = grayscale
        self.return_flat = return_flat
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]
        y = int(row["binary"])

        img = Image.open(path).convert("RGB")
        img = preprocess_image(img, out_size=self.img_size, brighten=self.brighten, grayscale=self.grayscale)

        x = torch.from_numpy(np.array(img)).float()
        x = x.permute(2, 0, 1)

        if self.normalize:
            x = x / 255.0

        y_oh = torch.tensor([1.0, 0.0]) if y == 0 else torch.tensor([0.0, 1.0])

        if self.return_flat:
            x = x.reshape(-1)

        return x, y_oh
