from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .config import Config
from .dataset import make_binary_balanced_df, RetinaBinaryDataset
from .models import LogisticRegressionBaseline, SimpleCNN, build_resnet18
from .utils import confusion_from_logits, accuracy_from_logits


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "cnn", "resnet18"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_path = cfg.data_root / cfg.labels_csv
    images_dir = cfg.data_root / cfg.images_dir
    df = make_binary_balanced_df(csv_path, images_dir, seed=cfg.seed)

    _, val_df = train_test_split(
        df, test_size=cfg.val_frac, random_state=cfg.seed, stratify=df["binary"]
    )

    if args.model == "logreg":
        ds = RetinaBinaryDataset(val_df, img_size=cfg.img_size, brighten=1.2, grayscale=False, return_flat=True)
        in_features = 3 * cfg.img_size * cfg.img_size
        model = LogisticRegressionBaseline(in_features=in_features, dropout_p=0.3)
    elif args.model == "cnn":
        ds = RetinaBinaryDataset(val_df, img_size=cfg.img_size, brighten=1.2, grayscale=False, return_flat=False)
        model = SimpleCNN(dropout_p=0.2)
    else:
        ds = RetinaBinaryDataset(val_df, img_size=cfg.img_size, brighten=1.2, grayscale=False, return_flat=False)
        model = build_resnet18(num_classes=2, pretrained=False)

    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=cfg.num_workers)
    model = model.to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).cpu()
        all_logits.append(logits)
        all_y.append(y)

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)

    acc = accuracy_from_logits(all_logits, all_y)
    cm = confusion_from_logits(all_logits, all_y)
    print("Accuracy:", acc)
    print("Confusion [[TN,FP],[FN,TP]]:\n", cm)


if __name__ == "__main__":
    main()
