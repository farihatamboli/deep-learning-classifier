from __future__ import annotations
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .config import Config
from .dataset import make_binary_balanced_df, RetinaBinaryDataset
from .models import LogisticRegressionBaseline, SimpleCNN, build_resnet18
from .utils import ensure_dirs, accuracy_from_logits, confusion_from_logits, plot_curves


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    all_logits = []
    all_y = []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        total_n += bs

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)
    cm = confusion_from_logits(all_logits, all_y)

    return {"loss": total_loss / total_n, "acc": total_acc / total_n, "cm": cm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "cnn", "resnet18"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    ensure_dirs(cfg.outputs_root, cfg.ckpt_dir, cfg.plot_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_path = cfg.data_root / cfg.labels_csv
    images_dir = cfg.data_root / cfg.images_dir

    df = make_binary_balanced_df(csv_path, images_dir, seed=cfg.seed)

    train_df, val_df = train_test_split(
        df, test_size=cfg.val_frac, random_state=cfg.seed, stratify=df["binary"]
    )

    if args.model == "logreg":
        train_ds = RetinaBinaryDataset(train_df, img_size=cfg.img_size, brighten=1.2, grayscale=False, return_flat=True)
        val_ds = RetinaBinaryDataset(val_df, img_size=cfg.img_size, brighten=1.2, grayscale=False, return_flat=True)

        in_features = 3 * cfg.img_size * cfg.img_size
        model = LogisticRegressionBaseline(in_features=in_features, dropout_p=0.3)
        epochs = args.epochs if args.epochs is not None else 10

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = None

    elif args.model == "cnn":
        train_ds = RetinaBinaryDataset(train_df, img_size=cfg.img_size, brighten=1.2, grayscale=False, return_flat=False)
        val_ds = RetinaBinaryDataset(val_df, img_size=cfg.img_size, brighten=1.2, grayscale=False, return_flat=False)

        model = SimpleCNN(dropout_p=0.2)
        epochs = args.epochs if args.epochs is not None else 10

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = None

    else:
        train_ds = RetinaBinaryDataset(train_df, img_size=cfg.img_size, brighten=1.2, grayscale=False, return_flat=False)
        val_ds = RetinaBinaryDataset(val_df, img_size=cfg.img_size, brighten=1.2, grayscale=False, return_flat=False)

        model = build_resnet18(num_classes=2, pretrained=True)
        epochs = args.epochs if args.epochs is not None else 25

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.001)

    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_path = cfg.ckpt_dir / f"best_{args.model}.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
        )
        print("Val confusion [[TN,FP],[FN,TP]]:\n", val_metrics["cm"])

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({"model": model.state_dict(), "history": history}, best_path)

    plot_curves(history, cfg.plot_dir / f"{args.model}_curves.png", title=f"{args.model} training curves")
    print("Saved best checkpoint:", best_path)
    print("Saved plot:", cfg.plot_dir / f"{args.model}_curves.png")


if __name__ == "__main__":
    main()
