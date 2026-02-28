# Diagnosing Diabetic Retinopathy 

This repo recreates a 2023 course project that trains models to detect diabetic retinopathy (DR) from retina images using the Kaggle EyePACS dataset (or a compatible resized variant). Please read the PDF for a full overview of the project. 

## What it does
- Loads retina images + labels (0–4)
- Converts to binary labels:
  - 0 = no DR
  - 1–4 = DR present
- Balances the dataset by downsampling class 0 to match class 1
- Preprocesses images:
  - center-crop + resize to 512x512
  - brightness enhancement for contrast
- Trains 3 models:
  - Logistic regression baseline (flattened pixels)
  - Small CNN (3 conv + 3 maxpool + dropout + MLP)
  - Finetuned ResNet-18 (pretrained)

## Repo layout
```
dr-diagnosis/
  data/raw/            # put Kaggle dataset here (NOT committed)
  outputs/             # generated checkpoints + plots (NOT committed)
  src/                 # code
```

## Setup

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Put Kaggle data in `data/raw/`
Expected layout:
```
data/raw/
  trainLabels.csv
  train/
    <image files>.jpeg (or .jpg or .png)
```

**Important:** Do NOT upload Kaggle images to GitHub (license/size). Commit only code + instructions.

### 3) Train (also generates plots)
```bash
python -m src.train --model logreg
python -m src.train --model cnn
python -m src.train --model resnet18
```

Checkpoints save to `outputs/checkpoints/` and curves to `outputs/plots/`.

### 4) Evaluate
```bash
python -m src.eval --model resnet18 --ckpt outputs/checkpoints/best_resnet18.pt
```

## Quick start using Kaggle CLI (recommended)
1. Create a Kaggle token: Kaggle → Account → "Create New API Token"
2. Put `kaggle.json` in `~/.kaggle/kaggle.json` and run:
```bash
pip install kaggle
kaggle datasets download -d diabeticinretinopathy/diabetic-retinopathy-detection -p data/raw --unzip
```

Then ensure:
- `trainLabels.csv` is in `data/raw/`
- image folder is `data/raw/train/` (rename if needed)

## Notes
This project was initially created in Feb 2023 (typo in PDF). It was enhanced using GPT-5.2 in Feb 2026.
