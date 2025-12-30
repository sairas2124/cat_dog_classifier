# CatraClassifier 🐱🐶 (Cats vs Dogs Image Classifier)

A complete, runnable **end-to-end** project you can open in VS Code, train a model, run predictions, and then push to GitHub.

## What it does
- Downloads the dataset via **kagglehub**
- Prepares a clean folder structure (`data/processed/train|val/{cat,dog}`)
- Trains a **supervised** image classifier (transfer learning with ResNet-18, PyTorch)
- Saves the best model to `models/best.pt`
- Predicts on a single image from the command line

---

## 1) Setup (Windows / VS Code)

Open a terminal in the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> If you get CUDA issues, don't worry—this runs on CPU too.

---

## 2) Download dataset (kagglehub)

First, set your Kaggle credentials safely (do **NOT** commit your API token):
- **Option A (recommended):** put your Kaggle `kaggle.json` in:
  - `C:\Users\<YOU>\.kaggle\kaggle.json`
- **Option B:** set environment variables:
  - `KAGGLE_USERNAME`
  - `KAGGLE_KEY`

Then run:

```bash
python src/prepare_data.py --dataset "karakaggle/kaggle-cat-vs-dog-dataset" --val-split 0.2 --seed 42
```

This will create:
- `data/processed/train/cat` and `data/processed/train/dog`
- `data/processed/val/cat` and `data/processed/val/dog`

---

## 3) Train

```bash
python src/train.py --data-dir data/processed --epochs 5 --batch-size 32 --lr 0.0003
```

Outputs:
- `models/best.pt` (best model by validation accuracy)
- `models/metadata.json` (class mapping + settings)

---

## 4) Predict (single image)

```bash
python src/predict.py --weights models/best.pt --image "path/to/your/image.jpg"
```

---

## 5) Push to GitHub (GitHub Desktop)

1. Open **GitHub Desktop**
2. `File → Add local repository…` and select this project folder
3. Add a repository name like: `CatraClassifier`
4. Click **Commit to main**
5. Click **Publish repository** (choose Public)
6. Done ✅

### Important
- `models/` is **gitignored** by default (keeps repo clean).
- If you want to show the model file, remove `models/` from `.gitignore`.

---

## Common issues
- If `prepare_data.py` can't find cats/dogs folders automatically, open the printed dataset path and check folder names, then pass `--source-dir <path>`.

---

## Project structure
```
CatraClassifier/
  src/
    prepare_data.py
    train.py
    predict.py
    utils.py
  data/                 (created after prepare_data)
  models/               (created after training)
  .gitignore
  requirements.txt
  README.md
```
