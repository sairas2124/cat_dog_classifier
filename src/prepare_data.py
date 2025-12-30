import argparse
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

import kagglehub

from utils import ensure_dir, find_class_dirs, is_image_file

def copy_split_images(cat_dir: str, dog_dir: str, out_dir: str, val_split: float, seed: int):
    random.seed(seed)

    out_train_cat = os.path.join(out_dir, "train", "cat")
    out_train_dog = os.path.join(out_dir, "train", "dog")
    out_val_cat   = os.path.join(out_dir, "val", "cat")
    out_val_dog   = os.path.join(out_dir, "val", "dog")
    for p in [out_train_cat, out_train_dog, out_val_cat, out_val_dog]:
        ensure_dir(p)

    def split_and_copy(src_dir: str, dst_train: str, dst_val: str):
        files = [f for f in os.listdir(src_dir) if is_image_file(f)]
        files.sort()
        random.shuffle(files)

        n_val = int(len(files) * val_split)
        val_files = set(files[:n_val])

        for f in tqdm(files, desc=f"Copying {os.path.basename(src_dir)}", unit="img"):
            src = os.path.join(src_dir, f)
            if f in val_files:
                dst = os.path.join(dst_val, f)
            else:
                dst = os.path.join(dst_train, f)
            shutil.copy2(src, dst)

    split_and_copy(cat_dir, out_train_cat, out_val_cat)
    split_and_copy(dog_dir, out_train_dog, out_val_dog)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="karakaggle/kaggle-cat-vs-dog-dataset",
                    help="kagglehub dataset id, e.g. karakaggle/kaggle-cat-vs-dog-dataset")
    ap.add_argument("--source-dir", type=str, default=None,
                    help="If you already downloaded the dataset, pass the root path here.")
    ap.add_argument("--out-dir", type=str, default="data/processed",
                    help="Output folder for processed dataset.")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.source_dir:
        root = args.source_dir
    else:
        print(f"Downloading dataset via kagglehub: {args.dataset}")
        root = kagglehub.dataset_download(args.dataset)
    print("Dataset root path:", root)

    found = find_class_dirs(root)
    if not found:
        raise SystemExit(
            "Could not auto-find cat/dog folders. " 
            "Open the printed dataset root and find the folders manually, then re-run with --source-dir."
        )
    base, cat_name, dog_name = found
    cat_dir = os.path.join(base, cat_name)
    dog_dir = os.path.join(base, dog_name)

    print("Using folders:")
    print(" - cats:", cat_dir)
    print(" - dogs:", dog_dir)

    out_dir = args.out_dir
    # Clean out_dir if it already exists
    if os.path.exists(out_dir):
        print("Cleaning existing output folder:", out_dir)
        shutil.rmtree(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    copy_split_images(cat_dir, dog_dir, out_dir, args.val_split, args.seed)
    print("\nDone. Prepared dataset at:", os.path.abspath(out_dir))

if __name__ == "__main__":
    main()
