import os
import json
from pathlib import Path

def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_class_dirs(root: str):
    """Try to find cats and dogs directories inside an arbitrary dataset tree."""
    root = os.path.abspath(root)
    candidates = []
    for dirpath, dirnames, _ in os.walk(root):
        dset = set([d.lower() for d in dirnames])
        if "cats" in dset and "dogs" in dset:
            candidates.append((dirpath, "cats", "dogs"))
        if "cat" in dset and "dog" in dset:
            candidates.append((dirpath, "cat", "dog"))
    # Prefer deeper folders that actually contain images
    def score(item):
        base, c, d = item
        return len(base.split(os.sep))
    candidates.sort(key=score, reverse=True)
    return candidates[0] if candidates else None

def is_image_file(name: str) -> bool:
    name = name.lower()
    return name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
