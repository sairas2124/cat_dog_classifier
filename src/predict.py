import argparse
import os

import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="models/best.pt")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit("Weights not found. Train first (src/train.py).")
    if not os.path.isfile(args.image):
        raise SystemExit("Image file not found.")

    ckpt = torch.load(args.weights, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    if args.device in ["cpu", "cuda"]:
        device = args.device

    model = build_model(len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    tfm = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        p, idx = probs.max(dim=0)

    label = classes[int(idx)]
    print(f"Prediction: {label}  (prob={float(p):.4f})")
    print("All class probs:")
    for c, pr in zip(classes, probs.tolist()):
        print(f" - {c}: {pr:.4f}")

if __name__ == "__main__":
    main()
