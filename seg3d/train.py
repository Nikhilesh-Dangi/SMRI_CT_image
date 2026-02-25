import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.metrics import soft_dice_loss
from datasets.totalseg import TotalSegDataset
from models.unet3d import UNet3D


def main(cfg_path):
    cfg = load_config(cfg_path)
    torch.manual_seed(cfg["project"]["seed"])

    classes = cfg["labels"]["classes"]
    num_classes = len(classes)

    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    train_ds = TotalSegDataset(
        cfg["paths"]["totalseg_manifest"],
        classes,
        cfg["preprocess"]["target_spacing"],
        cfg["preprocess"]["patch_size"],
        cfg["preprocess"]["intensity_window"],
        cfg["preprocess"]["normalize"],
        split="train",
        val_split=cfg["training"]["val_split"],
        seed=cfg["project"]["seed"],
    )
    val_ds = TotalSegDataset(
        cfg["paths"]["totalseg_manifest"],
        classes,
        cfg["preprocess"]["target_spacing"],
        cfg["preprocess"]["patch_size"],
        cfg["preprocess"]["intensity_window"],
        cfg["preprocess"]["normalize"],
        split="val",
        val_split=cfg["training"]["val_split"],
        seed=cfg["project"]["seed"],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg["training"]["num_workers"])

    model = UNet3D(1, num_classes + 1).to(device)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

    best_val = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        for img, lbl in train_loader:
            img, lbl = img.to(device), lbl.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(img)
            loss = 0.5 * ce(logits, lbl) + 0.5 * soft_dice_loss(logits, lbl, num_classes)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                logits = model(img)
                loss = 0.5 * ce(logits, lbl) + 0.5 * soft_dice_loss(logits, lbl, num_classes)
                val_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(cfg["paths"]["output_dir"], "best_model.pt")
            torch.save({"model": model.state_dict(), "config": cfg}, ckpt)

    with open(os.path.join(cfg["paths"]["output_dir"], "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
