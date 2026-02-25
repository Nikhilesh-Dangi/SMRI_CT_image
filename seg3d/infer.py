import argparse
import numpy as np
import torch
import SimpleITK as sitk

from models.unet3d import UNet3D
from utils.config import load_config
from utils.labels import build_label_lut, remap_labels


def load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing = img.GetSpacing()
    return arr, spacing, img


def normalize(img, window, mode="zscore"):
    if window:
        lo, hi = window
        img = np.clip(img, lo, hi)
    if mode == "zscore":
        mean = img.mean()
        std = img.std() + 1e-6
        return (img - mean) / std
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    classes = cfg["labels"]["classes"]
    num_classes = len(classes)
    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")

    img, spacing, sitk_img = load_nifti(args.image)
    img = normalize(img, cfg["preprocess"]["intensity_window"], cfg["preprocess"]["normalize"])

    x = torch.from_numpy(img[None, None].astype(np.float32)).to(device)

    model = UNet3D(1, num_classes + 1).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    out_img = sitk.GetImageFromArray(pred)
    out_img.SetSpacing(sitk_img.GetSpacing())
    out_img.SetOrigin(sitk_img.GetOrigin())
    out_img.SetDirection(sitk_img.GetDirection())
    sitk.WriteImage(out_img, args.out)


if __name__ == "__main__":
    main()
