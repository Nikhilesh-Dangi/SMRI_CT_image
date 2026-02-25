import csv
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from utils.labels import build_label_lut, remap_labels


def _load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing = img.GetSpacing()  # x,y,z
    return arr, spacing


def _resample(img, spacing, target_spacing, is_label=False):
    # SimpleITK resample to target spacing
    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetSpacing((spacing[2], spacing[1], spacing[0]))
    out_spacing = (target_spacing[2], target_spacing[1], target_spacing[0])
    size = [int(round(sz * spc / tspc)) for sz, spc, tspc in zip(sitk_img.GetSize(), sitk_img.GetSpacing(), out_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputDirection(sitk_img.GetDirection())
    out = resampler.Execute(sitk_img)
    out_arr = sitk.GetArrayFromImage(out)
    return out_arr


def _normalize(img, window, mode="zscore"):
    if window:
        lo, hi = window
        img = np.clip(img, lo, hi)
    if mode == "zscore":
        mean = img.mean()
        std = img.std() + 1e-6
        return (img - mean) / std
    return img


def _random_crop(img, lbl, patch):
    z, y, x = img.shape
    pz, py, px = patch
    if z < pz or y < py or x < px:
        pad_z = max(0, pz - z)
        pad_y = max(0, py - y)
        pad_x = max(0, px - x)
        img = np.pad(img, ((0, pad_z), (0, pad_y), (0, pad_x)))
        lbl = np.pad(lbl, ((0, pad_z), (0, pad_y), (0, pad_x)))
        z, y, x = img.shape
    zs = random.randint(0, z - pz)
    ys = random.randint(0, y - py)
    xs = random.randint(0, x - px)
    return img[zs:zs+pz, ys:ys+py, xs:xs+px], lbl[zs:zs+pz, ys:ys+py, xs:xs+px]


class TotalSegDataset(Dataset):
    def __init__(self, manifest_csv, classes, target_spacing, patch_size, window, normalize, split="train", val_split=0.15, seed=42):
        with open(manifest_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        random.seed(seed)
        random.shuffle(rows)
        split_idx = int(len(rows) * val_split)
        if split == "val":
            self.rows = rows[:split_idx]
        else:
            self.rows = rows[split_idx:]

        self.lut = build_label_lut(classes)
        self.target_spacing = target_spacing
        self.patch_size = patch_size
        self.window = window
        self.normalize = normalize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img, spacing = _load_nifti(row["image"])
        lbl, _ = _load_nifti(row["label"])

        img = _resample(img, spacing, self.target_spacing, is_label=False)
        lbl = _resample(lbl, spacing, self.target_spacing, is_label=True)
        lbl = remap_labels(lbl, self.lut)

        img = _normalize(img, self.window, self.normalize)
        img, lbl = _random_crop(img, lbl, self.patch_size)

        img = torch.from_numpy(img[None].astype(np.float32))
        lbl = torch.from_numpy(lbl.astype(np.int64))
        return img, lbl
