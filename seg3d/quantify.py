import json
import argparse
import numpy as np
import SimpleITK as sitk


def load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing = img.GetSpacing()  # x,y,z
    return arr, spacing


def compute_volume(mask, spacing):
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return float(mask.sum() * voxel_vol)


def compute_csa(mask, spacing, axis="axial"):
    # CSA per slice; return max CSA
    if axis == "axial":
        areas = mask.sum(axis=(1, 2)) * spacing[0] * spacing[1]
    elif axis == "coronal":
        areas = mask.sum(axis=(0, 2)) * spacing[0] * spacing[2]
    else:
        areas = mask.sum(axis=(0, 1)) * spacing[1] * spacing[2]
    return float(areas.max())


def compute_mean_hu(img, mask):
    vals = img[mask > 0]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="CT image NIfTI")
    ap.add_argument("--label", required=True, help="Label NIfTI (class ids)")
    ap.add_argument("--class-map", required=True, help="JSON mapping class_id -> name")
    ap.add_argument("--axis", default="axial")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img, spacing = load_nifti(args.image)
    lbl, _ = load_nifti(args.label)

    with open(args.class_map, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    results = {}
    for class_id_str, name in class_map.items():
        class_id = int(class_id_str)
        mask = (lbl == class_id).astype(np.uint8)
        results[name] = {
            "volume_mm3": compute_volume(mask, spacing),
            "csa_mm2": compute_csa(mask, spacing, axis=args.axis),
            "mean_hu": compute_mean_hu(img, mask),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
