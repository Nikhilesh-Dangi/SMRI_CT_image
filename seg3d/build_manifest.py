import argparse
import os
from glob import glob

from utils.manifest import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing ct.nii.gz files")
    ap.add_argument("--label-name", required=True, help="Label filename to pair (e.g., muscles_label.nii.gz)")
    ap.add_argument("--out", required=True, help="Output CSV manifest path")
    args = ap.parse_args()

    images = glob(os.path.join(args.root, "**", "ct.nii.gz"), recursive=True)
    rows = []
    for img in images:
        lbl = os.path.join(os.path.dirname(img), args.label_name)
        if os.path.exists(lbl):
            rows.append([img, lbl])

    if not rows:
        raise SystemExit("No image/label pairs found. Check root and label-name.")

    write_manifest(rows, args.out)
    print(f"Wrote {len(rows)} pairs to {args.out}")


if __name__ == "__main__":
    main()
