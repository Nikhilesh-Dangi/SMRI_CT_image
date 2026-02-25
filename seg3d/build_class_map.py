import argparse
import json
from utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    classes = cfg["labels"]["classes"]
    class_map = {str(i + 1): cls["name"] for i, cls in enumerate(classes)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)


if __name__ == "__main__":
    main()
