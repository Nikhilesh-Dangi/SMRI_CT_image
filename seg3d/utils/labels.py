import numpy as np


def build_label_lut(classes):
    """
    Build a lookup for class index by label id.
    classes: list of dicts with keys: name, ids
    Returns dict label_id -> class_index (1..K). Background is 0.
    """
    lut = {}
    for idx, cls in enumerate(classes, start=1):
        for lab_id in cls["ids"]:
            lut[int(lab_id)] = idx
    return lut


def remap_labels(label_volume, lut):
    """Map original label ids to contiguous class ids; background stays 0."""
    out = np.zeros_like(label_volume, dtype=np.uint8)
    for lab_id, class_id in lut.items():
        out[label_volume == lab_id] = class_id
    return out
