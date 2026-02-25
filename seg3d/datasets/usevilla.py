import zipfile
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def _load_mat_from_zip(zf, name):
    with zf.open(name) as f:
        return sio.loadmat(f)


class USevillaDataset(Dataset):
    """
    External validation dataset.
    Contains CT volumes (CASO) and GT_BONE / GT_MUSCLE masks.
    Output labels are mapped to class indices:
      0 background, 1 bone_all, 2 muscle_specific
    """

    def __init__(self, zip_path, split="test"):
        self.zip_path = zip_path
        self.split = split
        self.names = []
        with zipfile.ZipFile(zip_path) as zf:
            prefix = "TEST DATASET/" if split == "test" else "TRAIN DATASET/"
            self.names = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".mat")]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.zip_path) as zf:
            data = _load_mat_from_zip(zf, self.names[idx])
        img = data["CASO"].astype(np.float32)  # 512x512x10
        bone = data["GT_BONE"].astype(np.uint8)
        muscle = data["GT_MUSCLE"].astype(np.uint8)

        # build label volume
        lbl = np.zeros_like(bone, dtype=np.uint8)
        lbl[bone > 0] = 1
        lbl[muscle > 0] = 2

        img = torch.from_numpy(img[None])
        lbl = torch.from_numpy(lbl.astype(np.int64))
        return img, lbl
