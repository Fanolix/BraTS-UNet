import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, h5_files):
        self.h5_files = h5_files 
        print(f"Chargment de {len(self.h5_files)} images.")

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        with h5py.File(self.h5_files[idx], "r") as f:
            image = np.array(f["image"])  # (240, 240, 4)
            mask = np.array(f["mask"])    # (240, 240, 3)

        # Normalisation
        denom = image.max() - image.min()
        image = (image - image.min()) / denom if denom != 0 else np.zeros_like(image)

        # On converti le masque "one-hot" (en trois dimension) en un masque plan
        mask = np.argmax(mask, axis=-1)

        # On les converti en tenseur pour qu'ils soient accepté par pytorch 
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (4, 240, 240)
        mask = torch.tensor(mask, dtype=torch.long)  # (240, 240)

        return image, mask

