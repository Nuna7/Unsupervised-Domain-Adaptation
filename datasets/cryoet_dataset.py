"""
cryoet_dataset.py  —  Dataset classes for Vox-UDA.

Loads preprocessed .npy files produced by preprocess.py.

Source : data/processed/source/volumes/*.npy  +  .../masks/*.npy
         Paired by SORTED filename (preprocess.py guarantees same stem).

Target : data/processed/target/volumes/*.npy
         NO labels — unlabelled experimental data.

Target eval (optional, if supervisor provides masks):
         data/processed/target_eval/volumes/*.npy  +  .../masks/*.npy
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple


# ---------------------------------------------------------------------------
# Source Dataset
# ---------------------------------------------------------------------------

class SourceDataset(Dataset):
    """
    Labelled source domain (simulated cryo-ET subtomograms).

    Returns (volume, mask):
        volume : float32 tensor (1, 32, 32, 32) — normalised to zero-mean unit-std
        mask   : long   tensor  (32, 32, 32)    — binary {0, 1}
    """

    def __init__(
        self,
        volume_dir: str,
        mask_dir:   str,
        normalize:  bool = True,
        transform:  Optional[Callable] = None,
    ):
        self.volume_paths = sorted(glob.glob(os.path.join(volume_dir, "*.npy")))
        self.mask_paths   = sorted(glob.glob(os.path.join(mask_dir,   "*.npy")))

        if len(self.volume_paths) == 0:
            raise FileNotFoundError(f"No .npy files in {volume_dir}")
        if len(self.volume_paths) != len(self.mask_paths):
            raise ValueError(
                f"Volume/mask count mismatch: "
                f"{len(self.volume_paths)} vs {len(self.mask_paths)}\n"
                f"Make sure preprocess.py completed successfully."
            )
        # Verify pairing by stem
        for vp, mp in zip(self.volume_paths[:5], self.mask_paths[:5]):
            vs = os.path.splitext(os.path.basename(vp))[0]
            ms = os.path.splitext(os.path.basename(mp))[0]
            if vs != ms:
                raise ValueError(
                    f"Volume/mask mismatch at index:\n"
                    f"  volume: {vp}\n  mask:   {mp}\n"
                    f"Stems don't match: '{vs}' vs '{ms}'"
                )

        self.normalize = normalize
        self.transform = transform

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vol  = np.load(self.volume_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx])
        mask = (mask > 0).astype(np.int64)

        if self.normalize:
            std = vol.std()
            if std > 1e-8:
                vol = (vol - vol.mean()) / std
            else:
                vol = vol - vol.mean()

        vol_t  = torch.from_numpy(vol).unsqueeze(0)   # (1, 32, 32, 32)
        mask_t = torch.from_numpy(mask).long()          # (32, 32, 32)

        if self.transform is not None:
            vol_t, mask_t = self.transform(vol_t, mask_t)

        return vol_t, mask_t


# ---------------------------------------------------------------------------
# Target Dataset  (NO labels — unlabelled)
# ---------------------------------------------------------------------------

class TargetDataset(Dataset):
    """
    Unlabelled target domain (real Poly-GA subtomograms).

    Returns:
        volume : float32 tensor (1, 32, 32, 32)
    """

    def __init__(
        self,
        volume_dir: str,
        normalize:  bool = True,
        transform:  Optional[Callable] = None,
    ):
        self.volume_paths = sorted(glob.glob(os.path.join(volume_dir, "*.npy")))
        if len(self.volume_paths) == 0:
            raise FileNotFoundError(f"No .npy files in {volume_dir}")
        self.normalize = normalize
        self.transform = transform

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vol = np.load(self.volume_paths[idx]).astype(np.float32)
        if self.normalize:
            std = vol.std()
            if std > 1e-8:
                vol = (vol - vol.mean()) / std
            else:
                vol = vol - vol.mean()
        vol_t = torch.from_numpy(vol).unsqueeze(0)
        if self.transform is not None:
            vol_t = self.transform(vol_t)
        return vol_t


# ---------------------------------------------------------------------------
# Labelled target  (evaluation ONLY — never used for training)
# ---------------------------------------------------------------------------

class TargetDatasetLabelled(Dataset):
    """
    Target with GT masks — only for quantitative evaluation.
    Use only when your supervisor provides Poly-GA segmentation masks.
    """

    def __init__(self, volume_dir: str, mask_dir: str, normalize: bool = True):
        self.volume_paths = sorted(glob.glob(os.path.join(volume_dir, "*.npy")))
        self.mask_paths   = sorted(glob.glob(os.path.join(mask_dir,   "*.npy")))
        if len(self.volume_paths) == 0:
            raise FileNotFoundError(f"No .npy files in {volume_dir}")
        if len(self.volume_paths) != len(self.mask_paths):
            raise ValueError("Volume/mask count mismatch in eval split.")
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vol  = np.load(self.volume_paths[idx]).astype(np.float32)
        mask = (np.load(self.mask_paths[idx]) > 0).astype(np.int64)
        if self.normalize:
            std = vol.std()
            vol = (vol - vol.mean()) / (std + 1e-8)
        return torch.from_numpy(vol).unsqueeze(0), torch.from_numpy(mask).long()


# ---------------------------------------------------------------------------
# NGM noise subset
# ---------------------------------------------------------------------------

class TargetNoiseSubset:
    """
    Random N_sampled-sized subset of target volumes for the NGM.
    Call resample() once per epoch to refresh.
    """

    def __init__(self, target_dataset: TargetDataset, n_sampled: int = 10):
        self.dataset   = target_dataset
        self.n_sampled = min(n_sampled, len(target_dataset))
        self.resample()

    def resample(self) -> None:
        idx         = np.random.choice(len(self.dataset), self.n_sampled, replace=False)
        self.items  = [self.dataset[int(i)] for i in idx]  # list of (1,32,32,32)

    def as_batch_tensor(self) -> torch.Tensor:
        return torch.stack(self.items, dim=0)   # (N, 1, 32, 32, 32)

    def __len__(self) -> int:
        return self.n_sampled


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def build_source_loader(
    volume_dir:  str,
    mask_dir:    str,
    batch_size:  int  = 16,
    num_workers: int  = 4,
    shuffle:     bool = True,
) -> DataLoader:
    return DataLoader(
        SourceDataset(volume_dir, mask_dir),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def build_target_loader(
    volume_dir:  str,
    batch_size:  int  = 16,
    num_workers: int  = 4,
    shuffle:     bool = True,
) -> DataLoader:
    return DataLoader(
        TargetDataset(volume_dir),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def build_eval_loader(
    volume_dir:  str,
    mask_dir:    str,
    batch_size:  int = 1,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        TargetDatasetLabelled(volume_dir, mask_dir),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )