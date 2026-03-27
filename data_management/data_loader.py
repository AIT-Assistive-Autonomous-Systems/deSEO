import torch
from torch.utils.data import DataLoader
import os 
from data_management.preprocessing import MyMSIAugment
from data_management.dataset_handler import build_dataset
from data_management.deSEO import DeSEODataset
import re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

class Chain:
    """Minimal compose for sample-dict transforms."""
    def __init__(self, *ops):
        self.ops = [op for op in ops if op is not None]
    def __call__(self, sample):
        for op in self.ops:
            sample = op(sample)
        return sample


class ForceSizeCrop:
    """
    Enforce (H,W)=size on all HxW or CxHxW tensors in the sample dict.
    - train: random crop (pads if smaller)
    - eval:  center crop (pads if smaller)
    """
    def __init__(self, size=256, mode="train"):
        if isinstance(size, int):
            self.th, self.tw = size, size
        else:
            self.th, self.tw = int(size[0]), int(size[1])
        self.mode = mode

    def _crop_or_pad(self, arr, y0, x0):
        # arr: (H,W) or (C,H,W), returns same rank, cropped/padded to (th,tw)
        if arr.ndim == 3:
            C, H, W = arr.shape
            out = np.zeros((C, self.th, self.tw), dtype=arr.dtype) if isinstance(arr, np.ndarray) else arr.new_zeros(C, self.th, self.tw)
            ys, xs = max(0, y0), max(0, x0)
            ye, xe = min(H, y0 + self.th), min(W, x0 + self.tw)
            oy, ox = ys - y0, xs - x0
            out[..., oy:oy+(ye-ys), ox:ox+(xe-xs)] = arr[..., ys:ye, xs:xe]
            return out
        elif arr.ndim == 2:
            H, W = arr.shape
            out = np.zeros((self.th, self.tw), dtype=arr.dtype) if isinstance(arr, np.ndarray) else arr.new_zeros(self.th, self.tw)
            ys, xs = max(0, y0), max(0, x0)
            ye, xe = min(H, y0 + self.th), min(W, x0 + self.tw)
            oy, ox = ys - y0, xs - x0
            out[oy:oy+(ye-ys), ox:ox+(xe-xs)] = arr[ys:ye, xs:xe]
            return out
        return arr  # leave non-image fields as-is

    def __call__(self, sample: dict):
        # pick a reference tensor to compute crop coords
        ref = None
        for v in sample.values():
            if hasattr(v, "shape") and getattr(v, "ndim", 0) in (2, 3):
                ref = v
                break
        if ref is None:
            return sample

        H, W = ref.shape[-2:]
        # choose top-left of crop (with padding-aware negatives)
        if self.mode == "train":
            y0 = np.random.randint(-(self.th - H), H) if H < self.th else np.random.randint(0, H - self.th + 1)
            x0 = np.random.randint(-(self.tw - W), W) if W < self.tw else np.random.randint(0, W - self.tw + 1)
        else:
            y0 = (H - self.th) // 2
            x0 = (W - self.tw) // 2

        out = {}
        for k, v in sample.items():
            if hasattr(v, "shape") and getattr(v, "ndim", 0) in (2, 3):
                # convert torch<->numpy seamlessly
                is_torch = torch.is_tensor(v)
                arr = v.detach().cpu().numpy() if is_torch else v
                cropped = self._crop_or_pad(arr, y0, x0)
                out[k] = torch.from_numpy(cropped) if is_torch else cropped
            else:
                out[k] = v
        return out


def collate_dict(batch):
    # batch is a list of dicts, e.g. [{ 'target_mask': Tensor, … }, …]
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        for key in batch[0]
    }


# deSEO_dataloader
class deSEO_dataloader(DataLoader):
    def __init__(self,
                 dataset, 
                 batch_size=4, 
                 shuffle=True, 
                 num_workers=7, 
                 pin_memory=True, 
                 collate_fn=None, 
                 persistent_workers=True,
                 prefetch_factor=4,
                 drop_last=True):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_dict,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last
        )


def _ensure_dataset_built(configs):
    """Create results dir + dataset once, reuse everywhere."""
    paths = configs.dataset.paths

    os.makedirs(paths["PATH_RESULTS"], exist_ok=True)

    root = Path(paths["PATH_myDataset"])
    if not root.exists():
        print(f"[INFO] Creating deSEO dataset in {root}...")
        root.mkdir(parents=True, exist_ok=True)
        build_dataset(
            path_masks=paths["PATH_MASKS"],
            path_rgb=paths["PATH_RGB"],
            path_metadata=paths["PATH_METADATA"],
            configs=configs,
        )
    else:
        print(f"[INFO] Dataset already exists at {root}, skipping creation.")

    return root


def _create_split_loader(configs, split: str):
    """
    Generic split loader: 'train' | 'val' | 'test'.
    No config fallbacks: expects configs.train / configs.val / configs.test
    to all have batch_size, num_workers, prefetch_factor.
    """
    # Ensure dataset is present
    root = _ensure_dataset_built(configs)
    cache_dir = configs.dataset.paths["PATH_CACHE"]
    splits_dir = root / "splits"

    is_train = (split == "train")
    split_cfg = getattr(configs, split)  # configs.train / configs.val / configs.test

    # Transform: augmentation only for train
    if is_train:
        transform = Chain(MyMSIAugment(configs=configs))
    else:
        transform = None

    # Dataset: use manifest if available, otherwise full dataset
    ds_kwargs = dict(
        configs=configs,
        root_yaml_dir=str(root),
        cache_dir=cache_dir,
        transform=transform,
    )

    manifest = splits_dir / f"{split}.txt"
    if manifest.exists():
        print(f"[INFO] Loading '{split}' split from manifests at {manifest}")
        ds_kwargs["split"] = split
    else:
        print(f"[INFO] No manifest for split '{split}' found. Using full dataset.")

    dataset = DeSEODataset(**ds_kwargs)
    print(f"[INFO] Loaded {len(dataset)} samples for split '{split}'.")

    # Dataloader args
    loader_kwargs = dict(
        batch_size=split_cfg.batch_size,
        shuffle=is_train,
        num_workers=split_cfg.num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=(split_cfg.num_workers > 0),
    )

    # Only use prefetch_factor when workers > 0 and > 0
    if split_cfg.num_workers > 0 and split_cfg.prefetch_factor > 0:
        loader_kwargs["prefetch_factor"] = split_cfg.prefetch_factor

    return deSEO_dataloader(dataset, **loader_kwargs)


# Public API
def create_training_loader(configs):
    return _create_split_loader(configs, "train")


def create_validation_loader(configs):
    return _create_split_loader(configs, "val")


def create_test_loader(configs):
    return _create_split_loader(configs, "test")
    """Test loader: no data augmentation, no shuffling, keep all samples."""
    return _create_split_loader(configs, "test")