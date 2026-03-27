import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.utils.data

from data_management.offline_align_cache import _cache_path


def _own_c_contig(a: np.ndarray, dtype) -> np.ndarray:
    a = np.asarray(a)
    need_copy = (
        isinstance(a, np.memmap)
        or (not a.flags.writeable)
        or (not a.flags["C_CONTIGUOUS"])
        or (a.dtype != dtype)
    )
    return np.array(a, dtype=dtype, order="C", copy=True) if need_copy else a


def _to_torch_f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(_own_c_contig(a, np.float32))


def _to_torch_u8_as_f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(_own_c_contig(a, np.uint8)).to(torch.float32)


def _ensure_hw_u8(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        if a.shape[0] == 1:      # CHW
            a = a[0]
        elif a.shape[-1] == 1:   # HWC
            a = a[..., 0]
    a = a.astype(np.uint8, copy=False)
    return np.ascontiguousarray(a)


def _ensure_chw_f32(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        a = a[None, ...]
    elif a.ndim == 3 and a.shape[-1] in (1, 3):  # HWC -> CHW
        a = np.transpose(a, (2, 0, 1))
    a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)


def _ensure_1hw_f32(a: np.ndarray, H: int, W: int) -> np.ndarray:
    if a.ndim == 2:
        a = a[None, ...]
    elif a.ndim == 3 and a.shape[-1] == 1:  # HWC single band
        a = np.transpose(a, (2, 0, 1))
    a = a.astype(np.float32, copy=False)
    if a.shape[-2:] != (H, W):
        raise ValueError(f"Cached array shape {a.shape[-2:]} != expected {(H, W)}")
    return np.ascontiguousarray(a)


def _load_yaml_list(yaml_paths):
    import yaml  # simple import instead of __import__
    out = []
    for y in yaml_paths:
        with open(y, "r") as f:
            out += yaml.safe_load(f)
    return out


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    arr = np.load(path, mmap_mode=None)
    if (not arr.flags.writeable) or (not arr.flags.c_contiguous):
        arr = np.array(arr, copy=True, order="C")
    return arr


def _normalize_01(t: torch.Tensor) -> torch.Tensor:
    t_min, t_max = t.aminmax()
    return (t - t_min) / (t_max - t_min + 1e-8)


def _read_split_paths(root_yaml_dir: Path, split: str) -> list[Path]:
    splits_dir = root_yaml_dir / "splits"
    split_file = splits_dir / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    dirs = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if line:
            dirs.append(Path(line))
    return dirs


class DeSEODataset(torch.utils.data.Dataset):
    """PyTorch Dataset for deSEO samples with on-disk cache."""

    def __init__(
        self,
        root_yaml_dir: str | Path,
        cache_dir: str | Path,
        split: str,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        configs=None,
    ) -> None:
        self.root_yaml_dir = Path(root_yaml_dir)
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.transform = transform
        self.configs = configs

        # Build sample list strictly from split manifests
        listed_dirs = _read_split_paths(self.root_yaml_dir, split)
        yaml_paths = [d / "paths.yaml" for d in listed_dirs if (d / "paths.yaml").exists()]
        self.samples = _load_yaml_list(yaml_paths)

        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"[INFO] DeSEODataset '{split}' with {len(self.samples)} samples.")
        print(f"[INFO] Cache dir: {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        samp = self.samples[idx]
        tm, im, trgb, irgb, valid_mask, idem, tdem = self._load_cache(samp)

        out = {
            "target_mask": _to_torch_u8_as_f32(tm),
            "input_mask":  _to_torch_u8_as_f32(im),
            "target_rgb":  _to_torch_f32(trgb),
            "input_rgb":   _to_torch_f32(irgb),
            "valid_mask":  _to_torch_f32(valid_mask),
        }

        if self.configs is not None and self.configs.train.use_DEM:
            if idem is None or tdem is None:
                raise RuntimeError("use_DEM=True but DEM files are missing in cache.")
            out["input_dem"]  = _normalize_01(_to_torch_f32(idem))
            out["target_dem"] = _normalize_01(_to_torch_f32(tdem))

        if self.transform:
            out = self.transform(out)

        return out

    def _load_cache(
        self,
        samp: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        out_dir = _cache_path(samp, cache_root=self.cache_dir, read_only=True)

        f_tm   = out_dir / "target_mask.npy"
        f_im   = out_dir / "input_mask.npy"
        f_trgb = out_dir / "target_rgb.npy"
        f_irgb = out_dir / "input_rgb.npy"
        f_val  = out_dir / "valid_mask.npy"
        f_idem = out_dir / "input_dem_sensor.npy"
        f_tdem = out_dir / "target_dem_sensor.npy"

        tm   = _ensure_hw_u8(_load_npy(f_tm))
        im   = _ensure_hw_u8(_load_npy(f_im))
        trgb = _ensure_chw_f32(_load_npy(f_trgb))
        irgb = _ensure_chw_f32(_load_npy(f_irgb))

        H, W = trgb.shape[-2], trgb.shape[-1]

        if f_val.exists():
            valid = _ensure_1hw_f32(_load_npy(f_val), H, W)
        else:
            valid = np.ones((1, H, W), dtype=np.float32)

        idem = _ensure_1hw_f32(_load_npy(f_idem), H, W) if f_idem.exists() else None
        tdem = _ensure_1hw_f32(_load_npy(f_tdem), H, W) if f_tdem.exists() else None

        return tm, im, trgb, irgb, valid, idem, tdem
