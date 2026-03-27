import itertools
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image as PILImage
from datasets import load_dataset, Features, Value
from datasets.features import Image as ImageFeature
import os
from typing import Sequence
import cv2
import torch


def inspect_features(data_files: str, split: str = "train") -> None:
    """
    Load a dataset in streaming mode and print its declared feature columns.

    Args:
        data_files: Path or pattern to the WebDataset archive.
        split: Dataset split name (default 'train').
    """
    ds = load_dataset(
        "webdataset",
        data_files=data_files,
        split=split,
        streaming=True
    )
    feats = ds.info.features
    print("Declared feature columns:")
    for name, feat in feats.items():
        print(f"  {name:12s} => {feat}")


def load_raw_dataset(data_files: str,
                     image_key: str,
                     split: str = "train"):
    """
    Load a WebDataset in streaming mode, overriding the specified image_key
    to be raw binary bytes (to disable HF auto-decoding).

    Args:
        data_files: Path or pattern to the WebDataset archive.
        image_key: The feature name to treat as binary.
        split: Dataset split name (default 'train').

    Returns:
        An IterableDataset with the image_key field as bytes.
    """
    info_ds = load_dataset(
        "webdataset",
        data_files=data_files,
        split=split,
        streaming=True
    )
    orig_feats = info_ds.info.features
    raw_feats = Features({
        name: Value("binary") if name == image_key else feat
        for name, feat in orig_feats.items()
    })
    return load_dataset(
        "webdataset",
        data_files=data_files,
        split=split,
        streaming=True,
        features=raw_feats
    )


def collect_tif_arrays(ds,
                       key: str,
                       n: int = 9,
                       max_scan: int = 3000) -> list:
    """
    Collect and decode up to n multi-band TIFF arrays from the dataset.

    Args:
        ds: IterableDataset with the TIFF field as raw bytes.
        key: Feature name of the TIFF bytes.
        n: Number of valid arrays to collect.
        max_scan: Max samples to scan through.

    Returns:
        List of normalized NumPy arrays (H×W or H×W×C).
    """
    arrs = []
    for sample in itertools.islice(ds, max_scan):
        b = sample.get(key)
        if not isinstance(b, (bytes, bytearray)):
            continue
        try:
            arr = tifffile.imread(BytesIO(b))
        except Exception:
            continue
        # Channel-first -> H×W×C
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1,2,0))
        # Limit to 3 bands
        if arr.ndim == 3 and arr.shape[2] > 3:
            img = arr[..., :3]
        else:
            img = arr
        # Normalize 0–1
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img, dtype=float)
        arrs.append(img)
        if len(arrs) >= n:
            break
    return arrs


def collect_png_masks(ds,
                      key: str = "png",
                      n: int = 9,
                      max_scan: int = 2000) -> list:
    """
    Collect and decode up to n PNG-encoded masks from the dataset.

    Args:
        ds: IterableDataset with mask field (PIL.Image or bytes).
        key: Feature name of the PNG masks.
        n: Number of masks to collect.
        max_scan: Max samples to scan through.

    Returns:
        List of normalized binary NumPy arrays (H×W).
    """
    masks = []
    for sample in itertools.islice(ds, max_scan):
        val = sample.get(key)
        arr = None
        # Already decoded by HF?
        if isinstance(val, PILImage.Image):
            arr = np.array(val.convert("L"))
        elif isinstance(val, (bytes, bytearray)):
            try:
                arr = np.array(PILImage.open(BytesIO(val)).convert("L"))
            except Exception:
                continue
        if arr is None:
            continue
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            mask01 = (arr - mn) / (mx - mn)
        else:
            mask01 = (arr != mn).astype(float)
        masks.append(mask01)
        if len(masks) >= n:
            break
    return masks


def plot_images(images: list,
                rows: int = 3,
                cols: int = 3,
                cmap: str = None) -> None:
    """
    Plot a list of image arrays in a grid.

    Args:
        images: List of 2D or 3D NumPy arrays.
        rows: Number of grid rows.
        cols: Number of grid columns.
        cmap: Matplotlib colormap for single-channel arrays.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    for ax, img in zip(axes, images):
        if img.ndim == 2 and cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def report_matching_subfolders(path_masks, matching_folders) -> None:
    """
    Print a summary of how many matching subfolders were found,
    and what percentage they represent of all subfolders in path_masks.
    """
    total = len(os.listdir(path_masks))
    found = len(matching_folders)

    # Avoid division by zero if the directory is empty
    pct = (found / total * 100) if total > 0 else 0.0

    print(
        f"Found {found} matching subfolders, "
        f"which account for {pct:.2f}% of the total number of subfolders in {path_masks}."
    )


def save_first_image(model, visualizer, epoch, first_img_debug):
    if epoch == 0 and first_img_debug:
        # disable gradients for inference
        model.set_requires_grad(model.netG, False)
        # compute the image
        model.forward()
        # save an image for debugging
        model.compute_visuals()
        visualizer.display_current_results(model.get_current_visuals(), epoch, True)
        # reactivate gradients
        model.set_requires_grad(model.netG, True)
        return False
    return first_img_debug


def _ensure_gray_u8(arr: np.ndarray) -> np.ndarray:
    """Accept (H,W), (3,H,W), (H,W,3) of various dtypes => uint8 (H,W)."""
    a = arr
    if a.ndim == 3 and a.shape[0] in (1, 3):  # (C,H,W) -> (H,W,C)
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        maxv = float(a.max()) if a.size else 1.0
        a = (a * (255.0 if maxv <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return a

def _rgb_to_hsv01(x):
    """
    Convert RGB in [-1,1] to HSV in [0,1]. x: (B,3,H,W) float.
    Hue is in [0,1) with circular wrap-around.
    """
    eps = 1e-8
    x01 = x * 0.5 + 0.5  # [-1,1] -> [0,1]
    r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
    maxc, _ = x01.max(dim=1, keepdim=True)
    minc, _ = x01.min(dim=1, keepdim=True)
    delta = maxc - minc
    # Saturation
    s = delta / (maxc + eps)
    # Hue
    hr = ((g - b) / (delta + eps))
    hg = ((b - r) / (delta + eps)) + 2.0
    hb = ((r - g) / (delta + eps)) + 4.0
    h = torch.zeros_like(delta)
    h = torch.where((maxc - r).abs() < eps, hr, h)
    h = torch.where((maxc - g).abs() < eps, hg, h)
    h = torch.where((maxc - b).abs() < eps, hb, h)
    # normalize to [0,1)
    h = (h / 6.0) % 1.0
    v = maxc
    return torch.cat([h, s, v], dim=1)
   
