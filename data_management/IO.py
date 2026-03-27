# this folder contains all the important I/O functions for the data management
import os
import yaml
from collections import defaultdict
import numpy as np
import rasterio
from PIL import Image as PILImage

def save_samples_to_yaml(paths_data_samples, base_path, subfolder):
    """
    Writes <base_path>/<subfolder>/paths.yaml.

    Accepts either:
      - list[dict]  (new format produced by extract_and_save_subfolder)
      - list[tuple] (old format: tm, im, trgb, irgb, tmeta, imeta)
    """
    import os, yaml
    from collections import abc

    out_root = os.path.join(base_path, subfolder)
    os.makedirs(out_root, exist_ok=True)

    normalized = []
    for item in paths_data_samples:
        if isinstance(item, dict):
            # new format: use as-is, but keep key order stable
            normalized.append({
                "input_mask":      item.get("input_mask"),
                "target_mask":     item.get("target_mask"),
                "input_rgb":       item.get("input_rgb"),
                "target_rgb":      item.get("target_rgb"),
                "input_metadata":  item.get("input_metadata"),
                "target_metadata": item.get("target_metadata"),
                "crop":            item.get("crop"),
                "dem_path":        item.get("dem_path"),
                "scene":           item.get("scene"),
                "input_idx":       item.get("input_idx"),
                "target_idx":      item.get("target_idx"),
            })
        elif isinstance(item, abc.Sequence) and len(item) == 6:
            # old tuple format -> upgrade to dict
            tm, im, trgb, irgb, tmeta, imeta = item
            normalized.append({
                "input_mask":      im,
                "target_mask":     tm,
                "input_rgb":       irgb,
                "target_rgb":      trgb,
                "input_metadata":  imeta,
                "target_metadata": tmeta,
                "crop":            None,
                "dem_path":        None,
                "scene":           subfolder,
                "input_idx":       None,
                "target_idx":      None,
            })
        else:
            raise ValueError(f"Unexpected sample entry: {type(item)}")

    yaml_path = os.path.join(out_root, "paths.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(normalized, f, sort_keys=False)


def read_tiff_image(path_tif: str) -> np.ndarray:
    # Load raw DN bands
    with rasterio.open(path_tif) as src:
        return src.read().astype(float)  # shape (4, H, W)

def read_png_image(path_png: str) -> np.ndarray:
    with PILImage.open(path_png) as img:
        img_np = np.array(img)
        # if the dimensions are (H, W, 3), convert to (3, H, W)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = img_np.transpose(2, 0, 1)  # to visualise you need to transpose!
        # close the image file
    PILImage.Image.close(img)
    return img_np
    


