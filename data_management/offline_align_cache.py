import os
import glob
import yaml
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from warnings import warn as Warnings
from osgeo import gdal

# Remove redundant imports
from data_management.preprocessing import normalize_bands_pix2pix
from data_management.geo_crop import open_png_as_raster, align_and_crop_geo_images
from data_management.utils import _ensure_gray_u8
from data_management.features_matchers import LoFTRMatcher, KorniaLoFTR
from data_management.dataset_utils import get_dem_scene, read_dem_arr, _idx_from_name, scene_from_path, scene_idx_from_path
from omegaconf import DictConfig, OmegaConf
import hydra

import numpy as np

def _hist_match_channel(src_u8_hw, ref_u8_hw):
    src_hist = np.bincount(src_u8_hw.ravel(), minlength=256).astype(np.float64)
    ref_hist = np.bincount(ref_u8_hw.ravel(), minlength=256).astype(np.float64)

    src_cdf = np.cumsum(src_hist);  src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist);  ref_cdf /= ref_cdf[-1]

    lut = np.clip(np.round(np.interp(src_cdf, ref_cdf, np.arange(256))), 0, 255).astype(np.uint8)
    return lut[src_u8_hw]

    src_hist = np.bincount(src_vals, minlength=256).astype(np.float64)
    ref_hist = np.bincount(ref_vals, minlength=256).astype(np.float64)
    src_cdf = np.cumsum(src_hist);  src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist);  ref_cdf /= ref_cdf[-1]
    lut = np.clip(np.round(np.interp(src_cdf, ref_cdf, np.arange(256))), 0, 255).astype(np.uint8)
    return lut[src_u8_hw]


def _hist_match_rgb(src_u8_chw, ref_u8_chw, mask_hw=None, mode="l_only"):
    """
    src/ref: (3,H,W) uint8 in **RGB**.
    mode="rgb": per-channel RGB matching
    mode="l_only": match only L (Lab) channel
    """
    assert src_u8_chw.ndim == 3 and src_u8_chw.shape[0] == 3
    assert ref_u8_chw.ndim == 3 and ref_u8_chw.shape[0] == 3
    if mode == "rgb":
        out = np.empty_like(src_u8_chw)
        for c in range(3):
            out[c] = _hist_match_channel(src_u8_chw[c], ref_u8_chw[c])
        return out
    elif mode == "l_only":
        # CHW -> HWC (RGB)
        src_rgb = np.transpose(src_u8_chw, (1, 2, 0))
        ref_rgb = np.transpose(ref_u8_chw, (1, 2, 0))
        # RGB <-> Lab (OpenCV has direct RGB conversions)
        src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB)
        ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB)
        Ls, As, Bs = cv2.split(src_lab)
        Lr, _, _   = cv2.split(ref_lab)
        Ls_m = _hist_match_channel(Ls, Lr)
        out_lab = cv2.merge([Ls_m, As, Bs])
        out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
        return np.transpose(out_rgb, (2, 0, 1))
    else:
        return src_u8_chw  # no-op


def align_rasters(s):
    """Align and crop rasters to a common grid."""
    rasters_aligned = align_and_crop_geo_images([
        open_png_as_raster(s["input_mask"], s["input_metadata"]),
        open_png_as_raster(s["input_rgb"], s["input_metadata"]),
        open_png_as_raster(s["target_mask"], s["target_metadata"]),
        open_png_as_raster(s["target_rgb"], s["target_metadata"]),
        open_png_as_raster(s["dem_path"], s["input_metadata"])
    ])
    if len(rasters_aligned) != 5:
        raise RuntimeError(f"Failed to align/crop {s['input_rgb']} and {s['target_rgb']}")
    return rasters_aligned


def read_arr(ds):
    n = ds.RasterCount
    if n == 1:
        return ds.GetRasterBand(1).ReadAsArray()
    bands = [ds.GetRasterBand(i).ReadAsArray() for i in range(1, min(n, 3) + 1)]
    output = np.dstack(bands)  # (H, W, C)
    return np.transpose(output, (2, 0, 1))  # (C, H, W)


def _estimate_homography(src_xy, dst_xy, th):
    if len(src_xy) < 4:
        return np.eye(3, dtype=np.float32), 0
    robust_flag = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    H, inliers = cv2.findHomography(src_xy.astype(np.float32),
                                    dst_xy.astype(np.float32),
                                    method=robust_flag,
                                    ransacReprojThreshold=th,
                                    maxIters=50000, confidence=0.999)
    if H is None or inliers is None:
        return np.eye(3, dtype=np.float32), 0
    return H.astype(np.float32), int(inliers.sum())


def _warp_perspective(img, H, out_hw, is_mask=False):
    Ht, Wt = out_hw
    a = np.asarray(img)

    # Detect layout
    chw_input = False
    if a.ndim == 3:
        # Treat (1,H,W) or (3,H,W) as CHW; otherwise assume HWC
        if a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
            chw_input = True
    # else: HW (mask/gray/DEM single-band)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    warped = cv2.warpPerspective(
        a, H, (Wt, Ht), flags=interp,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    if chw_input:
        # OpenCV may drop the singleton channel and return HW
        if warped.ndim == 2:
            warped = warped[..., None]     # -> (H,W,1)
        return np.transpose(warped, (2, 0, 1))  # HWC -> CHW

    return warped  # HWC or HW stays as-is


def _warp_valid_mask_like_perspective(src_hw, H, dst_hw):
    Hs, Ws = src_hw; Ht, Wt = dst_hw
    ones = np.ones((Hs, Ws), np.uint8)
    return cv2.warpPerspective(ones, H, (Wt, Ht), flags=cv2.INTER_NEAREST, borderValue=0)


def _load_samples(cfg=None):
    root = Path(cfg.dataset.paths["ROOT_YAML_DIR"])
    yaml_paths = set()

    # If the user points directly to a single file (e.g., .../JAX_127/paths.yaml)
    if root.is_file() and root.name.lower().endswith(".yaml"):
        yaml_paths.add(root.resolve())
    else:
        # 1) honor split manifests if they exist
        splits_dir = root / "splits"
        if splits_dir.exists():
            listed_dirs = set()
            for name in ("train.txt", "val.txt", "test.txt"):
                p = splits_dir / name
                if p.exists():
                    for line in p.read_text().splitlines():
                        line = line.strip()
                        if line:
                            listed_dirs.add((root / line).resolve())
            for d in listed_dirs:
                y = d / "paths.yaml"
                if y.exists():
                    yaml_paths.add(y.resolve())

        # 2) recursive fallback (any depth)
        for y in root.rglob("paths.yaml"):
            yaml_paths.add(y.resolve())

    # 3) emit samples
    for y in sorted(yaml_paths):
        try:
            with open(y, "r") as f:
                entries = yaml.safe_load(f) or []
        except Exception as e:
            print(f"Skip unreadable {y}: {e}")
            continue
        for sample in entries:
            yield y, sample


def _crop_with_zero_pad(arr, y, x, h, w):
    """
    Safe crop for HW, CHW or HWC arrays. Out-of-bounds areas are zero-padded.
    """
    a = np.asarray(arr)
    was_chw = False
    # Normalize to HWC/HW for slicing
    if a.ndim == 3 and a.shape[0] in (1, 3):  # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
        was_chw = True

    H, W = a.shape[:2]
    y1, x1 = max(0, y), max(0, x)
    y2, x2 = min(y + h, H), min(x + w, W)

    if a.ndim == 2:
        out = np.zeros((h, w), dtype=a.dtype)
    else:
        C = a.shape[2]
        out = np.zeros((h, w, C), dtype=a.dtype)

    if y1 < y2 and x1 < x2:
        oy1, ox1 = y1 - y, x1 - x
        oy2, ox2 = oy1 + (y2 - y1), ox1 + (x2 - x1)
        out[oy1:oy2, ox1:ox2, ...] = a[y1:y2, x1:x2, ...]

    if was_chw:
        # back to CHW
        if out.ndim == 2:
            out = out[None, ...]  # (H,W) -> (1,H,W)
        else:
            out = np.transpose(out, (2, 0, 1))
    return out


def _cache_path(sample, cache_root="datasets/cache", read_only=False) -> Path:
    in_mask = Path(sample["input_mask"])
    cache_root = Path(cache_root)

    scene = scene_from_path(in_mask)        # e.g., 'JAX_225'
    idx   = scene_idx_from_path(in_mask)    # e.g., '225'

    # example: shadow/.../JAX_170/JAX_170_7_pan.json
    path = sample.get("input_metadata")
    filename = os.path.basename(path)       # "JAX_170_7_pan.json"
    stem, _ = os.path.splitext(filename)    # "JAX_170_7_pan"

    parts = stem.split("_")
    time = parts[-2] if len(parts) >= 2 else None

    # base subdir
    subdir = cache_root / scene / idx
    if time:
        subdir = subdir / time

    if read_only:
        # Do NOT create folder or generate a new unique name
        return subdir

    # For writing: ensure unique directory and create it
    subdir = unique_subdir(subdir)
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def unique_subdir(base: Path) -> Path:
    """
    Return a unique subdir path. If base exists, append _1, _2, ...
    """
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{counter}")
        counter += 1
    return candidate


def process_sample(cfg, s, loftr):
    """
    Align/cook one YAML sample 's' and write .npy tensors into the cache.
    Uses YAML's 'dem_path' when present and crops to YAML's 'crop': {y,x,h,w}.
    """
    cache_root = cfg.dataset.paths["PATH_CACHE"]
    out_dir = _cache_path(cache_root=cache_root, sample=s, read_only=True) # TODO check this
 
    # Skip if already looks complete (now includes DEM files + valid mask)
    required = ["input_mask.npy", "target_mask.npy", "input_rgb.npy",
                "target_rgb.npy", "valid_mask.npy",
                "input_dem.npy", "target_dem.npy",
                "input_dem_sensor.npy", "target_dem_sensor.npy"]
    if all((out_dir / fn).exists() for fn in required):
        Warnings(f"Skipping existing sample: {out_dir}")
        return
    # --- 3) Align everything to a COMMON ORTHO grid (north-up) ---
    # This uses your existing aligner; all returned datasets share the same SRS/bounds/size.
    in_mask_ds_al, in_rgb_ds_al, tg_mask_ds_al, tg_rgb_ds_al, dem_ds_al = align_rasters(s)

    # Read arrays (uint8 for mask/RGB, float for DEM if needed)
    in_rgb_ortho = read_arr(in_rgb_ds_al)       # (3,H,W) u8
    tg_rgb_ortho = read_arr(tg_rgb_ds_al)       # (3,H,W) u8
    in_mask_ortho = read_arr(in_mask_ds_al)     # (H,W)   u8
    tg_mask_ortho = read_arr(tg_mask_ds_al)     # (H,W)   u8

    dem_ortho = dem_ds_al.GetRasterBand(1).ReadAsArray().astype(np.float32)  # (H,W)

    # --- 4) Estimate fine alignment in ORTHO space (input -> target) ---
    Hs = np.eye(3, dtype=np.float32)
    Ht, Wt = tg_rgb_ortho.shape[-2], tg_rgb_ortho.shape[-1]
    Hi, Wi = in_rgb_ortho.shape[-2], in_rgb_ortho.shape[-1]

    if loftr is not None:
        g_in = _ensure_gray_u8(in_rgb_ortho)   # CHW -> HxW u8
        g_tg = _ensure_gray_u8(tg_rgb_ortho)
        k0, k1, conf = loftr.match(g_in, g_tg)
        if len(conf) >= 4:
            th = cfg.tools.get("LOFTR_RANSAC_THRESH", 4.0)
            min_inliers = cfg.tools.get("LOFTR_MIN_INLIERS", 15)
            Hs, ninl = _estimate_homography(k0, k1, th=th)
            if ninl < min_inliers:
                print(f"WARN: only {ninl} inliers; keeping coarse ortho alignment")

    # --- 5) Warp INPUT ortho stack into TARGET ortho grid ---
    in_rgb_aligned   = _warp_perspective(in_rgb_ortho, Hs, (Ht, Wt), is_mask=False)  # (3,Ht,Wt) u8
    in_mask_aligned  = _warp_perspective(in_mask_ortho, Hs, (Ht, Wt), is_mask=True)  # (Ht,Wt) u8
    in_dem_aligned   = _warp_perspective(dem_ortho,     Hs, (Ht, Wt), is_mask=False)  # (Ht,Wt) float32
    valid_after      = _warp_valid_mask_like_perspective((Hi, Wi), Hs, (Ht, Wt))      # (Ht,Wt) u8


    # TARGET is already our reference ortho image
    tg_rgb  = tg_rgb_ortho
    tg_mask = tg_mask_ortho
    tg_dem_aligned = dem_ortho.astype(np.float32, copy=False)  # same DEM in target grid

    # --- 6) (Optional) histogram matching (sensor domain) BEFORE normalization ---
    if bool(cfg.tools.get("HIST_MATCH")):
        Cov, Hov, Wov = tg_mask.shape
        in_rgb_aligned[:, :Hov, :Wov] = _hist_match_rgb(
            in_rgb_aligned[:, :Hov, :Wov], tg_rgb[:, :Hov, :Wov],
            mask_hw=(in_mask_aligned[:Hov, :Wov] > 0) & (tg_mask[:Hov, :Wov] > 0),
            mode=str(cfg.tools.get("HIST_MATCH_MODE"))
        )

    # --- 7) Normalize to [-1,1] if your training expects it ---
    in_rgbN, tg_rgbN = normalize_bands_pix2pix(in_rgb_aligned), normalize_bands_pix2pix(tg_rgb)

    # --- 8) Optional border crop / YAML crop (sensor coordinates!) ---
    c = int(cfg.tools.get("CROP_BORDER", 0))
    if c > 0:
        in_mask_aligned = in_mask_aligned[:, c:-c, c:-c]
        tg_mask         = tg_mask[:, c:-c, c:-c]
        in_rgbN         = in_rgbN[:, c:-c, c:-c]
        tg_rgbN         = tg_rgbN[:, c:-c, c:-c]
        in_dem_aligned  = in_dem_aligned[c:-c, c:-c]
        tg_dem_aligned  = tg_dem_aligned[c:-c, c:-c]
        valid_after     = valid_after[c:-c, c:-c]


    cinfo = s.get("crop")
    if cinfo is not None:
        y = int(cinfo.get("y", 0)); x = int(cinfo.get("x", 0))
        Hdef, Wdef = int(tg_rgbN.shape[-2]), int(tg_rgbN.shape[-1])
        h = int(cinfo.get("h", Hdef)); w = int(cinfo.get("w", Wdef))
        # Apply to everything IN SENSOR FRAME
        in_mask_aligned = _crop_with_zero_pad(in_mask_aligned, y, x, h, w)
        tg_mask         = _crop_with_zero_pad(tg_mask,         y, x, h, w)
        in_rgbN         = _crop_with_zero_pad(in_rgbN,         y, x, h, w)
        tg_rgbN         = _crop_with_zero_pad(tg_rgbN,         y, x, h, w)
        in_dem_aligned  = _crop_with_zero_pad(in_dem_aligned,  y, x, h, w)
        tg_dem_aligned  = _crop_with_zero_pad(tg_dem_aligned,  y, x, h, w)
        valid_after     = _crop_with_zero_pad(valid_after,     y, x, h, w)


    # --- 9) Save (all in TARGET sensor space) ---
    np.save(out_dir / "input_mask.npy",        in_mask_aligned.astype(np.uint8,  copy=False))
    np.save(out_dir / "target_mask.npy",       tg_mask.astype(np.uint8,          copy=False))
    np.save(out_dir / "input_rgb.npy",         in_rgbN.astype(np.float32,        copy=False))
    np.save(out_dir / "target_rgb.npy",        tg_rgbN.astype(np.float32,        copy=False))
    np.save(out_dir / "input_dem_sensor.npy",  in_dem_aligned.astype(np.float32, copy=False))
    np.save(out_dir / "target_dem_sensor.npy", tg_dem_aligned.astype(np.float32, copy=False))
    np.save(out_dir / "valid_mask.npy",        (valid_after > 0).astype(np.uint8, copy=False))


def main(cfg, use_loftr=True, loftr_conf=0.20, loftr_max_size=1024):
    print(cfg.keys())
    os.makedirs(cfg.dataset.paths["PATH_CACHE"], exist_ok=True)

    loftr = None
    if use_loftr and KorniaLoFTR is not None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        loftr = LoFTRMatcher(device=device, pretrained="outdoor",
                             max_size=loftr_max_size, conf_thresh=loftr_conf)

    samples = list(_load_samples(cfg))
    print(f"Found {len(samples)} samples under {cfg.dataset.paths['ROOT_YAML_DIR']}.")
    
    # Use tqdm for the progress bar and process sequentially
    for _, s in tqdm(samples, desc="Processing samples", ncols=100):
        process_sample(cfg, s, loftr)  # Process each sample sequentially
    
    print(f"Done. Cache root: {cfg.dataset.paths['PATH_CACHE']}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def _entry(cfg: DictConfig):
    """
    Compose config from conf/, inherit dataset group (e.g. dataset=deSEO),
    then set globals and run the pipeline.
    """
    # build globals from hydra cfg so main() can use them
    tool = OmegaConf.select(cfg, "tools.offline_align_cache", default=None)
    if tool is None:
        tool = OmegaConf.select(cfg, "tools", default={})
    use_loftr      = bool(tool.get("use_loftr", True))
    loftr_conf     = float(tool.get("loftr_conf", 0.20))
    loftr_max_size = int(tool.get("loftr_max_size", 1024))

    # call your existing main *without* cfg if it expects globals,
    # OR pass the knobs if your signature allows.
    # If your main currently has main(cfg), keep that but gate the globals below (see hotfix).
    main(cfg, use_loftr=use_loftr, loftr_conf=loftr_conf, loftr_max_size=loftr_max_size)

if __name__ == "__main__":
    _entry()
