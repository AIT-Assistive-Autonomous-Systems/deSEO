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
from data_management.rpc_perspective import warp_mask_ortho_to_sensor, warp_dsm_ortho_to_sensor, warp_rgb_ortho_to_sensor
from data_management.dataset_utils import get_dem_scene, read_dem_arr, _idx_from_name, scene_from_path, scene_idx_from_path
from omegaconf import DictConfig, OmegaConf
import hydra

def _hist_match_channel(src_u8_hw, ref_u8_hw, mask_hw=None):
    H, W = src_u8_hw.shape
    if mask_hw is None:
        mask_hw = np.ones((H, W), dtype=bool)
    else:
        mask_hw = mask_hw.astype(bool)
        if mask_hw.shape != (H, W):          # <-- guard
            mask_hw = mask_hw[:H, :W]

    src_vals = src_u8_hw[mask_hw].ravel()
    ref_vals = ref_u8_hw[mask_hw].ravel()
    if src_vals.size == 0 or ref_vals.size == 0:
        return src_u8_hw  # nothing to do

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
            out[c] = _hist_match_channel(src_u8_chw[c], ref_u8_chw[c], mask_hw)
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
        Ls_m = _hist_match_channel(Ls, Lr, mask_hw)
        out_lab = cv2.merge([Ls_m, As, Bs])
        out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
        return np.transpose(out_rgb, (2, 0, 1))
    else:
        return src_u8_chw  # no-op


def _bounds_from_gt(gt, xsize, ysize):
    xmin = gt[0]
    xmax = gt[0] + gt[1] * xsize
    ymax = gt[3]
    ymin = gt[3] + gt[5] * ysize
    # Ensure min/max ordering
    return (min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))


def warp_ds_to_template(src_ds, template_ds, resample="bilinear"):
    """
    Warp src_ds into the exact grid (SRS, bounds, width, height) of template_ds.
    Returns an in-memory GDAL dataset.
    """
    alg = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic
    }.get(resample, gdal.GRA_Bilinear)

    xsize = template_ds.RasterXSize
    ysize = template_ds.RasterYSize
    gt = template_ds.GetGeoTransform()
    prj = template_ds.GetProjection()
    xmin, ymin, xmax, ymax = _bounds_from_gt(gt, xsize, ysize)

    opts = gdal.WarpOptions(
        format="MEM",
        dstSRS=prj,
        outputBounds=(xmin, ymin, xmax, ymax),
        width=xsize,
        height=ysize,
        resampleAlg=alg,
        srcNodata=None,
        dstNodata=0
    )
    warped = gdal.Warp("", src_ds, options=opts)
    if warped is None:
        raise RuntimeError("gdal.Warp failed to warp DEM to template grid")
    return warped


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
    bands = ds.RasterCount
    if bands == 1:
        return ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    arr = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
    if arr.shape[0] > 3: arr = arr[:3]
    return arr.astype(np.uint8)


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




def clean_scene_path(scene_path: str) -> str:
    """
    Extracts the scene ID (e.g., 'JAX_188') from the full scene path.
    Strips crop-related directories (e.g., 'crops_512') from the scene path.
    """
    # Get the grandparent directory (which should be the scene folder like 'JAX_188')
    grandparent_dir = Path(scene_path).parent.parent.name
    # Extract the scene ID (assuming scene is the first part like 'JAX_188')
    scene_parts = grandparent_dir.split("_")
    # If we have multiple parts, the scene name is the first two parts
    scene_id = "_".join(scene_parts[:2]) if len(scene_parts) > 1 else grandparent_dir
    return scene_id



def _cache_path(sample, cache_root="datasets/cache"):
    in_mask = Path(sample["input_mask"])
    cache_root = Path(cache_root)
    scene = scene_from_path(in_mask)            # e.g., 'JAX_225'
    idx   = scene_idx_from_path(in_mask)        # e.g., '225'
    subdir = cache_root / scene / idx
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def gdal_resize(ds, target_size, resample=gdal.GRA_NearestNeighbour, src_nodata=None, dst_nodata=None):
    """
    Return a new in-memory GDAL dataset resized to (width, height) pixels.
    Georeferencing is preserved; the geographic extent stays the same and the
    pixel size changes accordingly.
    """
    width, height = target_size

    warp_opts = {
        "format": "MEM",
        "width": width,
        "height": height,
        "resampleAlg": resample,
    }
    # Optional nodata handling for masks
    if src_nodata is not None:
        warp_opts["srcNodata"] = src_nodata
    if dst_nodata is not None:
        warp_opts["dstNodata"] = dst_nodata

    out = gdal.Warp('', ds, **warp_opts)
    if out is None:
        raise RuntimeError("gdal.Warp failed to resize dataset")
    return out

def process_sample(cfg, s, loftr):
    """
    Align/cook one YAML sample 's' and write .npy tensors into the cache.
    Uses YAML's 'dem_path' when present and crops to YAML's 'crop': {y,x,h,w}.
    """
    cache_root = cfg.dataset.paths["PATH_CACHE"]
    out_dir = _cache_path(cache_root=cache_root, sample=s)

    # Skip if already looks complete (now includes DEM files + valid mask)
    required = ["input_mask.npy", "target_mask.npy", "input_rgb.npy",
                "target_rgb.npy", "valid_mask.npy",
                "input_dem.npy", "target_dem.npy",
                "input_dem_sensor.npy", "target_dem_sensor.npy"]
    if all((out_dir / fn).exists() for fn in required):
        Warnings(f"Skipping existing sample: {out_dir}")
        return

    # --- 1) Open north-up rasters (for lon/lat) ---
    in_mask_ds = open_png_as_raster(s["input_mask"],  s["input_metadata"])
    in_rgb_ds  = open_png_as_raster(s["input_rgb"],   s["input_metadata"])
    tg_mask_ds = open_png_as_raster(s["target_mask"], s["target_metadata"])
    tg_rgb_ds  = open_png_as_raster(s["target_rgb"],  s["target_metadata"])

    # --- 2) Resolve DEM paths (same logic you already use) ---
    in_scene_path = clean_scene_path(s["input_mask"])
    tg_scene_path = clean_scene_path(s["target_mask"])
    dem_pref = s.get("dem_path", None)
    if dem_pref and os.path.exists(dem_pref):
        in_dem_path = Path(dem_pref); tg_dem_path = Path(dem_pref)
    else:
        in_dem_path = get_dem_scene(Path(cfg.dataset.paths["DEM_TIF_DIR"]) / in_scene_path)
        tg_dem_path = get_dem_scene(Path(cfg.dataset.paths["DEM_TIF_DIR"]) / tg_scene_path)

    # --- 3) Project everything ORTHO -> SENSOR for each scene ---
    # RGB
    in_rgb_sensor = warp_rgb_ortho_to_sensor(in_rgb_ds, s["input_metadata"],  in_dem_path, gaussian_sigma=None)
    tg_rgb_sensor = warp_rgb_ortho_to_sensor(tg_rgb_ds, s["target_metadata"], tg_dem_path, gaussian_sigma=None)
    # masks
    in_mask_sensor = warp_mask_ortho_to_sensor(in_mask_ds, s["input_metadata"],  in_dem_path)  # (Hs,Ws) u8
    tg_mask_sensor = warp_mask_ortho_to_sensor(tg_mask_ds, s["target_metadata"], tg_dem_path)  # (Ht,Wt) u8
    # DEMs
    in_dem_sensor = warp_dsm_ortho_to_sensor(in_dem_path, s["input_metadata"],  agg="max", fill_value=np.nan)
    tg_dem_sensor = warp_dsm_ortho_to_sensor(tg_dem_path, s["target_metadata"], agg="max", fill_value=np.nan)

    # --- 4) Estimate fine alignment on SENSOR RGBs (input -> target) ---
    Hs = np.eye(3, dtype=np.float32)
    Ht, Wt = tg_rgb_sensor.shape[-2], tg_rgb_sensor.shape[-1]
    Hi, Wi = in_rgb_sensor.shape[-2], in_rgb_sensor.shape[-1]

    if loftr is not None:
        g_in = _ensure_gray_u8(in_rgb_sensor)  # expects CHW, returns HxW u8
        g_tg = _ensure_gray_u8(tg_rgb_sensor)
        k0, k1, conf = loftr.match(g_in, g_tg)
        if len(conf) >= 4:
            th = cfg.tools.get("LOFTR_RANSAC_THRESH", 4.0)
            min_inliers = cfg.tools.get("LOFTR_MIN_INLIERS", 15)
            Hs, ninl = _estimate_homography(k0, k1, th=th)
            if ninl < min_inliers:
                print(f"WARN: only {ninl} inliers; keeping coarse sensor alignment")

    # --- 5) Warp INPUT sensor stack into TARGET sensor grid ---
    in_rgb_aligned   = _warp_perspective(in_rgb_sensor,   Hs, (Ht, Wt), is_mask=False)  # (3,Ht,Wt)
    in_mask_aligned  = _warp_perspective(in_mask_sensor,  Hs, (Ht, Wt), is_mask=True)   # (Ht,Wt)
    in_dem_aligned   = _warp_perspective(in_dem_sensor,   Hs, (Ht, Wt), is_mask=False)  # (Ht,Wt) float32
    valid_after      = _warp_valid_mask_like_perspective((Hi, Wi), Hs, (Ht, Wt))        # (Ht,Wt) u8

    # TARGET stays as reference (already in sensor)
    tg_dem_aligned = tg_dem_sensor
    tg_rgb         = tg_rgb_sensor
    tg_mask        = tg_mask_sensor

    # --- 6) (Optional) histogram matching (sensor domain) BEFORE normalization ---
    if bool(cfg.tools.get("HIST_MATCH", False)):
        Hov, Wov = tg_mask.shape
        in_rgb_aligned[:, :Hov, :Wov] = _hist_match_rgb(
            in_rgb_aligned[:, :Hov, :Wov], tg_rgb[:, :Hov, :Wov],
            mask_hw=(in_mask_aligned[:Hov, :Wov] > 0) & (tg_mask[:Hov, :Wov] > 0),
            mode=str(cfg.tools.get("HIST_MATCH_MODE", "l_only"))
        )

    # --- 7) Normalize to [-1,1] if your training expects it ---
    in_rgbN, tg_rgbN = normalize_bands_pix2pix(in_rgb_aligned), normalize_bands_pix2pix(tg_rgb)

    # --- 8) Optional border crop / YAML crop (sensor coordinates!) ---
    c = int(cfg.tools.get("CROP_BORDER", 0))
    if c > 0:
        in_mask_aligned = in_mask_aligned[c:-c, c:-c]
        tg_mask         = tg_mask[c:-c, c:-c]
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
