import os
import math
from typing import List, Tuple, Sequence, Union
import numpy as np
from data_management.IO import save_samples_to_yaml, read_tiff_image, read_png_image
from data_management.utils import report_matching_subfolders
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_descriptor')  # avoids POSIX semaphores
from typing import Dict
from data_management.dataset_utils import _acq_datetime, _bbox_from_geojson, _bbox_iou, _compute_view_geometry, _listdir_sorted, exp_term
from data_management.dataset_utils import _load_meta, _idx_from_name, _angular_diff_deg, get_dem_scene

import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from osgeo import gdal


# MAX_DELTA_SUN_AZ_DEG = float(FILTERS.get("MAX_DELTA_SUN_AZ_DEG", 10.0))  # disabled 
def _pattern(cfg, key, default):
    try:
        return getattr(cfg.dataset, "patterns")[key]
    except Exception:
        return default


def name_for_rgb(idx: str, subfolder, configs) -> str:
    tmpl = _pattern(configs, "RGB_NAME", "{subfolder}_{idx}_rgb.png")
    return tmpl.format(subfolder=subfolder, idx=idx)


def name_for_mask(idx: str, subfolder, configs) -> str:
    tmpl = _pattern(configs, "MASK_NAME", "{subfolder}_{idx}_shadow.png")
    return tmpl.format(subfolder=subfolder, idx=idx)


def name_for_meta(idx: str, subfolder, configs) -> str:
    tmpl = _pattern(configs, "META_NAME", "{subfolder}_{idx}_pan.json")
    return tmpl.format(subfolder=subfolder, idx=idx)


def split_dataset_paths(
    locations,
    out_root,
    train_p: float,
    val_p: float,
    test_p: float,
    seed: int = 1337,
    materialize: str = "manifest"):  # "manifest" | "symlink" | "copy"
    """
    locations: list[str | Path]  # paths of locations/scenes you want to split
    out_root: str | Path         # where to save the 3 splits
    materialize:
      - "manifest": write lists to train.txt / val.txt / test.txt
      - "symlink":  create directories with symlinks to the original locations
      - "copy":     create directories with deep copies (slower)
    """
    # --- validate proportions ---
    S = train_p + val_p + test_p
    if not (0.999 <= S <= 1.001):
        raise ValueError(f"train+val+test must sum to 1.0, got {S:.3f}")
    if any(p < 0 for p in (train_p, val_p, test_p)):
        raise ValueError("Proportions must be non-negative.")

    # --- shuffle reproducibly ---
    rng = random.Random(seed)
    locs = [Path(p) for p in locations]
    rng.shuffle(locs)

    # --- compute split sizes with leftover handling ---
    n = len(locs)
    n_train = math.floor(n * train_p)
    n_val   = math.floor(n * val_p)
    # put any rounding remainder into test to guarantee coverage
    n_test  = n - n_train - n_val

    train = locs[:n_train]
    val   = locs[n_train:n_train + n_val]
    test  = locs[n_train + n_val:]

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if materialize == "manifest":
        (out_root / "train.txt").write_text("\n".join(map(str, train)) + "\n")
        (out_root / "val.txt").write_text("\n".join(map(str, val)) + "\n")
        (out_root / "test.txt").write_text("\n".join(map(str, test)) + "\n")
    elif materialize in {"symlink", "copy"}:
        for split_name, items in [("train", train), ("val", val), ("test", test)]:
            split_dir = out_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for p in items:
                # each "location" might be a dir; replicate as a link/copy inside split_dir
                target = split_dir / p.name
                if materialize == "symlink":
                    try:
                        if target.exists() or target.is_symlink():
                            target.unlink() if target.is_symlink() else shutil.rmtree(target)
                        os.symlink(p, target, target_is_directory=p.is_dir())
                    except OSError:
                        # fallback to copy if symlinks not permitted (e.g., on Windows without admin)
                        if p.is_dir():
                            shutil.copytree(p, target, dirs_exist_ok=True)
                        else:
                            shutil.copy2(p, target)
                else:  # copy
                    if p.is_dir():
                        shutil.copytree(p, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(p, target)
    else:
        raise ValueError("materialize must be 'manifest', 'symlink', or 'copy'.")

    return {"train": train, "val": val, "test": test}


def _shadow_frac_crop(mask_hw_u8, y, x, sz=576):
    m = mask_hw_u8[y:y+sz, x:x+sz]
    if m.size == 0:  # out-of-bounds guard
        return 1.0
    if m.ndim == 3:
        m = m[..., 0]
    return float((m > 0).mean())


def _sample_windows(H, W, sz=576, K=8):
    if H < sz or W < sz:
        return []
    rng = np.random.default_rng(42)
    ys = rng.integers(0, H - sz + 1, size=K)
    xs = rng.integers(0, W - sz + 1, size=K)
    return [(int(y), int(x), sz, sz) for y, x in zip(ys, xs)]


def extract_and_save_subfolder(configs, list_matching_subfolder: List[str]):
    all_processed_subfolders = []

    # --- small local helpers ---
    for subfolder in tqdm(list_matching_subfolder, desc="Processing subfolders"):
        # get the masks, rgbs and metadata paths
        mask_path = os.path.join(configs.dataset.paths["PATH_MASKS"], subfolder)
        rgb_path = os.path.join(configs.dataset.paths["PATH_RGB"], subfolder)
        metadata_path = os.path.join(configs.dataset.paths["PATH_METADATA"], subfolder)

        if not (os.path.exists(mask_path) and os.path.exists(rgb_path) and os.path.exists(metadata_path)):
            print(f"Skipping {subfolder} — missing one of: {mask_path}, {rgb_path}, {metadata_path}")
            continue

        # gather indices present in all three modalities
        rgb_pngs  = _listdir_sorted(rgb_path, ".png")
        mask_pngs = _listdir_sorted(mask_path, ".png")
        meta_jsons= _listdir_sorted(metadata_path, ".json")
        common_endings = [str(i).strip() for i in _common_indices(mask_pngs, rgb_pngs, meta_jsons)]
        if len(common_endings) < 2:
            print(f"Skipping {subfolder} — not enough common indices across modalities.")
            continue

        # build per-index meta/geo/bbox/date maps (used by filters)
        meta_map, geo_map, bbox_map, dt_map = {}, {}, {}, {}
        for idx in common_endings:
            met_path = os.path.join(metadata_path, name_for_meta(idx, subfolder))  # simple helper
            meta = _load_meta(met_path)
            meta_map[idx] = met_path
            geo_map[idx]  = _compute_view_geometry(meta)
            bbox_map[idx] = _bbox_from_geojson(meta)
            dt_map[idx]   = _acq_datetime(meta)

        # determine a safe image size for sampling windows
        # use min H/W across masks (in case of slight differences)
        H = min(read_png_image(os.path.join(mask_path, name_for_mask(idx, subfolder))).shape[0] for idx in common_endings)
        W = min(read_png_image(os.path.join(mask_path, name_for_mask(idx, subfolder))).shape[1] for idx in common_endings)

        K     = int(getattr(configs.dataset, "CROPS_PER_SCENE"))
        CROP  = int(getattr(configs.dataset, "CROP_SIZE"))
        min_shadow_diff_tresh = float(getattr(configs.dataset.filters, "MIN_SHADOW_FRAC"))
        windows = _sample_windows(H, W, sz=CROP, K=K)  # this is a list!

        paths_data_samples_all = []

        added = False
        for (y, x, h, w) in windows:
            # pick the least-shadowed target FOR THIS WINDOW
            crop_shadow = {}
            for idx in common_endings:
                m = read_png_image(os.path.join(mask_path, name_for_mask(idx, subfolder)))
                crop_shadow[idx] = _shadow_frac_crop(m, y, x, sz=CROP)
            target_idx = max(crop_shadow, key=crop_shadow.get)

            # pair target with other dates that pass filters AND differ in shadows in THIS crop
            for input_idx in common_endings:
                if input_idx == target_idx:
                    continue

                # restrict candidates to within 20 days of the reference acquisition date
                # We do a quick metadata read to check the date window before heavier calculations.
                met_cand_quick = os.path.join(metadata_path, name_for_meta(input_idx, subfolder))
                meta_c_quick = _load_meta(met_cand_quick)
                cand_dt = _acq_datetime(meta_c_quick)
                d = abs((cand_dt - dt_map[target_idx]).days)
                diff_days = min(d, 365 - d)  # proper wrap-around

                # in the winter months, be more selective
                if dt_map[target_idx].month in [11, 12, 1, 2]:
                    if diff_days > int(configs.dataset.filters["DELTA_DAYS_WINTER"]):
                        continue

                if cand_dt.month in [11, 12, 1, 2]:
                    if diff_days > int(configs.dataset.filters["DELTA_DAYS_WINTER"]):
                        continue
                    
                if diff_days > int(configs.dataset.filters["DELTA_DAYS"]):
                    continue

                # crop-wise shadow difference, to discard scenes with too few shadows
                m_in = read_png_image(os.path.join(mask_path, name_for_mask(input_idx,  subfolder)))
                m_tg = read_png_image(os.path.join(mask_path, name_for_mask(target_idx, subfolder)))
                if m_in.ndim == 3: 
                    m_in = m_in[..., 0]
                if m_tg.ndim == 3: 
                    m_tg = m_tg[..., 0]
                a = (m_in[y:y+CROP, x:x+CROP] > 0)
                b = (m_tg[y:y+CROP, x:x+CROP] > 0)
                shadow_diff_fraction = float((a != b).mean())
                if shadow_diff_fraction < min_shadow_diff_tresh:
                    continue
                
                # filters based on orbital parameters
                geo_ref  = geo_map.get(target_idx);   bbox_ref = bbox_map.get(target_idx)
                geo_cand = geo_map.get(input_idx);    bbox_c   = bbox_map.get(input_idx)
                if any(v is None for v in (geo_ref, geo_cand, bbox_ref, bbox_c)):
                    continue

                iou = _bbox_iou(bbox_ref, bbox_c)
                nodata = _nodata_fraction_rgb(os.path.join(rgb_path, name_for_rgb(input_idx, subfolder)))  # simple helper
                if not _passes_filters(configs, geo_ref, geo_cand, iou, nodata):
                    continue  # keep your thresholds/logic :contentReference[oaicite:7]{index=7}

                # optional DEM per scene
                dem_dir  = Path(configs.dataset.paths.get("DEM_TIF_DIR")) / subfolder
                dem_path = None
                if dem_dir.exists():
                    dem_path = get_dem_scene(dem_dir)

                # write manifest entry expected by DeSEODataset (plus crop info)
                paths_data_samples_all.append({
                    "input_mask":  os.path.join(mask_path, name_for_mask(input_idx,  subfolder)),
                    "target_mask": os.path.join(mask_path, name_for_mask(target_idx, subfolder)),
                    "input_rgb":   os.path.join(rgb_path,  name_for_rgb(input_idx,   subfolder)),
                    "target_rgb":  os.path.join(rgb_path,  name_for_rgb(target_idx,  subfolder)),
                    "input_metadata":  meta_map[input_idx],
                    "target_metadata": meta_map[target_idx],
                    "crop": {"y": int(y), "x": int(x), "h": int(CROP), "w": int(CROP)},
                    "dem_path": str(dem_path) if dem_path else None,
                    "scene": subfolder,
                    "target_idx": str(target_idx),
                    "input_idx":  str(input_idx),
                })

                break  # only one input per target

        if not paths_data_samples_all:
            print(f"[INFO] {subfolder}: no valid crop-wise pairs — skipping YAML.")
            continue

        # persist to <PATH_myDataset>/<subfolder>/paths.yaml
        save_samples_to_yaml(paths_data_samples_all, configs.dataset.paths["PATH_myDataset"], subfolder)
        print(f"[OK] {subfolder}: wrote a crop-wise pairs.")
        all_processed_subfolders.append(subfolder)

    # split into train/val/test (paths-only manifests)
    if all_processed_subfolders:
        out_root = os.path.join(configs.dataset.paths["PATH_myDataset"], "splits")
        split_dataset_paths(
            locations=[os.path.join(configs.dataset.paths["PATH_myDataset"], sf) for sf in all_processed_subfolders],
            out_root=out_root, train_p=0.7, val_p=0.15, test_p=0.15, seed=42, materialize="manifest"
        )


def build_dataset(path_masks, path_rgb, path_metadata, configs=None):
    if configs.dataset["NAME"] == "deSEO":
        matching_folders = find_matching_subfolders(path_masks, path_rgb)
        report_matching_subfolders(path_masks, matching_folders)
        matching_folders = find_matching_subfolders(matching_folders, path_metadata)
        report_matching_subfolders(path_masks, matching_folders)
        extract_and_save_subfolder(configs, matching_folders)
    elif configs.dataset["NAME"] == "UAV_SC":
        # For UAV-SC, we assume all data is already organized in PATH_myDataset as required
        print(f"[INFO] Dataset {configs.dataset['NAME']} assumed to be pre-organized in {configs.dataset.paths['PATH_myDataset']}. No action taken.")
    else:
        raise ValueError(f"Unsupported dataset name: {configs.dataset['NAME']}")


def find_matching_subfolders(
    path_or_list1: Union[str, Sequence[str]],
    path_or_list2: Union[str, Sequence[str]]
) -> List[str]:
    def to_set(x: Union[str, Sequence[str]]) -> set:
        if not isinstance(x, str):
            return set(x)
        return {
            name
            for name in os.listdir(x)
            if os.path.isdir(os.path.join(x, name))
        }
    sub1 = to_set(path_or_list1)
    sub2 = to_set(path_or_list2)
    common = sub1 & sub2
    return sorted(common)


def _nodata_fraction_rgb(path: str) -> float:
    try:
        img = read_png_image(path) if path.lower().endswith(".png") else read_tiff_image(path)
        if img.ndim == 3 and img.shape[2] == 4:  # RGBA
            alpha = img[..., 3]
            return float((alpha == 0).mean())
        # Fallback: if float dtype, treat NaNs as NoData
        if np.issubdtype(img.dtype, np.floating):
            return float(np.isnan(img).mean())
        # Otherwise unknown; assume none
        return 0.0
    except Exception:
        return 0.0


def _common_indices(mask_pngs: List[str], rgb_pngs: List[str], meta_jsons: List[str]) -> List[str]:
    m = {_idx_from_name(p) for p in mask_pngs}
    r = {_idx_from_name(p) for p in rgb_pngs}
    j = {_idx_from_name(p) for p in meta_jsons}
    common = sorted(m & r & j, key=lambda x: int(x))
    return common


def _passes_filters(configs, geo_ref: Dict[str,float], geo_cand: Dict[str,float], bbox_iou: float, nodata_frac: float) -> bool:
    if bbox_iou < float(configs.dataset.filters["MIN_BBOX_IOU"]): 
        print(f"[FILTER] IoU {bbox_iou:.2f} < {float(configs.dataset.filters['MIN_BBOX_IOU']):.2f} for candidate.")
        return False
    if abs(geo_ref["off_nadir_deg"] - geo_cand["off_nadir_deg"]) > float(configs.dataset.filters["MAX_DELTA_OFF_NADIR_DEG"]): 
        print(f"[FILTER] Off-nadir candidate {geo_cand['off_nadir_deg']:.2f}° ")
        print(f" vs ref {geo_ref['off_nadir_deg']:.2f}° ")
        return False
    if _angular_diff_deg(geo_ref["look_azimuth_deg"], geo_cand["look_azimuth_deg"]) > float(configs.dataset.filters["MAX_DELTA_AZIMUTH_DEG"]): 
        print(f"[FILTER] Look azimuth {geo_cand['look_azimuth_deg']:.2f}° ")
        print(f" vs ref {geo_ref['look_azimuth_deg']:.2f}° ")
        return False
    #if _angular_diff_deg(geo_ref["sun_azimuth_deg"], geo_cand["sun_azimuth_deg"]) > MAX_DELTA_SUN_AZ_DEG: 
    #    print(f"[FILTER] Sun azimuth {geo_cand['sun_azimuth_deg']:.2f}° ")
    #    return False  # (disabled for now)
    if abs(geo_ref["sun_elevation_deg"] - geo_cand["sun_elevation_deg"]) > float(configs.dataset.filters["MAX_DELTA_SUN_EL_DEG"]): 
        print(f"[FILTER] Sun elevation {geo_cand['sun_elevation_deg']:.2f}° ")
        print(f" vs ref {geo_ref['sun_elevation_deg']:.2f}° ")
        return False
    if nodata_frac > float(configs.dataset.filters["MAX_NODATA_FRAC"]): 
        print(f"[FILTER] NoData fraction {nodata_frac:.2f} > {float(configs.dataset.filters['MAX_NODATA_FRAC']):.2f} for candidate.")
        print(f" vs ref {nodata_frac:.2f} ")
        return False
    return True


def name_for_rgb(idx: str, subfolder) -> str:
    return f"{subfolder}_{idx}_rgb.png"


def name_for_mask(idx: str, subfolder) -> str:
    return f"{subfolder}_{idx}_shadow.png"


def name_for_meta(idx: str, subfolder) -> str:
    return f"{subfolder}_{idx}_pan.json"

    return os.path.join(metadata_path, name_for_meta(idx, subfolder))