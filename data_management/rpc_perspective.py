# data_management/rpc_perspective.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union, Optional

import json
import numpy as np
from osgeo import gdal, osr  # type: ignore

__all__ = ["warp_mask_ortho_to_sensor"]

# --------------------------- RPC forward model ---------------------------

def _forward_rpc_vec(lat: np.ndarray,
                     lon: np.ndarray,
                     alt: np.ndarray,
                     rpc: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized RPC forward projection: (lat, lon, alt) -> (row, col)."""
    # Normalize
    Ln = (lat - float(rpc["lat_offset"])) / float(rpc["lat_scale"])
    Pn = (lon - float(rpc["lon_offset"])) / float(rpc["lon_scale"])
    Hn = (alt - float(rpc["alt_offset"])) / float(rpc["alt_scale"])

    # 20 basis terms, shape (20, N)
    T = np.vstack([
        np.ones_like(Ln),
        Ln, Pn, Hn,
        Ln * Pn, Ln * Hn, Pn * Hn,
        Ln * Ln, Pn * Pn, Hn * Hn,
        Ln * Ln * Ln,
        (Ln * Ln) * Pn, (Ln * Ln) * Hn,
        Ln * (Pn * Pn), Pn * Pn * Pn, (Pn * Pn) * Hn,
        Ln * (Hn * Hn), Pn * (Hn * Hn), Hn * Hn * Hn,
        Ln * Pn * Hn,
    ])  # (20, N)

    row_num = np.asarray(rpc["row_num"], dtype=float)
    row_den = np.asarray(rpc["row_den"], dtype=float)
    col_num = np.asarray(rpc["col_num"], dtype=float)
    col_den = np.asarray(rpc["col_den"], dtype=float)

    row_r = row_num @ T
    row_d = row_den @ T
    col_r = col_num @ T
    col_d = col_den @ T

    rows = float(rpc["row_offset"]) + float(rpc["row_scale"]) * (row_r / row_d)
    cols = float(rpc["col_offset"]) + float(rpc["col_scale"]) * (col_r / col_d)
    return rows, cols


# ------------------------ Geo helpers & DSM sampling ------------------------

def _ensure_same_crs(
    ds: gdal.Dataset,
    *,
    epsg: int = 4326,
    assume_epsg_if_missing: bool = True,
    resample_alg = gdal.GRA_NearestNeighbour,  # good for masks
) -> gdal.Dataset:
    """
    Return a dataset in EPSG:<epsg>. If the source has no CRS:
      - attach EPSG:<epsg> if assume_epsg_if_missing=True
      - otherwise raise RuntimeError.
    For masks, nearest-neighbour resampling is the right default.
    """
    src_wkt = ds.GetProjection()  # GetProjectionRef() is deprecated
    if not src_wkt:
        if not assume_epsg_if_missing:
            raise RuntimeError("Dataset has no projection information")
        # Attach EPSG and return an in-memory clone
        return gdal.Translate(
            "", ds, format="MEM", outputSRS=f"EPSG:{epsg}"
        )

    src_sr = osr.SpatialReference(); src_sr.ImportFromWkt(src_wkt)
    tgt_sr = osr.SpatialReference(); tgt_sr.ImportFromEPSG(epsg)

    # Already in target CRS? return as-is
    if src_sr.IsSame(tgt_sr):
        return ds

    # Preserve (src,dst) nodata if present
    b1 = ds.GetRasterBand(1)
    nodata = b1.GetNoDataValue() if b1 is not None else None

    warp_opts = gdal.WarpOptions(
        format="MEM",
        dstSRS=tgt_sr.ExportToWkt(),
        resampleAlg=resample_alg,
        errorThreshold=0.0,
        multithread=True,
        srcNodata=nodata,
        dstNodata=nodata,
    )
    out = gdal.Warp("", ds, options=warp_opts)
    if out is None:
        raise RuntimeError("gdal.Warp failed")
    return out


def _sample_dsm_heights_at_fg(xs: np.ndarray,
                              ys: np.ndarray,
                              mask_gt: Tuple[float, float, float, float, float, float],
                              dsm_ds: gdal.Dataset) -> np.ndarray:
    """
    Sample heights from `dsm_ds` at lon/lat positions defined by (xs, ys) indices
    of the mask and its geotransform `mask_gt`. All in EPSG:4326.
    """
    dsm_ds = _ensure_same_crs(dsm_ds)

    # mask pixel centers -> lon/lat (EPSG:4326)
    lon = mask_gt[0] + (xs + 0.5) * mask_gt[1] + (ys + 0.5) * mask_gt[2]
    lat = mask_gt[3] + (xs + 0.5) * mask_gt[4] + (ys + 0.5) * mask_gt[5]

    inv = gdal.InvGeoTransform(dsm_ds.GetGeoTransform())

    # World -> DSM pixel indices (column=x, row=y)
    dc = inv[0] + lon * inv[1] + lat * inv[2]
    dr = inv[3] + lon * inv[4] + lat * inv[5]

    dc_i = np.clip(np.rint(dc).astype(np.int64), 0, dsm_ds.RasterXSize - 1)
    dr_i = np.clip(np.rint(dr).astype(np.int64), 0, dsm_ds.RasterYSize - 1)

    dsm = dsm_ds.GetRasterBand(1).ReadAsArray()
    if dsm is None:
        raise RuntimeError("Failed to read DSM band data.")

    return dsm[dr_i, dc_i].astype(np.float32)


def _sensor_size_from_meta(meta: dict) -> Tuple[int, int]:
    """Return (H, W) from metadata JSON; must exist."""
    H = int(round(meta.get("height", 0)))
    W = int(round(meta.get("width", 0)))
    return H, W

@lru_cache(maxsize=32)
def _disk_offsets(radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute integer (dy, dx) offsets for a filled disk of given radius.
    Includes center (0,0).
    """
    r = int(max(0, radius))

    coords: List[Tuple[int, int]] = []
    r2 = r * r
    for dy in range(-r, r + 1):
        max_dx = int(np.floor(np.sqrt(r2 - dy * dy)))
        for dx in range(-max_dx, max_dx + 1):
            coords.append((dy, dx))

    arr = np.asarray(coords, dtype=np.int64)
    return arr[:, 0], arr[:, 1]


# --------------------------- Main API function ---------------------------

def warp_mask_ortho_to_sensor(
    mask_ds: gdal.Dataset,
    meta_json_path: Union[str, Path],
    dsm_ds_path: Optional[Union[str, Path]] = None,
    radius_px: int = 1,
    *,
    assume_mask_epsg4326_if_missing: bool = True,
) -> np.ndarray:
    """
    Forward-map + splat a **georeferenced** foreground mask into the sensor grid.

    Pipeline:
      1) Ensure `mask_ds` is in EPSG:4326 (optionally attach if missing).
      2) Extract centers of foreground pixels (>0).
      3) Heights from DSM (if provided) or RPC alt_offset fallback.
      4) RPC forward project (lat, lon, h) -> (row, col).
      5) Splat each hit into a filled disk of `radius_px`.

    Returns
    -------
    np.ndarray : (H, W) uint8 mask in sensor space with values {0, 255}.
    """
    # --- 1) Read meta (RPC + sensor size) ---
    meta_json_path = Path(meta_json_path)
    with meta_json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    Hs, Ws = _sensor_size_from_meta(meta)
    rpc = meta["rpc"]

    # --- 2) Ensure source mask is EPSG:4326 and read as boolean FG ---
    src = _ensure_same_crs(mask_ds)

    arr = src.ReadAsArray()
    if arr.ndim == 3:
        arr = arr[0]

    fg = arr > 0

    # --- 3) Build lon/lat arrays at pixel centers from GeoTransform ---
    gt = src.GetGeoTransform()  # (x0, px_w, rot_x, y0, rot_y, px_h)
    ys, xs = np.nonzero(fg)
    lon = gt[0] + (xs + 0.5) * gt[1] + (ys + 0.5) * gt[2]
    lat = gt[3] + (xs + 0.5) * gt[4] + (ys + 0.5) * gt[5]

    # --- 4) Heights: DSM or alt_offset fallback ---
    dsm = gdal.Open(str(dsm_ds_path), gdal.GA_ReadOnly)
    h = _sample_dsm_heights_at_fg(xs, ys, gt, dsm)

    # --- 5) RPC forward projection ---
    rows, cols = _forward_rpc_vec(
        lat.astype(np.float64),
        lon.astype(np.float64),
        h.astype(np.float64),
        rpc,
    )

    valid = np.isfinite(rows) & np.isfinite(cols)

    rr = np.rint(rows[valid]).astype(np.int64)
    cc = np.rint(cols[valid]).astype(np.int64)

    inb = (rr >= 0) & (rr < Hs) & (cc >= 0) & (cc < Ws)

    rr = rr[inb]
    cc = cc[inb]

    # Deduplicate seeds to avoid re-splatting same pixel
    if rr.size > 1:
        seeds = np.unique(np.stack((rr, cc), axis=1), axis=0)
        rr, cc = seeds[:, 0], seeds[:, 1]

    # --- 6) Splat ---
    out = np.zeros((Hs, Ws), dtype=np.uint8)
    r = int(max(0, radius_px))

    if r == 0:
        out[rr, cc] = 255
        return out

    dy, dx = _disk_offsets(r)
    for y0, x0 in zip(rr, cc):
        ys_off = y0 + dy
        xs_off = x0 + dx
        ib = (ys_off >= 0) & (ys_off < Hs) & (xs_off >= 0) & (xs_off < Ws)
        if np.any(ib):
            out[ys_off[ib], xs_off[ib]] = 255

    return out
