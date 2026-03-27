# Support functions for dataset creation and filtering
import os, json, math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_descriptor')  # avoids POSIX semaphores
from datetime import datetime, timezone
import glob
from pathlib import Path
from datetime import datetime, timezone
import re
from typing import Any, Dict, Optional

def scene_from_path(p):
    """Return the scene folder name, e.g. 'JAX_225'."""
    p = Path(p)
    return Path(p).parent.name


def scene_idx_from_path(p):
    """
    Prefer digits in the scene folder name (e.g. 'JAX_225' -> '225').
    If not present, try to read a leading integer after '<scene>_' in the basename.
    Finally, fall back to the first integer in the basename that is not a 4-digit year.
    """
    p = Path(p)
    scene = p.parent.name
    m = re.search(r'(\d+)$', scene)
    if m:
        return m.group(1)

    base = p.stem
    if base.startswith(scene + "_"):
        rest = base[len(scene) + 1:]
        m2 = re.match(r'(\d+)', rest)
        if m2:
            return m2.group(1)

    nums = re.findall(r'\d+', base)
    for g in nums:
        v = int(g)
        if v < 1900 or v > 2100:  # avoid years like 2018
            return g
    return nums[0] if nums else "0"


def _acq_datetime(meta: Dict[str, Any]) -> Optional[datetime]:
    """
    Try to extract an acquisition datetime from various common fields / formats.
    Returns timezone-aware UTC datetime if possible, otherwise None.
    Supports compact vendor formats like 'YYYYMMDDHHMMSS' (e.g., '20150925163525').
    """
    # Common keys to look for
    candidates = [
        meta.get("acq_datetime"),
        meta.get("acquisition_datetime"),
        meta.get("acquisition_date"),   # <- your example lives here
        meta.get("datetime"),
        meta.get("timestamp"),
        meta.get("date"),
        meta.get("imaging_time"),
        meta.get("imaging_datetime"),
        meta.get("properties", {}).get("datetime"),
        meta.get("properties", {}).get("acquisitionDate"),
        meta.get("properties", {}).get("timestamp"),
    ]
    for v in candidates:
        dt = _try_parse(v)
        if dt is not None:
            return dt
    return None


def _try_parse(s: Any) -> Optional[datetime]:
    if s is None:
        return None

    # UNIX timestamp (seconds or ms)
    if isinstance(s, (int, float)):
        if s > 1e12:  # likely milliseconds
            s = s / 1000.0
        return datetime.fromtimestamp(float(s), tz=timezone.utc)

    if isinstance(s, str):
        s = s.strip()

        # Normalize trailing Z to an explicit UTC offset for fromisoformat
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        # ISO-8601 (with or without TZ)
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

        # Compact numeric vendor formats:
        # YYYYMMDDHHMMSS[fff...], YYYYMMDDHHMM, or just YYYYMMDD
        m = re.fullmatch(r"(\d{8})(\d{2})?(\d{2})?(\d{2})?(\d+)?", s)
        if m:
            ymd, hh, mm, ss, frac = m.groups()
            try:
                year  = int(ymd[0:4])
                month = int(ymd[4:6])
                day   = int(ymd[6:8])
                hour  = int(hh) if hh is not None else 0
                minute= int(mm) if mm is not None else 0
                sec   = int(ss) if ss is not None else 0

                # Fractional seconds (milliseconds/microseconds), normalize to microseconds
                micro = 0
                if frac:
                    # keep up to 6 digits (microseconds), right-pad if shorter
                    f = (frac + "000000")[:6]
                    micro = int(f)

                return datetime(year, month, day, hour, minute, sec, micro, tzinfo=timezone.utc)
            except Exception:
                pass

        # Date-only with separators
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                d = datetime.strptime(s, fmt)
                return d.replace(tzinfo=timezone.utc)
            except Exception:
                continue

    return None


def _listdir_sorted(path: str, ext: str) -> List[str]:
    return sorted([f for f in os.listdir(path) if f.endswith(ext)])


def _idx_from_name(name: str) -> str:
    """
    Robustly extract an index from a filename by taking the last integer group.
    Works with many patterns (IMG_000123.png, frame123.tif, scene_12_mask.png).
    """
    base = os.path.splitext(os.path.basename(name))[0]
    m = re.findall(r"\d+", base)
    if not m:
        raise ValueError(f"Could not find an integer index in filename: {name}")
    return m[-1]  # last group is the index


def _angular_diff_deg(a: float, b: float) -> float:
    d = abs((a - b + 180) % 360 - 180)
    return float(d)


def _bbox_from_geojson(meta: Dict[str, Any]) -> Optional[Tuple[float,float,float,float]]:
    try:
        coords = meta["geojson"]["coordinates"][0]
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        return (min(lons), min(lats), max(lons), max(lats))  # (xmin, ymin, xmax, ymax)
    except Exception:
        return None


def _bbox_iou(b1, b2) -> float:
    if b1 is None or b2 is None:
        return 1.0  # If missing, do not penalize
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter if (a1 + a2 - inter) > 0 else 1e-9
    return float(inter / union)


def _invert_rpc(row, col, alt, rpc, lat0=None, lon0=None, max_iter=30, tol=1e-8):
    lat = rpc["lat_offset"] if lat0 is None else float(lat0)
    lon = rpc["lon_offset"] if lon0 is None else float(lon0)
    for _ in range(max_iter):
        r, c = _forward_rpc(lat, lon, alt, rpc)
        fr, fc = r - row, c - col
        if max(abs(fr), abs(fc)) < tol:
            break
        d = 1e-6
        r1, c1 = _forward_rpc(lat + d, lon, alt, rpc)
        r2, c2 = _forward_rpc(lat, lon + d, alt, rpc)
        J = np.array([[r1 - r, r2 - r], [c1 - c, c2 - c]], dtype=float) / d
        try:
            delta = np.linalg.solve(J, np.array([fr, fc], dtype=float))
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(J, np.array([fr, fc], dtype=float), rcond=None)
        lat -= float(delta[0]); lon -= float(delta[1])
        if max(abs(delta[0]), abs(delta[1])) < 1e-10:
            break
    return lat, lon


def _forward_rpc(lat, lon, alt, rpc):
    Ln = (lat - rpc["lat_offset"]) / rpc["lat_scale"]
    Pn = (lon - rpc["lon_offset"]) / rpc["lon_scale"]
    Hn = (alt - rpc["alt_offset"]) / rpc["alt_scale"]
    T = _rpc_terms(Ln, Pn, Hn)
    row = rpc["row_offset"] + rpc["row_scale"] * (np.dot(rpc["row_num"], T) / np.dot(rpc["row_den"], T))
    col = rpc["col_offset"] + rpc["col_scale"] * (np.dot(rpc["col_num"], T) / np.dot(rpc["col_den"], T))
    return float(row), float(col)


def _rpc_terms(L: float, P: float, H: float) -> np.ndarray:
    return np.array([
        1.0,
        L, P, H,
        L*P, L*H, P*H,
        L*L, P*P, H*H,
        L*L*L,
        (L*L)*P, (L*L)*H,
        L*(P*P), P*P*P, (P*P)*H,
        L*(H*H), P*(H*H), H*H*H,
        L*P*H
    ], dtype=float)


def _compute_view_geometry(meta: Dict[str, Any]) -> Dict[str, float]:
    rpc = meta["rpc"]
    row_c, col_c = rpc["row_offset"], rpc["col_offset"]
    lat0, lon0 = meta["geojson"]["center"][1], meta["geojson"]["center"][0]
    alt1 = rpc["alt_offset"] - rpc["alt_scale"]
    alt2 = rpc["alt_offset"] + rpc["alt_scale"]
    lat1, lon1 = _invert_rpc(row_c, col_c, alt1, rpc, lat0=lat0, lon0=lon0)
    lat2, lon2 = _invert_rpc(row_c, col_c, alt2, rpc, lat0=lat0, lon0=lon0)
    lat_mean = (lat1 + lat2) / 2.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_mean))
    dx = (lon2 - lon1) * m_per_deg_lon
    dy = (lat2 - lat1) * m_per_deg_lat
    dz = alt2 - alt1
    horiz = math.hypot(dx, dy)
    off_nadir = math.degrees(math.atan2(horiz, abs(dz)))
    az = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    sun_el = float(meta.get("sun_elevation", 0) or 0)
    sun_az = float(meta.get("sun_azimuth", 0) or 0)
    return {
        "off_nadir_deg": float(off_nadir),
        "look_azimuth_deg": float(az),
        "sun_elevation_deg": sun_el,
        "sun_azimuth_deg": sun_az
    }


def _load_meta(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    # clean up numeric strings
    for k in ("sun_elevation", "sun_azimuth"):
        if k in meta and isinstance(meta[k], str):
            try:
                meta[k] = float(meta[k].replace("+",""))
            except Exception:
                pass
    return meta


def exp_term(delta, sigma):
    return math.exp(-abs(delta)/sigma)


def get_dem_scene(dem_dir: Path) -> str:
    files = glob.glob(f"{dem_dir}/*.tif")
    if len(files) == 0:
        files = glob.glob(f"{dem_dir}/*.png")
    if len(files) == 0:
        raise ValueError(f"No DEM files found in {dem_dir}")
    if len(files) > 1:
        raise ValueError(f"Multiple DEM files found in {dem_dir}: {files}")
    return files[0]


def read_dem_arr(ds):
    """Read DEM as float32 (1, H, W)."""
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    return arr[None, ...]
