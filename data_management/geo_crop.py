import json
import numpy as np
import imageio
from typing import List, Tuple, Dict, Optional
from osgeo import gdal, osr
from math import floor, ceil
from .rpc_correction import detect_orientation_from_rpc


# GeoJSON / corners utilities
def get_coordinates_bbox(geojson_path: str):
    """Return the exterior ring (list of [lon, lat]) from a GeoJSON file."""
    with open(geojson_path) as f:
        meta = json.load(f)
    ring = meta['geojson']['coordinates'][0]
    return ring


def ring_to_corners(ring: List[List[float]]) -> Dict[str, Tuple[float, float]]:
    # read the first four points as corners
    if len(ring) < 4:
        raise ValueError("GeoJSON ring must have at least 4 points for corners")
    corners = {
        'TL': (ring[0][0], ring[0][1]),  # Top Left
        'TR': (ring[1][0], ring[1][1]),  # Top Right
        'BR': (ring[2][0], ring[2][1]),  # Bottom Right
        'BL': (ring[3][0], ring[3][1])   # Bottom Left
    }
    return corners


def vflip_corners(C: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Swap top↔bottom labels for a vertical flip while preserving clockwise order."""
    TL, TR, BR, BL = C['TL'], C['TR'], C['BR'], C['BL']
    return {'TL': BL, 'TR': BR, 'BR': TR, 'BL': TL}


# GeoTransform helpers
def _invert_gt(gt):
    """
    Return the inverted geotransform as a 6-tuple, compatible across GDAL versions.
    In some GDAL versions, InvGeoTransform returns (ok, inv_gt); in others it returns inv_gt.
    """
    res = gdal.InvGeoTransform(gt)
    # Newer GDAL: returns inv_gt directly (a 6-tuple)
    if isinstance(res, (list, tuple)) and len(res) == 6 and not isinstance(res[0], (bool, int)):
        return tuple(res)
    # Older GDAL: returns (ok, inv_gt)
    if isinstance(res, (list, tuple)) and len(res) == 2:
        ok, inv_gt = res
        if not ok:
            raise RuntimeError("Cannot invert geotransform")
        return tuple(inv_gt)
    # Last resort: treat as direct 6-tuple
    return tuple(res)


def build_gt(C: Dict[str, Tuple[float, float]], rows: int, cols: int) -> tuple:
    """
    Build a north-up GeoTransform from ordered corners C and image size (rows, cols).
    C keys: 'TL','TR','BR','BL'; values are (lon, lat) tuples.
    Returns (originX, pixelWidth, rotX, originY, rotY, pixelHeight).
    """
    TL, TR, BR, BL = C['TL'], C['TR'], C['BR'], C['BL']

    # Compute pixel resolution (center-to-center spacing)
    xres = (TR[0] - TL[0]) / float(max(cols - 1, 1))
    yres = (TL[1] - BL[1]) / float(max(rows - 1, 1))

    # Shift from center coords to corner origin (GDAL expects UL corner)
    originX = TL[0] - 0.5 * xres
    originY = TL[1] + 0.5 * yres  # pixelHeight will be negative below

    return (originX, xres, 0.0, originY, 0.0, -yres)


# Pixel-window computation
def _bbox_to_pixel_window(ds, left, bottom, right, top):
    gt = ds.GetGeoTransform()
    inv_gt = _invert_gt(gt)

    # project bbox to pixel/line
    ulx, uly = gdal.ApplyGeoTransform(inv_gt, left,  top)
    urx, ury = gdal.ApplyGeoTransform(inv_gt, right, top)
    lrx, lry = gdal.ApplyGeoTransform(inv_gt, right, bottom)
    llx, lly = gdal.ApplyGeoTransform(inv_gt, left,  bottom)

    xs = [ulx, urx, lrx, llx]
    ys = [uly, ury, lry, lly]

    x_min = int(np.floor(min(xs)))
    y_min = int(np.floor(min(ys)))
    x_max = int(np.ceil (max(xs)))
    y_max = int(np.ceil (max(ys)))

    # clamp to dataset bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(ds.RasterXSize, x_max)
    y_max = min(ds.RasterYSize, y_max)

    xsize = max(0, x_max - x_min)
    ysize = max(0, y_max - y_min)
    return x_min, y_min, xsize, ysize


# Core I/O
def open_png_as_raster(path_png: str, geojson_path: str) -> gdal.Dataset:
    """
    Open a PNG and its associated GeoJSON metadata, returning a GDAL in-memory dataset.
    Orientation is decided strictly from the RPC (ascending/descending), not from corner order.
    We enforce north-up by flipping/rotating as needed and relabeling corners accordingly.
    Additionally, we attach the TL/TR/BR/BL corner coordinates as:
      - GCPs (so they travel with the dataset),
      - and mirrored in the 'CORNERS' metadata domain as JSON.
    """
    # Load pixels
    arr = imageio.imread(path_png)
    # squeeze odd shapes but keep 2D or 3D
    if arr.ndim not in (2, 3):
        arr = np.squeeze(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {arr.shape} for {path_png}")

    # Normalize corners dict from GeoJSON ring
    ring = get_coordinates_bbox(geojson_path)
    corners = ring_to_corners(ring)  # dict with 'TL','TR','BR','BL' -> (lon, lat)

    # Decide orientation purely from RPC
    action = detect_orientation_from_rpc(geojson_path)
    print(f"Detected orientation action: {action}")

    # Apply orientation
    if action == 'vflip':
        arr = arr[::-1, ...]
        corners = vflip_corners(corners)
    elif action == 'hflip':
        arr = arr[:, ::-1, ...]
        corners = {'TL': corners['TR'], 'TR': corners['TL'],
                   'BR': corners['BL'], 'BL': corners['BR']}
    elif action == 'rot180':
        arr = arr[::-1, ::-1, ...]
        corners = relabel_corners(corners, 2)
    elif action == 'rot90ccw':
        arr = np.rot90(arr, 1)
        corners = relabel_corners(corners, 1)
    elif action == 'rot90cw':
        arr = np.rot90(arr, -1)
        corners = relabel_corners(corners, 3)
    elif action in (None, 'none'):
        pass
    else:
        print(f"Warning: unknown orientation action '{action}', leaving as-is.")

    # Recompute shape AFTER orientation
    if arr.ndim == 2:
        rows, cols, bands = arr.shape[0], arr.shape[1], 1
    else:
        rows, cols, bands = arr.shape[0], arr.shape[1], arr.shape[2]

    # some version of GDAL sometimes dislikes non-contiguous views
    arr = np.ascontiguousarray(arr)

    # Build north-up geotransform from  corners 
    gt = build_gt(corners, rows, cols)
    gdal_dtype = gdal.GDT_Byte

    # Create GDAL dataset (xsize = cols, ysize = rows)
    mem = gdal.GetDriverByName('MEM')
    ds = mem.Create('', cols, rows, bands, gdal_dtype)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    # Write pixel data
    if bands == 1:
        ds.GetRasterBand(1).WriteArray(arr if arr.ndim == 2 else arr[:, :, 0])
    else:
        for i in range(bands):
            ds.GetRasterBand(i + 1).WriteArray(arr[:, :, i])

    # Attach corner coordinates as GCPs and metadata
    # Use half-pixel convention: pixel centers at (0.5, 0.5), (cols-0.5, 0.5), ...
    pix_corners = {
        'TL': (0.5, 0.5),
        'TR': (cols - 0.5, 0.5),
        'BR': (cols - 0.5, rows - 0.5),
        'BL': (0.5, rows - 0.5),
    }
    gcps = []
    for key in ('TL', 'TR', 'BR', 'BL'):
        lon, lat = corners[key]
        px, py = pix_corners[key]
        g = gdal.GCP()
        g.GCPX = float(lon)   # longitude / easting
        g.GCPY = float(lat)   # latitude  / northing
        g.GCPZ = 0.0
        g.GCPPixel = float(px)
        g.GCPLine  = float(py)
        g.Id = key
        gcps.append(g)

    gcp_srs = osr.SpatialReference(); gcp_srs.ImportFromEPSG(4326)
    ds.SetGCPs(gcps, gcp_srs.ExportToWkt())

    # Also stash JSON copy
    ds.SetMetadata({
        'corners_json': json.dumps(corners),   # TL/TR/BR/BL dict as JSON
        'corner_order': 'TL,TR,BR,BL',
        'pixel_corner_convention': 'centers at (0.5,0.5) ... (cols-0.5,rows-0.5)'
    }, 'CORNERS')

    return ds

# Bounds helpers
def bbox_from_gt(ds):
    gt = ds.GetGeoTransform()
    cols, rows = ds.RasterXSize, ds.RasterYSize
    # use pixel-corner convention: centers at .5 => corners at [-0.5, cols-0.5], [-0.5, rows-0.5]
    xs = [-0.5, cols - 0.5]
    ys = [-0.5, rows - 0.5]
    world = [gdal.ApplyGeoTransform(gt, x, y) for x in xs for y in ys]
    lons, lats = zip(*world)
    return (min(lons), min(lats), max(lons), max(lats))


# Common-grid alignment (resampling)
def _intersect_bounds(bounds_list: List[Tuple[float, float, float, float]]):
    left   = max(b[0] for b in bounds_list)
    bottom = max(b[1] for b in bounds_list)
    right  = min(b[2] for b in bounds_list)
    top    = min(b[3] for b in bounds_list)
    return (left, bottom, right, top)


def _snap_to_grid(bounds, px_w, px_h):
    """Align outputBounds to the pixel grid implied by (px_w, px_h)."""
    left, bottom, right, top = bounds
    # assume px_w > 0 (east), px_h < 0 (south) for north-up
    left   = np.floor(left  / px_w) * px_w
    right  = np.ceil (right / px_w) * px_w
    top    = np.ceil (top   / px_h) * px_h  # px_h negative -> ceil grows up
    bottom = np.floor(bottom/ px_h) * px_h  # px_h negative -> floor goes down
    return (left, bottom, right, top)


def build_template_grid(datasets: List[gdal.Dataset],
                        pixel_size: Optional[Tuple[float, float]] = None,
                        dst_srs_wkt: Optional[str] = None) -> Dict[str, object]:
    """
    Choose a common pixel size and world extent (intersection), aligned to grid.
    Returns dict with: outputBounds, xRes, yRes, width, height, dstSRS.
    Assumes all datasets are in the same SRS already unless dst_srs_wkt is provided.
    """
    # pick SRS (use first by default)
    if dst_srs_wkt is None:
        dst_srs_wkt = datasets[0].GetProjection()

    # collect bounds (same SRS)
    bounds_list = [bbox_from_gt(ds) for ds in datasets]
    inter = _intersect_bounds(bounds_list)
    if not (inter[0] < inter[2] and inter[1] < inter[3]):
        raise ValueError("No world-space overlap across inputs")

    # hoose pixel size, notice that default: median abs pixel size across inputs
    if pixel_size is None:
        px_ws = [abs(ds.GetGeoTransform()[1]) for ds in datasets]
        px_hs = [abs(ds.GetGeoTransform()[5]) for ds in datasets]
        px_w = float(np.median(px_ws))
        px_h = float(np.median(px_hs))
    else:
        px_w = float(pixel_size[0]); px_h = float(pixel_size[1])

    # enforce north-up convention
    px_h = -abs(px_h)
    px_w =  abs(px_w)

    # snap bounds to this grid and compute integer width/height
    out_bounds = _snap_to_grid(inter, px_w, px_h)
    left, bottom, right, top = out_bounds
    width  = int(round((right - left) / px_w))
    height = int(round((bottom - top) / px_h))  # px_h negative

    return {
        "outputBounds": out_bounds,
        "xRes": px_w,
        "yRes": abs(px_h),
        "width": width,
        "height": height,
        "dstSRS": dst_srs_wkt
    }


def warp_to_template(ds: gdal.Dataset,
                     template: Dict[str, object],
                     resample: str = "bilinear") -> gdal.Dataset:
    """
    Warp a GDAL dataset to the template grid (returns an in-memory dataset).
    resample: "nearest" for masks, "bilinear"/"cubic" for RGB.
    """
    alg = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic
    }[resample]

    out = gdal.Warp(
        destNameOrDestDS = "",
        srcDSOrSrcDSTab  = ds,
        format           = "MEM",
        outputBounds     = template["outputBounds"],
        xRes             = template["xRes"],
        yRes             = template["yRes"],
        dstSRS           = template["dstSRS"],
        width            = template["width"],
        height           = template["height"],
        resampleAlg      = alg,
        targetAlignedPixels = True,
        multithread      = True,
        errorThreshold   = 0.0
    )
    return out


def align_to_common_grid(rasters: List[gdal.Dataset],
                         pixel_size: Optional[Tuple[float, float]] = None,
                         dst_srs_wkt: Optional[str] = None,
                         resample_map: Optional[List[str]] = None) -> Tuple[List[gdal.Dataset], Dict[str, object]]:
    """
    Resample all rasters to a single common grid so that the same world bbox maps to the same
    (xsize, ysize) everywhere. Returns (aligned_rasters, template).

    resample_map: optional list parallel to `rasters` with values in {"nearest","bilinear","cubic"}.
                   If None, a heuristic is used: single-band or name contains 'mask'/'shadow' -> nearest, else bilinear.
    """
    template = build_template_grid(rasters, pixel_size=pixel_size, dst_srs_wkt=dst_srs_wkt)

    aligned = []
    for i, ds in enumerate(rasters):
        if resample_map is not None:
            method = resample_map[i]
        else:
            # Heuristic: treat masks as nearest
            meta = ds.GetMetadata() or {}
            src_path = meta.get('SRC_PATH', '').lower()
            is_mask = (ds.RasterCount == 1) or ('mask' in src_path) or ('shadow' in src_path)
            method = 'nearest' if is_mask else 'bilinear'
        aligned.append(warp_to_template(ds, template, resample=method))
    return aligned, template


# Cropping
def crop_geo_images(rasters: List[gdal.Dataset]):
    """Crop a list of GDAL datasets to their overlapping lon/lat bbox derived from GTs."""
    # get the gts bboxs
    corners = [bbox_from_gt(ds) for ds in rasters]
    # Compute intersection across all images
    left   = max(b[0] for b in corners)
    bottom = max(b[1] for b in corners)
    right  = min(b[2] for b in corners)
    top    = min(b[3] for b in corners)

    # No overlap? make a safe empty return to raise upstream
    if not (left < right and bottom < top):
        return []

    # Crop each dataset to the common bbox
    out = []
    for ds in rasters:
        xoff, yoff, xsize, ysize = _bbox_to_pixel_window(ds, left, bottom, right, top)
        print(f"Cropping ds to xoff={xoff}, yoff={yoff}, xsize={xsize}, ysize={ysize}")
        if xsize == 0 or ysize == 0:
            continue

        num_bands = ds.RasterCount
        driver = gdal.GetDriverByName('MEM')
        out_ds = driver.Create('', xsize, ysize, num_bands, ds.GetRasterBand(1).DataType)

        # shift the origin using forward GT
        gt = ds.GetGeoTransform()
        ulx_world, uly_world = gdal.ApplyGeoTransform(gt, xoff, yoff)
        new_gt = (ulx_world, gt[1], gt[2], uly_world, gt[4], gt[5])
        out_ds.SetGeoTransform(new_gt)
        out_ds.SetProjection(ds.GetProjection())

        for b in range(1, num_bands + 1):
            data = ds.GetRasterBand(b).ReadAsArray(xoff, yoff, xsize, ysize)
            out_ds.GetRasterBand(b).WriteArray(data)

        out.append(out_ds)

    return out


def align_and_crop_geo_images(rasters: List[gdal.Dataset],
                              pixel_size: Optional[Tuple[float, float]] = None,
                              dst_srs_wkt: Optional[str] = None,
                              resample_map: Optional[List[str]] = None,
                              skip_extra_crop: bool = False) -> List[gdal.Dataset]:
    """
    Convenience wrapper: first align all rasters to a common grid (resampling), then crop them to
    their common intersection. If `skip_extra_crop` is True, returns the aligned rasters directly
    (they already share the same extent and grid defined by the template intersection).
    """
    aligned, template = align_to_common_grid(rasters, pixel_size=pixel_size, dst_srs_wkt=dst_srs_wkt, resample_map=resample_map)
    if skip_extra_crop:
        return aligned
    # crop using the template bounds for perfect parity
    left, bottom, right, top = template["outputBounds"]
    out = []
    for ds in aligned:
        xoff, yoff, xsize, ysize = _bbox_to_pixel_window(ds, left, bottom, right, top)
        print(f"[aligned] Cropping ds to xoff={xoff}, yoff={yoff}, xsize={xsize}, ysize={ysize}")
        driver = gdal.GetDriverByName('MEM')
        out_ds = driver.Create('', xsize, ysize, ds.RasterCount, ds.GetRasterBand(1).DataType)
        gt = ds.GetGeoTransform()
        ulx_world, uly_world = gdal.ApplyGeoTransform(gt, xoff, yoff)
        new_gt = (ulx_world, gt[1], gt[2], uly_world, gt[4], gt[5])
        out_ds.SetGeoTransform(new_gt)
        out_ds.SetProjection(ds.GetProjection())
        for b in range(1, ds.RasterCount + 1):
            data = ds.GetRasterBand(b).ReadAsArray(xoff, yoff, xsize, ysize)
            out_ds.GetRasterBand(b).WriteArray(data)
        out.append(out_ds)
    return out


# Corner relabeling helper
def relabel_corners(corners: Dict[str, Tuple[float, float]], rotation: int) -> Dict[str, Tuple[float, float]]:
    """
    Relabel corners in clockwise order based on the rotation:
    0 - no change, 1 - rotate 90° CCW, 2 - rotate 180°, 3 - rotate 90° CW.
    """
    if rotation == 0:
        return corners
    elif rotation == 1:
        return {'TL': corners['BL'], 'TR': corners['TL'], 'BR': corners['TR'], 'BL': corners['BR']}
    elif rotation == 2:
        return {'TL': corners['BR'], 'TR': corners['BL'], 'BR': corners['TL'], 'BL': corners['TR']}
    elif rotation == 3:
        return {'TL': corners['TR'], 'TR': corners['BR'], 'BR': corners['BL'], 'BL': corners['TL']}
    else:
        raise ValueError("Rotation must be one of: 0, 1, 2, or 3.")
