import json
import numpy as np
import numpy as np
from math import isfinite

def _rpc_poly(num, den, L, P, H):
    T = np.array([
        1.0, L, P, H, L*P, L*H, P*H, L*L, P*P, H*H,
        L*L*L, L*P*P, L*H*H, L*L*P, P*P*P, P*H*H, L*L*H, P*P*H, L*P*H, H*H*H
    ], dtype=float)
    num = np.asarray(num, float).reshape(-1)
    den = np.asarray(den, float).reshape(-1)
    if num.size != 20 or den.size != 20:
        raise ValueError("RPC coeff vectors must have 20 terms")
    numv = float(np.dot(num, T)); denv = float(np.dot(den, T))
    if not isfinite(numv) or not isfinite(denv) or denv == 0:
        raise ValueError("Non‑finite or zero denominator in RPC eval")
    return numv / denv

def _norm(val, off, sc):
    sc = float(sc)
    if sc == 0: raise ValueError("RPC scale cannot be zero")
    return (float(val) - float(off)) / sc

def _project_row_col(r, lon, lat, h):
    L = _norm(lon, r['lon_off'], r['lon_scale'])
    P = _norm(lat, r['lat_off'], r['lat_scale'])
    H = _norm(h,   r['h_off'],   r['h_scale'])
    Ln = _rpc_poly(r['line_num'], r['line_den'], L, P, H)
    Sn = _rpc_poly(r['samp_num'], r['samp_den'], L, P, H)
    row = r['line_off'] + r['line_scale'] * Ln
    col = r['samp_off'] + r['samp_scale'] * Sn
    return row, col

def _load_rpc_dict(rpc_json_path):
    r0 = json.load(open(rpc_json_path))
    r  = r0.get('rpc', r0.get('RPC', r0))
    # normalize keys usually used (TODO refine this compatibility in next vesions)
    return {
        'lon_off':  r.get('lon_offset',  r.get('LON_OFF',  r.get('long_offset'))),
        'lat_off':  r.get('lat_offset',  r.get('LAT_OFF',  r.get('latitude_offset'))),
        'h_off':    r.get('alt_offset',  r.get('HGT_OFF',  r.get('height_offset', 0.0))),
        'lon_scale':r.get('lon_scale',   r.get('LON_SCALE', r.get('long_scale', 1.0))),
        'lat_scale':r.get('lat_scale',   r.get('LAT_SCALE', r.get('latitude_scale', 1.0))),
        'h_scale':  r.get('alt_scale',   r.get('HGT_SCALE', r.get('height_scale', 1.0))),
        'line_off': r.get('row_offset',  r.get('LINE_OFF',  r.get('line_offset'))),
        'samp_off': r.get('col_offset',  r.get('SAMP_OFF',  r.get('samp_offset'))),
        'line_scale':r.get('row_scale',  r.get('LINE_SCALE',r.get('line_scale', 1.0))),
        'samp_scale':r.get('col_scale',  r.get('SAMP_SCALE',r.get('samp_scale', 1.0))),
        'line_num': np.asarray(r.get('row_num',  r.get('LINE_NUM_COEFF')), dtype=float),
        'line_den': np.asarray(r.get('row_den',  r.get('LINE_DEN_COEFF')), dtype=float),
        'samp_num': np.asarray(r.get('col_num',  r.get('SAMP_NUM_COEFF')), dtype=float),
        'samp_den': np.asarray(r.get('col_den',  r.get('SAMP_DEN_COEFF')), dtype=float),
    }

def detect_orientation_from_rpc(rpc_json_path, dlon=1e-6, dlat=1e-6):
    """
    Returns one of: 'none','vflip','hflip','rot180','rot90ccw','rot90cw'
    based on the Jacobian of (lon,lat)->(row,col) near the RPC reference.
    """
    r = _load_rpc_dict(rpc_json_path)
    lon0, lat0, h0 = r['lon_off'], r['lat_off'], r['h_off']

    # central differences
    row_e,col_e = _project_row_col(r, lon0 + dlon, lat0, h0)
    row_w,col_w = _project_row_col(r, lon0 - dlon, lat0, h0)
    row_n,col_n = _project_row_col(r, lon0, lat0 + dlat, h0)
    row_s,col_s = _project_row_col(r, lon0, lat0 - dlat, h0)

    drow_dlon = (row_e - row_w) / (2*dlon)
    dcol_dlon = (col_e - col_w) / (2*dlon)
    drow_dlat = (row_n - row_s) / (2*dlat)
    dcol_dlat = (col_n - col_s) / (2*dlat)

    # Heuristic: are axes swapped (row depends more on lon; col on lat)?
    axes_swapped = (abs(drow_dlon) > abs(drow_dlat)) and (abs(dcol_dlat) > abs(dcol_dlon))

    if axes_swapped:
        # Determine 90 deg direction from signs
        if drow_dlon > 0 and dcol_dlat > 0:
            return 'rot90ccw'   # row<=east, col<=north
        else:
            return 'rot90cw'    # row<=west, col<=south
    else:
        need_v = drow_dlat > 0   # rows increase to the north -> upside‑down
        need_h = dcol_dlon < 0   # cols decrease to the east  -> mirrored left/right
        if need_v and need_h:
            return 'rot180'
        elif need_v:
            return 'vflip'
        elif need_h:
            return 'hflip'
        else:
            return 'none'
