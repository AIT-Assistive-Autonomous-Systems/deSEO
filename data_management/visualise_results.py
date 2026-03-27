#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Please install Pillow: pip install pillow")

# ---------- helpers ----------
def infer_hw_from_meta(meta):
    """Try common keys inside meta to get (H, W). Returns (None, None) if unknown."""
    if meta is None:
        return None, None
    # meta might be numpy object; try to make it dict-ish
    try:
        m = meta.item() if hasattr(meta, "item") else meta
    except Exception:
        m = meta
    if isinstance(m, dict):
        for h_key, w_key in [
            ("height","width"), ("H","W"), ("h","w"),
            ("img_h","img_w"), ("im_h","im_w"), ("rows","cols"),
            ("target_h","target_w"), ("input_h","input_w"),
        ]:
            if h_key in m and w_key in m:
                try:
                    H = int(m[h_key]); W = int(m[w_key])
                    return H, W
                except Exception:
                    pass
    return None, None

def reshape_flat(arr, key, meta):
    """
    Convert flattened-ish arrays to image-like shapes.
    Handles:
      - (1,1,N) or (1,N) or (N,) -> (H,W[,C]) with C in {1,3,4}
      - Leaves proper shapes unchanged.
    Prefers sizes from meta if available; falls back to reasonable inference.
    """
    a = np.asarray(arr)

    # Already image-like?
    if a.ndim == 2:          # HxW
        return a
    if a.ndim == 3 and a.shape[-1] in (1,3,4):  # HxWxC
        return a
    if a.ndim == 4:          # NxHxWxC -> pass through
        return a

    # Squeeze trivial dims
    a = np.squeeze(a)

    # If still 3D but last dim huge (channels>>4), treat as flat vector
    if a.ndim == 3 and a.shape[-1] > 4 and a.shape[0] == 1 and a.shape[1] == 1:
        flat = a.reshape(-1)
    elif a.ndim == 2 and 1 in a.shape:
        flat = a.reshape(-1)
    elif a.ndim == 1:
        flat = a
    else:
        return a  # Unknown, let downstream try

    n = int(flat.size)

    # 1) Try meta-reported sizes
    Hm, Wm = infer_hw_from_meta(meta)
    if Hm and Wm:
        # choose C based on divisibility (prefer masks for "*mask*" keys)
        if "mask" in (key or "").lower():
            Ccands = [1, 3, 4]
        else:
            Ccands = [3, 4, 1]
        for C in Ccands:
            if Hm * Wm * C == n:
                return flat.reshape(Hm, Wm, C) if C > 1 else flat.reshape(Hm, Wm)

    # 2) Try to infer square images with common channels
    # Prefer mask (C=1) for keys containing 'mask' or 'valid'
    if "mask" in (key or "").lower() or "valid" in (key or "").lower():
        channel_order = [1, 3, 4]
    else:
        channel_order = [3, 4, 1]

    for C in channel_order:
        if n % C == 0:
            hw = n // C
            r = int(round(hw ** 0.5))
            if r * r == hw:  # perfect square
                return flat.reshape(r, r, C) if C > 1 else flat.reshape(r, r)

    # 3) As a last resort, try a near-square factorization
    def near_square_hw(total):
        r = int(np.floor(np.sqrt(total)))
        for h in range(r, 0, -1):
            if total % h == 0:
                return h, total // h
        return None, None

    for C in channel_order:
        if n % C == 0:
            hw = n // C
            H, W = near_square_hw(hw)
            if H and W:
                return flat.reshape(H, W, C) if C > 1 else flat.reshape(H, W)

    # Give up: return original flattened (caller will min-max normalize and save a 1×N "strip")
    return flat

def to_uint8_img(arr):
    a = np.asarray(arr)

    # If reshape produced 1D, try to present as a tall strip for visibility
    if a.ndim == 1:
        # Make a tall image ~512 px wide
        width = min(a.size, 512)
        height = int(np.ceil(a.size / width))
        b = np.zeros((height * width,), dtype=a.dtype)
        b[:a.size] = a
        a = b.reshape(height, width)

    # Ensure channel axis is last
    if a.ndim == 3 and a.shape[0] in (1,3,4) and a.shape[-1] not in (1,3,4):
        a = np.moveaxis(a, 0, -1)

    # If grayscale 2D, keep as 2D
    # If single-channel 3D, squeeze
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]

    # dtype/scale handling
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif np.issubdtype(a.dtype, np.integer):
        a = np.clip(a, 0, 255).astype(np.uint8)
    else:
        amin, amax = float(a.min()), float(a.max())
        if amax <= 1.0 and amin >= 0.0:
            a = (a * 255.0).round().astype(np.uint8)
        elif amax <= 1.0 and amin >= -1.0:
            a = ((a + 1.0) * 0.5 * 255.0).round().astype(np.uint8)
        else:
            if amax == amin:
                a = np.zeros_like(a, dtype=np.uint8)
            else:
                a = ((a - amin) / (amax - amin) * 255.0).round().astype(np.uint8)
    return a

def save_png(array, out_path_base, key_name, meta=None):
    arr = reshape_flat(array, key_name, meta)

    def _save_single(img_arr, idx_suffix=None):
        img_u8 = to_uint8_img(img_arr)
        out_path = Path(f"{out_path_base}_{key_name}.png") if idx_suffix is None \
                   else Path(f"{out_path_base}_{key_name}_{idx_suffix:04d}.png")
        # Ensure valid shape for PIL
        if img_u8.ndim == 2:
            Image.fromarray(img_u8).save(out_path)
        elif img_u8.ndim == 3 and img_u8.shape[-1] in (3,4):
            Image.fromarray(img_u8).save(out_path)
        else:
            # Fallback: save as grayscale strip
            Image.fromarray(img_u8 if img_u8.ndim == 2 else img_u8[...,0]).save(out_path)

    a = np.asarray(arr)
    if a.ndim in (2, 3):
        _save_single(a, None)
    elif a.ndim == 4:
        for i in range(a.shape[0]):
            _save_single(a[i], i)
    else:
        _save_single(a, None)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Extract arrays from NPZ and save PNGs to test/ folder.")
    parser.add_argument("npz_path", type=str, help="Path to the .npz file")
    parser.add_argument("--outdir", type=str, default="test", help="Output directory (default: test)")
    parser.add_argument("--prefix", type=str, default="", help="Optional filename prefix")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)

    base = args.prefix if args.prefix else npz_path.stem
    base_path = outdir / base

    meta = data["meta"] if "meta" in data else None

    for key in ["input_rgb","target_rgb","input_mask","target_mask","valid_mask"]:
        if key in data:
            try:
                save_png(data[key], str(base_path), key, meta)
                print(f"Saved {key} to {outdir}")
            except Exception as e:
                print(f"Warning: could not save {key} as PNGs: {e}")

    if "align_homography" in data:
        np.savetxt(f"{base_path}_align_homography.txt", np.array(data["align_homography"]), fmt="%.6f")
        print(f"Wrote homography to {base_path}_align_homography.txt")

    if "align_inliers" in data:
        with open(f"{base_path}_align_inliers.txt", "w") as f:
            f.write(str(int(data["align_inliers"])))
        print(f"Wrote number of inliers to {base_path}_align_inliers.txt")

    if "meta" in data:
        try:
            with open(f"{base_path}_meta.txt", "w", encoding="utf-8") as f:
                f.write(repr(data["meta"]))
            print(f"Wrote meta to {base_path}_meta.txt")
        except Exception as e:
            print(f"Warning: could not write meta: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
