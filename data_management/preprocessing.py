from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
import torch
import gc

# Pix2Pix-style per-band normalization
def normalize_bands_pix2pix(bands: np.ndarray) -> np.ndarray:
    """
    Robust per-channel min-max to [-1, 1].
    Accepts (H,W), (C,H,W), or (H,W,C). Returns (C,H,W), float32.
    """
    a = np.asarray(bands)
    # ensure CHW
    if a.ndim == 2:
        a = a[None, ...]                       # (1,H,W)
    elif a.ndim == 3 and a.shape[0] not in (1, 3):
        a = np.transpose(a, (2, 0, 1))         # HWC -> CHW

    a = a.astype(np.float32, copy=False)

    # per-channel mins/maxes
    cmins = np.nanmin(a, axis=(1, 2), keepdims=True)
    cmaxs = np.nanmax(a, axis=(1, 2), keepdims=True)
    denom = np.maximum(cmaxs - cmins, 1e-6)   # avoid divide-by-zero

    out = (a - cmins) / denom  # [0,1]
    out = out * 2.0 - 1.0  # [-1,1]
    return out

import numpy as np
import torch
from typing import Dict, Tuple

class MyMSIAugment:
    def __init__(self, rng=42,
                 prefer_in_bounds: bool = True, in_bounds_prob: float = 0.9,
                 min_valid_frac: float = 0.90, max_resamples: int = 8,
                 pix2pix_norm: bool = False, configs=None):
        self.crop_h = self.crop_w = int(configs.train.aug_crop_size)
        self.rng = np.random.default_rng(int(rng))
        self.prefer_in_bounds = bool(prefer_in_bounds)
        self.in_bounds_prob = float(in_bounds_prob)
        self.min_valid_frac = float(min_valid_frac)
        self.max_resamples = int(max_resamples)
        self.pix2pix_norm = bool(pix2pix_norm)
        self.configs = configs

    @staticmethod
    def _to_np(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    @staticmethod
    def _crop_or_pad_hw(arr: np.ndarray, y0: int, x0: int, out_h: int, out_w: int) -> np.ndarray:
        a = np.asarray(arr)

        if a.ndim == 2:
            H, W = a.shape
            out = np.zeros((out_h, out_w), dtype=a.dtype)
            ys, xs = max(0, y0), max(0, x0)
            ye, xe = min(H, y0 + out_h), min(W, x0 + out_w)
            if ye > ys and xe > xs:
                oy, ox = ys - y0, xs - x0
                out[oy:oy+(ye-ys), ox:ox+(xe-xs)] = a[ys:ye, xs:xe]
            return out

        if a.ndim == 3:
            # Decide CHW vs HWC by where the small channel dim (2 or 3) is
            if a.shape[0] in (2, 3) and a.shape[0] != a.shape[-1]:
                # CHW
                C, H, W = a.shape
                out = np.zeros((C, out_h, out_w), dtype=a.dtype)
                ys, xs = max(0, y0), max(0, x0)
                ye, xe = min(H, y0 + out_h), min(W, x0 + out_w)
                if ye > ys and xe > xs:
                    oy, ox = ys - y0, xs - x0
                    out[:, oy:oy+(ye-ys), ox:ox+(xe-xs)] = a[:, ys:ye, xs:xe]
                return out
            elif a.shape[-1] in (2, 3):
                # HWC
                H, W, C = a.shape
                out = np.zeros((out_h, out_w, C), dtype=a.dtype)
                ys, xs = max(0, y0), max(0, x0)
                ye, xe = min(H, y0 + out_h), min(W, x0 + out_w)
                if ye > ys and xe > xs:
                    oy, ox = ys - y0, xs - x0
                    out[oy:oy+(ye-ys), ox:ox+(xe-xs), :] = a[ys:ye, xs:xe, :]
                return out
            else:
                raise ValueError(f"Expected 2 or 3 channels, got shape {a.shape}")

        raise ValueError(f"Expected 2D or 3D array, got {a.ndim}D with shape {a.shape}")

    @staticmethod
    def _crop_or_pad_chw(arr: np.ndarray, y0: int, x0: int, out_h: int, out_w: int) -> np.ndarray:
        assert arr.ndim == 3, f"CHW expected, got {arr.shape}"
        C, H, W = arr.shape
        out = np.zeros((C, out_h, out_w), dtype=arr.dtype)
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(H, y0 + out_h), min(W, x0 + out_w)
        if ye > ys and xe > xs:
            oy, ox = ys - y0, xs - x0
            out[:, oy:oy+(ye-ys), ox:ox+(xe-xs)] = arr[:, ys:ye, xs:xe]
        return out

    def _sample_origin(self, H: int, W: int) -> Tuple[int, int]:
        """Choose (y0, x0); optionally prefer fully in-bounds windows."""
        ch, cw = self.crop_h, self.crop_w
        can_inbounds = (H >= ch) and (W >= cw)
        if self.prefer_in_bounds and can_inbounds and self.rng.random() < self.in_bounds_prob:
            y0 = 0 if H == ch else self.rng.integers(0, H - ch + 1)
            x0 = 0 if W == cw else self.rng.integers(0, W - cw + 1)
            return int(y0), int(x0)
        # allow padding
        y0 = self.rng.integers(-(ch - H) if H < ch else 0, max(0, H - ch) + 1)
        x0 = self.rng.integers(-(cw - W) if W < cw else 0, max(0, W - cw) + 1)
        return int(y0), int(x0)

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        # --- unwrap to numpy w/ expected shapes
        tm   = self._to_np(sample['target_mask'])  # HW
        im   = self._to_np(sample['input_mask'])   # HW
        trgb = self._to_np(sample['target_rgb'])   # CHW
        irgb = self._to_np(sample['input_rgb'])    # CHW

        use_dem = bool(getattr(self.configs.train, "use_DEM"))
        if use_dem:
            idem = self._to_np(sample["input_dem"])   # CHW (1,H,W)
            tdem = self._to_np(sample["target_dem"])  # CHW (1,H,W)

        vm_in = sample.get('valid_mask')
        vm_in = self._to_np(vm_in)

        # --- compute common valid area (by height/width only)
        H = min(tm.shape[-2], im.shape[-2], trgb.shape[-2], irgb.shape[-2])
        W = min(tm.shape[-1], im.shape[-1], trgb.shape[-1], irgb.shape[-1])

        best = None
        best_frac = -1.0
        for _ in range(max(1, self.max_resamples)):
            y0, x0 = self._sample_origin(H, W)
            tm_c   = self._crop_or_pad_hw(tm,   y0, x0, self.crop_h, self.crop_w)
            im_c   = self._crop_or_pad_hw(im,   y0, x0, self.crop_h, self.crop_w)
            trgb_c = self._crop_or_pad_chw(trgb, y0, x0, self.crop_h, self.crop_w)
            irgb_c = self._crop_or_pad_chw(irgb, y0, x0, self.crop_h, self.crop_w)
            vm_c   = self._crop_or_pad_chw(vm_in, y0, x0, self.crop_h, self.crop_w)  # (1,H,W)
            frac = float(vm_c.mean())

            if use_dem:
                idem_c = self._crop_or_pad_chw(idem, y0, x0, self.crop_h, self.crop_w)
                tdem_c = self._crop_or_pad_chw(tdem, y0, x0, self.crop_h, self.crop_w)

            if frac >= self.min_valid_frac:
                best = (tm_c, im_c, trgb_c, irgb_c, vm_c) if not use_dem else (tm_c, im_c, trgb_c, irgb_c, vm_c, idem_c, tdem_c)
                break

            if frac > best_frac:
                best = (tm_c, im_c, trgb_c, irgb_c, vm_c) if not use_dem else (tm_c, im_c, trgb_c, irgb_c, vm_c, idem_c, tdem_c)
                best_frac = frac

        if best is None:
            # should never happen, but keep it explicit
            tm_c, im_c, trgb_c, irgb_c, vm_c = tm, im, trgb, irgb, vm_in
            if use_dem:
                idem_c, tdem_c = idem, tdem
        else:
            if use_dem:
                tm_c, im_c, trgb_c, irgb_c, vm_c, idem_c, tdem_c = best
            else:
                tm_c, im_c, trgb_c, irgb_c, vm_c = best

        # optional Pix2Pix-style normalization (expects CHW, float32)
        if self.pix2pix_norm:
            trgb_c = normalize_bands_pix2pix(trgb_c)
            irgb_c = normalize_bands_pix2pix(irgb_c)

        # --- pack to torch with fixed dtypes
        out = {
            'target_mask': torch.as_tensor(tm_c, dtype=torch.uint8),
            'input_mask':  torch.as_tensor(im_c, dtype=torch.uint8),
            'target_rgb':  torch.as_tensor(trgb_c, dtype=torch.float32),
            'input_rgb':   torch.as_tensor(irgb_c, dtype=torch.float32),
            'valid_mask':  torch.as_tensor(vm_c,  dtype=torch.uint8),  # keep as byte mask
        }
        if use_dem:
            out['input_dem']  = torch.as_tensor(idem_c, dtype=torch.float32)
            out['target_dem'] = torch.as_tensor(tdem_c, dtype=torch.float32)
        return out
