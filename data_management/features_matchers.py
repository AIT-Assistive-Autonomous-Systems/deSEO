try:
    from kornia.feature import LoFTR as KorniaLoFTR
except Exception:
    KorniaLoFTR = None  # optional
from data_management.utils import _ensure_gray_u8    
import torch
import numpy as np

class LoFTRMatcher:
    def __init__(self, device=None, pretrained="outdoor", max_size=1024, conf_thresh=0.40, max_matches=8000):
        if KorniaLoFTR is None:
            raise ImportError("kornia not installed; pip install kornia")
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = KorniaLoFTR(pretrained=pretrained).to(self.device).eval()
        self.max_size = int(max_size)
        self.conf_thresh = float(conf_thresh)
        self.max_matches = int(max_matches)

    @torch.inference_mode()
    def match(self, gray0_u8, gray1_u8):
        g0 = _ensure_gray_u8(gray0_u8); g1 = _ensure_gray_u8(gray1_u8)
        t0 = torch.from_numpy(g0)[None,None].float()/255.0
        t1 = torch.from_numpy(g1)[None,None].float()/255.0
        def resize_limit(t):
            H,W = t.shape[-2:]; scale=1.0; long=max(H,W)
            if long>self.max_size:
                scale=self.max_size/float(long)
                newH=int(round(H*scale/8.0))*8; newW=int(round(W*scale/8.0))*8
                t=torch.nn.functional.interpolate(t,size=(newH,newW),mode="bilinear",align_corners=False)
            return t, scale
        t0,s0 = resize_limit(t0); t1,s1 = resize_limit(t1)
        t0,t1 = t0.to(self.device), t1.to(self.device)
        out = self.model({"image0": t0, "image1": t1})
        k0 = out["keypoints0"].detach().cpu().numpy().astype(np.float32)
        k1 = out["keypoints1"].detach().cpu().numpy().astype(np.float32)
        conf = out.get("confidence", torch.ones(len(k0), device=self.device)).detach().cpu().numpy().astype(np.float32)
        if s0 != 1.0: k0 /= s0
        if s1 != 1.0: k1 /= s1
        keep = conf >= self.conf_thresh
        if keep.any(): k0, k1, conf = k0[keep], k1[keep], conf[keep]
        if len(conf) > self.max_matches:
            idx = np.argsort(-conf)[: self.max_matches]
            k0, k1, conf = k0[idx], k1[idx], conf[idx]
        return k0, k1, conf
