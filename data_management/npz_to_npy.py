from pathlib import Path
import numpy as np
import tqdm

cache_npz_root = Path("datasets/deSEO/cache_offline")
# use tqdm to make things trackable
for npz_path in tqdm.tqdm(cache_npz_root.rglob("*.npz")):
    print(f"Processing {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    scene = npz_path.parent.name
    stem  = npz_path.stem
    out_dir = cache_npz_root / scene / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # npz_to_npy.py
    np.save(out_dir / "input_rgb.npy",  data["input_rgb"].astype(np.float16, copy=False))
    np.save(out_dir / "target_rgb.npy", data["target_rgb"].astype(np.float16, copy=False))
    np.save(out_dir / "input_mask.npy",  data["input_mask"].astype(np.uint8,  copy=False))
    np.save(out_dir / "target_mask.npy", data["target_mask"].astype(np.uint8,  copy=False))

    if "valid_mask" in data:
        np.save(out_dir / "valid_mask.npy", data["valid_mask"])
    if "align_affine" in data:
        np.save(out_dir / "align_affine.npy", data["align_affine"])

    # optional: persist meta dict if present
    #if "meta" in data:
    #    (out_dir / "meta.json").write_text(json.dumps(dict(data["meta"].tolist()), indent=2))

    # (optionally) remove old file:
    # npz_path.unlink()
