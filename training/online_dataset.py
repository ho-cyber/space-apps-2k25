#!/usr/bin/env python3
"""
online_dataset.py

Download Kepler/TESS light curves from MAST and preprocess them for training.
"""

from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    import lightkurve as lk
except ImportError:
    lk = None

try:
    from astroquery.mast import Observations
except ImportError:
    Observations = None

from train_model import preprocess_flux_array, preprocess_and_cache_fits, SEQ_LEN

# ------------------------
# Safe naming
# ------------------------
def safe_name(name: str):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")

# ------------------------
# Download light curves using Lightkurve or MAST
# ------------------------
def download_lightcurve(target, mission="Kepler", out_dir=Path("./online_fits"), max_files=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    fits_files = []

    if lk is not None:
        try:
            search = lk.search_lightcurvefile(target, mission=mission)
            if len(search) > 0:
                selection = search[:max_files]
                try:
                    _ = selection.download_all(download_dir=str(out_dir))
                except TypeError:
                    _ = selection.download_all()
                fits_files = sorted(out_dir.rglob("*.fits"))
        except Exception:
            pass

    # Fallback to MAST
    if len(fits_files) == 0 and Observations is not None:
        try:
            obs_table = Observations.query_object(target, radius="0.05 deg")
            if len(obs_table) > 0:
                products = Observations.get_product_list(obs_table)
                candidates = [
                    i
                    for i, row in enumerate(products)
                    if "llc.fits" in str(row.get("productFilename", "")).lower()
                    or "slc.fits" in str(row.get("productFilename", "")).lower()
                ]
                keep = candidates[:max_files]
                _ = Observations.download_products(
                    products[keep], download_dir=str(out_dir), mrp_only=False
                )
                fits_files = sorted(out_dir.rglob("*.fits"))
        except Exception:
            pass

    return [str(p) for p in fits_files[:max_files]]

# ------------------------
# Main function for training integration
# ------------------------
def download_light_curves_and_preprocess(target_samples=20, cache_dir=Path("./preprocessed_cache")):
    """
    Downloads a small number of stars (default 20) from online, preprocesses them,
    returns features and heuristic labels (1=dip, 0=no dip).
    """
    features = []
    labels = []

    # Example targets for demonstration; in practice replace with real KIC/TIC IDs
    example_targets = [f"KIC {i}" for i in range(11490410, 11490410 + target_samples)]

    for t in tqdm(example_targets, desc="Downloading online light curves"):
        fits_paths = download_lightcurve(t, max_files=1)
        for fp in fits_paths:
            arr = preprocess_and_cache_fits(Path(fp), cache_dir, target_len=SEQ_LEN)
            if arr is None:
                continue
            # heuristic labeling: dip deeper than -0.005 -> label 1
            label = 1 if np.percentile(arr, 5) < -0.005 else 0
            features.append(arr)
            labels.append(label)
            if len(features) >= target_samples:
                break
        if len(features) >= target_samples:
            break

    return features, labels
