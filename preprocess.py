"""
preprocess.py  —  Complete data preparation for Vox-UDA.

What this script does
---------------------
SOURCE  (simulated H5):
  • Reads each .h5 file (keys: 'image', 'label'), both shape (68, 68, 68).
  • Resizes to 32³ via scipy zoom (paper: "all subtomograms are resized to 32³").
  • Binarises the label (the H5 label is uint8; paper threshold is 300 for
    greyscale masks, but H5 labels appear already binary 0/1 — we check both).
  • Saves volume → data/processed/source/volumes/<name>.npy
           label  → data/processed/source/masks/<name>.npy

TARGET  (Poly-GA MRC):
  • Reads each .map/.mrc file using mrcfile.
  • Pads the volume so all three dimensions are divisible by 32.
  • Extracts NON-OVERLAPPING 32³ patches (stride = 32).
  • Saves each patch → data/processed/target/volumes/<file>_patch<k>.npy
  • No labels saved (target is unlabelled during training).


Usage
-----
  python preprocess.py                         # uses default paths below
  python preprocess.py --sim_dir  data/simulated_h5_data_ds20
                        --polyga_dir data/Poly-GA
                        --out_dir    data/processed

Output directory layout
-----------------------
  data/processed/
    source/
      volumes/   <name>.npy   float32  (32, 32, 32)
      masks/     <name>.npy   uint8    (32, 32, 32)  binary {0, 1}
    target/
      volumes/   <name>_patch<k>.npy  float32  (32, 32, 32)
    target_eval/                     (only if GT labels provided)
      volumes/   same as target/
      masks/     <name>_patch<k>.npy  uint8
    stats.json   dataset statistics for reference
"""

import os
import sys
import glob
import json
import argparse
import logging
import numpy as np
import h5py
import mrcfile
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Defaults  (match your directory layout)
# ---------------------------------------------------------------------------
BASE = "/shared/scratch/0/home/v_zonunmawia_zadeng/UDA"
DEFAULT_SIM_DIR   = os.path.join(BASE, "data/simulated_h5_data_ds20")
DEFAULT_POLYGA_DIR= os.path.join(BASE, "data/Poly-GA")
DEFAULT_OUT_DIR = os.path.join(BASE, "data/processed")
PATCH_SIZE        = 32          # paper: "all subtomograms resized to 32³"
MASK_THRESHOLD    = 0        
        


# ---------------------------------------------------------------------------
# Utility: resize a 3D volume to a target cubic size
# ---------------------------------------------------------------------------


def resize_volume(vol: np.ndarray, target: int = 32) -> np.ndarray:
    """
    Resize a 3D volume to (target, target, target) using trilinear zoom.

    The zoom factor along each axis = target / current_size.
    Uses order=1 (linear) for volumes and order=0 (nearest) for masks
    to avoid creating non-binary values.

    Args:
        vol    : (D, H, W) numpy array
        target : desired side length
    Returns:
        resized (target, target, target) array (same dtype as input)
    """
    D, H, W = vol.shape
    factors = (target / D, target / H, target / W)
    order   = 0 if vol.dtype == np.uint8 else 1
    resized = zoom(vol.astype(np.float32), factors, order=order)
    return resized.astype(vol.dtype)

def atomic_save_npy(path, arr):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.save(f, arr)
    os.replace(tmp_path, path)

# ---------------------------------------------------------------------------
# Utility: pad a volume so dims are multiples of patch_size
# ---------------------------------------------------------------------------

def pad_to_multiple(vol: np.ndarray, multiple: int = 32) -> np.ndarray:
    """
    Symmetrically pad a 3D volume with zeros so every dimension is a
    multiple of `multiple`.
    """
    D, H, W = vol.shape
    pd = (multiple - D % multiple) % multiple
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple
    return np.pad(vol, [(0, pd), (0, ph), (0, pw)], mode="constant")


# ---------------------------------------------------------------------------
# Utility: extract non-overlapping 32³ patches
# ---------------------------------------------------------------------------

def extract_patches(vol: np.ndarray, patch_size: int = 32) -> list:
    """
    Extract all non-overlapping patch_size³ patches from a 3D volume.
    The volume is first padded so dimensions are exact multiples of patch_size.

    Returns:
        list of (patch_size, patch_size, patch_size) arrays
    """
    vol   = pad_to_multiple(vol, patch_size)
    D, H, W = vol.shape
    patches = []
    for z in range(0, D, patch_size):
        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
                patch = vol[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
    return patches


# ---------------------------------------------------------------------------
# Process simulated H5 source data
# ---------------------------------------------------------------------------

def process_source(sim_dir: str, out_dir: str, patch_size: int = 32):
    """
    Load every .h5 file in sim_dir, resize image+label to 32³, save as .npy.

    H5 file structure:
        'image' : (68, 68, 68) float32  — subtomogram
        'label' : (68, 68, 68) uint8    — segmentation mask

    Naming convention in filenames:
        snrSNR003_prot1c5u_386.h5
        snrSNR005_prot2ane_381.h5
        snrSNRinfinity_prot6t3e_217.h5
    """
    vol_dir  = os.path.join(out_dir, "source", "volumes")
    mask_dir = os.path.join(out_dir, "source", "masks")
    os.makedirs(vol_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(sim_dir, "*.h5")))
    if not h5_files:
        log.error(f"No .h5 files found in {sim_dir}")
        return 0

    log.info(f"Processing {len(h5_files)} simulated H5 files → {patch_size}³ ...")

    n_saved = 0
    skipped = 0
    label_max_vals = []

    for fpath in h5_files:
        stem = os.path.splitext(os.path.basename(fpath))[0]  # e.g. snrSNR003_prot1c5u_386

        try:
            with h5py.File(fpath, "r") as f:
                # Verify expected keys
                keys = list(f.keys())
                if "image" not in keys or "label" not in keys:
                    log.warning(f"  Skipping {stem}: expected keys 'image','label', got {keys}")
                    skipped += 1
                    continue

                vol   = f["image"][()].astype(np.float32)    # (68,68,68)
                label = f["label"][()].astype(np.uint8)      # (68,68,68)

        except Exception as e:
            log.warning(f"  Error reading {fpath}: {e}")
            skipped += 1
            continue

        # --- Sanity check ---
        if vol.shape != label.shape:
            log.warning(f"  {stem}: vol shape {vol.shape} != label shape {label.shape}, skipping")
            skipped += 1
            continue

        label_max_vals.append(int(label.max()))

        # --- Resize to 32³ ---
        vol_32   = resize_volume(vol,   patch_size)    # float32 (32,32,32)
        label_32 = resize_volume(label, patch_size)    # uint8   (32,32,32)

        # --- Binarise label ---
        # H5 labels appear to be already binary (0/1).
        # If max value is much larger (e.g., from greyscale 0–65535),
        # threshold at MASK_THRESHOLD.
        if label_32.max() > 1:
            label_32 = (label_32 > MASK_THRESHOLD).astype(np.uint8)
        else:
            label_32 = (label_32 > 0).astype(np.uint8)

        # --- Save ---
        vol_path  = os.path.join(vol_dir,  f"{stem}.npy")
        mask_path = os.path.join(mask_dir, f"{stem}.npy")

        atomic_save_npy(vol_path, vol_32)
        atomic_save_npy(mask_path, label_32)
        n_saved += 1

    log.info(f"  Source: saved {n_saved} samples  |  skipped {skipped}")
    if label_max_vals:
        log.info(f"  Label max values seen: min={min(label_max_vals)} max={max(label_max_vals)}")
    return n_saved


# ---------------------------------------------------------------------------
# Process Poly-GA real target data (NO labels)
# ---------------------------------------------------------------------------

def process_target(polyga_dir: str, out_dir: str, patch_size: int = 32):
    """
    Load every .map/.mrc file in polyga_dir, extract 32³ patches, save .npy.

    From your preprocess.py exploration, the volumes have shapes like:
        (60, 60, 60)   → 1 patch  (padding needed to reach 32 multiple, but 60 < 64)
        (90, 90, 90)   → 8 patches (after pad to 96)
        (71, 926, 926) → 1568 patches

    The paper states Poly-GA has 1,033 samples total:
        66  × 26S subtomograms
        66  × TRiC subtomograms
        901 × Ribosome subtomograms
    All re-scaled to 32³.

    NOTE: Some of the MRC volumes you have appear to ALREADY be individual
    subtomograms (60³, 90³) and some are full tomogram slabs (71×926×926).
    We handle both by patch extraction.
    """
    vol_dir = os.path.join(out_dir, "target", "volumes")
    os.makedirs(vol_dir, exist_ok=True)

    mrc_files = sorted(
        glob.glob(os.path.join(polyga_dir, "*.map")) +
        glob.glob(os.path.join(polyga_dir, "*.mrc")) +
        glob.glob(os.path.join(polyga_dir, "*.rec"))
    )
    if not mrc_files:
        log.error(f"No .map/.mrc/.rec files found in {polyga_dir}")
        return 0

    log.info(f"Processing {len(mrc_files)} Poly-GA MRC files ...")

    n_patches = 0
    shape_report = []

    for fpath in mrc_files:
        stem = os.path.splitext(os.path.basename(fpath))[0]

        try:
            with mrcfile.open(fpath, permissive=True) as mrc:
                vol = mrc.data.copy().astype(np.float32)
        except Exception as e:
            log.warning(f"  Error reading {fpath}: {e}")
            continue

        if vol is None or vol.ndim != 3:
            log.warning(f"  {stem}: unexpected volume shape/None, skipping")
            continue

        shape_report.append((stem, vol.shape))
        D, H, W = vol.shape

        # ── Strategy: if volume is already ~32–64³, resize directly.
        #              If it's a large tomogram, extract patches.
        max_dim = max(D, H, W)

        if max_dim <= 80:
            # Small subtomogram → resize to exactly 32³
            vol_32 = resize_volume(vol, patch_size)   # (32,32,32) float32
            vol_32 = vol_32.astype(np.float32)
            out_path = os.path.join(vol_dir, f"{stem}.npy")
            np.save(out_path, vol_32)
            n_patches += 1

        else:
            # Large volume or slab → patch extraction
            patches = extract_patches(vol, patch_size)
            for k, patch in enumerate(patches):
                # Skip patches that are almost entirely zero (empty background)
                if patch.std() < 1e-6:
                    continue
                out_path = os.path.join(vol_dir, f"{stem}_patch{k:04d}.npy")
                atomic_save_npy(out_path, patch.astype(np.float32))
                n_patches += 1

    log.info(f"  Target: saved {n_patches} patches")
    log.info("  Shape summary:")
    for stem, shp in shape_report:
        log.info(f"    {stem:30s}  {shp}")

    return n_patches


# ---------------------------------------------------------------------------
# Process evaluation labels (optional — if supervisor provides them)
# ---------------------------------------------------------------------------

def process_target_eval(eval_vol_dir: str, eval_mask_dir: str, out_dir: str,
                         patch_size: int = 32):
    """
    Expected: eval_vol_dir and eval_mask_dir have paired .npy or .mrc/.map
              files with the same filenames.
    """
    out_vol  = os.path.join(out_dir, "target_eval", "volumes")
    out_mask = os.path.join(out_dir, "target_eval", "masks")
    os.makedirs(out_vol,  exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    vol_files = sorted(glob.glob(os.path.join(eval_vol_dir, "*.npy")) +
                       glob.glob(os.path.join(eval_vol_dir, "*.mrc")) +
                       glob.glob(os.path.join(eval_vol_dir, "*.map")))

    if not vol_files:
        log.warning(f"No eval volumes found in {eval_vol_dir}")
        return

    log.info(f"Processing {len(vol_files)} evaluation volume+mask pairs ...")
    n = 0
    for vp in vol_files:
        stem = os.path.splitext(os.path.basename(vp))[0]
        mp   = os.path.join(eval_mask_dir, os.path.basename(vp))
        if not os.path.exists(mp):
            mp = os.path.join(eval_mask_dir, stem + ".npy")
        if not os.path.exists(mp):
            log.warning(f"  No mask found for {stem}, skipping")
            continue

        vol  = np.load(vp) if vp.endswith(".npy") else _load_mrc(vp)
        mask = np.load(mp) if mp.endswith(".npy") else _load_mrc(mp)

        vol  = resize_volume(vol.astype(np.float32), patch_size)
        mask = resize_volume(mask.astype(np.uint8),  patch_size)
        mask = (mask > 0).astype(np.uint8)

        np.save(os.path.join(out_vol,  f"{stem}.npy"), vol)
        np.save(os.path.join(out_mask, f"{stem}.npy"), mask)
        n += 1

    log.info(f"  Eval: saved {n} pairs")


def _load_mrc(path):
    with mrcfile.open(path, permissive=True) as mrc:
        return mrc.data.copy()


# ---------------------------------------------------------------------------
# Print dataset statistics
# ---------------------------------------------------------------------------

def print_stats(out_dir: str):
    stats = {}
    for split in ["source/volumes", "source/masks", "target/volumes"]:
        d = os.path.join(out_dir, split)
        if os.path.isdir(d):
            files = glob.glob(os.path.join(d, "*.npy"))
            stats[split] = len(files)

    log.info("=" * 55)
    log.info("  DATASET STATISTICS")
    log.info("=" * 55)
    for k, v in stats.items():
        log.info(f"  {k:<35s} {v:>6d} files")

    # Check a source sample
    src_vols = glob.glob(os.path.join(out_dir, "source/volumes/*.npy"))
    if src_vols:
        s = np.load(src_vols[0])
        log.info(f"  Source volume sample shape  : {s.shape}  dtype: {s.dtype}")
    src_masks = glob.glob(os.path.join(out_dir, "source/masks/*.npy"))
    if src_masks:
        m = np.load(src_masks[0])
        uniq = np.unique(m)
        log.info(f"  Source mask sample shape    : {m.shape}  unique values: {uniq}")

    tgt_vols = glob.glob(os.path.join(out_dir, "target/volumes/*.npy"))
    if tgt_vols:
        t = np.load(tgt_vols[0])
        log.info(f"  Target volume sample shape  : {t.shape}  dtype: {t.dtype}")

    log.info("=" * 55)

    # Save JSON summary
    json_path = os.path.join(out_dir, "stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"  Stats saved to {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Vox-UDA data")
    p.add_argument("--sim_dir",    default=DEFAULT_SIM_DIR)
    p.add_argument("--polyga_dir", default=DEFAULT_POLYGA_DIR)
    p.add_argument("--out_dir",    default=DEFAULT_OUT_DIR)
    p.add_argument("--patch_size", type=int, default=PATCH_SIZE)
    p.add_argument("--eval_vol_dir",  default=None,
                   help="Dir with Poly-GA volumes for evaluation (if labels available)")
    p.add_argument("--eval_mask_dir", default=None,
                   help="Dir with Poly-GA binary masks for evaluation")
    return p.parse_args()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 55)
    log.info("  VOX-UDA DATA PREPROCESSING")
    log.info("=" * 55)
    log.info(f"  Simulated data : {args.sim_dir}")
    log.info(f"  Poly-GA data   : {args.polyga_dir}")
    log.info(f"  Output dir     : {args.out_dir}")
    log.info(f"  Patch size     : {args.patch_size}³")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Source (simulated H5)
    n_src = process_source(args.sim_dir, args.out_dir, args.patch_size)

    # 2. Target (Poly-GA MRC, no labels)
    n_tgt = process_target(args.polyga_dir, args.out_dir, args.patch_size)

    # 3. Evaluation split (only if supervisor provides masks)
    if args.eval_vol_dir and args.eval_mask_dir:
        process_target_eval(args.eval_vol_dir, args.eval_mask_dir,
                            args.out_dir, args.patch_size)

    # 4. Print summary
    print_stats(args.out_dir)

    log.info("Preprocessing complete.")
    log.info("")
    log.info("  Next step — update configs/config.py paths:")
    log.info(f"    source_volume_dir = '{args.out_dir}/source/volumes'")
    log.info(f"    source_mask_dir   = '{args.out_dir}/source/masks'")
    log.info(f"    target_volume_dir = '{args.out_dir}/target/volumes'")