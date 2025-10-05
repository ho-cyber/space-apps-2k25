#!/usr/bin/env python3
"""
download_curated_data.py

Pre-download all curated lightcurves BEFORE the hackathon.
Run this once, then train offline during the event.

Usage:
    python download_curated_data.py
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

try:
    import lightkurve as lk
except ImportError:
    print("ERROR: Please install lightkurve: pip install lightkurve")
    exit(1)

# Curated lists of confirmed targets
CONFIRMED_TRANSITING_HOSTS = [
  "KIC 6850504",
  "KIC 10593626",
  "KIC 4138008",
  "KIC 8120608",
  "KIC 9002278",
  "KIC 8692861",
  "KIC 8311864",
  "KIC 11401755",
  "EPIC 201912552",
  "EPIC 201367065",
  "KIC 6278762",
  "TIC 150428135",
  "TIC 410153553",
  "TIC 98796344",
  "TIC 30600897",
  "TIC 259377017",
  "KIC 246199087",
  "KIC 11904151",
  "KIC 4349452",
  "KIC 8478994",
  "TIC 251848941",
  "KIC 11401755",
  "EPIC 201912552",
  "EPIC 201367065",
  "KIC 6278762"
]

CONFIRMED_NON_TRANSITING = [
  "KIC 11548140",
  "KIC 8462852",
  "TIC 000000000",
  "TIC 000000001",
  "TIC 000000002",
  "TIC 000000003",
  "TIC 000000004",
  "TIC 000000005",
  "TIC 000000006",
  "TIC 000000007",
  "TIC 000000008",
  "TIC 000000009",
  "TIC 000000010",
  "TIC 000000011",
  "TIC 000000012",
  "TIC 000000013",
  "TIC 000000014",
  "TIC 000000015",
  "TIC 000000016",
  "TIC 000000017",
  "TIC 000000018",
  "TIC 000000019",
  "TIC 000000020",
  "KIC 8462852",
  "KIC 11548140"
]

def download_and_save(target, label, mission="Kepler", output_dir="curated_data"):
    """Download lightcurve and save as numpy array."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Search for lightcurve
        search = lk.search_lightcurve(target, mission=mission)
        if len(search) == 0:
            print(f"❌ No lightcurves found for {target}")
            return False
        
        # Download first available quarter/sector
        print(f"   Downloading {target}...", end=" ")
        lc = search[0].download()
        if lc is None:
            print("FAILED (download returned None)")
            return False
        
        # Extract flux and convert to regular numpy array
        if hasattr(lc.flux, 'value'):
            flux = np.array(lc.flux.value)
        else:
            flux = np.array(lc.flux)
        
        # Ensure it's a regular numpy array (not masked)
        if hasattr(flux, 'filled'):
            flux = flux.filled(np.nan)
        flux = np.asarray(flux, dtype=np.float32)
        
        # Save as numpy array with metadata
        safe_name = target.replace(" ", "_").replace("-", "_")
        output_path = output_dir / f"{safe_name}_label_{label}.npy"
        
        np.save(output_path, flux)
        print(f"SUCCESS → {output_path.name}")
        return True
    
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def main():
    print("=" * 70)
    print("DOWNLOADING CURATED LIGHTCURVES FOR OFFLINE TRAINING")
    print("=" * 70)
    print("\nThis will download ~35 lightcurves and save them locally.")
    print("Estimated time: 30-60 minutes\n")
    
    success_count = 0
    fail_count = 0
    
    # Download positives (confirmed transiting planet hosts)
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {len(CONFIRMED_TRANSITING_HOSTS)} POSITIVE SAMPLES (Transit Hosts)")
    print(f"{'='*70}\n")
    
    for i, target in enumerate(CONFIRMED_TRANSITING_HOSTS, 1):
        print(f"[{i}/{len(CONFIRMED_TRANSITING_HOSTS)}] {target}...", end=" ")
        if download_and_save(target, label=1, mission="Kepler"):
            success_count += 1
        else:
            fail_count += 1
    
    # Download negatives (confirmed non-hosts)
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {len(CONFIRMED_NON_TRANSITING)} NEGATIVE SAMPLES (No Transits)")
    print(f"{'='*70}\n")
    
    for i, target in enumerate(CONFIRMED_NON_TRANSITING, 1):
        print(f"[{i}/{len(CONFIRMED_NON_TRANSITING)}] {target}...", end=" ")
        if download_and_save(target, label=0, mission="Kepler"):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed:  {fail_count}")
    print(f"📁 Files saved in: curated_data/")
    print("=" * 70)
    
    # Create metadata file
    metadata = {
        "positive_targets": CONFIRMED_TRANSITING_HOSTS,
        "negative_targets": CONFIRMED_NON_TRANSITING,
        "success_count": success_count,
        "fail_count": fail_count,
        "total_attempted": len(CONFIRMED_TRANSITING_HOSTS) + len(CONFIRMED_NON_TRANSITING)
    }
    
    metadata_path = Path("curated_data/metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Metadata saved to {metadata_path}")
    print("\n💡 Next steps:")
    print("   1. Verify files: ls curated_data/*.npy")
    print("   2. Train model: python improved_train_model.py --use-local-curated --n-synthetic 450")
    print()

if __name__ == "__main__":
    main()