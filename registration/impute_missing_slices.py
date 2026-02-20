import cv2
from pathlib import Path
import shutil

# =====================================================
# CONFIG
# =====================================================
INPUT_DIR = Path("results_stable_clean/best")
OUTPUT_DIR = Path("data/results_stable_clean_imputed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

BAD_SLICE_INDICES = [8, 26, 44, 62]

# =====================================================
# STEP 1: COPY ALL EXISTING SLICES
# =====================================================

for file in sorted(INPUT_DIR.glob("slice_*.png")):
    shutil.copy(file, OUTPUT_DIR / file.name)

print("Existing slices copied.")

# =====================================================
# STEP 2: COPY PREVIOUS SLICE FOR MISSING ONES
# =====================================================

for idx in BAD_SLICE_INDICES:

    prev_path = INPUT_DIR / f"slice_{idx-1:03d}.png"
    new_path  = OUTPUT_DIR / f"slice_{idx:03d}.png"

    if not prev_path.exists():
        print(f"⚠ Cannot create slice_{idx:03d}, previous slice missing")
        continue

    print(f"Creating slice_{idx:03d} by copying slice_{idx-1:03d}")

    shutil.copy(prev_path, new_path)

print("Missing slices filled by copying previous slice.")
