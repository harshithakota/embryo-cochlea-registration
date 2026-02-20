import os
import tifffile as tiff
import numpy as np
from tqdm import tqdm

# ---------------------------
# Paths
# ---------------------------
base_dir = os.path.join(os.path.dirname(__file__), "./", "data")
he_dir = os.path.join(base_dir, "he")
output_dir = os.path.join(base_dir, "he_downsample")

os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Downsample factor
# ---------------------------
DOWNSAMPLE = 16

# ---------------------------
# Process all H&E files
# ---------------------------
he_files = sorted([f for f in os.listdir(he_dir) if f.endswith(".tif")])

print(f"Found {len(he_files)} H&E slices")

for he_file in tqdm(he_files, desc="Downsampling H&E"):

    input_path = os.path.join(he_dir, he_file)
    output_path = os.path.join(output_dir, he_file)

    # ---------------------------
    # Load large TIFF safely
    # ---------------------------
    with tiff.TiffFile(input_path) as tif:
        img = tif.pages[0].asarray()

    # ---------------------------
    # Downsample by slicing
    # ---------------------------
    img_small = img[::DOWNSAMPLE, ::DOWNSAMPLE]

    # ---------------------------
    # Save downsampled image
    # ---------------------------
    tiff.imwrite(output_path, img_small.astype(img.dtype))

print("✅ Done! Downsampled images saved in:", output_dir)
