import numpy as np
import imageio.v2 as imageio
import os
from glob import glob
import SimpleITK as sitk

# =========================
# CONFIG
# =========================
SLICE_DIR = "results_stable/best"
OUT_NII = "results_stable/volume_registered.nii.gz"


# =========================
# LOAD SLICES 
# =========================
slice_paths = sorted(
    glob(os.path.join(SLICE_DIR, "slice_*.png")),
    key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
)

if len(slice_paths) == 0:
    raise RuntimeError("No slices found")

print(f"Found {len(slice_paths)} slices")

# =========================
# FIND MAX CANVAS SIZE
# =========================
max_h, max_w = 0, 0
for p in slice_paths:
    img = imageio.imread(p)
    max_h = max(max_h, img.shape[0])
    max_w = max(max_w, img.shape[1])

print(f"Canvas size: {max_h} x {max_w}")

# =========================
# STACK WITH CENTER PADDING
# =========================
volume = []

for p in slice_paths:
    img = imageio.imread(p)
    h, w = img.shape

    padded = np.zeros((max_h, max_w), dtype=np.float32)
    y0 = (max_h - h) // 2
    x0 = (max_w - w) // 2
    padded[y0:y0+h, x0:x0+w] = img

    volume.append(padded)

volume = np.stack(volume, axis=0)  # (Z, Y, X)
print("Final volume shape:", volume.shape)

# =========================
# SAVE AS NIFTI (.nii.gz)
# =========================
img_3d = sitk.GetImageFromArray(volume)
img_3d.SetSpacing((1.0, 1.0, 1.0)) 

sitk.WriteImage(img_3d, OUT_NII)
print(f"Saved 3D volume: {OUT_NII}")
