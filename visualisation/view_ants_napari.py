import numpy as np
import imageio.v2 as imageio
import napari
from glob import glob
import os

# =========================
# CONFIG
# =========================
SLICE_DIR = "data/warped_slices/137.001"



# =========================
# LOAD SLICES
# =========================
slice_paths = sorted(glob(os.path.join(SLICE_DIR, "*.png")))

if len(slice_paths) == 0:
    raise RuntimeError("No registered slices found!")

print(f"Found {len(slice_paths)} registered slices")

stack = []
max_h, max_w = 0, 0

# First pass: find max shape
for p in slice_paths:
    img = imageio.imread(p)
    max_h = max(max_h, img.shape[0])
    max_w = max(max_w, img.shape[1])

print(f"Max slice size: {max_h} x {max_w}")

# Second pass: pad + stack
for p in slice_paths:
    img = imageio.imread(p)

    padded = np.zeros((max_h, max_w), dtype=img.dtype)
    padded[: img.shape[0], : img.shape[1]] = img

    stack.append(padded)

volume = np.stack(stack, axis=0)
print("Final volume shape:", volume.shape)

# =========================
# VIEW IN NAPARI
# =========================
viewer = napari.Viewer()
viewer.add_image(
    volume,
    name="Registered MALDI",
    colormap="gray",
    contrast_limits=(
        np.percentile(volume, 1),
        np.percentile(volume, 99)
    )
)

napari.run()
