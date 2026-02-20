import ants
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import napari

# =========================
# CONFIG
# =========================
TRANSFORM_DIR = Path("results/transforms")


INPUT_DIR = Path("data/slices/130.889_gray")
OUTPUT_DIR = Path("data/warped/130.889.new")

REFERENCE_SLICE = "slice_074.png"   # anchor slice 

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD REFERENCE (FIXED IMAGE)
# =========================
ref_path = INPUT_DIR / REFERENCE_SLICE
ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

if ref_img is None:
    raise RuntimeError(f"Failed to load reference slice: {REFERENCE_SLICE}")

ref_img = ref_img.astype(np.float32) / 255.0
fixed = ants.from_numpy(ref_img)

# 🔧 ADD PADDING (CRITICAL)
PAD = 80   # start with 60–100, tune if needed
fixed = ants.pad_image(
    fixed,
    pad_width=[PAD, PAD]
)

# fixed = ants.from_numpy(ref_img)

H, W = ref_img.shape
print(f"✔ Reference grid: {H} x {W}")

# =========================
# APPLY TRANSFORMS
# =========================
warped_stack = []
slice_names = []

slice_paths = sorted(INPUT_DIR.glob("slice_*.png"))

print("\n🔹 Applying transforms to slices")

for sp in tqdm(slice_paths):
    sid = sp.stem          # e.g. slice_026
    tdir = TRANSFORM_DIR / sid

    if not tdir.exists():
        print(f"⚠️ Missing transform folder for {sid}, skipping")
        continue

    affine = list(tdir.glob("*Affine.mat"))
    warp = list(tdir.glob("*Warp.nii.gz"))

    if not affine or not warp:
        print(f"⚠️ Missing transforms for {sid}, skipping")
        continue

    # load slice
    img = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Failed to read {sid}, skipping")
        continue

    img = img.astype(np.float32) / 255.0
    moving = ants.from_numpy(img)

    # apply transforms
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(warp[0]), str(affine[0])],
        interpolator="linear"
    )

    warped_np = warped.numpy()

    # save warped slice
    out_path = OUTPUT_DIR / f"{sid}.png"
    cv2.imwrite(
        str(out_path),
        (np.clip(warped_np, 0, 1) * 255).astype(np.uint8)
    )

    warped_stack.append(warped_np)
    slice_names.append(sid)

print(f"\n✅ Saved {len(warped_stack)} warped slices to {OUTPUT_DIR}")

# =========================
# NAPARI VISUALIZATION
# =========================
print("Launching napari...")

volume = np.stack(warped_stack, axis=0)

viewer = napari.Viewer()
viewer.add_image(
    volume,
    name="Warped slices",
    colormap="gray"
)

napari.run()
