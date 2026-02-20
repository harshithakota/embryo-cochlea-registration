import ants
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_DIR = Path("data/grayscale_slices")
OUTPUT_DIR = Path("results_stablee/best")
TRANSFORM_DIR = Path("results_stablee/transforms")

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TRANSFORM_DIR.mkdir(exist_ok=True, parents=True)

BAD_SLICE_NAMES = {"slice_008", "slice_026", "slice_044", "slice_062"}

# =========================
# HELPERS
# =========================
def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def save_uint8(img, path):
    cv2.imwrite(str(path), (np.clip(img, 0, 1) * 255).astype(np.uint8))

# =========================
# LOAD SLICES
# =========================
slice_paths = sorted(INPUT_DIR.glob("slice_*.png"))
n = len(slice_paths)

if n < 2:
    raise RuntimeError("Not enough slices found")

print(f"Found {n} slices")

# =========================
# GLOBAL ANCHOR = LAST SLICE
# =========================
anchor_idx = n - 1
anchor_path = slice_paths[anchor_idx]

print(f"Global anchor slice: {anchor_path.name} (index {anchor_idx})")

anchor_img = load_gray(anchor_path)
anchor_ants = ants.from_numpy(anchor_img)
anchor_ants = ants.pad_image(anchor_ants, pad_width=[80, 80])

save_uint8(anchor_img, OUTPUT_DIR / anchor_path.name)

# =========================
# BACKWARD SEQUENTIAL REGISTRATION
# Rigid → SyN
# =========================
prev = anchor_ants

for i in tqdm(range(anchor_idx - 1, -1, -1), desc="Registering backward"):

    slice_name = slice_paths[i].stem

    # 🚫 Skip bad slices
    if slice_name in BAD_SLICE_NAMES:
        print(f"⚠️ Skipping bad slice: {slice_paths[i].name}")
        continue

    img = load_gray(slice_paths[i])
    moving = ants.from_numpy(img)

    # =========================
    # STAGE 1: RIGID
    # =========================
    rigid = ants.registration(
        fixed=prev,
        moving=moving,
        type_of_transform="Rigid",
        aff_metric="MI",
        reg_iterations=(40, 20, 0),
        shrink_factors=(2, 1),
        smoothing_sigmas=(1, 0),
        grad_step=0.05,
        verbose=False
    )

    # =========================
    # STAGE 2: SyN (NO affine)
    # =========================
    syn = ants.registration(
        fixed=prev,
        moving=rigid["warpedmovout"],
        type_of_transform="SyN",
        syn_metric="CC",
        reg_iterations=(20, 10, 0),
        shrink_factors=(2, 1),
        smoothing_sigmas=(1, 0),
        grad_step=0.04,
        verbose=False
    )

    warped = syn["warpedmovout"].numpy()

    # =========================
    # SAVE TRANSFORMS
    # =========================
    slice_tf_dir = TRANSFORM_DIR / slice_name
    slice_tf_dir.mkdir(exist_ok=True)

    for tf in rigid["fwdtransforms"] + syn["fwdtransforms"]:
        tf_path = Path(tf)
        tf_path.rename(slice_tf_dir / tf_path.name)

    # =========================
    # SAVE WARPED IMAGE
    # =========================
    save_uint8(warped, OUTPUT_DIR / slice_paths[i].name)

    # Update prev with stable warped result
    prev = ants.from_numpy(warped)

print("\n✅ Rigid → SyN sequential anchoring complete")
print(f"Results saved to: {OUTPUT_DIR}")

