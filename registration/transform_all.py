import ants
import numpy as np
import cv2
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm

# =========================
# CONFIG
# =========================
SLICES_ROOT = Path("data/slices_from_trimmed")
TRANSFORM_ROOT = Path("results_stable_clean/transforms")
OUTPUT_ROOT = Path("data/volumes_new")

REFERENCE_SLICE_NAME = "slice_078.png"

BAD_SLICE_INDICES = [8, 26, 44, 62]


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# UTIL
# =========================
def numeric_slice_sort(p):
    return int(p.stem.split("_")[1])

def load_gray_np(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    return img

# =========================
# PROCESS EACH m/z FOLDER
# =========================
for mz_dir in sorted(SLICES_ROOT.glob("*_gray")):
    mz = mz_dir.name.replace("_gray", "")
    print(f"\n🚀 Processing m/z {mz}")

    slice_paths = sorted(
        mz_dir.glob("slice_*.png"),
        key=numeric_slice_sort
    )

    if not slice_paths:
        print(f"⚠️ No slices found for {mz}, skipping")
        continue

    # =========================
    # LOAD REGISTERED REFERENCE
    # =========================
    ref_path = Path("results_stable_clean/best") / REFERENCE_SLICE_NAME

    if not ref_path.exists():
        raise RuntimeError(f"Registered reference slice not found at {ref_path}")

    ref_img_np = load_gray_np(ref_path)
    fixed = ants.from_numpy(ref_img_np)

    warped_stack = []

    # Folder to save warped PNG slices
    # save_slice_dir = WARPED_SLICE_ROOT / mz
    # save_slice_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # WARP EACH SLICE
    # =========================
    for sp in tqdm(slice_paths, desc=f"  Warping slices ({mz})"):
        sid = sp.stem
        idx = int(sid.split("_")[1])

        # -----------------------------
        # Handle BAD slices (impute)
        # -----------------------------
        if idx in BAD_SLICE_INDICES:
            print(f"⚠️ Imputing bad slice {sid} by copying previous slice")

            if warped_stack:
                warped_stack.append(warped_stack[-1])
            else:
                warped_stack.append(ref_img_np)

            continue

        # -----------------------------
        # Anchor slice
        # -----------------------------
        if sid == REFERENCE_SLICE_NAME.replace(".png", ""):
            # warped_stack.append(ref_img_np)
            continue

        # -----------------------------
        # Normal transform logic
        # -----------------------------
        moving_np = load_gray_np(sp)
        moving = ants.from_numpy(moving_np)

        tdir = TRANSFORM_ROOT / sid
        if not tdir.exists():
            print(f"⚠️ Missing transforms for {sid}, copying previous slice")

            if warped_stack:
                warped_stack.append(warped_stack[-1])
            else:
                warped_stack.append(ref_img_np)

            continue

        affines = sorted(tdir.glob("*GenericAffine.mat"))
        warps = sorted(tdir.glob("*Warp.nii.gz"))

        transform_list = []

        if warps:
            warp_file = warps[0]
            prefix = warp_file.name.replace("1Warp.nii.gz", "")

            syn_affine = None
            rigid_affine = None

            for a in affines:
                if a.name.startswith(prefix):
                    syn_affine = a
                else:
                    rigid_affine = a

            if syn_affine:
                transform_list.append(str(warp_file))
                transform_list.append(str(syn_affine))
            if rigid_affine:
                transform_list.append(str(rigid_affine))

        elif affines:
            transform_list = [str(affines[0])]

        else:
            print(f"No usable transforms for {sid}, copying previous slice")

            if warped_stack:
                warped_stack.append(warped_stack[-1])
            else:
                warped_stack.append(ref_img_np)

            continue

        warped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=transform_list,
            interpolator="linear"
        )

        warped_stack.append(warped.numpy())


        # Save warped PNG slice
        # out_png = save_slice_dir / f"{sid}.png"
        # cv2.imwrite(str(out_png), (arr * 255).astype(np.uint8))

    if not warped_stack:
        print(f"⚠️ No slices warped for {mz}")
        continue

    # =========================
    # STACK → NIFTI
    # =========================
    volume = np.stack(warped_stack, axis=0)

    sitk_img = sitk.GetImageFromArray(volume)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))

    out_path = OUTPUT_ROOT / f"{mz}.nii.gz"
    sitk.WriteImage(sitk_img, str(out_path))

    print(f"Saved {out_path}")

print("\nALL m/z volumes generated successfully")
