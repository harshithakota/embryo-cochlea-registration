import os
import ants
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tifffile import imwrite

# ---------------------------
# Paths
# ---------------------------
base_dir = os.path.join(os.path.dirname(__file__), "./")
he_dir = os.path.join(base_dir, "data/he_downsample")
maldi_dir = os.path.join(base_dir, "results/best")

output_dir_images = os.path.join(os.path.dirname(__file__), "", "he_maldi_reg")
os.makedirs(output_dir_images, exist_ok=True)

# ---------------------------
# Pair files by prefix 
# ---------------------------
import re

# ---------------------------
# Pair files by numeric index
# ---------------------------
he_files = sorted([f for f in os.listdir(he_dir) if f.endswith(".tif")])
maldi_files = sorted([f for f in os.listdir(maldi_dir) if f.endswith(".png")])

# Create index → filename dictionary for MALDI
maldi_dict = {}

for f in maldi_files:
    match = re.search(r'\d+', f)
    if match:
        idx = int(match.group())
        maldi_dict[idx] = f

paired_files = []

for he_file in he_files:
    match = re.search(r'\d+', he_file)
    if match:
        idx = int(match.group())
        if idx in maldi_dict:
            paired_files.append((he_file, maldi_dict[idx]))
        else:
            print(f"⚠️ No MALDI slice found for H&E index {idx}")

print(f"Found {len(paired_files)} matched slice pairs")


# ---------------------------
# Process each pair
# ---------------------------
for idx, (he_file, maldi_file) in enumerate(tqdm(paired_files, desc="Processing pairs")):

    # ---------------------------
    # Load images
    # ---------------------------
    he_gray = cv2.imread(os.path.join(he_dir, he_file), cv2.IMREAD_GRAYSCALE)
    he_rgb  = cv2.imread(os.path.join(he_dir, he_file), cv2.IMREAD_COLOR)
    he_rgb  = cv2.cvtColor(he_rgb, cv2.COLOR_BGR2RGB)
    # Flip H&E left-right (horizontal flip)
    he_gray = cv2.flip(he_gray, 1)
    he_rgb  = cv2.flip(he_rgb, 1)


    maldi_img = cv2.imread(os.path.join(maldi_dir, maldi_file), cv2.IMREAD_GRAYSCALE)

    if he_gray is None or he_rgb is None or maldi_img is None:
        print(f"⚠️ Skipping due to missing image for {he_file}")
        continue

    # ---------------------------
    # Resize to MALDI resolution
    # ---------------------------
    target_shape = (maldi_img.shape[1], maldi_img.shape[0])

    he_gray_resized = cv2.resize(he_gray, target_shape).astype(np.float32)
    he_rgb_resized  = cv2.resize(he_rgb, target_shape).astype(np.float32) / 255.0

    # ---------------------------
    # Normalize
    # ---------------------------
    he_gray_resized = (he_gray_resized - he_gray_resized.min()) / (
        he_gray_resized.max() - he_gray_resized.min() + 1e-8
    )

    maldi_img_norm = (maldi_img - maldi_img.min()) / (
        maldi_img.max() - maldi_img.min() + 1e-8
    )

    # ---------------------------
    # ANTs Registration (UNCHANGED)
    # ---------------------------
    fixed = ants.from_numpy(maldi_img_norm)
    moving = ants.from_numpy(he_gray_resized)

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN"
    )

    warped = reg["warpedmovout"].numpy()
    warped_norm = (warped - warped.min()) / (warped.max() - warped.min() + 1e-8)

    # ---------------------------
    # Warp RGB H&E using SAME transform (NEW)
    # ---------------------------
    warped_rgb = np.zeros_like(he_rgb_resized)

    for c in range(3):
        channel = ants.from_numpy(he_rgb_resized[..., c])
        warped_channel = ants.apply_transforms(
            fixed=fixed,
            moving=channel,
            transformlist=reg["fwdtransforms"]
        )
        warped_rgb[..., c] = warped_channel.numpy()

    # ---------------------------
    # Overlay generation (UNCHANGED)
    # ---------------------------
    background_mask = maldi_img_norm < 0.1
    maldi_img_norm[background_mask] = 0

    he_vis = warped_norm ** 0.7
    maldi_vis = maldi_img_norm ** 0.5

    overlay = np.zeros((maldi_img_norm.shape[0], maldi_img_norm.shape[1], 3))
    overlay[..., 0] = he_vis
    overlay[..., 1] = maldi_vis
    overlay[..., 2] = 0

    # ---------------------------
    # Visualization (UNCHANGED)
    # ---------------------------
    plt.figure(figsize=(20, 5))

    panels = [
        (maldi_img_norm, "MALDI (Fixed)"),
        (he_gray_resized, "H&E (Resized)"),
        (warped_norm, "Warped H&E"),
        (overlay, "Overlay (RGB)")
    ]

    for i, (img, title) in enumerate(panels, 1):
        plt.subplot(1, 4, i)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()

    prefix = he_file.replace(".tif", "")
    if prefix.endswith("_he"):
        prefix = prefix[:-3]

    plt.savefig(os.path.join(output_dir_images, f"alignment_{prefix}.png"),
                bbox_inches="tight")
    plt.close()

    # ---------------------------
    # Save outputs
    # ---------------------------
    imwrite(
        os.path.join(output_dir_images, f"warped_he_{prefix}.tif"),
        (warped_norm * 255).astype(np.uint8)
    )

    imwrite(
        os.path.join(output_dir_images, f"overlay_{prefix}.tif"),
        (overlay * 255).astype(np.uint8)
    )

    imwrite(
        os.path.join(output_dir_images, f"warped_he_color_{prefix}.tif"),
        (np.clip(warped_rgb, 0, 1) * 255).astype(np.uint8)
    )

    print(f"Saved outputs for: {prefix}")

print("Done! Results saved in", output_dir_images)