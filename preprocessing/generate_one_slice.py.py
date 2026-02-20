import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import imageio.v2 as imageio


# =========================
# CONFIG
# =========================

CSV_PATH = "data/trimmed_csv/Cochlea_3D_mz309.281.csv"
MZ_COL = "m.z.309.281"

GRAY_DIR = "data/slices/309.281_gray"
# COLOR_DIR = "data/slices/885.551_color"

os.makedirs(GRAY_DIR, exist_ok=True)
# os.makedirs(COLOR_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

slice_ids = sorted(df["tissue_id"].unique())
print("Number of slices:", len(slice_ids))

# =========================
# PROCESS EACH SLICE
# =========================
for sid in slice_ids:
    print(f"Processing slice {sid}")

    slice_df = df[df["tissue_id"] == sid]
    if slice_df.empty:
        continue

    xs = np.sort(slice_df["x"].unique())
    ys = np.sort(slice_df["y"].unique())

    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys)}

    img = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for _, row in slice_df.iterrows():
        img[y_to_row[row["y"]], x_to_col[row["x"]]] = row[MZ_COL]

    # =========================
    # REGISTRATION IMAGE
    # =========================
    log_img = np.log1p(img)

    # Light smoothing (helps registration)
    reg_img = log_img


    # Percentile clipping (stable contrast)
    nonzero = reg_img[reg_img > 0]
    if len(nonzero) == 0:
        continue

    vmin = np.percentile(nonzero, 10)
    vmax = np.percentile(nonzero, 90)

    # =========================
    # SAVE GRAYSCALE (FOR REGISTRATION) — NO BORDER
    # =========================
    reg_norm = np.clip((reg_img - vmin) / (vmax - vmin), 0, 1)
    reg_uint8 = (reg_norm * 255).astype(np.uint8)

    imageio.imwrite(
        os.path.join(GRAY_DIR, f"slice_{sid:03d}.png"),
        reg_uint8
    )


    # =========================
    # SAVE COLORED (FOR VISUALIZATION)
    # =========================
    # colored = plt.cm.magma(reg_norm)[:, :, :3]  # RGB only
    # colored_uint8 = (colored * 255).astype(np.uint8)

    # imageio.imwrite(
    #     os.path.join(COLOR_DIR, f"slice_{sid:03d}.png"),
    #     colored_uint8
    # )


print("All slices saved with registration-ready preprocessing.")
