import os
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from pathlib import Path

# =========================
# CONFIG
# =========================
TRIMMED_DIR = Path("data/trimmed_csvs_200-400")   # folder with trimmed CSVs
OUT_ROOT = Path("data/slices_from_trimmed")   # output root
OUT_ROOT.mkdir(exist_ok=True)

# =========================
# PROCESS EACH TRIMMED CSV
# =========================
csv_files = sorted(TRIMMED_DIR.glob("Cochlea_3D_m.z.*.csv"))
print(f"Found {len(csv_files)} trimmed CSVs")

for csv_path in csv_files:
    print(f"\n Processing {csv_path.name}")

    # --- infer m/z column ---
    cols = pd.read_csv(csv_path, nrows=0).columns
    mz_cols = [c for c in cols if c.startswith("m.z.")]
    if len(mz_cols) != 1:
        print("  Skipping (could not uniquely identify m/z column)")
        continue

    mz_col = mz_cols[0]
    mz_val = mz_col.replace("m.z.", "")

    gray_dir = OUT_ROOT / f"{mz_val}_gray"
    gray_dir.mkdir(exist_ok=True)

    # --- load CSV ---
    df = pd.read_csv(csv_path)
    slice_ids = sorted(df["tissue_id"].unique())
    print(f"  → {len(slice_ids)} slices")

    # =========================
    # PROCESS EACH SLICE
    # =========================
    for sid in slice_ids:
        slice_df = df[df["tissue_id"] == sid]
        if slice_df.empty:
            continue

        xs = np.sort(slice_df["x"].unique())
        ys = np.sort(slice_df["y"].unique())

        x_to_col = {x: i for i, x in enumerate(xs)}
        y_to_row = {y: i for i, y in enumerate(ys)}

        img = np.zeros((len(ys), len(xs)), dtype=np.float32)

        for _, row in slice_df.iterrows():
            img[y_to_row[row["y"]], x_to_col[row["x"]]] = row[mz_col]

        # =========================
        # REGISTRATION IMAGE
        # =========================
        log_img = np.log1p(img)
        nz = log_img[log_img > 0]
        if nz.size == 0:
            continue

        vmin, vmax = np.percentile(nz, [10, 90])
        reg_norm = np.clip((log_img - vmin) / (vmax - vmin), 0, 1)
        reg_uint8 = (reg_norm * 255).astype(np.uint8)

        imageio.imwrite(
            gray_dir / f"slice_{sid:03d}.png",
            reg_uint8
        )

    print(f" Finished m/z {mz_val}")

print("\n All slices generated from trimmed CSVs")
