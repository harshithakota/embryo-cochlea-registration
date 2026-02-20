# 🧠 Embryo Cochlea MALDI 3D Registration Pipeline
End-to-end pipeline for spectral slice reconstruction, deformable registration, 3D volume generation, and multimodal H&E alignment of embryonic cochlea MALDI imaging data.

## 🔬 Overview
This repository provides a complete workflow to:

- Process large MALDI CSV datasets
- Generate grayscale tissue slices per m/z channel
- Perform slice-to-slice deformable registration
- Reuse transforms across all spectral channels
- Construct 3D volumes (.nii.gz)
- Optionally impute missing slices
- Align registered MALDI slices with H&E histology

## 📦 Installation

```pip install antspyx numpy pandas opencv-python SimpleITK imageio tifffile tqdm matplotlib napari scipy```

## 📂 Folder Structure

preprocessing/       → CSV trimming and slice generation

registration/        → Registration + 3D reconstruction

he_alignment/        → H&E ↔ MALDI alignment

visualisation/       → Napari and volume viewers

data/                → Input/output data (ignored)


## 🚀 FULL PIPELINE (Step-by-Step)

### STEP 1 — Trim Large MALDI CSV

File:
```text
preprocessing/trim_csv.py
```

Edit:

```INPUT = "data/Cochlea_3D_TIC.csv"
OUT_DIR = Path("data/trimmed_csvs_0_100")
START = 0
END = 100
```

Run:
```python preprocessing/trim_csv.py```

### STEP 2 — Generate Grayscale Slices

File:
```text
preprocessing/generate_all_slices.py
```

Edit:

```TRIMMED_DIR = Path("data/trimmed_csvs_0_100")
OUT_ROOT = Path("data/slices_from_trimmed")
```

Run:
```python preprocessing/generate_all_slices.py```


Each m/z gets its own folder:
data/slices_from_trimmed/130.889_gray/

### STEP 3 — Register ONE Reference m/z Channel

⚠️ Important: Register one good m/z channel.

File:
```text
registration/main_registration.py
```

Edit:

```INPUT_DIR = Path("data/slices_from_trimmed/130.889_gray")
OUTPUT_DIR = Path("results_stable/best")
TRANSFORM_DIR = Path("results_stable/transforms")
REFERENCE_SLICE_NAME = "slice_078.png"
```

Run:
```python registration/main_registration.py```

Output:
```results_stable/
    best/
    transforms/
```
    
### STEP 4 — Apply Transforms to All m/z Channels

File:
```text
registration/transform_all.py
```

Edit:

```SLICES_ROOT = Path("data/slices_from_trimmed")
TRANSFORM_ROOT = Path("results_stable/transforms")
OUTPUT_ROOT = Path("data/volumes_new")
REFERENCE_SLICE_NAME = "slice_078.png"
```

Run:
```python registration/transform_all.py```

Output:
```data/volumes_new/*.nii.gz```

Each m/z now has a 3D volume.

### OPTIONAL — Build Volume from Registered PNGs

File:
```text
registration/reconstruct_3d.py
```

Run:
```python registration/reconstruct_3d.py```

### OPTIONAL — Impute Missing Slices

File:
```text
registration/impute_missing_slices.py
```

Run:
```python registration/impute_missing_slices.py```


## 🧪 H&E ↔ MALDI Alignment

### STEP 1 — Downsample H&E
Place TIFF files in:
data/he/
Run:
python he_alignment/downsample_he.py

### STEP 2 — Register H&E to MALDI
File:
he_alignment/maldi_he_reg.py
Edit:
maldi_dir = "results_stable/best"
Run:
python he_alignment/maldi_he_reg.py
Output:
he_maldi_reg/
    alignment_*.png
    warped_he_*.tif
    overlay_*.tif
    
## 🧠 Registration Strategy
- Global anchor slice
- Backward sequential registration
- Rigid (MI) → SyN (CC)
- Transform reuse across all m/z channels
- Padding for stability
- Manual bad slice handling

## 📌 Notes

- Always register only one m/z channel.
- Reuse transforms for all other spectral channels.
- Large data folders are excluded from Git.
- Ensure folder names match exactly.

## 🔬 Research Context

This pipeline reconstructs 3D volumetric structure of embryonic cochlea tissue from MALDI spectral imaging data and aligns it with histological H&E sections using deformable multimodal registration.
