import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom, gaussian_filter

# =========================
# CONFIG
# =========================
IN_NII = "results/volume_registered.nii.gz"
OUT_NII = "results/volume_registered_imputed.nii.gz"

UPSAMPLE_FACTOR = 2   # inserts 1 slice between each pair
Z_SMOOTH_SIGMA = 0.5  # light smoothing along Z only

# =========================
# LOAD VOLUME
# =========================
img = sitk.ReadImage(IN_NII)
vol = sitk.GetArrayFromImage(img)   # (Z, Y, X)

print("Original shape:", vol.shape)

# =========================
# OPTIONAL: LIGHT Z-SMOOTHING
# =========================
vol = gaussian_filter(vol, sigma=(Z_SMOOTH_SIGMA, 0, 0))

# =========================
# CORRECT Z-INTERPOLATION
# =========================
Z = vol.shape[0]
new_Z = UPSAMPLE_FACTOR * (Z - 1) + 1

zoom_factor = new_Z / Z

vol_interp = zoom(
    vol,
    zoom=(zoom_factor, 1, 1),
    order=1          # linear in Z
)

print("Imputed shape:", vol_interp.shape)

# =========================
# SAVE WITH CORRECT SPACING
# =========================
out_img = sitk.GetImageFromArray(vol_interp)

sx, sy, sz = img.GetSpacing()
out_img.SetSpacing((sx, sy, sz / UPSAMPLE_FACTOR))
out_img.SetOrigin(img.GetOrigin())
out_img.SetDirection(img.GetDirection())

sitk.WriteImage(out_img, OUT_NII)

print("Saved:", OUT_NII)
