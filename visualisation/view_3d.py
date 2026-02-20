import SimpleITK as sitk
import napari
import numpy as np

# load volume
img = sitk.ReadImage("data/volumes_new/102.057.nii.gz")
volume = sitk.GetArrayFromImage(img)

print("Volume shape:", volume.shape)

lo = np.percentile(volume, 5)
hi = np.percentile(volume, 98)

viewer = napari.Viewer(ndisplay=3)

layer = viewer.add_image(
    volume,
    name="Registered cochlea",
    colormap="gray",
    rendering="attenuated_mip",
    gamma=0.6,
    contrast_limits=(lo, hi),
    blending="translucent_no_depth",
)

layer.interpolation = "nearest"

napari.run()
