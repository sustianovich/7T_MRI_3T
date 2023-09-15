import os
from helpers import *

import ants
import SimpleITK as sitk

print(f"AntsPy version = {ants.__version__}")
print(f"SimpleITK version = {sitk.__version__}")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
print(f"project folder = {BASE_DIR}")
raw_examples = [
    "brain-lesion_T1w.nii.gz",
]
# Raw image
raw_example = raw_examples[0]
raw_img_path = os.path.join(BASE_DIR, "assets", "raw_examples", raw_example)
raw_img_ants = ants.image_read(raw_img_path, reorient="IAL")

explore_3D_array(arr=raw_img_ants.numpy())
mask_example = add_suffix_to_filename(raw_example, suffix="mask")
mask_img_path = os.path.join(BASE_DIR, "assets", "raw_examples", mask_example)
mask_img_ants = ants.image_read(mask_img_path, reorient="IAL")

explore_3D_array_with_mask_contour(arr=raw_img_ants.numpy(), mask=mask_img_ants.numpy())
# Template image
template_img_path = os.path.join(
    BASE_DIR, "assets", "templates", "mni_icbm152_t1_tal_nlin_sym_09a.nii"
)
template_img_ants = ants.image_read(template_img_path, reorient="IAL")

explore_3D_array(arr=template_img_ants.numpy())
print("\t\tRAW IMG")
print(raw_img_ants)

print("\t\tTEMPLATE IMG")
print(template_img_ants)
# Registration
transformation = ants.registration(
    fixed=template_img_ants, moving=raw_img_ants, type_of_transform="SyN", verbose=True
)
print(transformation)
registered_img_ants = transformation["warpedmovout"]

explore_3D_array(arr=registered_img_ants.numpy())
out_folder = os.path.join(BASE_DIR, "assets", "preprocessed")
out_folder = os.path.join(
    out_folder, raw_example.split(".")[0]
)  # create folder with name of the raw file
os.makedirs(out_folder, exist_ok=True)  # create folder if not exists

out_filename = add_suffix_to_filename(raw_example, suffix="registered")
out_path = os.path.join(out_folder, out_filename)

print(raw_img_path[len(BASE_DIR) :])
print(out_path[len(BASE_DIR) :])
registered_img_ants.to_file(out_path)
# Move raw mask from native space.
registered_mask_img_ants = ants.apply_transforms(
    moving=mask_img_ants,
    fixed=transformation["warpedmovout"],
    transformlist=transformation["fwdtransforms"],
    verbose=True,
)
explore_3D_array_with_mask_contour(
    arr=registered_img_ants.numpy(), mask=registered_mask_img_ants.numpy()
)
out_filename = add_suffix_to_filename(mask_example, suffix="registered")
out_path = os.path.join(out_folder, out_filename)

print(out_path[len(BASE_DIR) :])
registered_mask_img_ants.to_file(out_path)
