# pad_masks_to_ct.py
import os
import SimpleITK as sitk
import numpy as np

# Point this to your Task folder
raw_task = os.path.expanduser('~/all_data/nnUNet/nnUNet_raw/Task11_CTPelvic1K')
img_dir = os.path.join(raw_task, 'imagesTr')
lab_dir = os.path.join(raw_task, 'labelsTr')

for img_fname in os.listdir(img_dir):
    if not img_fname.endswith('.nii.gz'): continue
    pat = img_fname[:-7]  # drop .nii.gz
    img_path = os.path.join(img_dir, img_fname)
    mask_path = os.path.join(lab_dir, pat + '.nii.gz')
    if not os.path.exists(mask_path):
        print(f'⚠️  No mask found for {pat}, skipping.')
        continue

    # Read
    img = sitk.ReadImage(img_path)
    m  = sitk.ReadImage(mask_path)
    im_arr = sitk.GetArrayFromImage(img)    # shape = (z,y,x)
    ma_arr = sitk.GetArrayFromImage(m)

    if im_arr.shape == ma_arr.shape:
        continue

    # Compute per‑axis pad or crop
    target_shape = im_arr.shape
    src_shape    = ma_arr.shape
    padded = ma_arr
    for axis in range(3):
        ts = target_shape[axis]
        ss = padded.shape[axis]
        delta = ts - ss
        if delta > 0:
            # pad: split front/back
            bef = delta // 2
            aft = delta - bef
            pad_spec = [(0,0)]*3
            pad_spec[axis] = (bef, aft)
            padded = np.pad(padded, pad_spec, mode='constant', constant_values=0)
        elif delta < 0:
            # crop: split front/back
            delta = -delta
            bef = delta // 2
            aft = delta - bef
            slicer = [slice(None)]*3
            slicer[axis] = slice(bef, ss - aft)
            padded = padded[tuple(slicer)]

    assert padded.shape == target_shape, f"Failed to pad {pat}"
    # Write back
    new_m = sitk.GetImageFromArray(padded)
    new_m.SetOrigin(m.GetOrigin())
    new_m.SetSpacing(m.GetSpacing())
    new_m.SetDirection(m.GetDirection())
    sitk.WriteImage(new_m, mask_path)
    print(f'✅  Padded mask {pat} from {src_shape} → {target_shape}')
