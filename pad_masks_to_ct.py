#!/usr/bin/env python3
import os
import SimpleITK as sitk
import numpy as np

# Point this to your Task folder
raw_task = os.path.expanduser('~/all_data/nnUNet/nnUNet_raw_splitted/Task11_CTPelvic1K')
img_dir   = os.path.join(raw_task, 'imagesTr')
lab_dir   = os.path.join(raw_task, 'labelsTr')

for img_fname in sorted(os.listdir(img_dir)):
    if not img_fname.endswith('_0000.nii.gz'):
        continue
    pat      = img_fname[:-12]  # strip "_0000.nii.gz"
    img_path = os.path.join(img_dir, img_fname)
    mask_path= os.path.join(lab_dir, pat + '.nii.gz')
    if not os.path.exists(mask_path):
        print(f'⚠️  No mask for {pat}, skipping.')
        continue

    img = sitk.ReadImage(img_path)
    m   = sitk.ReadImage(mask_path)
    ia  = sitk.GetArrayFromImage(img)  # (z,y,x)
    ma  = sitk.GetArrayFromImage(m)

    if ia.shape != ma.shape:
        padded = ma.copy()
        # pad or crop each axis so padded.shape == ia.shape
        for axis in range(3):
            ts = ia.shape[axis]
            ss = padded.shape[axis]
            delta = ts - ss
            if delta > 0:  # pad
                bef = delta // 2
                aft = delta - bef
                pad_spec = [(0,0)]*3
                pad_spec[axis] = (bef, aft)
                padded = np.pad(padded, pad_spec, mode='constant', constant_values=0)
            elif delta < 0:  # crop
                delta = -delta
                bef = delta // 2
                aft = delta - bef
                sl = [slice(None)]*3
                sl[axis] = slice(bef, ss - aft)
                padded = padded[tuple(sl)]

        assert padded.shape == ia.shape, f"{pat} → {padded.shape} != {ia.shape}"
        # write back
        new_m = sitk.GetImageFromArray(padded.astype(np.uint8))
        new_m.SetOrigin(   m.GetOrigin())
        new_m.SetSpacing(  m.GetSpacing())
        new_m.SetDirection(m.GetDirection())
        sitk.WriteImage(new_m, mask_path)
        print(f'✅  {pat}: {ma.shape} → {ia.shape}')
