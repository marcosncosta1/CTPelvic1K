#!/usr/bin/env python3
import os
import SimpleITK as sitk

# adjust these to your real task folder
base = os.path.expanduser('~/all_data/nnUNet/nnUNet_raw_splitted/Task11_CTPelvic1K/')
ct_dir   = os.path.join(base, 'imagesTr')
mask_dir = os.path.join(base, 'labelsTr')

bad = []
good = []
for fn in sorted(os.listdir(ct_dir)):
    if not fn.endswith('_0000.nii.gz'): continue
    case = fn.replace('_0000.nii.gz','')
    ct_path   = os.path.join(ct_dir,   fn)
    mask_path = os.path.join(mask_dir, f'{case}.nii.gz')
    if not os.path.exists(mask_path):
        bad.append((case, 'missing mask'))
        continue
    ct = sitk.ReadImage(ct_path)
    mk = sitk.ReadImage(mask_path)
    if tuple(ct.GetSize())[::-1] == tuple(mk.GetSize())[::-1]:
        good.append(case)
    else:
        bad.append((case,
            tuple(ct.GetSize())[::-1],
            tuple(mk.GetSize())[::-1]
        ))

print(f'TOTAL cases: {len(good)+len(bad)}')
print(f'  ✓ matching:   {len(good)}')
print(f'  ✗ mismatched: {len(bad)}')
if bad:
    print('\nMISMATCHED CASES:')
    for x in bad:
        print(' ', *x)
