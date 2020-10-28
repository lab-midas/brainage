import os
from pathlib import Path

import zarr
import h5py
import dotenv
import numpy as np
import nibabel as nib
import pandas as pd 
from omegaconf import OmegaConf
from tqdm import tqdm
dotenv.load_dotenv()

DATA = Path(os.getenv('DATA'))
CONFIG = Path(os.getenv('CONFIG'))
total = 0
mni_brain = nib.load(DATA/'nako/processed/mni/cropped_MNI152_T1_1mm_brain.nii.gz')
affine = mni_brain.affine
cfg = OmegaConf.load(str(CONFIG/'volume/config.yaml'))
map_diff = np.zeros([155, 185, 155])
map_sigma = np.zeros([155, 185, 155])
info = pd.read_csv(cfg.dataset.info).astype({'key': str, 'age': np.float64})
info = info.set_index('key')
with zarr.open(str(DATA/'nako/processed/patchwise/maps.zarr')) as zf:
    keys = list(zf['agemaps'])
    for key in tqdm(keys):
        age = info.loc[key]['age']
        map_diff = map_diff + zf['agemaps'][key][0] - age
        map_sigma = map_sigma + zf['agemaps'][key][1]
        total = total + 1
map_diff = map_diff/total
map_sigma = map_sigma/total
nii = nib.Nifti1Image(map_diff, affine)
nib.save(nii, str(DATA/'nako/processed/patchwise/export/map_diff_average.nii.gz'))
nii = nib.Nifti1Image(map_sigma, affine)
nib.save(nii, str(DATA/'nako/processed/patchwise/export/map_sigma_average.nii.gz'))