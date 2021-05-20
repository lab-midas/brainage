from genericpath import exists
from pathlib import Path

from p_tqdm import p_map
import nibabel as nib
import nilearn.image

fwhm = 10
cpus = 8
gm_data = Path('/mnt/qdata/raheppt1/data/brainage/nako/interim/vbm/test4/mri')
out_data = (gm_data/'smooth')
out_data.mkdir(exist_ok=True)
files = list(gm_data.glob('mwp1*.nii')) 

def process(f):
    img = nib.load(f)
    img_smooth = nilearn.image.smooth_img(img, fwhm)
    nib.save(img_smooth, str(out_data/(f.stem + f'_smooth{fwhm}.nii')))

added = p_map(process, files, num_cpus=cpus)