import time
from pathlib import Path

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import datasets
from nilearn.input_data import MultiNiftiMasker
from nilearn.image import get_data


from nilearn.mass_univariate import permuted_ols
from sklearn.feature_selection import VarianceThreshold

n_subjects = 2136
var_threshold = 0.001
smoothness = 12
permutations = 5000
jobs = 15

t0 = time.perf_counter()
print('loading and preprocessing data ...')


dir = Path('/mnt/qdata/raheppt1/data/brainage/nako/interim/vbm/test4')
files = sorted(list(dir.glob('*nii')))
keys = [f.stem[:6] for f in files]
tiv = Path('/mnt/qdata/raheppt1/data/brainage/nako/interim/vbm/test4/report/TIV_test4.txt')
tiv = np.array([float(l.split('\t')[0]) for l in tiv.open('r').readlines()])

info = pd.read_csv('/mnt/qdata/raheppt1/data/brainage/nako/interim/nako_age_labels.csv').astype({'key': str, 'age': np.float64})
info = info.set_index('key')
metadata = pd.merge(info.loc[keys]['age'], pd.DataFrame.from_dict({'key': keys, 'tiv': tiv}), how='inner', on='key')
metadata.set_index('key')
dir = Path('/mnt/qdata/raheppt1/data/brainage/nako/interim/vbm/test4/mri')
proc_files = sorted(list(dir.glob('mwp1*nii')))
proc_keys = [f.stem[4:10] for f in proc_files]
proc_metadata = metadata.set_index('key').loc[proc_keys]
print(len(proc_metadata))

gray_matter_map_filenames = proc_files
gray_matter_map_filenames = sorted([str(f) for f in gray_matter_map_filenames])[:n_subjects]
age = np.array(proc_metadata['age'].tolist())[:n_subjects]
tiv = np.array(proc_metadata['tiv'].tolist())[:n_subjects]
tiv[np.isnan(tiv)] = 0
tiv = tiv[:, np.newaxis]
nifti_masker = MultiNiftiMasker(standardize=False, smoothing_fwhm=smoothness, memory=None, n_jobs=jobs, verbose=1)  #, cache options
gm_maps_masked = nifti_masker.fit_transform(gray_matter_map_filenames)
gm_maps_masked = np.concatenate(gm_maps_masked, axis=0)
n_samples, n_features = gm_maps_masked.shape
print('%d samples, %d features' % (n_subjects, n_features)) 
print(f'{time.perf_counter() - t0} s')

### Inference with massively univariate model ###
print("Massively univariate model")

# Remove features with too low between-subject variance
variance_threshold = VarianceThreshold(threshold=var_threshold)

# Statistical inference
data = variance_threshold.fit_transform(gm_maps_masked)
#data = gm_maps_masked

neg_log_pvals, t_scores_original_data, _ = permuted_ols(
    age, data,  # + intercept as a covariate by default
    confounding_vars=tiv,
    n_perm=permutations,  # 1,000 in the interest of time; 10000 would be better
    n_jobs=jobs)  # CPUs
signed_neg_log_pvals = neg_log_pvals * np.sign(t_scores_original_data)
signed_neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    variance_threshold.inverse_transform(signed_neg_log_pvals))
print(f'{time.perf_counter() - t0} s')
nib.save(signed_neg_log_pvals_unmasked, 'test.nii.gz')