import tfrec
from pathlib import Path
import re

# todo adapt to changes in main tfrec.py
def convert_nii_files(filenames,
                      path_dir_tfrec,
                      series_tag=''):
    """ Converts list of nii files to tfrec.

    Args:
        filenames: List with nii files to be converted to tfrec.
        path_dir_tfrec: (pathlib.Path), where tfrec files should be saved.
        series_tag: Tags the series type in the tfrec filename (<series_tag><id>.tfrec)

    Returns:

    """
    for file in filenames:
        print(f'Nii file: {file.name}')
        # Get patient id from file name.
        m = re.match('.*IXI([0-9]{3}).*', file.name)
        pat_id = int(m.group(1))
        print(f'ID: {pat_id}')

        # Convert nii file to tfrec.
        tfrec.convert_nii_to_tfrec(file,
                                   path_dir_tfrec.joinpath(series_tag+str(pat_id).zfill(4)+'.tfrec'),
                                   idtag=str(pat_id))

def main():
    path_dir_nii = Path('/Users/tobiashepp/projects/age_prediction/data/IXI')
    path_dir_tfrec = Path('/Users/tobiashepp/projects/age_prediction/data/tfrec')

    # Get list with all nii files in path_dir_nii.
    filenames = list(path_dir_nii.glob('*.nii'))
    filenames.sort()
    # Convert all nii files to tfrec and save tfrec files in path_dir_tfrec.
    convert_nii_files(filenames, path_dir_tfrec, series_tag='PP')


if __name__ == '__main__':
    main()


