import pandas as pd
from pathlib import Path
import re

# Cleaning the IXI excel sheet for duplicate and non valid entries.
# -> 565 subjects remaining.

def check_IXIPP(df_info):

    print(f'Checking preprocessed IXI dataset (#{len(df_info.index)})...')
    # Get valid subjects.
    subject_list = df_info['IXI_ID'].values
    path_pp_nii = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/nii3d')
    path_pp_info = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/IXI_PP_cleaned.csv'
    pattern = 'IXI{:03d}*'
    # Check if image data is available for all subjects.
    for id in subject_list:
        if not list(path_pp_nii.glob(pattern.format(id))):
            print(f'- {id} not found')
            df_info = df_info[df_info['IXI_ID'] != id]
    print(f'-> #{len(df_info.index)} datasets left')
    # Save to IXI_cleaned csv file.
    df_info.to_csv(path_pp_info)


def check_IXIT(df_info, i):

    print(f'Checking IXI T{i} dataset (#{len(df_info.index)})...')
    # Get valid subjects.
    subject_list = df_info['IXI_ID'].values
    path_nii = Path(f'/mnt/share/raheppt1/project_data/brain/IXI/IXI_T{i}/IXI_T{i}')
    path_info = f'/mnt/share/raheppt1/project_data/brain/IXI/IXI_T{i}/IXI_T{i}_cleaned.csv'
    pattern = 'IXI{:03d}*'
    # Check if image data is available for all subjects.
    for id in subject_list:
        if not list(path_nii.glob(pattern.format(id))):
            print(f'- {id} not found')
            df_info = df_info[df_info['IXI_ID'] != id]
    print(f'-> #{len(df_info.index)} datasets left')
    # Save to IXI_cleaned csv file.
    df_info.to_csv(path_info)


def main():
    path_info = '/mnt/share/raheppt1/project_data/brain/IXI/IXI.xls'
    path_info_cleaned = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_cleaned.csv'

    # Read xls to dataframe.
    df_info = pd.read_excel(path_info, index_col=0)

    # Remove duplicated subjects
    df_info = df_info.loc[~df_info.index.duplicated(keep='first')]

    # Keep only subjects with valid age entries.
    df_info = df_info[df_info['AGE'] > 0]
    df_info = df_info[df_info['SEX_ID'] > 0]

    # Save to IXI_cleaned csv file.
    df_info.to_csv(path_info_cleaned)

    print(df_info)

    df_info = pd.read_csv(path_info_cleaned)
    print(len(df_info.index))
    check_IXIPP(df_info)
    check_IXIT(df_info, 1)
    print(len(df_info.index))

if __name__ == '__main__':
    main()