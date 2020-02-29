from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def slicedir(in_dir,
             out_dir,
             axis=2):
    """
    Converts 3D nii file directory into a new directory of 2D slices.
    Foreach 3D dataset a new directory for the slices is created.
    The filenames of the slices are <3d_file_name>_sli<slice_number>.nii

    Args:
        in_dir: dir with 3d nii (path as str or pathlib.Path)
        out_dir: dir to export 2d nii to (path as str or pathlib.Path)

    Returns:

    """
    print(in_dir)
    for file in in_dir.glob('*nii*'):
        img = sitk.ReadImage(str(file))
        sl_num = img.GetSize()[2]
        print(f'file {str(file)}, slices {sl_num}')
        # create directory in 2d directory
        target_dir = out_dir.joinpath(file.stem)
        target_dir.mkdir(exist_ok=True)

        # iterate slices
        for sl_idx in range(sl_num):
            sl_img = img[:, :, sl_idx]
            sl_img_a = sitk.GetArrayFromImage(sl_img).transpose([1, 0])

            # new_filename is filename_sl<sl_idx>.nii
            #new_filename = file.name.replace('.nii', f'_sl{str(sl_idx).zfill(4)}.nii')
            #new_file = target_dir.joinpath(new_filename)
            #sitk.WriteImage(sl_img, str(new_file))
            new_filename = file.name.replace('.nii', f'_sl{str(sl_idx).zfill(4)}.npy')
            new_file = target_dir.joinpath(new_filename)
            np.save(str(new_file), sl_img_a)


def main():
    work_dir = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/nii3d')
    out_dir =  Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/npy2dz')

    slicedir(work_dir, out_dir)

if __name__ == '__main__':
    main()

