from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Queue
from torchio.sampler import ImageSampler

from torchio.transforms import (
    ZNormalization,
    RandomNoise,
    RandomFlip,
    RandomAffine,
)
from torchvision.transforms import Compose


subject_images = [
    Image('t1', '/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/nii3d/smwc1rIXI002-Guys-0828-MPRAGESEN_-s256_-0301-00003-000001-01.nii', INTENSITY),
    Image('liver', '/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/nii3d/smwc1rIXI002-Guys-0828-MPRAGESEN_-s256_-0301-00003-000001-01.nii', LABEL)
]

subject_list = [subject_images]
dataset = ImagesDataset(subject_list)

# get first (full) image sample
sample = dataset[0]
print(sample)

# define transforms for data normalization and augmentation
transforms = (
    ZNormalization(),
    RandomNoise(std_range=(0, 0.25)),
    RandomAffine(scales=(0.9, 1.1), degrees=10),
    RandomFlip(axes=(0,)),
)
transform = Compose(transforms)
subjects_dataset = ImagesDataset(dataset, transform)

queue_dataset = Queue(
            dataset,
            max_length=10,
            samples_per_volume=2,
            patch_size=100,
            sampler_class=ImageSampler,
            num_workers=1,
            shuffle_patches=False,
            shuffle_subjects=False)

# get first patch
sample = queue_dataset[0]['t1']['data']
label =  queue_dataset[0]['liver']['data']
print(sample.size())