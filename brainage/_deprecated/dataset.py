#import sys
#sys.path.insert(0, '/home/raheppt1/projects/MedicalDataAugmentationTool')

import os
import SimpleITK as sitk

from datasets.graph_dataset import GraphDataset
from transformations.spatial import translation, rotation, scale, flip, composite
from transformations.intensity.sitk.shift_scale_clamp import ShiftScaleClamp
from iterators.id_list_iterator import IdListIterator
from generators.label_generator import LabelGenerator
from generators.image_generator import ImageGenerator
from datasources.label_datasource import LabelDatasource
from datasources.cached_image_datasource import CachedImageDataSource

class AgeData(object):
    def __init__(self,
                 config,
                 shuffle_training_images=True,
                 output_folder='',
                 save_debug_images=False,
                 path_training_csv=None,
                 path_validation_csv=None,
                 path_test_csv=None,
                 select_testset=False,
                 default_processing=True):
        
        print(select_testset)
        print(config['path_validation_csv'])

        # Image parameters
        self.dim = 3
        self.image_size = config['image_size']
        self.image_spacing = config['image_spacing']
        self.shuffle_images = shuffle_training_images

        # Save debug images.
        self.save_debug_images = save_debug_images
        self.output_folder = './'

        # Define data directory.
        self.base_folder = config['base_folder']
        self.file_prefix = config['file_prefix']
        self.file_suffix = config['file_suffix']
        self.file_ext = config['file_ext']
  
        # csv file with ids of the training/validation images.
        if path_training_csv:
            self.path_training_csv = path_training_csv
        else:
            self.path_training_csv = config['path_training_csv']

        if path_training_csv:
            self.path_validation_csv = path_validation_csv
        else:
            self.path_validation_csv = config['path_validation_csv']

        if path_test_csv:
            self.path_test_csv = path_test_csv
        else:
            self.path_test_csv = config['path_test_csv']

        if select_testset:
            self.path_validation_csv = self.path_test_csv

        # csv file with the age information for each subject.
        self.path_info_csv = config['path_info_csv']

        if default_processing:
            self.set_default_processing()
        else:
            self.train_intensity_shift = config['train_intensity_shift']
            self.train_intensity_scale = config['train_intensity_scale']
            self.train_intensity_clamp_min = config['train_intensity_clamp_min']
            self.train_intensity_clamp_max = config['train_intensity_clamp_max']
            self.train_intensity_random_scale = config['train_intensity_random_scale']
            self.train_intensity_random_shift = config['train_intensity_random_shift']

            self.val_intensity_shift = config['val_intensity_shift']
            self.val_intensity_scale = config['val_intensity_scale']
            self.val_intensity_clamp_min = config['val_intensity_clamp_min']
            self.val_intensity_clamp_max = config['val_intensity_clamp_max']

            self.augmentation_flip = config['augmentation_flip']
            self.augmentation_scale = config['augmentation_scale']
            self.augmentation_translation = config['augmentation_translation']
            self.augmentation_rotation = config['augmentation_rotation']

    def set_default_processing(self):
        self.train_intensity_shift = 0.0
        self.train_intensity_scale = 1.0
        self.train_intensity_clamp_min = 0.0
        self.train_intensity_clamp_max = 2.0
        self.train_intensity_random_scale = 0.4
        self.train_intensity_random_shift = 0.25

        self.val_intensity_shift = 0.0
        self.val_intensity_scale = 1.0
        self.val_intensity_clamp_min = 0.0
        self.val_intensity_clamp_max = 2.0

        self.augmentation_flip = [0.5, 0.0, 0.0]
        self.augmentation_scale = [0.1, 0.1, 0.1]
        self.augmentation_translation = [10.0, 10.0, 10.0]
        self.augmentation_rotation = [0.1, 0.1, 0.1]

    def crop_pad(self, image):
        image = sitk.Crop(image, [0, 0, 0], [0, 0, 0])
        image = sitk.MirrorPad(image, [0, 0, 0], [0, 0, 0])
        return image

    def post_processing_sitk_train(self, image):
        ssc = ShiftScaleClamp(shift=self.train_intensity_shift,
                              scale=self.train_intensity_scale,
                              clamp_min=self.train_intensity_clamp_min,
                              clamp_max=self.train_intensity_clamp_max,
                              random_scale=self.train_intensity_random_scale,
                              random_shift=self.train_intensity_random_shift)
        return ssc(image)

    def post_processing_sitk_val(self, image):
        ssc = ShiftScaleClamp(shift=self.val_intensity_shift,
                              scale=self.val_intensity_scale,
                              clamp_min=self.val_intensity_clamp_min,
                              clamp_max=self.val_intensity_clamp_max)
        return ssc(image)

    def data_generators(self, dim, post_processing_sitk, transformation, data_sources):
        # Create data generator with processed image samples and corresponding labels.
        data_generator_dict = {}
        data_generator_dict['image'] = ImageGenerator(dim, self.image_size, self.image_spacing,
                                                      name='image',
                                                      post_processing_sitk=post_processing_sitk,
                                                      resample_default_pixel_value=0.0,
                                                      parents=[data_sources['image'], transformation])
        data_generator_dict['age'] = LabelGenerator(name='age', parents=[data_sources['age']])

        return data_generator_dict

    def dataset_train(self):
        # Define training images.
        # csv "<id>, <subj>" -> OrderedDict ('image_id', '<id>'), ('unique_id', '<id><subj>')]
        iterator = IdListIterator(self.path_training_csv, random=self.shuffle_images)

        # Load image and label data.
        sources = self.datasources(self.base_folder, self.path_info_csv, iterator)

        # Data augmentation: spatial transformation.

        transformation_list = [translation.InputCenterToOrigin(self.dim)]

        if self.augmentation_flip:
            transformation_list.append(flip.Random(
                self.dim, self.augmentation_flip))
        
        if self.augmentation_scale:
            transformation_list.append(scale.Random(
                self.dim, self.augmentation_scale))
        
        if self.augmentation_translation:
            transformation_list.append(translation.Random(
                self.dim, self.augmentation_translation))

        if self.augmentation_rotation:
            transformation_list.append(rotation.Random(
                self.dim, self.augmentation_rotation))

        transformation_list.append(translation.OriginToOutputCenter(self.dim,
                                                                    self.image_size,
                                                                    self.image_spacing))

        transformation = composite.Composite(self.dim, transformation_list,
                                            kwparents={'image': sources['image']})

        generators = self.data_generators(self.dim,
                                          self.post_processing_sitk_train,
                                          transformation,
                                          sources)

        dataset = GraphDataset(data_sources=list(sources.values()),
                               data_generators=list(generators.values()),
                               iterator=iterator,
                               debug_image_folder=os.path.join(self.output_folder,
                                                               'debug_train')
                               if self.save_debug_images else None,
                               debug_image_type='gallery')
        return dataset


    def dataset_val(self):
        # Define validation images.
        # csv "<id>, <subj>" -> OrderedDict ('image_id', '<id>'), ('unique_id', '<id><subj>')]
        print(self.path_validation_csv)
        iterator = IdListIterator(self.path_validation_csv)
 
        # Load image and label data.
        sources = self.datasources(self.base_folder, self.path_info_csv, iterator)

        transformation = composite.Composite(self.dim, [translation.InputCenterToOrigin(self.dim),
                                                        translation.OriginToOutputCenter(self.dim,
                                                                                         self.image_size,
                                                                                         self.image_spacing)],
                                             kwparents={'image': sources['image']})

        generators = self.data_generators(self.dim,
                                          self.post_processing_sitk_val,
                                          transformation,
                                          sources)

        dataset = GraphDataset(data_sources=list(sources.values()),
                               data_generators=list(generators.values()),
                               iterator=iterator,
                               debug_image_folder=os.path.join(self.output_folder,
                                                               'debug_val')
                               if self.save_debug_images else None,
                               debug_image_type='single_image')
        ds = dataset.get_next()
        return dataset

    def datasources(self, image_folder, label_csv, iterator):
        datasources_dict = {}
        # Loading sitk image from the following location:
        # image_folder/<file_prefix><id_dict_preprocessing(id_dict['image_id'])><file_suffix><file_ext>
        datasources_dict['image'] = CachedImageDataSource(image_folder,
                                                          file_prefix=self.file_prefix,
                                                          file_suffix=self.file_suffix,
                                                          file_ext=self.file_ext,
                                                          preprocessing=self.crop_pad,
                                                          sitk_pixel_type=sitk.sitkFloat32,
                                                          cache_maxsize=24000,
                                                          name='image',
                                                          parents=[iterator])

        # Uses id_dict['image_id'] as the key for loading the label from a given .csv.
        # 'key_0', 'label_0'
        datasources_dict['age'] = LabelDatasource(label_csv, name='image', parents=[iterator])
        return datasources_dict
