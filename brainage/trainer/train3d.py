import os
import time
from pathlib import Path

import torch
import hydra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import NeptuneLogger
from dotenv import load_dotenv

from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.abstract_transforms import Compose

from brainage.model.model3d import AgeModel3DVolume
from brainage.dataset.dataset3d import BrainDataset, BrainPatchDataset

load_dotenv()

config = os.getenv('CONFIG')
@hydra.main(config_path=config, strict=False)
def main(cfg):
    # config
    project = cfg.project.name
    job = cfg.project.job
    data_path = cfg.dataset.data
    data_group = cfg.dataset.group
    info = cfg.dataset.info
    infocolumn = cfg.dataset.column
    train_set = cfg.dataset.train
    val_set = cfg.dataset.val
    debug_set = cfg.dataset.debug or None
    if debug_set:
        train_set = debug_set
        val_set = debug_set
    patch_size = cfg.dataset.patch_size
    data_mode = cfg.dataset.mode 
    data_augmentation = cfg.dataset.data_augmentation
    crop_size = np.array(cfg.dataset.crop_size)
    crop_margins = np.array(cfg.dataset.crop_margins)
    gamma_range = cfg.dataset.gamma_range 
    preload = cfg.dataset.preload
    seed = cfg.project.seed or 42
    seed_everything(seed)
    ts = time.gmtime()
    job_id = time.strftime("%Y-%m-%d-%H-%M-%S", ts) + f'-{cfg.dataset.fold}'

    # logging
    wandb_logger = WandbLogger(name=f'{job}-{job_id}', project=project, log_model=True)
    #neptune_logger = NeptuneLogger(project_name=f'lab-midas/{project}',
    #                               params=OmegaConf.to_container(cfg, resolve=True),
    #                               experiment_name=f'{job}-{job_id}',
    #                               tags=[job])
    
    # get keys and metadata
    train_keys = [l.strip() for l in Path(train_set).open().readlines()]
    val_keys = [l.strip() for l in Path(val_set).open().readlines()]

    assert data_mode in ['patchwise', 'volume']
    if data_mode == 'patchwise':
        val_transform = None
        transforms = [
            GammaTransform(gamma_range=gamma_range, data_key='data'),
            MirrorTransform(axes=[0], data_key='data',),]
        train_transform = Compose(transforms)
        if data_augmentation:
            train_transform = val_transform

        ds_train = BrainPatchDataset(data=data_path,
                            keys=train_keys,
                            info=info,
                            group=data_group,
                            column=infocolumn,
                            patch_size=patch_size,
                            preload=preload,
                            transform=train_transform)

        ds_val = BrainPatchDataset(data=data_path,
                            keys=val_keys, 
                            info=info,
                            column=infocolumn,
                            patch_size=patch_size,
                            group=data_group,
                            preload=preload,
                            transform=val_transform) 

    elif data_mode == 'volume':
        if np.any(crop_size):
            val_transform = CenterCropTransform(crop_size=crop_size, data_key='data')
        else:
            val_transform = None

        transforms = []
        if np.any(crop_size):
            transforms.append(RandomCropTransform(crop_size=crop_size, data_key='data', margins=crop_margins))
        transforms.append(GammaTransform(gamma_range=gamma_range, data_key='data'))
        transforms.append(MirrorTransform(axes=[0], data_key='data'))
        train_transform = Compose(transforms)
        if not data_augmentation:
            train_transform = val_transform

        ds_train = BrainDataset(data=data_path,
                            keys=train_keys,
                            info=info,
                            group=data_group,
                            column=infocolumn,
                            preload=preload,
                            transform=train_transform)

        ds_val = BrainDataset(data=data_path,
                            keys=val_keys, 
                            info=info,
                            column=infocolumn,
                            group=data_group,
                            preload=preload,
                            transform=val_transform)

    model = AgeModel3DVolume(OmegaConf.to_container(cfg, resolve=True),
                     ds_train, ds_val)

    trainer = Trainer(logger=[wandb_logger],
                      **OmegaConf.to_container(cfg.trainer))
    trainer.fit(model)


if __name__ == '__main__':
    main()
