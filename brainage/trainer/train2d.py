from pathlib import Path

import hydra
import dotenv
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from brainage.dataset.dataset2d import SliceDataset
from brainage.model.model2d import AgeModel2DSlices

# TODO patches
# TODO chunks
# TODO data augmentation
# TODO seeds/deterministic

@hydra.main(config_path='../../config/config.yaml', strict=False)
def main(cfg):
    print(cfg.pretty())
    # logging
    wandb_logger = WandbLogger(name=cfg.project.job,
                               project=cfg.project.name,
                               log_model=True)
    print(wandb_logger.experiment.dir)
    # get keys and metadata
    with Path(cfg.dataset.train).open('r') as f:
        train_keys = [l.strip() for l in f.readlines()]
    with Path(cfg.dataset.val).open('r') as f:
        val_keys = [l.strip() for l in f.readlines()]
    info_df = pd.read_feather(cfg.dataset.info)

    # slice selection
    if cfg.dataset.slicing == 'range':
        slice_selection = np.arange(start=cfg.dataset.slices[0], 
                                    stop=cfg.dataset.slices[1],
                                    step=cfg.dataset.slices[2])
        info_df = info_df[info_df['slice'].isin(slice_selection)]
    elif cfg.dataset.slicing == 'list':
        slice_selection = cfg.dataset.slices
        info_df = info_df[info_df['slice'].isin(slice_selection)]

    info_train =  info_df[info_df.key.isin(train_keys)]
    info_val = info_df[info_df.key.isin(val_keys)]

    ds_train = SliceDataset(cfg.dataset.data, 
                            info=info_train, 
                            preload=cfg.dataset.preload or False,
                            zoom=cfg.dataset.zoom)
    ds_val = SliceDataset(cfg.dataset.data, 
                          info=info_val, 
                          preload=cfg.dataset.preload or False,
                          zoom=cfg.dataset.zoom)

    model = AgeModel2DSlices(OmegaConf.to_container(cfg, resolve=True),
                             ds_train, ds_val)

    # train model
    trainer = Trainer(logger=wandb_logger,
                      **OmegaConf.to_container(cfg.trainer))
    trainer.fit(model)  

if __name__ == '__main__':
    main()