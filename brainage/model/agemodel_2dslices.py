import os
from pathlib import Path

import hydra
import torch
import dotenv
import wandb
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torchvision.models as models
from torch.nn import functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
dotenv.load_dotenv()

from brainage.dataset.dataset2d import SliceDataset

class AgeModel2DSlices(pl.LightningModule):

    def __init__(self,
                 hparams,   
                 train_ds=None,
                 val_ds=None):
        super().__init__()

        # copy over
        self.hparams = hparams
        cfg = OmegaConf.create(hparams)
        self.model_name = cfg.model.name
        self.inputs = cfg.model.inputs or 1
        self.outputs = cfg.model.outputs or 1
        self.pretrained = cfg.model.pretrained or False

        self.learning_rate = cfg.optimizer.learning_rate or 1e-4
        self.weight_decay = cfg.optimizer.weight_decay or 0.0
        self.batch_size = cfg.loader.batch_size or 64
        self.num_workers = cfg.loader.num_workers or 4
        
        #!change
        self.outputs=24

        # load data
        self.train_ds = train_ds
        self.val_ds = val_ds

        # define model
        assert self.model_name in ['resnet18', 'resnet50']
        if self.model_name == 'resnet18':
            self.net = models.resnet18(pretrained=self.pretrained)
            self.net.conv1 = torch.nn.Conv2d(self.inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.net.fc = torch.nn.Linear(in_features=512, out_features=self.outputs, bias=True)
        elif self.model_name == 'resnet50':
            self.net = models.resnet50(pretrained=self.pretrained)
            self.net.conv1 = torch.nn.Conv2d(self.inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.net.fc = torch.nn.Linear(in_features=2048, out_features=self.outputs, bias=True)

    def forward(self, x):
        return self.net(x)
        
    def log_sample_images(self, batch, batch_idxn, n=5):
        samples = batch['data'].detach().cpu().numpy()
        labels =  batch['label'][0].detach().cpu().numpy()
        samples = np.transpose(samples, [0,2,3,1])
        samples = [wandb.Image(samples[i]*255, 
                    caption=f'batch {batch_idxn} age {labels[i]}') 
                    for i in range(n)]
        wandb.log({'samples': samples})

    def training_step(self, batch, batch_idx):
        if self.global_step < 5:
            self.log_sample_images(batch, batch_idx)

        x = batch['data'].float()
        y = batch['label'][0].float()
        y = ((y.clamp(20.0, 80.0)-20.0)//2.5).long()

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        #loss = F.mse_loss(y_hat[:, 0], y)
        logs = {'train_loss': loss.item()}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = batch['label'][0].float()
        y_hat = (torch.argmax(self(x), dim=1)*2.5+20).float()
        return {'val_loss': F.mse_loss(y_hat, y),
                'mse': F.mse_loss(y_hat, y), # y_hat[:, 0]
                'mae': F.l1_loss(y_hat, y)} # y_hat[:, 0]

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['mae'] for x in outputs]).mean()
        avg_mse = torch.stack([x['mse'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'mae': avg_mae, 'mse': avg_mse}
        return {'val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def train_dataloader(self):
        dataset = self.train_ds
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True, pin_memory=True)
        return loader
    
    def val_dataloader(self):
        dataset = self.val_ds
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=False, pin_memory=True)
        return loader

# TODO aleatoric loss
# TODO quantile regression
# TODO data augmentation
# TODO sanity check

@hydra.main(config_path='../../config/config.yaml', strict=False)
def main(cfg):
    print(cfg.pretty())
    wandb_logger = WandbLogger(name=cfg.project.job,
                               project=cfg.project.name,
                               log_model=True)

    # get keys and metadata
    with Path(cfg.dataset.train).open('r') as f:
        train_keys = [l.strip() for l in f.readlines()]
    with Path(cfg.dataset.val).open('r') as f:
        val_keys = [l.strip() for l in f.readlines()]
    info_df = pd.read_feather(cfg.dataset.info)
    info_df = info_df[(info_df['slice']>55) & (info_df['slice']<57)]
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

    trainer = Trainer(logger=wandb_logger,
                      **OmegaConf.to_container(cfg.trainer))
    trainer.fit(model)  

if __name__ == '__main__':
    main()