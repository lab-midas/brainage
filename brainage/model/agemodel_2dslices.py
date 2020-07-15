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
                 train_ds,
                 val_ds):
        super().__init__()

        # copy over
        self.hparams = hparams
        self.learning_rate = hparams['optimizer']['learning_rate']
        self.weight_decay = hparams['optimizer']['weight_decay']
        self.batch_size = hparams['loader']['batch_size'] 
        self.num_workers = hparams['loader']['num_workers']
        self.inputs = hparams['model']['inputs']
        self.outputs = hparams['model']['outputs']
        
        # load data
        self.train_ds = train_ds
        self.val_ds = val_ds

        # define model
        features = {'resnet18': 512,
                    'resnet50': 2048}

        self.net = models.resnet18()
        self.net.conv1 = torch.nn.Conv2d(self.inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = torch.nn.Linear(in_features=512, out_features=self.outputs, bias=True)
        #self.net = models.resnet50()
        #self.net.conv1 = torch.nn.Conv2d(self.inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #self.net.fc = torch.nn.Linear(in_features=2048, out_features=self.outputs, bias=True)

    def forward(self, x):
        return self.net(x)
        
    def log_sample_images(self, batch, batch_idxn, n=5):
        sample = batch['data'].detach().cpu().numpy()
        label =  batch['label'][0].detach().cpu().numpy()
        sample = np.transpose(sample, [0,2,3,1])
        wandb.log({'samples': [wandb.Image(sample[i]*255, 
                                           caption=f'batch {batch_idxn} age {label[i]}') 
                                for i in range(n)]})

    def training_step(self, batch, batch_idx):
        if self.global_step < 5:
            self.log_sample_images(batch, batch_idx)

        x = batch['data'].float()
        y = batch['label'][0].float()
        y_hat = self(x)
        loss = F.mse_loss(y_hat[:, 0], y)
        logs = {'train_loss': loss.item()}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = batch['label'][0].float()
        y_hat = self(x)
        return {'val_loss': F.mse_loss(y_hat[:, 0], y),
                'mse': F.mse_loss(y_hat[:, 0], y),
                'mae': F.l1_loss(y_hat[:, 0], y)}

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

# TODO pretraining 
# TODO gradcam
# TODO aleatoric loss
# TODO quantile regression
# TODO data augmentation
# TODO plot sample images
# TODO sanity check

@hydra.main(config_path='../../config/config.yaml', strict=False)
def main(cfg):
    print(cfg.pretty())
    wandb_logger = WandbLogger(name=cfg.project.job,
                               project=cfg.project.name)

    with Path(cfg.dataset.train).open('r') as f:
        train_keys = [l.strip() for l in f.readlines()]

    with Path(cfg.dataset.val).open('r') as f:
        val_keys = [l.strip() for l in f.readlines()]

    info_df = pd.read_feather(cfg.dataset.info)
    shape = cfg.dataset.shape
    data = np.memmap(cfg.dataset.data,
                    mode='r',
                    dtype=np.float16,
                    shape=(len(info_df), shape[0], shape[1]))
    
    info_df = info_df[(info_df['slice']>70) & (info_df['slice']<90)]
    info_train =  info_df[info_df.key.isin(train_keys)]
    ds_train = SliceDataset(np.array(data[info_train.index]), info_train)
    info_val = info_df[info_df.key.isin(val_keys)]
    ds_val = SliceDataset(np.array(data[info_val.index]), info_val)

    model = AgeModel2DSlices(OmegaConf.to_container(cfg, resolve=True),
                             ds_train, ds_val)

    trainer = Trainer(logger=wandb_logger, 
                      **OmegaConf.to_container(cfg.trainer))
    trainer.fit(model)  

if __name__ == '__main__':
    main()