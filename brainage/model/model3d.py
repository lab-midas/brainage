import logging
from pathlib import Path

import wandb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from brainage.model.loss import l2_loss
from brainage.model.architecture.resnet3d import generate_model
from brainage.model.architecture.simple import SimpleCNN


class AgeModel3DVolume(pl.LightningModule):

    def __init__(self,
                hparams,
                train_ds=None,
                val_ds=None):
        super().__init__()

        # copy over
        self.hparams = hparams
        cfg = OmegaConf.create(hparams)
        self.model_depth = cfg.model.depth or 18
        self.inputs = cfg.model.inputs or 1
        self.outputs = cfg.model.outputs or 1
        self.use_position = cfg.model.position or False
        self.loss_type = cfg.model.loss or 'l2'
        self.heteroscedastic = cfg.model.heteroscedastic or False
        self.norm_type = cfg.model.norm or 'IN'
        self.learning_rate = cfg.optimizer.learning_rate or 1e-4
        self.weight_decay = cfg.optimizer.weight_decay or 0.0
        self.batch_size = cfg.loader.batch_size or 8
        self.num_workers = cfg.loader.num_workers or 4
        self.use_layer = cfg.model.use_layer or [1,1,1,1]
        self.strides = cfg.model.strides
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.no_max_pool = cfg.model.no_max_pool or False

        if self.loss_type == 'l2':
            self.loss_criterion = l2_loss(heteroscedastic=self.heteroscedastic)

        self.net = generate_model(model_depth=self.model_depth, 
                                n_input_channels=self.inputs , 
                                n_classes=self.outputs,
                                norm_type=self.norm_type, 
                                use_layer=self.use_layer,
                                strides=self.strides,
                                no_max_pool=self.no_max_pool,
                                use_position=False)

    def forward(self, x, pos=None, hook=False):
        return self.net(x, pos, hook)
    
    def gradcam(self, x, pos=None, channel=0):
        # compute gradients
        y = self(x, pos=pos, hook=True)
        y[0, channel].backward()
        gradients = self.net.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = self.net.get_activations(x).detach()
        # weight the channels by corresponding gradients
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        return y, heatmap

    def log_samples(self, batch, batch_idx):
        samples = []
        for img, label in zip(batch['data'], batch['label']):
            img = img[0, :, img.size()[1]//2, :].cpu().numpy()*255.0
            samples.append(wandb.Image(img, caption=f'batch {batch_idx} age {label}'))
        wandb.log({'samples': samples})

    def training_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = batch['label'].float()
        pos = batch['position'].float() if self.use_position else None
        y_hat = self(x, pos=pos)
        loss, y_pred = self.loss_criterion(y_hat, y)

        if self.global_step < 5:
            self.log_samples(batch, batch_idx)

        logs = {'train_loss': loss.item()}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = batch['label'].float()
        pos = batch['position'].float() if self.use_position else None
        y_hat = self(x, pos=pos)
        loss, y_pred = self.loss_criterion(y_hat, y)
        return {'val_loss': loss,
                'mse': F.mse_loss(y_pred, y), 
                'mae': F.l1_loss(y_pred, y)}  

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['mae'] for x in outputs]).mean()
        avg_mse = torch.stack([x['mse'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'mae': avg_mae, 'mse': avg_mse}
        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

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
