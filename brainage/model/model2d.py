import os
from pathlib import Path

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

from brainage.model.loss import class_loss, l2_loss
from brainage.dataset.dataset2d import SliceDataset

# TODO upload checkpoints
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
        self.use_position = cfg.model.position or False
        self.loss_type = cfg.model.loss or 'l2'
        self.heteroscedastic = cfg.model.heteroscedastic or False # only relevant for l2
        # classification labels (only relevant for ce)
        self.label_range = cfg.model.label_range or [20,80] # only relevant for ce
        self.label_step = cfg.model.label_step or 2.5 # only relevant for ce

        self.learning_rate = cfg.optimizer.learning_rate or 1e-4
        self.weight_decay = cfg.optimizer.weight_decay or 0.0
        self.batch_size = cfg.loader.batch_size or 64
        self.num_workers = cfg.loader.num_workers or 4
        
        # datasets
        self.train_ds = train_ds
        self.val_ds = val_ds
        # grad cam 
        self.gradient = None

        # select loss criterion
        if self.loss_type == 'ce':
            self.loss_criterion = class_loss(label_range=self.label_range,
                                             label_step=self.label_step)
            self.outputs = int((self.label_range[1]-self.label_range[0])//self.label_step)
        elif self.loss_type == 'l2':
            self.loss_criterion = l2_loss(heteroscedastic=self.heteroscedastic)

        # define model
        add_features = 1 if self.use_position else 0
        assert self.model_name in ['resnet18', 'resnet50']
        if self.model_name == 'resnet18':
            net = models.resnet18(pretrained=self.pretrained)
            net.conv1 = torch.nn.Conv2d(self.inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            net.fc = torch.nn.Linear(in_features=512+add_features, out_features=self.outputs, bias=True)
        elif self.model_name == 'resnet50':
            net = models.resnet50(pretrained=self.pretrained)
            net.conv1 = torch.nn.Conv2d(self.inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            net.fc = torch.nn.Linear(in_features=2048+add_features, out_features=self.outputs, bias=True)
        # split model
        self.features_conv = torch.nn.Sequential(*list(net.children())[:-2])
        self.avgpool = net.avgpool
        self.fc = net.fc

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)

    def forward(self, x, pos=None, hook=False):
        # extract features
        y = self.features_conv(x)
        # register the hook
        if hook:
            y.register_hook(self.activations_hook)
        # pooling and flatten
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        # if given, include slice position
        if self.use_position: 
            y = torch.cat([y, torch.unsqueeze(pos, dim=1)], dim=1)
        # regression
        y = self.fc(y)
        return y
        
    def log_samples(self, batch, batch_idxn, n=5):
        samples = batch['data'].detach().cpu().numpy()
        labels =  batch['label'][0].detach().cpu().numpy()
        slices =  batch['position'].detach().cpu().numpy()
        samples = np.transpose(samples, [0,2,3,1])
        samples = [wandb.Image(samples[i]*255, 
                    caption=f'batch {batch_idxn} age {labels[i]} slice {slices[i]}') 
                    for i in range(n)]
        wandb.log({'samples': samples})

    def training_step(self, batch, batch_idx):
        # logging
        if self.global_step < 5:
            self.log_samples(batch, batch_idx)

        x = batch['data'].float()
        y = batch['label'][0].float()
        pos = batch['position'].float()
        y_hat = self(x, pos=pos)
        loss, y_pred = self.loss_criterion(y_hat, y)
  
        logs = {'train_loss': loss.item()}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = batch['label'][0].float()
        pos = batch['position'].float()
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

