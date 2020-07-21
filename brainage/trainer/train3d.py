import logging
import numpy as np
import torch
import argparse
import operator
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from configargparse import ArgumentParser
from pathlib import Path

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging.neptune import NeptuneLogger
from pytorch_lightning import loggers

from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.abstract_transforms import Compose

from dl_downstream_tasks.dataset import BrainDataset, BrainPatchDataset
from dl_downstream_tasks.utils import _LOG_LEVEL_STRINGS, _log_level_string_to_int
from dl_downstream_tasks.models.resnet3d import generate_model


class AgeModel(pl.LightningModule):

    def __init__(self,
                hparams,
                train_ds=None,
                val_ds=None,
                test_ds=None,
                save_path=None):
        super().__init__()

        # copy over
        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size 
        self.num_workers = hparams.num_workers
        self.norm_type = hparams.norm_type
        self.weight_decay = hparams.weight_decay
        self.log_image_interval = hparams.log_image_interval
        self.norm_type = hparams.norm_type
        self.outputs = hparams.outputs
        self.hparams = hparams
        self.save_path = save_path

        # load data
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        # define model
        self.net = generate_model(model_depth=hparams.model_depth, 
                                n_input_channels=1, 
                                n_classes=self.outputs,
                                norm_type=self.norm_type)

    def forward(self, x):
        return self.net(x)
        
    def training_step(self, batch, batch_idx):
        x = batch["data"]
        y = batch["label"].float()
        y_hat = self(x)
        loss = F.mse_loss(y_hat[:, 0], y)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def log_sample(self, batch):
        for img in batch["data"]:
            content = img[0, :, img.size()[1]//2, :].cpu().numpy()
            content = (content - np.min(content))/(np.max(content) - np.min(content))*255.0
            self.logger[1].experiment.log_image("sagittal", content)
            content = img[0, :, :, img.size()[2]//2].cpu().numpy()
            content = (content - np.min(content))/(np.max(content) - np.min(content))*255.0
            self.logger[1].experiment.log_image("axial", content)

    def validation_step(self, batch, batch_idx):
        x = batch["data"]
        y = batch["label"].float()
        y_hat = self(x)

        # log images for the very first batch
        if self.current_epoch == 0 and batch_idx < 3:
            self.log_sample(batch)

        return {"val_loss": F.mse_loss(y_hat[:, 0], y),
                "mse": F.mse_loss(y_hat[:, 0], y),
                "mae": F.l1_loss(y_hat[:, 0], y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in outputs]).mean()
        avg_mse = torch.stack([x["mse"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "mae": avg_mae, "mse": avg_mse}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x = batch["data"]
        y = batch["label"].float()
        key = batch["key"]
        y_hat = self(x)
        return {"val_loss": F.mse_loss(y_hat[:, 0], y),
                "mse": F.mse_loss(y_hat[:, 0], y),
                "mae": F.l1_loss(y_hat[:, 0], y),
                "key": key,
                "label": y.cpu().numpy(),
                "prediction": y_hat.cpu().numpy()}

    def test_epoch_end(self, outputs):
        labels = np.concatenate([x["label"] for x in outputs])
        predictions = np.concatenate([x["prediction"][:, 0] for x in outputs])
        keys = reduce(operator.add, [x["key"] for x in outputs])
        result_dict = {"keys": keys, "labels": labels, "predictions": predictions}
        df = pd.DataFrame.from_dict(result_dict)
        df.to_csv(self.save_path)
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in outputs]).mean()
        avg_mse = torch.stack([x["mse"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "mae": avg_mae, "mse": avg_mse}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

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
    
    def test_dataloader(self):
        dataset = self.test_ds
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=False, pin_memory = True)
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_depth", type=int, default=18)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--norm_type", type=str, choices=["BN", "PN", "IN", "NO"], default="PN")
        parser.add_argument("--log_image_interval", type=int, default=1000)
        parser.add_argument("--outputs", type=int, default=1)
        return parser

def main():
    # arg parsing
    parser = ArgumentParser()
    # add experiment level args
    parser.add_argument('-c', '--config', is_config_file=True, default="/home/raheppt1/projects/MriAnonymization/dl_downstream_tasks/experiments/n_age_10.yaml")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--experiment_name', type=str, default="age_prediction")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_group', type=str, default="original")
    parser.add_argument('--info', type=str)
    parser.add_argument('--infocolumn', type=str, default="label")
    parser.add_argument('--train_set', type=str)
    parser.add_argument('--val_set', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--data_augmentation', action="store_true")
    parser.add_argument('--crop_size', type=int, nargs='+', default=[64, 96, 96])
    parser.add_argument('--crop_margins', type=int, nargs='+', default=[25, 5, 5])
    parser.add_argument('--gamma_range', type=float, nargs='+', default=[0.7, 1.3])
    parser.add_argument('--neptune_project', type=str, default='lab-midas/kth-age')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--log_level', default='INFO', type=_log_level_string_to_int, nargs='?',
                        help='Set the logging output level. {0}'.format(_LOG_LEVEL_STRINGS))
    # add model specific args
    parser = AgeModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # set seeds
    torch.manual_seed(hparams.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # False
    np.random.seed(hparams.seed)

    # logging
    # in order to use neptune logging:
    # export NEPTUNE_API_TOKEN = '...' !!! 
    logging.basicConfig(level=hparams.log_level)
    source_files = [__file__]
    if hparams.config:
        source_files.append(hparams.config)
    neptune_logger = NeptuneLogger(
        project_name=hparams.neptune_project,
        params=vars(hparams),
        experiment_name=hparams.experiment_name,
        tags=[hparams.experiment_name],
        upload_source_files=source_files)
    tb_logger = loggers.TensorBoardLogger(hparams.log_dir)

    # define data augmentation 
    hparams.crop_size = np.array(hparams.crop_size)
    hparams.crop_margins = np.array(hparams.crop_margins)
    #val_transform = CenterCropTransform(crop_size=hparams.crop_size, data_key="data")
    val_transform = None
    transforms = [
        #RandomCropTransform(crop_size=hparams.crop_size, data_key="data", margins=hparams.crop_margins),
        GammaTransform(gamma_range=hparams.gamma_range, data_key="data"),
        MirrorTransform(axes=[0], data_key="data",),]
    train_transform = Compose(transforms)
    if not hparams.data_augmentation:
        train_transform = val_transform

    # loading data to memory ...
    train_ds = BrainPatchDataset(data=hparams.data_path,
                        fkeys=hparams.train_set,
                        info=hparams.info,
                        group=hparams.data_group,
                        column=hparams.infocolumn,
                        patch_size=(48, 48, 48),
                        preload=hparams.preload,
                        transform=train_transform)

    val_ds = BrainPatchDataset(data=hparams.data_path,
                        fkeys=hparams.val_set,
                        info=hparams.info,
                        column=hparams.infocolumn,
                        patch_size=(48, 48, 48),
                        group=hparams.data_group,
                        preload=hparams.preload,
                        transform=val_transform)

    # initialize model
    model = AgeModel(hparams,
                    train_ds,
                    val_ds)

    # train model
    trainer = Trainer(gpus=hparams.gpus,
                      max_epochs=hparams.max_epochs,
                      default_save_path=hparams.model_dir,
                      logger=[tb_logger, neptune_logger])
    trainer.fit(model)


def test():
     # arg parsing
    parser = ArgumentParser()
    # add experiment level args
    parser.add_argument('-c', '--config', is_config_file=True, default=None)
    parser.add_argument('--data_path', type=str, default="/media/datastore1/thepp/ADNI_EVAL/msggan.h5")#"/home/thepp/data/ADNI/MRI_128.h5")
    parser.add_argument('--data_group', type=str, default="msggan")
    parser.add_argument('--info', type=str, default="/home/thepp/data/info/ADNI/ADNI_age_all.csv")
    parser.add_argument('--infocolumn', type=str, default="label")
    parser.add_argument('--test_set', type=str, default="/home/thepp/data/info/ADNI/test_msggan.dat")
    parser.add_argument('--save_dir', type=str, default="/home/thepp/results")
    parser.add_argument('--crop_size', type=int, nargs='+', default=[64, 96, 96])
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--log_level', default='INFO', type=_log_level_string_to_int, nargs='?',
                        help='Set the logging output level. {0}'.format(_LOG_LEVEL_STRINGS))
    parser.add_argument('--checkpoint_path', type=str, default="/home/thepp/data/models/default_age_test/version_20_KTHAG-132/checkpoints/epoch=196.ckpt")
    # add model specific args
    parser = AgeModel.add_model_specific_args(parser)    
    hparams = parser.parse_args()

    # define data augmentation 
    hparams.crop_size = np.array(hparams.crop_size)
    val_transform = CenterCropTransform(crop_size=hparams.crop_size, data_key="data")
   
    # loading data to memory ...
    test_ds = BrainDataset(data=hparams.data_path,
                        fkeys=hparams.test_set,
                        info=hparams.info,
                        column=hparams.infocolumn,
                        group=hparams.data_group,
                        transform=val_transform,
                        preload=hparams.preload)

    # initialize model
    model = AgeModel.load_from_checkpoint(checkpoint_path=hparams.checkpoint_path,
                                       map_location=None,
                                       test_ds=test_ds,
                                       save_path=Path(hparams.save_dir).joinpath(hparams.data_group + ".csv"))
    # test model
    trainer = Trainer(gpus=hparams.gpus)
    trainer.test(model)


def grad_cam():

     # arg parsing
    parser = ArgumentParser()
    # add experiment level args
    parser.add_argument('-c', '--config', is_config_file=True, default=None)
    parser.add_argument('--data_path', type=str, default="/media/datastore1/thepp/ADNI_EVAL/original.h5")
    parser.add_argument('--data_group', type=str, default="original")
    parser.add_argument('--info', type=str, default="/home/thepp/data/info/ADNI/ADNI_age_all.csv")
    parser.add_argument('--infocolumn', type=str, default="label")
    parser.add_argument('--test_set', type=str, default="/home/thepp/data/info/ADNI/test.dat")
    parser.add_argument('--crop_size', type=int, nargs='+', default=[64, 96, 96])
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--log_level', default='INFO', type=_log_level_string_to_int, nargs='?',
                        help='Set the logging output level. {0}'.format(_LOG_LEVEL_STRINGS))
    parser.add_argument('--checkpoint_path', type=str, default="/home/thepp/data/models/default_age_test/version_20_KTHAG-132/checkpoints/epoch=196.ckpt")
    # add model specific args
    parser = AgeModel.add_model_specific_args(parser)    
    hparams = parser.parse_args()

    # define data augmentation 
    hparams.crop_size = np.array(hparams.crop_size)
    val_transform = CenterCropTransform(crop_size=hparams.crop_size, data_key="data")
   
    print("Loading checkpoint ...")
    checkpoint = torch.load(hparams.checkpoint_path)
    a = checkpoint
    return
    # loading data to memory ...
    test_ds = BrainDataset(data=hparams.data_path,
                        fkeys=hparams.test_set,
                        info=hparams.info,
                        column=hparams.infocolumn,
                        group=hparams.data_group,
                        transform=val_transform)

    

if __name__ == "__main__":
    main()
    #import h5py
    #with h5py.File("/home/thepp/deface.h5", 'r') as fhandle:
    #    [print(key) for key in fhandle]
